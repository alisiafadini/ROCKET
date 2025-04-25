"""
LLG targets for cryoEM data
"""

import numpy as np
import torch
from SFC_Torch import SFcalculator

from rocket import utils as rk_utils
from rocket.cryo import utils as cryo_utils


class LLGloss(torch.nn.Module):
    """
    Object_oriented interface to calculate LLG loss

    # Initialization, only have to do it once
    sfc = llg_sf.initial_SFC(...)
    llgloss = LLGloss(sfc, tng_file, device)
    Ecalc = llgloss.compute_Ecalc(xyz_orth)
    TODO: sigmaA calculation/initialization

    # Loss calculation for each step
    loss = -llgloss(xyz_orth, bin_labels=[1,2,3], num_batch=10, sub_ratio=0.3)

    resol_min, resol_max: None | float
        resolution cutoff for the used miller index.
        Will use resol_min <= dHKL <= resol_max

    TODO:
    Currently the initialization needs Eobs, Eps, Centric, Dobs, Feff, Bin_labels.
    We load them through tng_data.
    Later we should be able to calculate everything from SFcalculator,
    all necesary information is ready there.

    """

    def __init__(
        self,
        sfc: SFcalculator,
        mtz_file: str,
        resol_min=None,
        resol_max=None,
    ) -> None:
        super().__init__()
        self.sfc = sfc
        self.device = sfc.device
        data_dict = cryo_utils.load_tng_data(mtz_file, device=self.device)

        self.register_buffer("Emean", data_dict["Emean"])
        self.Emean: torch.Tensor
        self.register_buffer("PHIEmean", data_dict["PHIEmean"])
        self.PHIEmean: torch.Tensor
        self.register_buffer("Dobs", data_dict["Dobs"])
        self.Dobs: torch.Tensor

        if resol_min is None:
            resol_min = min(self.sfc.dHKL)

        if resol_max is None:
            resol_max = max(self.sfc.dHKL)

        resol_bool = (self.sfc.dHKL >= (resol_min - 1e-4)) & (
            self.sfc.dHKL <= (resol_max + 1e-4)
        )
        self.working_set = resol_bool

    def assign_sigmaAs_sfc(self, Ec_HKL):
        """
        This function is copied from SFC, compute sigmaA using corr(Eo, Ec)

        # TODO: make sure the correlation works for complex number

        Args:
            Ec_HKL (torch.tensor) : [N_HKL]

        Returns:
            sigmaAs [N_bins]
        """

        Ecalc = Ec_HKL.abs().detach().clone()
        sigmaAs = torch.zeros(self.sfc.n_bins, dtype=torch.float32, device=self.device)
        for i in range(self.sfc.n_bins):
            index_i = self.sfc.bins == i
            Eobs_i = self.Emean[index_i].square()  # [N_subset]
            Ecalc_i = Ecalc[index_i].square()  # [N_subset]
            # Compute correlation coefficient
            Eoi_centered = Eobs_i - Eobs_i.mean()
            Eci_centered = Ecalc_i - torch.mean(Ecalc_i, dim=-1, keepdims=True)
            Covi = (Eci_centered @ Eoi_centered) / (Eoi_centered.shape[0] - 1)
            Eoi_std = torch.std(Eoi_centered, correction=0)
            Eci_std = torch.std(Eci_centered, dim=-1, correction=0)
            sigmaAs[i] = (Covi / (Eoi_std * Eci_std)).clamp(min=0.001, max=0.999).sqrt()
        return sigmaAs

    def assign_sigmaAs_read(self, Ec_HKL):
        """
        This function is copied from SFC, compute sigmaA using corr(Eo, Ec)

        # TODO: make sure the correlation works for complex number

        Args:
            Ec_HKL (torch.tensor) : [N_HKL]

        Returns:
            sigmaAs [N_bins]
        """

        # Ecalc = Ec_HKL.abs().detach().clone()
        Ecalc = Ec_HKL.detach().clone()

        oversampling_factor = 4
        Eobs_amp = self.Emean
        Eobs_phi = self.PHIEmean

        PI_on_180 = 0.017453292519943295
        Ecalc_amp = torch.abs(Ecalc)
        Ecalc_phi = torch.angle(Ecalc) / PI_on_180

        sigmaAs = cryo_utils.sigmaA_from_model_in_map(
            rk_utils.assert_numpy(Eobs_amp),
            rk_utils.assert_numpy(Eobs_phi),
            rk_utils.assert_numpy(self.Dobs),
            rk_utils.assert_numpy(Ecalc_amp),
            rk_utils.assert_numpy(Ecalc_phi),
            self.sfc,
            oversampling_factor,
            self.sfc.dHKL,
            plot=False,
        )

        return sigmaAs

    def freeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(False) for sigmaA in self.sigmaAs]

    def unfreeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(True) for sigmaA in self.sigmaAs]

    def compute_Ecalc(
        self,
        xyz_orth,
        update_scales=False,
        scale_steps=10,
        scale_initialize=False,
    ) -> torch.Tensor:
        self.sfc.calc_fprotein(atoms_position_tensor=xyz_orth)

        if update_scales:
            # We used Emean to override the Fo attribute in SFC,
            # so the scales are optimized to match Ep and Emean
            self.sfc.get_scales_adam(
                lr=0.01,
                n_steps=scale_steps,
                sub_ratio=0.7,
                initialize=scale_initialize,
            )

        # We have normalized the Ep above, and the scales are optimized to match Emean,
        # so we what we got is already Ecalc
        # Note @ Aug 26 by MH, do scaling first, then normalizationg
        # Fc_HKL = self.sfc.calc_ftotal()
        # replace with its normalized Ep
        # Ec_HKL = self.sfc.calc_Ec(Fc_HKL)

        ## testing
        Ep = self.sfc.calc_Ec(self.sfc.Fprotein_HKL)
        self.sfc.Fprotein_HKL = Ep
        Fc_HKL = self.sfc.calc_ftotal()
        Ec_HKL = self.sfc.calc_Ec(Fc_HKL)
        self.sigmaAs = self.assign_sigmaAs_read(Ec_HKL)

        return Ec_HKL

    def forward(
        self,
        xyz_ort: torch.Tensor,
        bin_labels=None,
        num_batch=1,
        sub_ratio=1.0,
        update_scales=False,
    ):
        """
        TODO: Use rfree label in the LLG calculation
        Args:
            xyz_orth: torch.Tensor, [N_atoms, 3] in angstroms
                Orthogonal coordinates of proteins, coming from AF2 model, send to SFC

            bin_labels: None or List[int]
                Labels of bins used in the loss calculation.
                If None, will use the whole miller indices.
                Serve as a proxy for resolution selection

            num_batch: int
                Number of batches

            sub_ratio: float between 0.0 and 1.0
                Fraction of mini-batch sampling over all miller indices,
                e.g. 0.3 meaning each batch sample 30% of miller indices

        """

        Ecalc = self.compute_Ecalc(
            xyz_ort,
            update_scales=update_scales,
        )
        llg = 0.0

        if bin_labels is None:
            bin_labels = self.sfc.n_bins

        # for i, label in enumerate(bin_labels):
        for i, label in enumerate(range(bin_labels)):
            index_i = self.sfc.bins[self.working_set] == label
            Ecalc_i = torch.abs(Ecalc[self.working_set][index_i])
            Ecalc_phi_i = torch.angle(Ecalc[self.working_set][index_i])
            Eob_i = self.Emean[self.working_set][index_i]
            Eob_phi_i = self.PHIEmean[self.working_set][index_i] * (torch.pi / 180)
            Dobs_i = self.Dobs[self.working_set][index_i]

            sigmaA_i = self.sigmaAs[int(i)]
            for _ in range(num_batch):
                sub_boolean_mask = np.random.rand(len(Eob_i)) < sub_ratio
                llg_ij = cryo_utils.llgcryo_calculate(
                    Eob_i[sub_boolean_mask],
                    Eob_phi_i[sub_boolean_mask],
                    Ecalc_i[sub_boolean_mask],
                    Ecalc_phi_i[sub_boolean_mask],
                    sigmaA_i,
                    Dobs_i[sub_boolean_mask],
                ).sum()
                llg = llg + llg_ij
        return llg
