"""
LLG targets for cryoEM data
"""

import time
import torch
import numpy as np
from SFC_Torch import SFcalculator
from rocket.cryo import utils as cryo_utils
from rocket.cryo import structurefactors as cryo_sf
from rocket import utils
from functools import partial


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
        resolution cutoff for the used miller index. Will use resol_min <= dHKL <= resol_max

    TODO:
    Currently the initialization needs inputs like Eobs, Eps, Centric, Dobs, Feff, Bin_labels. We do so by loading the tng_data.
    Later we should be able to calculate everything from SFcalculator, all necesary information is ready there.

    """

    def __init__(
        self,
        sfc: SFcalculator,
        mtz_file: str,
        device: torch.device,
        resol_min=None,
        resol_max=None,
    ) -> None:
        super().__init__()
        self.sfc = sfc
        self.device = device
        data_dict = cryo_utils.load_tng_data(mtz_file, device=device)

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

    def assign_sigmaAs(self, Ecalc, subset="working", requires_grad=True):
        # here can call the function from cctbx
        ...

    def freeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(False) for sigmaA in self.sigmaAs]

    def unfreeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(True) for sigmaA in self.sigmaAs]

    def compute_Ecalc(
        self,
        xyz_orth,
        solvent=True,
        return_Fc=False,
        update_scales=False,
        scale_steps=10,
        scale_initialize=False,
        added_chain_HKL=None,
        added_chain_asu=None,
    ) -> torch.Tensor:
        self.sfc.calc_fprotein(atoms_position_tensor=xyz_orth)

        if added_chain_HKL is not None:
            self.sfc.Fprotein_HKL = self.sfc.Fprotein_HKL + added_chain_HKL
            self.sfc.Fprotein_asu = self.sfc.Fprotein_asu + added_chain_asu

        if solvent:
            self.sfc.calc_fsolvent()
            if update_scales:
                self.sfc.get_scales_adam(
                    lr=0.01,
                    n_steps=scale_steps,
                    sub_ratio=0.7,
                    initialize=scale_initialize,
                )
            Fc = self.sfc.calc_ftotal()
        else:
            # MH note: we need scales here, even without solvent contribution
            self.sfc.Fmask_HKL = torch.zeros_like(self.sfc.Fprotein_HKL)
            if update_scales:
                self.sfc.get_scales_adam(
                    lr=0.01,
                    n_steps=scale_steps,
                    sub_ratio=0.7,
                    initialize=scale_initialize,
                )
            Fc = self.sfc.calc_ftotal()

        Fm = cryo_sf.ftotal_amplitudes(Fc, self.sfc.dHKL, sort_by_res=True)

        Ecalc = Fm  ### or do we want to calculate sigmaAs and then return Ecalc from there? # TODO

        if return_Fc:
            return Ecalc, Fc
        else:
            return Ecalc

    def forward(
        self,
        xyz_ort: torch.Tensor,
        bin_labels=None,
        num_batch=1,
        sub_ratio=1.0,
        solvent=True,
        update_scales=False,
        added_chain_HKL=None,
        added_chain_asu=None,
    ):
        """
        TODO: Use rfree label in the LLG calculation
        Args:
            xyz_orth: torch.Tensor, [N_atoms, 3] in angstroms
                Orthogonal coordinates of proteins, coming from AF2 model, send to SFC

            bin_labels: None or List[int]
                Labels of bins used in the loss calculation. If None, will use the whole miller indices.
                Serve as a proxy for resolution selection

            num_batch: int
                Number of batches

            sub_ratio: float between 0.0 and 1.0
                Fraction of mini-batch sampling over all miller indices,
                e.g. 0.3 meaning each batch sample 30% of miller indices

        """

        Ecalc = self.compute_Ecalc(
            xyz_ort,
            solvent=solvent,
            update_scales=update_scales,
            added_chain_HKL=added_chain_HKL,
            added_chain_asu=added_chain_asu,
        )
        llg = 0.0

        if bin_labels is None:
            bin_labels = self.unique_bins

        for i, label in enumerate(bin_labels):
            index_i = self.bin_labels[self.working_set] == label
            # if sum(index_i) == 0:
            #    continue
            Ecalc_i = Ecalc[self.working_set][index_i]
            Eob_i = self.Eobs[self.working_set][index_i]
            Centric_i = self.Centric[self.working_set][index_i]
            Dobs_i = self.Dobs[self.working_set][index_i]

            sigmaA_i = self.sigmaAs[int(i)]
            for j in range(num_batch):
                sub_boolean_mask = np.random.rand(len(Eob_i)) < sub_ratio
                llg_ij = cryo_utils.llgItot_calculate(
                    sigmaA_i,
                    Dobs_i[sub_boolean_mask],
                    Eob_i[sub_boolean_mask],
                    Ecalc_i[sub_boolean_mask],
                    Centric_i[sub_boolean_mask],
                ).sum()
                # print("Batch {}".format(j), llg_ij.item())
                llg = llg + llg_ij

        return llg
