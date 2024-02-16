"""
LLG targets for xtal data
"""
import time
import torch
import numpy as np
from SFC_Torch import SFcalculator
from rocket.llg import utils as llg_utils
from rocket.llg import structurefactors as llg_sf
from rocket import utils


class LLGloss(torch.nn.Module):
    """
    Objecte_oriented interface to calculate LLG loss

    # Initialization, only have to do it once
    sfc = llg_sf.initial_SFC(...)
    llgloss = LLGloss(sfc, tng_file, device)
    Ecalc = llgloss.compute_Ecalc(xyz_orth)
    llgloss.refine_sigmaA_adam(Ecalc.detach(), n_step=50)
    llgloss.freeze_sigmaA()

    # Loss calculation for each step
    loss = -llgloss(xyz_orth, bin_labels=[1,2,3], num_batch=10, sub_ratio=0.3)

    resol_min, resol_max: None | float
        resolution cutoff for the used miller index. Will use resol_min <= dHKL <= resol_max

    TODO:
    Currently the initialization needs inputs like Eobs, Eps, Centric, Dobs, Feff, Bin_labels. We do so by loading the tng_data.
    Later we should be able to calculate everything from SFcalculator, all necesary information is ready there.

    """

    def __init__(self, sfc: SFcalculator, tng_file: str, device: torch.device, resol_min=None, resol_max=None) -> None:
        super().__init__()
        self.sfc = sfc
        self.device = device
        data_dict = llg_utils.load_tng_data(tng_file, device=device)

        self.register_buffer("Eobs", data_dict["EDATA"])
        self.Eobs: torch.Tensor
        self.register_buffer("Eps", data_dict["EPS"])
        self.Eps: torch.Tensor
        self.register_buffer("Centric", data_dict["CENTRIC"])
        self.Centric: torch.Tensor
        self.register_buffer("Dobs", data_dict["DOBS"])
        self.Dobs: torch.Tensor
        self.register_buffer("Feff", data_dict["FEFF"])
        self.Feff: torch.Tensor
        self.register_buffer("bin_labels", data_dict["BIN_LABELS"])
        self.bin_labels: torch.Tensor
        self.unique_bins = torch.unique(self.bin_labels)

        if resol_min is None:
            resol_min = min(self.sfc.dHKL)
        
        if resol_max is None:
            resol_max = max(self.sfc.dHKL)
        
        resol_bool = (self.sfc.dHKL >= (resol_min - 1e-4)) & (self.sfc.dHKL <= (resol_max + 1e-4))
        self.working_set =  (~self.sfc.free_flag) & (~self.sfc.Outlier) & (resol_bool)

    def init_sigmaAs(self, Ecalc):
        self.sigmaAs = []
        for bin_i in self.unique_bins:
            index_i = self.bin_labels == bin_i
            Eobs_i = self.Eobs[index_i]
            Ecalc_i = Ecalc[index_i]

            # Initialize from correlation coefficient
            sigmaA_i = (
                torch.corrcoef(torch.stack([Eobs_i, Ecalc_i], dim=0))[1][0]
                .clamp(min=0.001, max=0.999)
                .sqrt()
                .to(device=self.device, dtype=torch.float32)
                .requires_grad_(True)
            )
            self.sigmaAs.append(sigmaA_i)

    def freeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(False) for sigmaA in self.sigmaAs]

    def unfreeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(True) for sigmaA in self.sigmaAs]

    def refine_sigmaA_adam(
        self, Ecalc, n_steps=25, lr=0.01, sub_ratio=0.3, initialize=True, verbose=False
    ):
        def adam_opt_i(i, index_i, n_steps, sub_ratio, lr, verbose):
            def adam_stepopt(sub_boolean_mask):
                loss = -llg_utils.llgTot_calculate(
                    self.sigmaAs[i],
                    Eobs_i[sub_boolean_mask],
                    Ecalc_i[sub_boolean_mask],
                    centric_i[sub_boolean_mask],
                )
                adam.zero_grad()
                loss.backward()
                adam.step()
                self.sigmaAs[i] = torch.clamp(self.sigmaAs[i], 0.015, 0.99)
                return loss

            Eobs_i = self.Eobs[index_i]
            Ecalc_i = Ecalc[index_i]
            centric_i = self.Centric[index_i]
            adam = torch.optim.Adam([self.sigmaAs[i]], lr=lr)
            for _ in range(n_steps):
                start_time = time.time()
                sub_boolean_mask = (
                    np.random.rand(
                        len(Eobs_i),
                    )
                    < sub_ratio
                )
                temp_loss = adam_stepopt(sub_boolean_mask)
                time_this_round = round(time.time() - start_time, 3)
                str_ = "Time: " + str(time_this_round)
                if verbose:
                    print(
                        f"SigmaA {i}", utils.assert_numpy(temp_loss), str_, flush=True
                    )

        if initialize:
            self.init_sigmaAs(Ecalc)

        for i, bin_i in enumerate(self.unique_bins):
            index_i = self.bin_labels == bin_i
            adam_opt_i(
                i, index_i, n_steps=n_steps, sub_ratio=sub_ratio, lr=lr, verbose=verbose
            )

    def compute_Ecalc(
        self,
        xyz_orth,
        solvent=True,
        return_Fc=False,
        update_scales=False,
        added_chain=None,
    ) -> torch.Tensor:
        self.sfc.calc_fprotein(atoms_position_tensor=xyz_orth)

        if added_chain is not None:
            self.sfc.Fprotein_HKL = self.sfc.Fprotein_HKL + added_chain

        if solvent:
            self.sfc.calc_fsolvent()
            if update_scales:
                self.sfc.get_scales_adam(
                    lr=0.01,
                    n_steps=10,
                    sub_ratio=0.7,
                    initialize=False,
                )
            Fc = self.sfc.calc_ftotal()
        else:
            # MH note: we need scales here, even without solvent contribution
            self.sfc.Fmask_HKL = torch.zeros_like(self.sfc.Fprotein_HKL)
            if update_scales:
                self.sfc.get_scales_adam(
                    lr=0.01,
                    n_steps=10,
                    sub_ratio=0.7,
                    initialize=False,
                )
            Fc = self.sfc.calc_ftotal()

        Fm = llg_sf.ftotal_amplitudes(Fc, self.sfc.dHKL, sort_by_res=True)
        sigmaP = llg_sf.calculate_Sigma_atoms(Fm, self.Eps, self.bin_labels)
        Ecalc = llg_sf.normalize_Fs(Fm, self.Eps, sigmaP, self.bin_labels)

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
        added_chain=None,
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
            added_chain=added_chain,
        )
        llg = 0.0

        if bin_labels is None:
            bin_labels = self.unique_bins

        for i, label in enumerate(bin_labels):
            index_i = self.bin_labels[self.working_set] == label
            if sum(index_i) == 0:
                continue
            Ecalc_i = Ecalc[self.working_set][index_i]
            Eob_i = self.Eobs[self.working_set][index_i]
            Centric_i = self.Centric[self.working_set][index_i]
            Dobs_i = self.Dobs[self.working_set][index_i]
            sigmaA_i = self.sigmaAs[i]
            for j in range(num_batch):
                sub_boolean_mask = np.random.rand(len(Eob_i)) < sub_ratio
                llg_ij = llg_utils.llgItot_calculate(
                    sigmaA_i,
                    Dobs_i[sub_boolean_mask],
                    Eob_i[sub_boolean_mask],
                    Ecalc_i[sub_boolean_mask],
                    Centric_i[sub_boolean_mask],
                ).sum()
                # print("Batch {}".format(j), llg_ij.item())
                llg = llg + llg_ij

        return llg
