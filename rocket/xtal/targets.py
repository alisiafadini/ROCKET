"""
LLG targets for xtal data
"""

import time
from functools import partial

import numpy as np
import torch
from SFC_Torch import SFcalculator

from rocket import utils
from rocket.xtal import structurefactors as llg_sf
from rocket.xtal import utils as llg_utils


class LLGloss(torch.nn.Module):
    """
    Object_oriented interface to calculate LLG loss

    # Initialization, only have to do it once
    sfc = llg_sf.initial_SFC(...)
    llgloss = LLGloss(sfc, tng_file, device)
    Ecalc = llgloss.compute_Ecalc(xyz_orth)
    llgloss.refine_sigmaA_adam(Ecalc.detach(), n_step=50)
    llgloss.freeze_sigmaA()

    # Loss calculation for each step
    loss = -llgloss(xyz_orth, bin_labels=[1,2,3], num_batch=10, sub_ratio=0.3)

    resol_min, resol_max: None | float
        resolution cutoff for used miller index. Will use resol_min <= dHKL <= resol_max

    TODO:
    Currently the initialization needs Eobs, Eps, Centric, Dobs, Feff, Bin_labels.
    We do so by loading the tng_data.
    Later we should be able to calculate everything from SFcalculator,
    all necesary information is ready there.

    """

    def __init__(
        self,
        sfc: SFcalculator,
        tng_file: str,
        device: torch.device,
        resol_min=None,
        resol_max=None,
    ) -> None:
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
        self.register_buffer("bin_dHKL", data_dict["BIN_dHKLS"])
        self.bin_dHKL: torch.Tensor

        if resol_min is None:
            resol_min = min(self.sfc.dHKL)

        if resol_max is None:
            resol_max = max(self.sfc.dHKL)

        resol_bool = (self.sfc.dHKL >= (resol_min - 1e-4)) & (
            self.sfc.dHKL <= (resol_max + 1e-4)
        )
        self.working_set = (~self.sfc.free_flag) & (~self.sfc.Outlier) & (resol_bool)
        self.free_set = (self.sfc.free_flag) & (~self.sfc.Outlier) & (resol_bool)

    def init_sigmaAs(self, Ecalc, subset="working", requires_grad=True):
        if subset == "working":
            subset_boolean = (~self.sfc.free_flag) & (~self.sfc.Outlier)
        elif subset == "free":
            subset_boolean = (self.sfc.free_flag) & (~self.sfc.Outlier)

        Ecalc = Ecalc.detach().clone()
        self.sigmaAs = []
        for bin_i in self.unique_bins:
            index_i = self.bin_labels[subset_boolean] == bin_i
            Eobs_i = self.Eobs[subset_boolean][index_i]
            Ecalc_i = Ecalc[subset_boolean][index_i]

            # Initialize from correlation coefficient
            sigmaA_i = (
                torch.corrcoef(torch.stack([Eobs_i**2, Ecalc_i**2], dim=0))[1][0]
                .clamp(min=0.001, max=0.999)
                .sqrt()
                .to(device=self.device, dtype=torch.float32)
                .requires_grad_(requires_grad)
            )
            self.sigmaAs.append(sigmaA_i)

    def init_sigmaAs_nodata(self, frac=1.0, rms=0.7, requires_grad=True):
        self.sigmaAs = []
        for bin_i in self.unique_bins:
            index_i = self.bin_labels == bin_i
            s = self.bin_dHKL[index_i][0]
            sigmaA_i = torch.sqrt(torch.tensor(frac)) * torch.exp(
                -2 / 3 * torch.pi**2 * torch.tensor(rms) ** 2 * s**-2
            ).to(device=self.device, dtype=torch.float32).requires_grad_(requires_grad)
            self.sigmaAs.append(sigmaA_i)

    def freeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(False) for sigmaA in self.sigmaAs]

    def unfreeze_sigmaA(self):
        self.sigmaAs = [sigmaA.requires_grad_(True) for sigmaA in self.sigmaAs]

    def refine_sigmaA_adam(
        self, Ecalc, n_steps=50, lr=0.01, sub_ratio=0.3, initialize=True, verbose=False
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
                self.sigmaAs[i].data = torch.clamp(self.sigmaAs[i].data, 0.015, 0.99)
                return loss

            Eobs = self.Eobs.detach().clone()
            Ecalc_cloned = Ecalc.detach().clone()
            Eobs_i = Eobs[index_i]
            Ecalc_i = Ecalc_cloned[index_i]
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
            self.init_sigmaAs(Ecalc, requires_grad=True)

        for i, bin_i in enumerate(self.unique_bins):
            index_i = self.bin_labels == bin_i
            adam_opt_i(
                i, index_i, n_steps=n_steps, sub_ratio=sub_ratio, lr=lr, verbose=verbose
            )

    def refine_sigmaA_newton(
        self,
        Ecalc,
        n_steps=2,
        initialize=True,
        method="autodiff",
        subset="test",
        edge_weights=0.25,
        smooth_overall_weight=200.0,
    ):
        """
        subset : str, "working" or "free"

        method : str, "autodiff" or "analytical"

        TODO: include smooth_constraint
        """
        if initialize:
            self.init_sigmaAs(Ecalc, subset="working", requires_grad=False)
            # self.init_sigmaAs_nodata()

        if subset == "working":
            subset_boolean = (~self.sfc.free_flag) & (~self.sfc.Outlier)
        elif subset == "test":
            subset_boolean = (self.sfc.free_flag) & (~self.sfc.Outlier)

        for _n in range(n_steps):
            lps = []
            lpps = []
            ls = []
            for i, label in enumerate(self.unique_bins):
                # for i in range(0, len(self.unique_bins) - 1, 2):
                # TODO: tackle the corner case where no freeset in a bin
                index_i = self.bin_labels[subset_boolean] == label

                Ecalc_i = Ecalc[subset_boolean][index_i]
                Eob_i = self.Eobs[subset_boolean][index_i]

                Centric_i = self.Centric[subset_boolean][index_i]
                Dobs_i = self.Dobs[subset_boolean][index_i]
                sigmaA_i = self.sigmaAs[int(i)].detach().clone()
                l, lp, lpp = llg_utils.llgItot_with_derivatives2sigmaA(  # noqa: E741
                    sigmaA=sigmaA_i,
                    dobs=Dobs_i,
                    Eeff=Eob_i,
                    Ec=Ecalc_i,
                    centric_tensor=Centric_i,
                    method=method,
                )
                ls.append(l)
                lps.append(lp)
                lpps.append(lpp)

            dL1 = torch.stack(lps).detach()
            H1 = torch.diag(torch.stack(lpps).detach())

            # Smooth terms
            sigmaAs_tensor = (
                torch.stack(self.sigmaAs).detach().clone().requires_grad_(True)
            )
            smooth_calculator = partial(
                llg_utils.interpolate_smooth,
                edge_weights=edge_weights,
                total_weight=smooth_overall_weight,
            )
            Ls = smooth_calculator(sigmaAs_tensor)
            dL2 = torch.autograd.grad(Ls, sigmaAs_tensor, create_graph=True)[0]
            H2 = torch.autograd.functional.hessian(smooth_calculator, sigmaAs_tensor)

            # Combine the two terms
            dL_total = dL1 - dL2.detach()
            Htotal = H1 - H2.detach()

            sigmaAs_updated = llg_utils.newton_step(sigmaAs_tensor, dL_total, Htotal)
            sigmaAs_new = torch.clamp(sigmaAs_updated, 0.015, 0.99)
            self.sigmaAs = [s.detach().requires_grad_(False) for s in sigmaAs_new]

    def compute_Ecalc(
        self,
        xyz_orth,
        solvent=True,
        return_Fc=False,
        return_Rfactors=True,
        update_scales=False,
        scale_steps=10,
        scale_initialize=False,
        added_chain_HKL=None,
        added_chain_asu=None,
    ) -> torch.Tensor:
        """
        Compute normalized structure factors (Ecalc).

        Args:
            xyz_orth: Orthogonal coordinates of atoms
            solvent: Whether to include solvent contribution
            return_Fc: Whether to return calculated structure factors Fc
            return_Rfactors: Whether to calculate and return R-work and R-free
            update_scales: Whether to update scaling factors
            scale_steps: Number of steps for scale refinement
            scale_initialize: Whether to initialize scales
            added_chain_HKL: Additional HKL contributions
            added_chain_asu: Additional ASU contributions

        Returns:
            If return_Rfactors=True and return_Fc=True: (Ecalc, Fc, R_work, R_free)
            If return_Rfactors=True and return_Fc=False: (Ecalc, R_work, R_free)
            If return_Rfactors=False and return_Fc=True: (Ecalc, Fc)
            If return_Rfactors=False and return_Fc=False: Ecalc

        Note:
            R-factors are calculated using:
            R_FEFF = Sum[DOBS^2 * |FEFF - FCALC|] / Sum[DOBS^2 * FEFF]
            R_work uses the working set (~free_flag), R_free uses the free set
            (free_flag)
        """
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

        Fm = llg_sf.ftotal_amplitudes(Fc, self.sfc.dHKL, sort_by_res=True)
        sigmaP = llg_sf.calculate_Sigma_atoms(Fm, self.Eps, self.bin_labels)
        Ecalc = llg_sf.normalize_Fs(Fm, self.Eps, sigmaP, self.bin_labels)

        if return_Rfactors:
            # Calculate R-work and R-free using the formula:
            # R_FEFF = Sum[DOBS^2 * |FEFF - FCALC|] / Sum[DOBS^2 * FEFF]

            # Calculate R-work (working set)
            working_mask = self.working_set
            dobs_work = self.Dobs[working_mask]
            feff_work = self.Feff[working_mask]
            fcalc_work = Fc[working_mask]

            numerator_work = torch.sum(dobs_work**2 * torch.abs(feff_work - fcalc_work))
            denominator_work = torch.sum(dobs_work**2 * feff_work)
            R_work = numerator_work / (denominator_work + 1e-8)  # avoid divide by zero

            # Calculate R-free (free set)
            free_mask = self.free_set
            dobs_free = self.Dobs[free_mask]
            feff_free = self.Feff[free_mask]
            fcalc_free = Fc[free_mask]

            numerator_free = torch.sum(dobs_free**2 * torch.abs(feff_free - fcalc_free))
            denominator_free = torch.sum(dobs_free**2 * feff_free)
            R_free = numerator_free / (denominator_free + 1e-8)  # avoid divide by zero

            if return_Fc:
                return Ecalc, Fc, R_work, R_free
            else:
                return Ecalc, R_work, R_free
        elif return_Fc:
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
        return_Rfactors=False,
    ):
        """
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

            return_Rfactors: bool
                Whether to calculate and return R-work and R-free factors.
                If True, returns (llg, r_work, r_free). If False, returns llg only.

        """

        if return_Rfactors:
            Ecalc, r_work, r_free = self.compute_Ecalc(
                xyz_ort,
                solvent=solvent,
                update_scales=update_scales,
                added_chain_HKL=added_chain_HKL,
                added_chain_asu=added_chain_asu,
                return_Rfactors=True,
            )
        else:
            Ecalc = self.compute_Ecalc(
                xyz_ort,
                solvent=solvent,
                update_scales=update_scales,
                added_chain_HKL=added_chain_HKL,
                added_chain_asu=added_chain_asu,
                return_Rfactors=False,
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
            for _j in range(num_batch):
                sub_boolean_mask = np.random.rand(len(Eob_i)) < sub_ratio
                llg_ij = llg_utils.llgItot_calculate(
                    sigmaA_i,
                    Dobs_i[sub_boolean_mask],
                    Eob_i[sub_boolean_mask],
                    Ecalc_i[sub_boolean_mask],
                    Centric_i[sub_boolean_mask],
                ).sum()
                llg = llg + llg_ij

        if return_Rfactors:
            return llg, r_work, r_free
        else:
            return llg
