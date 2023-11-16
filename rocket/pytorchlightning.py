"""
Include customized pytorch lightning module implementation
"""

import torch
import pytorch_lightning as pl

from openfold.config import model_config

import rocket
from rocket.llg import utils as llg_utils
from rocket import coordinates as rk_coordinates
from rocket.llg import structurefactors as llg_sf


class XYZBiasRefiner(pl.LightningModule):
    def __init__(
        self,
        lr,
        device,
        preset,
        feats,
        ref_pdb,
        tng_file,
        phi_true,
        update=50,
        batch_num=1,
    ):
        super().__init__()
        self.lr = lr
        self.device = device
        self.feats = feats
        self.pdb_in = ref_pdb
        self.update = update
        self.batch_num = batch_num
        self.phitrue = phi_true

        # Initialize SFC instance and reference positions
        sfc = llg_sf.initial_SFC(self.pdb_in, tng_file, "FP", "SIGFP")
        self.ref_xyz = sfc.atom_pos_orth.clone()

        # Initialize MSA bias model
        self.af_bias = rocket.MSABiasAF(model_config(preset, train=True), preset).to(
            device
        )

        # Initialize LLGloss instance
        self.llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

    def training_step(self, update=True):
        # AF2 pass
        af2_results = self.af_bias(
            self.feats, num_iters=1, biasMSA=True
        )  # TODO: no recycling hard coded?

        # position alignment
        xyz_orth_sfc = rk_coordinates.extract_allatoms(af2_results, self.feats)
        aligned_xyz = rk_coordinates.align_positions(xyz_orth_sfc, self.ref_xyz)

        if update:
            # sigmaA calculation
            Ecalc, Fc = self.llgloss.compute_Ecalc(aligned_xyz, return_Fc=True)
            sigmas = llg_utils.sigmaA_from_model(
                rocket.utils.assert_numpy(self.llgloss.Eobs),
                self.phitrue,
                Ecalc,
                self.llgloss.Fc,
                self.llgloss.sfc.dHKL,
                self.llgloss.bin_labels,
            )
            self.llgloss.sigmaAs = sigmas

        # LLG loss calculation
        loss = -self.llgloss(
            aligned_xyz, bin_labels=None, num_batch=self.batch_num, sub_ratio=1.1
        )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_epoch_end(self, outputs):
        # Log mean loss at the end of each epoch
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log("avg_loss", avg_loss, prog_bar=True)

        # TODO: is there anything else to log? pLDDT, MSE loss to truth, etc.
