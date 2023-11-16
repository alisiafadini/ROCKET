"""
Include customized pytorch lightning module implementation
"""

import pytorch_lightning as pl
from pytorch_lightning.profiler import PyTorchProfiler

from openfold.config import model_config

import rocket
from rocket.llg import utils as llg_utils
from rocket import coordinates as rk_coordinates
from rocket.llg import structurefactors as llg_sf

"""
class XYZRefiner(....):
   def __init__(...):
      sfc = initialize_SFC(...)
      self.llgloss = LLGloss(sfc, tng_file)
      self.af2 = AF2bias(...config)

  def training_step(...):
      pred_xyz = self.af2(...)
      aligend_xyz = align(pred_xyz, target_xyz)
      loss = self.llgloss(aligned_xyz) 
      
"""


class XYZBiasRefiner(pl.LightningModule):
    def __init__(
        self,
        lr,
        num_epochs,
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
        self.num_epochs = num_epochs
        self.device = device
        self.feats = feats
        self.pdb_in = ref_pdb
        self.update = update
        self.batch_num = batch_num
        self.phitrue = phi_true

        # Initialize SFC instance and reference positions
        sfc = llg_sf.initial_SFC(self.pdb_in, tng_file, "FP", "SIGFP")
        self.ref_xyz = sfc.atom_pos_orth

        # Initialize MSA bias model
        self.af_bias = rocket.MSABiasAF(model_config(preset, train=True), preset).to(
            device
        )

        # Initialize LLGloss instance
        self.llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

    def training_step(self, batch, update=True):
        # AF2 pass
        af2_results = self.af_bias(
            self.feats, num_iters=1, biasMSA=True
        )  # TODO: no recycling hard coded?

        # position alignment
        xyz_orth_sfc = rk_coordinates.extract_allatoms(af2_results, self.feats)
        aligned_xyz = rk_coordinates.align_positions(xyz_orth_sfc, self.ref_xyz)

        if update:
            # sigmaA calculation
            Ecalc = self.llgloss.compute_Ecalc(aligned_xyz)
            sigmas = llg_utils.sigmaA_from_model(
                rocket.utils.assert_numpy(self.llgloss.Eobs),
                self.phitrue,
                self.llgloss.Ecalc,
                self.llgloss.Fc,
                self.llgloss.sfc.dHKL,
                self.llgloss.bin_labels,
            )
            self.llgloss.sigmaAs = sigmas

        loss = -self.llgloss(
            aligned_xyz, bin_labels=None, num_batch=self.batch_num, sub_ratio=1.1
        )

        return loss
