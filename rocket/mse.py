"""
MSE target for a target PDB file
"""

import numpy as np
import torch
from SFC_Torch import PDBParser

from rocket import refinement_utils as rkrf_utils


class MSElossBB:
    """
    Object_oreinted interface to calculate MSE loss for backbone or CA atoms
    """

    def __init__(
        self, target: PDBParser, moving: PDBParser, device: torch.device, selection="BB"
    ):
        """
        selection: str
            "BB" for backbone, "CA" for Calpha
        """

        self.device = device
        self.target_cra = target.cra_name
        self.moving_cra = moving.cra_name
        self.pdb_obj = moving
        self.target_pos = torch.tensor(
            target.atom_pos, device=device, dtype=torch.float32
        ).clone()
        self.selection = selection
        if selection == "BB":
            self.ind1, self.ind2 = rkrf_utils.get_common_bb_ind(moving, target)
        elif selection == "CA":
            self.ind1, self.ind2 = rkrf_utils.get_common_ca_ind(moving, target)

    def forward(self, xyz_ort: torch.Tensor, weights: torch.Tensor):
        mse_loss = torch.mean(
            torch.sum(
                (xyz_ort[self.ind1] - self.target_pos[self.ind2]) ** 2,
                dim=-1,
            )
            * weights[self.ind1]
        )
        return mse_loss


class MSEloss:
    """
    Object_oreinted interface to calculate MSE loss
    """

    def __init__(self, target: PDBParser, moving: PDBParser, device: torch.device):
        assert target.sequence == moving.sequence, (
            "target pdb has different sequence with moving pdb!"
        )
        # get the intersect atoms index
        self.target_pdb = target
        self.device = device
        index_target = []
        index_moving = []
        for i, name in enumerate(moving.cra_name):
            try:
                id_t = target.cra_name.index(name)
                index_moving.append(i)
                index_target.append(id_t)
            except:
                pass
        assert (
            np.array(target.cra_name)[index_target]
            == np.array(moving.cra_name)[index_moving]
        ).all()
        self.index_target = np.array(index_target)
        self.index_moving = np.array(index_moving)
        self.target_pos = torch.tensor(
            target.atom_pos, device=device, dtype=torch.float32
        )

    @property
    def sequence(self):
        return self.target_pdb.sequence

    def forward(self, xyz_ort: torch.Tensor, subratio=1.0):
        sub_boolean_mask = np.random.rand(len(self.index_moving)) < subratio
        mse_loss = torch.mean(
            torch.sum(
                (
                    xyz_ort[self.index_moving][sub_boolean_mask]
                    - self.target_pos[self.index_target][sub_boolean_mask]
                )
                ** 2,
                dim=-1,
            )
        )
        return mse_loss
