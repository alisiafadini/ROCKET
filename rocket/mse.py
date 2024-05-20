"""
MSE target for a target PDB file
"""

import torch
import numpy as np
from SFC_Torch import PDBParser
from rocket import utils


class MSEloss:
    """
    Object_oreinted interface to calculate MSE loss
    """

    def __init__(self, target: PDBParser, moving: PDBParser, device: torch.device):
        assert (
            target.sequence == moving.sequence
        ), "target pdb has different sequence with moving pdb!"
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

    def forward(self, xyz_ort: torch.Tensor):
        mse_loss = torch.mean(
            torch.sum(
                (xyz_ort[self.index_moving] - self.target_pos[self.index_target]) ** 2,
                dim=-1,
            )
        )
        return mse_loss
