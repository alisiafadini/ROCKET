"""
Real-space refinement by comparing model-generated maps with target CCP4 maps.
"""

import numpy as np
import torch
from SFC_Torch import PDBParser
from SFC_Torch.mask import reciprocal_grid

from .realspace_utils import create_spherical_mask


class RealSpace:
    """
    Calculate loss between target CCP4 map and model-generated map from coordinates.
    """

    def __init__(
        self,
        target_map,
        pdb_obj: PDBParser,
        device: torch.device,
        mask_position=None,
        mask_radius=None,
        loss_type: str = "cc",
    ):
        """Initialize RealSpace loss calculator"""
        if loss_type not in ["cc", "l2"]:
            raise ValueError("loss_type must be 'cc' or 'l2'")

        self.device = device
        self.pdb_obj = pdb_obj
        self.loss_type = loss_type
        self.target_ccp4_map = target_map
        self.mask_position = mask_position
        self.mask_radius = mask_radius

        # Load target map grid
        target_grid_np = np.array(target_map.grid, copy=False)
        self.target_map_grid = torch.tensor(
            target_grid_np, device=device, dtype=torch.float32
        ).clone()

        # Create mask
        if mask_position is not None and mask_radius is not None:
            mask_array = create_spherical_mask(
                target_map.grid, mask_position, mask_radius
            )
            self.target_mask = torch.tensor(mask_array, device=device, dtype=torch.bool)
        else:
            self.target_mask = torch.ones_like(self.target_map_grid, dtype=torch.bool)

        # Set all atoms as valid for alignment
        total_atoms = len(pdb_obj.atom_pos)
        self.valid_atom_indices = np.arange(total_atoms)
        self.ind1 = self.valid_atom_indices
        self.ind2 = self.valid_atom_indices

    def model2map(self, model_coordinates: torch.Tensor, sfc) -> torch.Tensor:
        """Generate normalized map from model coordinates"""
        Fprotein = sfc.calc_fprotein(model_coordinates, Return=True)
        rs_grid = reciprocal_grid(sfc.Hasu_array, Fprotein, sfc.gridsize)
        map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
        map_grid_norm = (map_grid - map_grid.mean()) / map_grid.std()

        if self.is_masked:
            map_grid_norm = map_grid_norm * self.target_mask.float()

        return map_grid_norm

    def get_correlation(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """Calculate correlation coefficient between target and model maps"""
        model_map = self.model2map(model_coordinates, dcp)

        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} vs "
                f"{self.target_map_grid.shape}"
            )

        target_masked = self.target_map_grid[self.target_mask]
        model_masked = model_map[self.target_mask]

        if target_masked.numel() > 50:
            correlation = torch.corrcoef(torch.stack([target_masked, model_masked]))[
                0, 1
            ]
            if torch.isnan(correlation):
                correlation = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            correlation = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        return correlation

    def forward(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """Compute loss based on selected loss type"""
        if self.loss_type == "cc":
            return self._forward_cc(model_coordinates, dcp)
        elif self.loss_type == "l2":
            return self._forward_l2(model_coordinates, dcp)

    def _forward_cc(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """CC loss (negative correlation coefficient)"""
        return -self.get_correlation(model_coordinates, dcp)

    def _forward_l2(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """L2 loss on normalized maps"""
        model_map = self.model2map(model_coordinates, dcp)

        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: {model_map.shape} vs "
                f"{self.target_map_grid.shape}"
            )

        target_masked = self.target_map_grid[self.target_mask]
        model_masked = model_map[self.target_mask]

        if target_masked.numel() < 50:
            return torch.tensor(1e6, device=self.device, dtype=torch.float32)

        # Normalize both maps using same statistics
        target_mean, target_std = target_masked.mean(), target_masked.std()
        model_mean, model_std = model_masked.mean(), model_masked.std()

        if target_std > 1e-8 and model_std > 1e-8:
            target_normalized = (target_masked - target_mean) / target_std
            model_normalized = (model_masked - model_mean) / model_std
        else:
            target_normalized = target_masked - target_mean
            model_normalized = model_masked - model_mean

        return torch.mean((target_normalized - model_normalized) ** 2)

    @property
    def is_masked(self):
        """Return True if masking is applied"""
        return self.mask_position is not None and self.mask_radius is not None
