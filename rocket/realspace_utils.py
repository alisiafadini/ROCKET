"""
Utility functions for real-space refinement operations.
"""

import gemmi
import numpy as np
import torch


def create_spherical_mask(
    map_grid: gemmi.FloatGrid, position: np.ndarray, radius: float
) -> np.ndarray:
    """
    Create spherical boolean mask for a map grid

    Args:
        map_grid: Input map grid (used for dimensions and metadata)
        position: Center position for masking
        radius: Radius for spherical masking

    Returns:
        Boolean numpy array (True inside sphere, False outside)
    """
    temp_mask = map_grid.clone()
    temp_mask.fill(0)
    temp_mask.set_points_around(
        gemmi.Position(position[0], position[1], position[2]),
        radius=radius,
        value=1,
    )
    temp_mask.symmetrize_max()
    return np.array(temp_mask, copy=False).astype(bool)


def save_mask_as_ccp4(target_ccp4_map, mask_position, mask_radius, output_path: str):
    """Save spherical mask as CCP4 file"""
    mask_grid = target_ccp4_map.grid.clone()
    mask_grid.fill(0)
    mask_grid.set_points_around(
        gemmi.Position(mask_position[0], mask_position[1], mask_position[2]),
        radius=mask_radius,
        value=1,
    )
    mask_grid.symmetrize_max()

    mask_ccp4 = gemmi.Ccp4Map()
    mask_ccp4.grid = mask_grid
    mask_ccp4.update_ccp4_header()
    mask_ccp4.write_ccp4_map(output_path)
    print(f"Mask saved to: {output_path}")


def save_map_as_ccp4(model_map: torch.Tensor, target_ccp4_map, output_path: str):
    """Save torch tensor map as CCP4 file"""
    try:
        model_map_np = model_map.detach().cpu().numpy()

        output_ccp4 = gemmi.Ccp4Map()
        output_ccp4.grid = gemmi.FloatGrid()
        output_ccp4.grid.copy_metadata_from(target_ccp4_map.grid)
        output_ccp4.grid.set_size_without_checking(*model_map_np.shape)
        output_ccp4.grid.set_values(model_map_np.flatten())
        output_ccp4.update_ccp4_header()
        output_ccp4.write_ccp4_map(output_path)
        print(f"Map saved to: {output_path}")

    except Exception as e:
        print(f"Error saving map: {e}")
        fallback_path = output_path.replace(".ccp4", "_numpy.npy")
        np.save(fallback_path, model_map_np)
        print(f"Saved as numpy array: {fallback_path}")
