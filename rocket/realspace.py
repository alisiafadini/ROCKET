"""
Real space target for comparing model-generated maps with target CCP4 maps

This module provides the RealSpace class that implements real-space refinement
by comparing model-generated electron density maps with target experimental maps
in CCP4 format. The class handles map loading, grid compatibility checking,
masking of invalid regions, and maximization of correlation coefficient.
"""

import gemmi
import numpy as np
import torch
from SFC_Torch import PDBParser
from SFC_Torch.mask import reciprocal_grid


class RealSpace:
    """
    Object-oriented interface to calculate real space correlation coefficient
    between a target CCP4 map and a model-generated map from PDB coordinates.
    Maximizes correlation coefficient.
    """

    def __init__(
        self,
        target_map: gemmi.Ccp4Map,
        pdb_obj: PDBParser,
        device: torch.device,
        mask_position: np.ndarray = None,
        mask_radius: float = None,
        loss_type: str = "cc",
    ):
        """
        Initialize RealSpace loss calculator

        Args:
            target_map:  CCP4 map object
            pdb_obj: PDBParser object containing the model structure
            device: torch device for computations
            mask_position: Position for spherical masking (3D coordinates)
            mask_radius: Radius for spherical masking
            loss_type: Type of loss - "cc" for correlation coefficient, "l2" for L2 loss
        """
        self.device = device
        self.pdb_obj = pdb_obj

        # Store loss type
        if loss_type not in ["cc", "l2"]:
            raise ValueError(
                "loss_type must be 'cc' (correlation coefficient) or 'l2' (L2 loss)"
            )
        self.loss_type = loss_type

        # Load target CCP4 map
        self.target_ccp4_map = target_map

        # Store masking parameters
        self.mask_position = mask_position
        self.mask_radius = mask_radius

        # Extract target map grid and apply masking if specified
        if mask_position is not None and mask_radius is not None:
            print(
                f"Applying spherical mask at position {mask_position} "
                f"with radius {mask_radius}"
            )
            print("grid is", self.target_ccp4_map.grid)
            masked_map_grid = self._mask_map(
                self.target_ccp4_map.grid, mask_position, mask_radius
            )
            target_grid_np = np.array(masked_map_grid, copy=False)
        else:
            target_grid_np = np.array(self.target_ccp4_map.grid, copy=False)

        self.target_map_grid = torch.tensor(
            target_grid_np, device=device, dtype=torch.float32
        ).clone()

        # Get grid information
        self.grid_shape = self.target_map_grid.shape
        self.unit_cell = self.target_ccp4_map.grid.unit_cell
        self.space_group = self.target_ccp4_map.grid.spacegroup

        # Create not-nan mask for target map BEFORE calling _find_valid_atom_indices
        self.target_mask = ~torch.isnan(self.target_map_grid)

        # Also create a mask for zero values if we applied spherical masking
        if mask_position is not None and mask_radius is not None:
            zero_mask = self.target_map_grid != 0
            self.target_mask = self.target_mask & zero_mask
            print(
                f"Combined mask (non-NaN and non-zero): "
                f"{self.target_mask.sum().item()} voxels"
            )

        # NOW find atom indices that correspond to valid (non-NaN) map regions
        self._find_valid_atom_indices()

        print(f"Loaded target map with shape: {self.grid_shape}")
        print(f"Unit cell: {self.unit_cell}")
        print(f"Space group: {self.space_group}")
        print(f"Loss type: {self.loss_type}")
        print(f"Valid voxels in target: {self.target_mask.sum().item()}")
        print(f"Valid atoms for alignment: {len(self.valid_atom_indices)}")

    def _find_valid_atom_indices(self):
        """
        Find atom indices that correspond to non-NaN regions of the target map
        for use in Kabsch alignment
        """
        # Get fractional coordinates of atoms
        atom_pos_cart = self.pdb_obj.atom_pos  # Cartesian coordinates

        # Convert to fractional coordinates using unit cell
        # This is a simplified conversion - in practice you might need
        # gemmi's coordinate conversion
        atom_pos_frac = self._cartesian_to_fractional(atom_pos_cart)

        # Convert fractional coordinates to grid indices
        grid_indices = self._fractional_to_grid_indices(atom_pos_frac)

        # Check which atoms fall in valid (non-masked) map regions
        valid_atoms = []
        for i, (gi, gj, gk) in enumerate(grid_indices):
            # Ensure indices are within grid bounds and check if voxel is valid
            if (
                0 <= gi < self.grid_shape[0]
                and 0 <= gj < self.grid_shape[1]
                and 0 <= gk < self.grid_shape[2]
                and self.target_mask[gi, gj, gk]
            ):
                valid_atoms.append(i)
        total_atoms = len(self.pdb_obj.atom_pos)
        self.valid_atom_indices = np.arange(total_atoms)
        # self.valid_atom_indices = np.array(valid_atoms)

        # For Kabsch alignment, we need indices for both moving and reference
        # Since we're working with the same structure, ind1 and ind2 are the same
        self.ind1 = self.valid_atom_indices
        self.ind2 = self.valid_atom_indices

    def _cartesian_to_fractional(self, cart_coords):
        """
        Convert Cartesian coordinates to fractional coordinates
        """
        # Get unit cell parameters
        a = self.unit_cell.a
        b = self.unit_cell.b
        c = self.unit_cell.c
        alpha = np.radians(self.unit_cell.alpha)
        beta = np.radians(self.unit_cell.beta)
        gamma = np.radians(self.unit_cell.gamma)

        # Create transformation matrix from Cartesian to fractional
        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)

        # Transformation matrix (simplified orthogonal case)
        frac_matrix = np.array([
            [
                1 / a,
                -cos_gamma / (a * sin_gamma),
                (cos_alpha * cos_gamma - cos_beta)
                / (a * sin_gamma * np.sqrt(1 - cos_gamma**2)),
            ],
            [
                0,
                1 / (b * sin_gamma),
                (cos_beta * cos_gamma - cos_alpha)
                / (b * sin_gamma * np.sqrt(1 - cos_gamma**2)),
            ],
            [0, 0, sin_gamma / (c * np.sqrt(1 - cos_gamma**2))],
        ])

        # Apply transformation
        frac_coords = np.dot(cart_coords, frac_matrix.T)
        return frac_coords

    def _fractional_to_grid_indices(self, frac_coords):
        """
        Convert fractional coordinates to grid indices
        """
        # Scale fractional coordinates to grid dimensions
        grid_coords = frac_coords * np.array(self.grid_shape)

        # Round to nearest integer indices
        grid_indices = np.round(grid_coords).astype(int)

        return grid_indices

    def get_valid_indices(self):
        """
        Get the atom indices that correspond to valid (non-zero ie non-masked)
        map regions

        Returns:
            tuple: (ind1, ind2) - indices for Kabsch alignment
        """
        return self.ind1, self.ind2

    def _mask_map(
        self, map_grid: gemmi.FloatGrid, position: np.ndarray, radius: float
    ) -> gemmi.FloatGrid:
        """
        Apply spherical masking to a map grid

        Args:
            map_grid: Input map grid to mask
            position: Center position for masking (3D coordinates)
            radius: Radius for spherical masking

        Returns:
            Masked map grid
        """
        mask = map_grid.clone()

        mask.fill(0)
        mask.set_points_around(
            gemmi.Position(position[0], position[1], position[2]),
            radius=radius,
            value=1,
        )
        mask.symmetrize_max()

        masked_array = np.array(map_grid, copy=True) * np.array(mask, copy=True)
        return gemmi.FloatGrid(masked_array)

    def set_mask(
        self, position: np.ndarray, radius: float, apply_to_model_maps: bool = True
    ):
        """
        Set or update the masking parameters and re-apply to target map

        Args:
            position: Center position for masking (3D coordinates)
            radius: Radius for spherical masking
            apply_to_model_maps: Whether to also apply masking to generated model maps
        """
        self.mask_position = position
        self.mask_radius = radius
        self.apply_mask_to_model = apply_to_model_maps

        print(f"Updating mask: position {position}, radius {radius}")

        # Re-apply masking to target map
        masked_map_grid = self._mask_map(self.target_ccp4_map.grid, position, radius)
        target_grid_np = np.array(masked_map_grid, copy=False)

        self.target_map_grid = torch.tensor(
            target_grid_np, device=self.device, dtype=torch.float32
        ).clone()

        # Update masks
        self.target_mask = ~torch.isnan(self.target_map_grid)
        zero_mask = self.target_map_grid != 0
        self.target_mask = self.target_mask & zero_mask

        # Update valid atom indices based on new mask
        self._find_valid_atom_indices()

    def clear_mask(self):
        """
        Remove masking and restore original target map
        """
        self.mask_position = None
        self.mask_radius = None
        self.apply_mask_to_model = False

        print("Clearing mask, restoring original target map")

        # Restore original target map
        target_grid_np = np.array(self.target_ccp4_map.grid, copy=False)
        self.target_map_grid = torch.tensor(
            target_grid_np, device=self.device, dtype=torch.float32
        ).clone()

        # Update masks
        self.target_mask = ~torch.isnan(self.target_map_grid)

        # Update valid atom indices
        self._find_valid_atom_indices()

    def model2map(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """
        Compute the 3D map from the model coordinates with optional masking

        Args:
            model_coordinates: Tensor of atomic coordinates
            dcp: Structure factor calculator object

        Returns:
            Raw map grid as torch tensor (optionally masked)
        """
        # Compute the 3D map from the model coordinates
        Fprotein = dcp.calc_fprotein(model_coordinates, Return=True)
        rs_grid = reciprocal_grid(dcp.Hasu_array, Fprotein, dcp.gridsize)
        map_grid = torch.real(torch.fft.fftn(rs_grid, dim=(-3, -2, -1)))
        map_grid_norm = (map_grid - map_grid.mean()) / map_grid.std()

        # Apply masking to model map if specified
        if (
            hasattr(self, "apply_mask_to_model")
            and self.apply_mask_to_model
            and self.mask_position is not None
            and self.mask_radius is not None
        ):
            # Convert torch tensor to numpy for masking
            map_np = map_grid_norm.detach().cpu().numpy()

            # Create gemmi FloatGrid from numpy array
            temp_grid = gemmi.FloatGrid()
            temp_grid.copy_metadata_from(self.target_ccp4_map.grid)
            # temp_grid.set_size_without_checking(*map_np.shape)
            temp_grid.set_values(map_np.flatten())

            # Apply mask
            masked_grid = self._mask_map(
                temp_grid, self.mask_position, self.mask_radius
            )

            # Convert back to torch tensor
            masked_np = np.array(masked_grid, copy=False)
            map_grid_norm = torch.tensor(
                masked_np, device=self.device, dtype=torch.float32
            )

        # Save map for debugging (optional)
        if hasattr(self, "_save_debug_maps") and self._save_debug_maps:
            zyx_array = map_grid_norm.detach().cpu().numpy().transpose(2, 1, 0)
            import mrcfile

            with mrcfile.new("rendered_volume.mrc", overwrite=True) as mrc:
                mrc.set_data(zyx_array)
                mrc.voxel_size = 0.7

        return map_grid_norm

    def get_correlation(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """
        Calculate correlation coefficient between target and model maps

        Args:
            model_coordinates: Tensor of atomic coordinates
            dcp: Structure factor calculator object

        Returns:
            Correlation coefficient (positive value)
        """
        # Generate model map
        model_map = self.model2map(model_coordinates, dcp)

        # Ensure compatible grids
        model_map = self._ensure_compatible_grids(model_map)

        # Create masks
        model_mask = ~torch.isnan(model_map)
        combined_mask = self.target_mask & model_mask

        # Extract valid regions (NO normalization)
        target_masked = self.target_map_grid[combined_mask]
        model_masked = model_map[combined_mask]

        # Calculate correlation coefficient
        if target_masked.numel() > 50:
            correlation = torch.corrcoef(torch.stack([target_masked, model_masked]))[
                0, 1
            ]
            if torch.isnan(correlation):
                correlation = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        else:
            correlation = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        return correlation

    def _ensure_compatible_grids(self, model_map: torch.Tensor) -> torch.Tensor:
        """
        Ensure model map has the same grid size as target map

        Args:
            model_map: Model-generated map tensor

        Returns:
            Model map (unchanged if grids match)

        Raises:
            ValueError: If grid shapes don't match
        """
        if model_map.shape != self.target_map_grid.shape:
            raise ValueError(
                f"Grid shape mismatch: model map has shape {model_map.shape}, "
                f"but target map has shape {self.target_map_grid.shape}. "
                f"Ensure both maps use the same grid parameters."
            )

        return model_map

    def forward(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """
        Forward pass: compute loss based on selected loss type

        Args:
            model_coordinates: Tensor of atomic coordinates
            dcp: Structure factor calculator object

        Returns:
            Loss value (negative correlation for CC, L2 loss for L2)
        """
        if self.loss_type == "cc":
            return self._forward_cc(model_coordinates, dcp)
        elif self.loss_type == "l2":
            return self._forward_l2(model_coordinates, dcp)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _forward_cc(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """
        Forward pass using correlation coefficient (maximize CC = minimize -CC)
        """
        # Use get_correlation() method for consistency
        correlation = self.get_correlation(model_coordinates, dcp)

        # Return negative correlation to convert maximization to minimization problem
        return -correlation

    def _forward_l2(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """
        Forward pass using L2 loss on normalized maps
        """
        # Generate model map
        model_map = self.model2map(model_coordinates, dcp)

        # Ensure compatible grids
        model_map = self._ensure_compatible_grids(model_map)

        # Create not-nan mask for model map
        model_mask = ~torch.isnan(model_map)

        # Combined mask: both maps must be valid (not NaN)
        combined_mask = self.target_mask & model_mask

        # Extract valid regions from both maps BEFORE normalization
        target_masked = self.target_map_grid[combined_mask]
        model_masked = model_map[combined_mask]

        if target_masked.numel() < 50:
            # If insufficient valid overlap, return a large loss
            return torch.tensor(1e6, device=self.device, dtype=torch.float32)

        # Normalize both maps using the SAME valid region statistics
        # This ensures consistent normalization
        target_mean = target_masked.mean()
        target_std = target_masked.std()
        model_mean = model_masked.mean()
        model_std = model_masked.std()

        if target_std > 1e-8 and model_std > 1e-8:
            target_normalized = (target_masked - target_mean) / target_std
            model_normalized = (model_masked - model_mean) / model_std
        else:
            # If std is too small, just center the data
            target_normalized = target_masked - target_mean
            model_normalized = model_masked - model_mean

        # Compute L2 loss
        l2_loss = torch.mean((target_normalized - model_normalized) ** 2)

        return l2_loss

    def set_loss_type(self, loss_type: str):
        """
        Change the loss type

        Args:
            loss_type: "cc" for correlation coefficient, "l2" for L2 loss
        """
        if loss_type not in ["cc", "l2"]:
            raise ValueError(
                "loss_type must be 'cc' (correlation coefficient) or 'l2' (L2 loss)"
            )

        old_type = self.loss_type
        self.loss_type = loss_type
        print(f"Changed loss type from '{old_type}' to '{loss_type}'")

    def get_l2_loss(self, model_coordinates: torch.Tensor, dcp) -> torch.Tensor:
        """
        Calculate L2 loss between normalized target and model maps

        Args:
            model_coordinates: Tensor of atomic coordinates
            dcp: Structure factor calculator object

        Returns:
            L2 loss value
        """
        return self._forward_l2(model_coordinates, dcp)

    @property
    def loss_type_info(self):
        """Return information about current loss type"""
        if self.loss_type == "cc":
            return "Correlation Coefficient (maximizing CC by minimizing -CC)"
        elif self.loss_type == "l2":
            return "L2 Loss on normalized maps"
        else:
            return f"Unknown loss type: {self.loss_type}"

    @property
    def target_shape(self):
        """Return the shape of the target map"""
        return self.grid_shape

    @property
    def valid_voxels(self):
        """Return the number of valid (non-NaN) voxels in target map"""
        return self.target_mask.sum().item()

    @property
    def is_masked(self):
        """Return True if masking is currently applied"""
        return self.mask_position is not None and self.mask_radius is not None

    @property
    def mask_info(self):
        """Return current mask parameters"""
        if self.is_masked:
            return {
                "position": self.mask_position,
                "radius": self.mask_radius,
                "apply_to_model": getattr(self, "apply_mask_to_model", False),
            }
        return None

    def save_mask_ccp4(self, output_path: str):
        """
        Save the current mask as a CCP4 file

        Args:
            output_path: Path where to save the mask CCP4 file
        """
        if not self.is_masked:
            print("No mask is currently applied")
            return

        # Create mask grid
        mask_grid = self.target_ccp4_map.grid.clone()
        mask_grid.fill(0)
        mask_grid.set_points_around(
            gemmi.Position(
                self.mask_position[0], self.mask_position[1], self.mask_position[2]
            ),
            radius=self.mask_radius,
            value=1,
        )
        mask_grid.symmetrize_max()

        # Save as CCP4
        mask_ccp4 = gemmi.Ccp4Map()
        mask_ccp4.grid = mask_grid
        mask_ccp4.update_ccp4_header()
        mask_ccp4.write_ccp4_map(output_path)
        print(f"Mask saved to: {output_path}")

    def save_model_map(self, model_coordinates: torch.Tensor, dcp, output_path: str):
        """
        Save the model-generated map to a CCP4 file for visualization/debugging

        Args:
            model_coordinates: Tensor of atomic coordinates
            dcp: Structure factor calculator object
            output_path: Path where to save the CCP4 map file
        """
        try:
            # Generate model map (no normalization)
            model_map = self.model2map(model_coordinates, dcp)

            # Ensure compatible grids
            model_map = self._ensure_compatible_grids(model_map)

            # Convert to numpy array for gemmi
            model_map_np = model_map.detach().cpu().numpy()

            # Create a new CCP4 map by copying the original target map structure
            # This preserves all the grid parameters properly
            output_ccp4 = gemmi.Ccp4Map()

            # Copy grid structure from target map
            output_ccp4.grid = gemmi.FloatGrid()
            output_ccp4.grid.copy_metadata_from(self.target_ccp4_map.grid)

            # Set the grid size and ensure it matches
            output_ccp4.grid.set_size_without_checking(*model_map_np.shape)

            # Copy the data
            output_ccp4.grid.set_values(model_map_np.flatten())

            # Update header information
            output_ccp4.update_ccp4_header()

            # Write to file
            output_ccp4.write_ccp4_map(output_path)
            print(f"Model map saved to: {output_path}")

        except Exception as e:
            print(f"Error saving model map: {e}")

            # Fallback: try a simpler approach
            try:
                # Alternative method: copy the target map and replace data
                output_ccp4 = gemmi.Ccp4Map()
                output_ccp4.grid = self.target_ccp4_map.grid.copy()

                # Replace the data with model map
                model_map_np = model_map.detach().cpu().numpy()
                np.copyto(np.array(output_ccp4.grid, copy=False), model_map_np)

                # Write to file
                output_ccp4.write_ccp4_map(output_path)
                print(f"Model map saved to: {output_path} (fallback method)")

            except Exception as e2:
                print(f"Both save methods failed: {e2}")

                # Last resort: save as numpy array for debugging
                fallback_path = output_path.replace(".ccp4", "_numpy.npy")
                np.save(fallback_path, model_map_np)
                print(f"Saved model map as numpy array to: {fallback_path}")
