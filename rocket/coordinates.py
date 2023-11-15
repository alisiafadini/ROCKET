"""
Functions relating model coordinates/PDB file modificaitons
"""

from rocket import utils
import torch
import numpy as np
from openfold.np import residue_constants
from SFC_Torch import SFcalculator


def write_pdb_with_positions(input_pdb_file, positions, output_pdb_file):
    # positions here expected to be rounded to 3 decimal points

    with open(input_pdb_file, "r") as f_in, open(output_pdb_file, "w") as f_out:
        for line in f_in:
            if line.startswith("ATOM"):
                atom_info = line[
                    :30
                ]  # Extract the first 30 characters containing atom information
                rounded_pos = positions.pop(
                    0
                )  # Pop the first rounded position from the list
                new_line = (
                    f"{atom_info}{rounded_pos[0]:8.3f}{rounded_pos[1]:8.3f}{rounded_pos[2]:8.3f}"
                    + line[54:]
                )
                f_out.write(new_line)
            else:
                f_out.write(line)


def fractionalize_torch(atom_pos_orth, unitcell, spacegroup, device=utils.try_gpu()):
    """
    Apply symmetry operations to real space asu model coordinates

    Parameters
    ----------
    atom_pos_orth: tensor, [N_atom, 3]
        ASU model ccordinates

    Will return fractional coordinates; Otherwise will return orthogonal coordinates

    Return
    ------
    atom_pos_sym_oped, [N_atoms, N_ops, 3] tensor in either fractional or orthogonal coordinates
    """
    atom_pos_orth.to(device=device)
    orth2frac_tensor = torch.tensor(
        unitcell.fractionalization_matrix.tolist(), device=device
    )
    atom_pos_frac = torch.tensordot(atom_pos_orth, orth2frac_tensor.T, 1)

    return atom_pos_frac


def extract_allatoms(outputs, feats):
    atom_types = residue_constants.atom_types
    pdb_lines = []
    atom_mask = outputs["final_atom_mask"]
    aatype = feats["aatype"]
    atom_positions = outputs["final_atom_positions"]
    # residue_index = feats["residue_index"].to(torch.int32)

    n = aatype.shape[0]
    # Add all atom sites.
    for i in range(n):
        for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue
            pdb_lines.append(pos)

    return torch.stack(pdb_lines)


def extract_atoms_and_backbone(outputs, feats):
    atom_types = residue_constants.atom_types
    atom_mask = outputs["final_atom_mask"]
    pdb_lines = []
    aatype = feats["aatype"]
    atom_positions = outputs["final_atom_positions"]
    selected_atoms_mask = []

    n = aatype.shape[0]
    for i in range(n):
        for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue
            pdb_lines.append(pos)

            if atom_name in ["C", "CA", "O", "N"]:
                selected_atoms_mask.append(torch.tensor(1, dtype=torch.bool))
            else:
                selected_atoms_mask.append(torch.tensor(0, dtype=torch.bool))

    return torch.stack(pdb_lines), torch.stack(selected_atoms_mask)


def extract_bfactors(prot):
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    b_factors = prot.b_factors

    b_factor_lines = []

    n = aatype.shape[0]
    # Add all atom sites.
    for i in range(n):
        for mask, b_factor in zip(atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue

            b_factor_lines.append(b_factor)

    return np.array(b_factor_lines)


def kabsch_align_matrices(tensor1, tensor2):
    # Center the atoms by subtracting their centroids
    centroid1 = torch.mean(tensor1, dim=0, keepdim=True)
    tensor1_centered = tensor1 - centroid1
    centroid2 = torch.mean(tensor2, dim=0, keepdim=True)
    tensor2_centered = tensor2 - centroid2

    # Calculate the covariance matrix
    covariance_matrix = torch.matmul(tensor2_centered.t(), tensor1_centered)

    # Perform Singular Value Decomposition (SVD) on the covariance matrix
    U, _, Vt = torch.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = torch.matmul(U, Vt)

    return centroid1, centroid2, rotation_matrix


def select_confident_atoms(current_pos, target_pos, bfacts=None):
    if bfacts is None:
        # If bfacts is None, set mask to all True
        reshaped_mask = torch.ones_like(current_pos, dtype=torch.bool)
    else:
        # Boolean mask for confident atoms
        mask = bfacts < 11.5
        reshaped_mask = mask.unsqueeze(1).expand_as(current_pos)

    # Select confident atoms using the mask
    current_pos_conf = torch.flatten(current_pos)[torch.flatten(reshaped_mask)]
    target_pos_conf = torch.flatten(target_pos)[torch.flatten(reshaped_mask)]

    N = current_pos_conf.numel() // 3

    return current_pos_conf.view(N, 3), target_pos_conf.view(N, 3)


def align_tensors(tensor1, centroid1, centroid2, rotation_matrix):
    tensor1_centered = tensor1 - centroid1

    # Apply the rotation and translation to align the first tensor to the second one
    aligned_tensor1 = torch.matmul(tensor1_centered, rotation_matrix.t()) + centroid2

    return aligned_tensor1


def align_positions(current_pos, target_pos, bfacts=None):
    # Perform alignment of positions
    # TO DO : add statement for when bfacts are not provided
    with torch.no_grad():
        current_pos_conf, target_pos_conf = select_confident_atoms(
            current_pos, target_pos, bfacts
        )

        centroid1, centroid2, rotation_matrix = kabsch_align_matrices(
            current_pos_conf, target_pos_conf
        )

    aligned_pos = align_tensors(current_pos, centroid1, centroid2, rotation_matrix)
    return aligned_pos


def set_new_positions(orth_pos, frac_pos, sfmodel, device=utils.try_gpu()):
    sfmodel.atom_pos_orth = torch.squeeze(orth_pos, dim=1).to(device)
    sfmodel.atom_pos_frac = torch.squeeze(frac_pos, dim=1).to(device)
    return sfmodel


def transfer_positions(
    aligned_pos, sfcalculator_model, b_factors, device=utils.try_gpu()
):
    # Transfer positions to sfcalculator
    frac_pos = fractionalize_torch(
        aligned_pos,
        sfcalculator_model.unit_cell,
        sfcalculator_model.space_group,
        device=device,
    )
    sfcalculator_model = set_new_positions(
        aligned_pos, frac_pos, sfcalculator_model, device=device
    )

    # add bfactor calculation based on plddt
    # sfcalculator_model.atom_b_iso = b_factors

    sfcalculator_model = update_sfcalculator(sfcalculator_model)
    return sfcalculator_model


def update_sfcalculator(sfmodel):
    sfmodel.inspect_data(verbose=False)
    sfmodel.calc_fprotein()
    sfmodel.calc_fsolvent()
    sfmodel.init_scales(requires_grad=True)
    sfmodel.calc_ftotal()
    return sfmodel


def initialize_model_frac_pos(model_file, tng_file, device=utils.try_gpu()):
    sfcalculator_model = SFcalculator(
        model_file,
        tng_file,
        expcolumns=["FP", "SIGFP"],
        set_experiment=True,
        testset_value=0,
        device=device,
    )
    target_pos = sfcalculator_model.atom_pos_orth
    sfcalculator_model.atom_pos_frac = sfcalculator_model.atom_pos_frac * 0.00
    sfcalculator_model.atom_pos_orth = sfcalculator_model.atom_pos_orth * 0.00
    sfcalculator_model.atom_pos_frac.requires_grad = True

    return sfcalculator_model, target_pos
