"""
Functions relating model coordinates/PDB file modificaitons
"""

from rocket import utils
import torch


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
