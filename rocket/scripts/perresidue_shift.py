import gemmi
import numpy as np
import argparse
import re
import pickle


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-root",
        "--file_root",
        required=True,
        help=("File path for directory to data"),
    )

    parser.add_argument(
        "-prot",
        "--protein_type",
        required=True,
        help=("Type of protein (prion, lyso, or abl)"),
    )

    return parser.parse_args()


def read_pdb(file_path):
    """Read PDB file and return a gemmi.Structure."""
    structure = gemmi.read_structure(file_path)
    return structure


def calculate_perresidue_distance(structure1, structure2):
    """Calculate the mean distance between atoms in corresponding residues."""
    mean_distances = []
    for model1, model2 in zip(structure1, structure2):
        for chain1, chain2 in zip(model1, model2):
            for residue1, residue2 in zip(chain1, chain2):
                if len(residue1) == len(
                    residue2
                ):  # Check if residues have the same number of atoms
                    distances = [
                        np.linalg.norm(
                            np.array(a1.pos.tolist()) - np.array(a2.pos.tolist())
                        )
                        for a1, a2 in zip(residue1, residue2)
                    ]
                    mean_distance = np.mean(distances)
                    mean_distances.append(mean_distance)
    return mean_distances


def main():
    args = parse_arguments()
    path = args.file_root
    iterations = int(re.search(r"it(\d+)", path).group(1))

    # Ranges of interest
    protein_to_ranges = {
        "prion": [(0, 3), (10, 16), (40, 50), (65, 75), (86, 89), (95, 103)],
        "lyso": [(7, 12), (14, 17), (18, 25), (37, 48), (49, 56), (57, 77)],
        "abl": [(16, 26), (27, 36), (38, 48), (70, 80), (145, 170), (195, 200)],
    }

    ranges = protein_to_ranges[args.protein_type]

    bfactor_data = {range_: [] for range_ in ranges}

    residue_numbers = []
    mean_distances_list = []
    mean_distances_tostart_list = []

    for i in range(iterations - 1):
        pdb_file1 = f"{path}/{i}_postRBR.pdb"
        pdb_file2 = f"{path}/{i+1}_postRBR.pdb"

        structure1 = read_pdb(pdb_file1)
        structure2 = read_pdb(pdb_file2)

        if i == 0:
            structure_start = structure1

        perres_distances = calculate_perresidue_distance(structure1, structure2)
        perres_distaces_tostart = calculate_perresidue_distance(
            structure_start, structure2
        )
        mean_distances_list.append(perres_distances)
        mean_distances_tostart_list.append(perres_distaces_tostart)

        # Assuming all structures have the same residue count, so just use the first model and chain
        residue_numbers = [residue.seqid.num for residue in structure1[0][0]]

        # Initialize a dictionary to store B-factor values for each residue range in the current iteration
        iteration_data = {range_: [] for range_ in ranges}

        # Iterate through atoms
        for chain in structure1[0]:
            for residue in chain:
                residue_number = residue.seqid.num

                # Check if the residue is within any specified range
                for range_ in ranges:
                    if range_[0] <= residue_number <= range_[1]:
                        # Extract B-factor value
                        bfactor = residue[0].b_iso

                        # Append B-factor value to the corresponding range and iteration
                        iteration_data[range_].append(bfactor)

        # Store the mean B-factor for each residue range in the current iteration
        for range_, data_b in iteration_data.items():
            mean_b_factor = np.mean(data_b)
            bfactor_data[range_].append((i, mean_b_factor))

            if i == 8:
                bfactor_data[range_].append((i + 1, mean_b_factor))

    mean_distances_array = np.array(mean_distances_list)
    mean_perresidue = np.mean(mean_distances_array, axis=0)

    mean_distances_tostart_array = np.array(mean_distances_tostart_list)
    mean_perresidue_tostart = np.mean(mean_distances_tostart_array, axis=0)

    np.save("{}/meanshift_perresidue.npy".format(path), mean_perresidue)
    np.save("{}/meanshift_perresidue_tostart.npy".format(path), mean_perresidue_tostart)
    np.save("{}/residue_numbers.npy".format(path), np.array(residue_numbers))

    with open(
        "{}/pseudoB_lineouts_data.pkl".format(path),
        "wb",
    ) as file:
        pickle.dump(bfactor_data, file)


if __name__ == "__main__":
    main()
