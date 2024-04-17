import argparse, os
from rocket.refinement import RocketRefinmentConfig, run_refinement

# WORKING_DIRECTORY = Path("/net/cci/alisia/openfold_tests/run_openfold/test_cases")
# ALL_DATASETS = ["6lzm"]

def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    parser.add_argument(
        "-p",
        "--path",
        default="./benchmark_data/",
        help=("Path to the parent folder"),
    )

    # Required arguments
    parser.add_argument(
        "-sys",
        "--systems",
        nargs='+',
        help=("PDB codes or filename roots for the dataset"),
    )

    parser.add_argument(
        "--note",
        default="",
        help=("note"),
    )
    
    return parser.parse_args()


def generate_phase1_config(
    *,
    working_path: str,
    file_root: str,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    note: str = ""
) -> RocketRefinmentConfig:

    phase1_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        batch_sub_ratio=0.7,
        number_of_batches=1,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        alignment_mode="B",
        additional_chain=False,
        verbose=False,
        bias_version=3,
        iterations=50,
        # iterations=2,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=True,
        additive_learning_rate=0.05,
        multiplicative_learning_rate=1.0,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=1e-11,
        b_threshold=10.0,
        min_resolution=3.0,
        note="phase1"+note,
    )

    return phase1_config


def generate_phase2_config(
    *,
    phase1_uuid: str,
    working_path: str,
    file_root: str,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    note : str = "",
) -> RocketRefinmentConfig:

    output_directory_path = f"{working_path}/{file_root}/outputs/{phase1_uuid}"

    starting_bias_path = f"{output_directory_path}/phase1{note}/best_msa_bias.pt"
    starting_weights_path = f"{output_directory_path}/phase1{note}/best_feat_weights.pt"

    for p in [starting_bias_path, starting_weights_path]:
        if not os.path.exists(p):
            raise IOError(f"no: {p}")

    phase2_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        batch_sub_ratio=1.0,
        number_of_batches=1,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        alignment_mode="B",
        additional_chain=False,
        verbose=False,
        bias_version=3,
        iterations=300,
        # iterations=2,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=True,
        additive_learning_rate=0.001,
        multiplicative_learning_rate=0.001,
        weight_decay=0.0001,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=0.0,
        b_threshold=10.0,
        note="phase2"+note,
        uuid_hex=phase1_uuid,
        starting_bias=starting_bias_path,
        starting_weights=starting_weights_path,
    )

    return phase2_config


def run_both_phases_single_dataset(*, working_path, file_root, note) -> None:

    phase1_config = generate_phase1_config(working_path=working_path, file_root=file_root, note=note)
    phase1_uuid = run_refinement(config=phase1_config)

    phase2_config = generate_phase2_config(phase1_uuid=phase1_uuid, working_path=working_path, file_root=file_root, note=note)
    phase2_uuid = run_refinement(config=phase2_config)


def run_both_phases_all_datasets() -> None:
    args = parse_arguments()
    datasets = args.systems
    for file_root in datasets:
        run_both_phases_single_dataset(working_path=args.path, file_root=file_root, note=args.note)

if __name__ == "__main__":
    run_both_phases_all_datasets()
