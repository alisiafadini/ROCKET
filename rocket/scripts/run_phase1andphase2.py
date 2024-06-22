import argparse, os, glob
from rocket.refinement import RocketRefinmentConfig, run_refinement
from typing import Union

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
        nargs="+",
        help=("PDB codes or filename roots for the dataset"),
    )

    parser.add_argument(
        "--note",
        default="",
        help=("note"),
    )

    parser.add_argument(
        "--additional_chain", action="store_true", help=("Additional Chain in ASU")
    )

    parser.add_argument(
        "--only_phase1",
        action="store_true",
        help=("Turn off phase 2 refinement in the pipeline"),
    )

    parser.add_argument(
        "--only_phase2",
        action="store_true",
        help=("Turn off phase 1 refinement in the pipeline"),
    )

    parser.add_argument(
        "--phase1_uuid", default=None, help=("uuid for phase 1 running")
    )

    return parser.parse_args()


def generate_phase1_config(
    *,
    working_path: str,
    file_root: str,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    additional_chain: bool = False,
    note: str = "",
) -> RocketRefinmentConfig:

    phase1_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        init_recycling=20,
        batch_sub_ratio=0.7,
        number_of_batches=1,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        alignment_mode="B",
        additional_chain=additional_chain,
        verbose=False,
        bias_version=3,
        num_of_runs=3,
        iterations=20,
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
        note="phase1" + note,
    )

    return phase1_config


def generate_phase2_config(
    *,
    phase1_uuid: Union[str, None],
    working_path: str,
    file_root: str,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    additional_chain: bool = False,
    note: str = "",
) -> RocketRefinmentConfig:

    if phase1_uuid is None:
        starting_bias_path = None
        starting_weights_path = None
    else:
        output_directory_path = f"{working_path}/{file_root}/outputs/{phase1_uuid}"
        phase1_path = glob.glob(f"{output_directory_path}/phase1*/")[0]
        starting_bias_path = glob.glob(os.path.join(phase1_path, "best_msa_bias*.pt"))[0]
        starting_weights_path = glob.glob(os.path.join(phase1_path, "best_feat_weights*.pt"))[0]

    for p in [starting_bias_path, starting_weights_path]:
        if not os.path.exists(p):
            raise IOError(f"no: {p}")

    phase2_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        batch_sub_ratio=1.0,
        number_of_batches=1,
        init_recycling=20,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        alignment_mode="B",
        additional_chain=additional_chain,
        verbose=False,
        bias_version=3,
        # iterations=300,
        iterations=40,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=True,
        additive_learning_rate=1e-3,
        multiplicative_learning_rate=1e-3,
        weight_decay=None,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=0.0,
        b_threshold=10.0,
        note="phase2" + note,
        uuid_hex=phase1_uuid,
        starting_bias=starting_bias_path,
        starting_weights=starting_weights_path,
        # starting_bias="/net/cci/alisia/large_dataset_folders/inputs_fromMH/benchmark_data/7pgk/outputs/eccedf4a7f9e4817bb28faa08241ecf0/phase2patienceboth50+150/best_msa_bias.pt",
        # starting_weights="/net/cci/alisia/large_dataset_folders/inputs_fromMH/benchmark_data/7pgk/outputs/eccedf4a7f9e4817bb28faa08241ecf0/phase2patienceboth50+150/best_feat_weights.pt",
    )

    return phase2_config


def run_both_phases_single_dataset(
    *, working_path, file_root, note, additional_chain
) -> None:

    phase1_config = generate_phase1_config(
        working_path=working_path,
        file_root=file_root,
        note=note,
        additional_chain=additional_chain,
    )
    phase1_uuid = run_refinement(config=phase1_config)

    phase2_config = generate_phase2_config(
        phase1_uuid=phase1_uuid,
        working_path=working_path,
        file_root=file_root,
        note=note,
        additional_chain=additional_chain,
    )
    phase2_uuid = run_refinement(config=phase2_config)


def run_both_phases_all_datasets() -> None:

    args = parse_arguments()
    datasets = args.systems

    for file_root in datasets:
        if args.only_phase1:
            phase1_config = generate_phase1_config(
                working_path=args.path,
                file_root=file_root,
                note=args.note,
                additional_chain=args.additional_chain,
            )
            phase1_uuid = run_refinement(config=phase1_config)

        elif args.only_phase2:
            phase2_config = generate_phase2_config(
                phase1_uuid=args.phase1_uuid,
                working_path=args.path,
                file_root=file_root,
                note=args.note,
                additional_chain=args.additional_chain,
            )
            phase2_uuid = run_refinement(config=phase2_config)

        else:
            run_both_phases_single_dataset(
                working_path=args.path,
                file_root=file_root,
                note=args.note,
                additional_chain=args.additional_chain,
            )


if __name__ == "__main__":
    run_both_phases_all_datasets()
