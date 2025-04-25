import argparse

from rocket.refinement_mse import RocketMSERefinmentConfig, run_refinement_mse


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
        "--add_lr",
        default=1e-3,
        type=float,
        help=("additive learning rate"),
    )

    parser.add_argument(
        "--mul_lr",
        default=1e-3,
        type=float,
        help=("multiplicative learning rate"),
    )

    parser.add_argument(
        "--num_of_runs",
        default=2,
        type=int,
        help=("number of trials"),
    )

    parser.add_argument(
        "--n_step",
        default=200,
        type=int,
        help=("number of steps"),
    )

    parser.add_argument(
        "--init_recycling",
        default=20,
        type=int,
        help=("number of initial recycling"),
    )

    parser.add_argument(
        "--free_flag",
        default="R-free-flags",
        type=str,
        help=("Column name of free flag"),
    )

    parser.add_argument(
        "--testset_value",
        default=1,
        type=int,
        help=("Value for test set"),
    )

    parser.add_argument(
        "--note",
        default="",
        help=("note"),
    )

    return parser.parse_args()


def generate_mse_config(
    *,
    working_path: str,
    file_root: str,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    num_of_runs: int = 1,
    n_step: int = 50,
    note: str = "",
    add_lr: float = 0.05,
    mul_lr: float = 1.0,
    init_recycling: int = 20,
) -> RocketMSERefinmentConfig:
    mse_config = RocketMSERefinmentConfig(
        file_root=file_root,
        path=working_path,
        init_recycling=init_recycling,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        bias_version=3,
        num_of_runs=num_of_runs,
        iterations=n_step,
        cuda_device=cuda_device,
        additive_learning_rate=add_lr,
        multiplicative_learning_rate=mul_lr,
        free_flag=free_flag,
        testset_value=testset_value,
        note="mse" + note,
    )

    return mse_config


def run_mse_all_datasets() -> None:
    args = parse_arguments()
    datasets = args.systems
    for file_root in datasets:
        mse_config = generate_mse_config(
            working_path=args.path,
            file_root=file_root,
            note=args.note,
            add_lr=args.add_lr,
            mul_lr=args.mul_lr,
            num_of_runs=args.num_of_runs,
            n_step=args.n_step,
            free_flag=args.free_flag,
            testset_value=args.testset_value,
            init_recycling=args.init_recycling,
        )
        phase1_uuid = run_refinement_mse(config=mse_config)


if __name__ == "__main__":
    run_mse_all_datasets()
