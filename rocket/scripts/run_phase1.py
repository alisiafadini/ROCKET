import argparse, os
from rocket.refinement import RocketRefinmentConfig, run_refinement
from typing import Union


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
        "--additional_chain", 
        action="store_true",
        help=("Additional Chain in ASU")
    )

    parser.add_argument(
        "--w_l2",
        default=1e-11,
        type=float,
        help=("weight for L2 loss"),
    )

    parser.add_argument(
        "--add_lr",
        default=0.05,
        type=float,
        help=("additive learning rate"),
    )

    parser.add_argument(
        "--mul_lr",
        default=1.0,
        type=float,
        help=("multiplicative learning rate"),
    )

    parser.add_argument(
        "--n_step",
        default=50,
        type=int,
        help=("number of steps"),
    )

    parser.add_argument(
        "--refine_sigmaA", 
        action="store_true",
        help=("refine sigmaAs")
    )

    parser.add_argument(
        "--min_resolution",
        default=0.30,
        type=float,
        help=("min resolution cut"),
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
    additional_chain: bool = False,
    w_l2: float = 1e-11,
    n_step: int = 50,
    min_resol: float = 0.3,
    note: str = "",
    refine_sigmaA: bool = True,
    add_lr: float = 0.05,
    mul_lr: float = 1.0,
) -> RocketRefinmentConfig:

    phase1_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        batch_sub_ratio=0.7,
        number_of_batches=1,
        kabsch_threshB=400.0,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        alignment_mode="B",
        additional_chain=additional_chain,
        verbose=False,
        bias_version=3,
        iterations=n_step,
        # iterations=2,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=refine_sigmaA,
        additive_learning_rate=add_lr,
        multiplicative_learning_rate=mul_lr,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=w_l2,
        b_threshold=10.0,
        min_resolution=min_resol,
        note="phase1"+note,
    )

    return phase1_config



def run_phase1_all_datasets() -> None:
    args = parse_arguments()
    datasets = args.systems
    for file_root in datasets:
        phase1_config = generate_phase1_config(working_path=args.path, 
                                               file_root=file_root, 
                                               note=args.note, 
                                               additional_chain=args.additional_chain,
                                               w_l2=args.w_l2,
                                               add_lr=args.add_lr,
                                               mul_lr=args.mul_lr,
                                               n_step=args.n_step,
                                               refine_sigmaA=args.refine_sigmaA,
                                               min_resol=args.min_resolution)
        phase1_uuid = run_refinement(config=phase1_config)

if __name__ == "__main__":
    run_phase1_all_datasets()