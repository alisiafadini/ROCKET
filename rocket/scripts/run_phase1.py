import argparse, os, glob
from rocket.refinement import RocketRefinmentConfig, run_refinement
from typing import Union, List


def int_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be an integer or 'None'.")
    

def float_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be an float or 'None'.")

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
        "--template_pdb",
        default=None,
        help=("template model name in the path"),
    )

    parser.add_argument(
        "--domain_segs",
        type=int,
        nargs="*",
        default=None,
        help=("A list of resid as domain boundaries"),
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
        "--sub_ratio",
        default=0.7,
        type=float,
        help=("Sub ratio of reflectios for each iteratio"),
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
        "--mse_uuid", default=None, help=("uuid for mse run")
    )

    parser.add_argument(
        "--phase2_final_lr",
        default=1e-3,
        type=float,
        help=("phase 2 final learning rate"),
    )

    parser.add_argument(
        "--smooth_stage_epochs",
        default=50,
        type=int_or_none,
        help=("number of smooth stages in phase1"),
    )
    parser.add_argument(
        "--num_of_runs",
        default=3,
        type=int,
        help=("number of trials"),
    )

    parser.add_argument(
        "--n_step",
        default=100,
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
        default=3.0,
        type=float,
        help=("min resolution cut"),
    )

    parser.add_argument(
        "--init_recycling",
        default=20,
        type=int,
        help=("number of initial recycling"),
    )

    parser.add_argument(
        "--note",
        default="",
        help=("note"),
    )

    parser.add_argument(
        "--voxel_spacing",
        default=4.5,
        type=float,
        help=("Voxel spacing for solvent percentage estimation"),
    )

    parser.add_argument(
        "--msa_subratio",
        default=None,
        type=float_or_none,
        help=("MSA subsampling ratio, between 0.0 and 1.0. Default None, no subsampling."),
    )

    parser.add_argument(
        "--input_msa",
        default=None,
        help=("path to msa file .a3m/.fasta for use, working path and system will prepend"),
    )

    parser.add_argument(
        "--bias_from_fullmsa", 
        action="store_true",
        help=("Generate initial profile via biasing from full msa profile")
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
    num_of_runs: int = 1,
    sub_ratio: float = 0.7, 
    n_step: int = 50,
    init_recycling: int = 20,
    min_resol: float = 3.0,
    note: str = "",
    refine_sigmaA: bool = True,
    input_msa: Union[str, None] = None,
    bias_from_fullmsa: bool = False,
    template_pdb: Union[str, None] = None,
    domain_segs: Union[List[int], None] = None,
    add_lr: float = 0.05,
    mul_lr: float = 1.0,
    phase2_final_lr: float = 1e-3,
    smooth_stage_epochs: int = 50,
    mse_uuid: Union[str, None] = None,
    voxel_spacing: float = 4.5,
    msa_subratio: Union[float, None] = None,
) -> RocketRefinmentConfig:

    if mse_uuid is None:
        starting_bias_path = None
        starting_weights_path = None
    else:
        output_directory_path = f"{working_path}/{file_root}/outputs/{mse_uuid}"
        mse_path = glob.glob(f"{output_directory_path}/mse*/")[0]
        starting_bias_path = glob.glob(os.path.join(mse_path, "best_msa_bias*.pt"))[
            0
        ]
        starting_weights_path = glob.glob(
            os.path.join(mse_path, "best_feat_weights*.pt")
        )[0]
    
    if template_pdb is not None:
        template_pdb = f"{working_path}/{file_root}/{template_pdb}"

    phase1_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        init_recycling=init_recycling,
        batch_sub_ratio=sub_ratio,
        number_of_batches=1,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        additional_chain=additional_chain,
        domain_segs=domain_segs,
        verbose=False,
        bias_version=3,
        num_of_runs=num_of_runs,
        iterations=n_step,
        # iterations=2,
        input_msa=input_msa,
        bias_from_fullmsa=bias_from_fullmsa,
        template_pdb=template_pdb,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=refine_sigmaA,
        additive_learning_rate=add_lr,
        multiplicative_learning_rate=mul_lr,
        phase2_final_lr=phase2_final_lr,
        smooth_stage_epochs=smooth_stage_epochs,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=w_l2,
        min_resolution=min_resol,
        starting_bias=starting_bias_path,
        starting_weights=starting_weights_path,
        note="phase1"+note,
        voxel_spacing=voxel_spacing,
        msa_subratio=msa_subratio,
    )

    return phase1_config



def run_phase1_all_datasets() -> None:
    args = parse_arguments()
    datasets = args.systems
    for file_root in datasets:
        phase1_config = generate_phase1_config(working_path=args.path, 
                                               file_root=file_root, 
                                               note=args.note,
                                               free_flag=args.free_flag,
                                               testset_value=args.testset_value, 
                                               additional_chain=args.additional_chain,
                                               domain_segs=args.domain_segs,
                                               init_recycling=args.init_recycling,
                                               w_l2=args.w_l2,
                                               sub_ratio=args.sub_ratio,
                                               add_lr=args.add_lr,
                                               mul_lr=args.mul_lr,
                                               num_of_runs=args.num_of_runs,
                                               n_step=args.n_step,
                                               input_msa=args.input_msa,
                                               bias_from_fullmsa=args.bias_from_fullmsa,
                                               template_pdb=args.template_pdb,
                                               refine_sigmaA=args.refine_sigmaA,
                                               min_resol=args.min_resolution,
                                               phase2_final_lr=args.phase2_final_lr,
                                               smooth_stage_epochs=args.smooth_stage_epochs,
                                               mse_uuid=args.mse_uuid,
                                               voxel_spacing=args.voxel_spacing,
                                               msa_subratio=args.msa_subratio,
                                               )
        phase1_uuid = run_refinement(config=phase1_config)

if __name__ == "__main__":
    run_phase1_all_datasets()