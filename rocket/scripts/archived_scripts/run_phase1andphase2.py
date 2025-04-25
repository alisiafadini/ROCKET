import argparse
import glob
import os

from rocket.refinement_xray import RocketRefinmentConfig, run_refinement


# WORKING_DIRECTORY = Path("/net/cci/alisia/openfold_tests/run_openfold/test_cases")
# ALL_DATASETS = ["6lzm"]
def int_or_none(value):
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Must be an integer or 'None'."
        )


def float_or_none(value):
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Must be an float or 'None'."
        )


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
        "--template_pdb",
        default=None,
        help=("template model name in the path"),
    )

    parser.add_argument(
        "--input_msa",
        default=None,
        help=(
            "path to msa file .a3m/.fasta for use, working path and system will prepend"
        ),
    )

    parser.add_argument(
        "--domain_segs",
        type=int,
        nargs="*",
        default=None,
        help=("A list of resid as domain boundaries"),
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

    parser.add_argument("--phase1_uuid", default=None, help=("uuid for phase 1 run"))

    parser.add_argument("--mse_uuid", default=None, help=("uuid for mse run"))

    parser.add_argument(
        "--init_recycling",
        default=20,
        type=int,
        help=("number of initial recycling"),
    )

    parser.add_argument(
        "--phase1_add_lr",
        default=0.05,
        type=float,
        help=("phase 1 additive learning rate"),
    )

    parser.add_argument(
        "--phase1_mul_lr",
        default=1.0,
        type=float,
        help=("phase 1 multiplicative learning rate"),
    )

    parser.add_argument(
        "--phase1_w_l2",
        default=1e-11,
        type=float,
        help=("phase 1 weights of L2 loss"),
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
        "--phase1_min_resol",
        default=3.0,
        type=float,
        help=("phase 1 resolution cut"),
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
        "--voxel_spacing",
        default=4.5,
        type=float,
        help=("Voxel spacing for solvent percentage estimation"),
    )

    parser.add_argument(
        "--w_plddt",
        default=0.0,
        type=float,
        help=("Weights for plddt loss"),
    )

    parser.add_argument(
        "--msa_subratio",
        default=None,
        type=float_or_none,
        help=(
            "MSA subsampling ratio, between 0.0 and 1.0. Default None, no subsampling."
        ),
    )

    return parser.parse_args()


def generate_phase1_config(
    *,
    working_path: str,
    file_root: str,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    template_pdb: str | None = None,
    input_msa: str | None = None,
    additional_chain: bool = False,
    additive_learning_rate: float = 0.05,
    multiplicative_learning_rate: float = 1.0,
    init_recycling: int = 20,
    phase1_min_resol: float = 4.0,
    phase1_w_l2: float = 1e-3,
    phase2_final_lr: float = 1e-3,
    smooth_stage_epochs: int | None = 50,
    domain_segs: list[int] | None = None,
    note: str = "",
    mse_uuid: str | None = None,
    voxel_spacing: float = 4.5,
    msa_subratio: float | None = None,
) -> RocketRefinmentConfig:
    if mse_uuid is None:
        starting_bias_path = None
        starting_weights_path = None
    else:
        output_directory_path = f"{working_path}/{file_root}/outputs/{mse_uuid}"
        mse_path = glob.glob(f"{output_directory_path}/mse*/")[0]
        starting_bias_path = glob.glob(os.path.join(mse_path, "best_msa_bias*.pt"))[0]
        starting_weights_path = glob.glob(
            os.path.join(mse_path, "best_feat_weights*.pt")
        )[0]

    if template_pdb is not None:
        template_pdb = f"{working_path}/{file_root}/{template_pdb}"

    phase1_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        init_recycling=init_recycling,
        batch_sub_ratio=0.7,
        number_of_batches=1,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        additional_chain=additional_chain,
        template_pdb=template_pdb,
        input_msa=input_msa,
        domain_segs=domain_segs,
        verbose=False,
        bias_version=3,
        num_of_runs=3,
        iterations=100,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=True,
        additive_learning_rate=additive_learning_rate,
        multiplicative_learning_rate=multiplicative_learning_rate,
        phase2_final_lr=phase2_final_lr,
        smooth_stage_epochs=smooth_stage_epochs,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=phase1_w_l2,
        # b_threshold=10.0,
        min_resolution=phase1_min_resol,
        note="phase1" + note,
        uuid_hex=mse_uuid,
        starting_bias=starting_bias_path,
        starting_weights=starting_weights_path,
        voxel_spacing=voxel_spacing,
        msa_subratio=msa_subratio,
    )

    return phase1_config


def generate_phase2_config(
    *,
    phase1_uuid: str | None,
    working_path: str,
    file_root: str,
    cuda_device: int = 0,
    free_flag: str = "R-free-flags",
    testset_value: int = 1,
    additional_chain: bool = False,
    phase1_add_lr: float = 0.05,
    phase1_mul_lr: float = 1.0,
    phase1_w_l2: float = 1e-3,
    phase2_final_lr: float = 1e-3,
    input_msa: str | None = None,
    template_pdb: str | None = None,
    domain_segs: list[int] | None = None,
    voxel_spacing: float = 4.5,
    init_recycling: int = 20,
    note: str = "",
    w_plddt: float = 0.0,
    # msa_subratio: Union[float, None] = None,
) -> RocketRefinmentConfig:
    if phase1_uuid is None:
        starting_bias_path = None
        starting_weights_path = None
    else:
        output_directory_path = f"{working_path}/{file_root}/outputs/{phase1_uuid}"
        phase1_path = glob.glob(f"{output_directory_path}/phase1*/")[0]
        starting_bias_path = glob.glob(os.path.join(phase1_path, "best_msa_bias*.pt"))[
            0
        ]
        starting_weights_path = glob.glob(
            os.path.join(phase1_path, "best_feat_weights*.pt")
        )[0]

    if input_msa is not None:
        msa_feat_init_path = glob.glob(os.path.join(phase1_path, "msa_feat_start.npy"))[
            0
        ]
    else:
        msa_feat_init_path = None
        # best_runid = os.path.basename(starting_bias_path).split("_")[-2]
        # if msa_subratio is not None:
        #     sub_msa_path = glob.glob(os.path.join(phase1_path, f"sub_msa_{best_runid}.npy"))[0]
        #     sub_delmat_path = glob.glob(os.path.join(phase1_path, f"sub_delmat_{best_runid}.npy"))[0]
        # else:
        #     sub_msa_path = None
        #     sub_delmat_path = None

    if template_pdb is not None:
        template_pdb = f"{working_path}/{file_root}/{template_pdb}"

    phase2_config = RocketRefinmentConfig(
        file_root=file_root,
        path=working_path,
        batch_sub_ratio=1.0,
        number_of_batches=1,
        init_recycling=init_recycling,
        rbr_opt_algorithm="lbfgs",
        rbr_lbfgs_learning_rate=150.0,
        smooth_stage_epochs=50,
        additional_chain=additional_chain,
        input_msa=input_msa,
        template_pdb=template_pdb,
        domain_segs=domain_segs,
        verbose=False,
        bias_version=3,
        iterations=500,
        cuda_device=cuda_device,
        solvent=True,
        sfc_scale=True,
        refine_sigmaA=True,
        additive_learning_rate=phase1_add_lr,
        multiplicative_learning_rate=phase1_mul_lr,
        phase2_final_lr=phase2_final_lr,
        weight_decay=None,
        free_flag=free_flag,
        testset_value=testset_value,
        l2_weight=phase1_w_l2,
        w_plddt=w_plddt,
        b_threshold=10.0,
        note="phase2" + note,
        uuid_hex=phase1_uuid,
        starting_bias=starting_bias_path,
        starting_weights=starting_weights_path,
        voxel_spacing=voxel_spacing,
        msa_subratio=None,
        msa_feat_init_path=msa_feat_init_path,
        # sub_msa_path=sub_msa_path,
        # sub_delmat_path=sub_delmat_path,
    )

    return phase2_config


def run_both_phases_single_dataset(
    *,
    working_path,
    file_root,
    note,
    free_flag,
    testset_value,
    additional_chain,
    phase1_add_lr,
    phase1_mul_lr,
    phase1_w_l2,
    w_plddt,
    phase2_final_lr,
    smooth_stage_epochs,
    init_recycling,
    phase1_min_resol,
    domain_segs,
    mse_uuid,
    voxel_spacing,
    template_pdb,
    msa_subratio,
    input_msa,
) -> None:
    phase1_config = generate_phase1_config(
        working_path=working_path,
        file_root=file_root,
        note=note,
        free_flag=free_flag,
        testset_value=testset_value,
        additional_chain=additional_chain,
        additive_learning_rate=phase1_add_lr,
        multiplicative_learning_rate=phase1_mul_lr,
        phase1_w_l2=phase1_w_l2,
        init_recycling=init_recycling,
        phase2_final_lr=phase2_final_lr,
        smooth_stage_epochs=smooth_stage_epochs,
        phase1_min_resol=phase1_min_resol,
        template_pdb=template_pdb,
        input_msa=input_msa,
        domain_segs=domain_segs,
        mse_uuid=mse_uuid,
        voxel_spacing=voxel_spacing,
        msa_subratio=msa_subratio,
    )
    phase1_uuid = run_refinement(config=phase1_config)

    phase2_config = generate_phase2_config(
        phase1_uuid=phase1_uuid,
        working_path=working_path,
        file_root=file_root,
        note=note,
        free_flag=free_flag,
        testset_value=testset_value,
        additional_chain=additional_chain,
        phase1_add_lr=phase1_add_lr,
        phase1_mul_lr=phase1_mul_lr,
        phase1_w_l2=phase1_w_l2,
        w_plddt=w_plddt,
        phase2_final_lr=phase2_final_lr,
        init_recycling=init_recycling,
        input_msa=input_msa,
        template_pdb=template_pdb,
        domain_segs=domain_segs,
        voxel_spacing=voxel_spacing,
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
                free_flag=args.free_flag,
                testset_value=args.testset_value,
                additional_chain=args.additional_chain,
                additive_learning_rate=args.phase1_add_lr,
                multiplicative_learning_rate=args.phase1_mul_lr,
                phase1_w_l2=args.phase1_w_l2,
                phase2_final_lr=args.phase2_final_lr,
                smooth_stage_epochs=args.smooth_stage_epochs,
                init_recycling=args.init_recycling,
                phase1_min_resol=args.phase1_min_resol,
                template_pdb=args.template_pdb,
                input_msa=args.input_msa,
                domain_segs=args.domain_segs,
                mse_uuid=args.mse_uuid,
                voxel_spacing=args.voxel_spacing,
                msa_subratio=args.msa_subratio,
            )
            phase1_uuid = run_refinement(config=phase1_config)

        elif args.only_phase2:
            phase2_config = generate_phase2_config(
                phase1_uuid=args.phase1_uuid,
                working_path=args.path,
                free_flag=args.free_flag,
                testset_value=args.testset_value,
                file_root=file_root,
                note=args.note,
                additional_chain=args.additional_chain,
                phase1_add_lr=args.phase1_add_lr,
                phase1_mul_lr=args.phase1_mul_lr,
                phase1_w_l2=args.phase1_w_l2,
                w_plddt=args.w_plddt,
                phase2_final_lr=args.phase2_final_lr,
                init_recycling=args.init_recycling,
                input_msa=arg.input_msa,
                template_pdb=args.template_pdb,
                domain_segs=args.domain_segs,
                voxel_spacing=args.voxel_spacing,
            )
            phase2_uuid = run_refinement(config=phase2_config)

        else:
            run_both_phases_single_dataset(
                working_path=args.path,
                file_root=file_root,
                note=args.note,
                free_flag=args.free_flag,
                testset_value=args.testset_value,
                additional_chain=args.additional_chain,
                phase1_add_lr=args.phase1_add_lr,
                phase1_mul_lr=args.phase1_mul_lr,
                phase1_w_l2=args.phase1_w_l2,
                w_plddt=args.w_plddt,
                smooth_stage_epochs=args.smooth_stage_epochs,
                phase2_final_lr=args.phase2_final_lr,
                init_recycling=args.init_recycling,
                phase1_min_resol=args.phase1_min_resol,
                template_pdb=args.template_pdb,
                domain_segs=args.domain_segs,
                mse_uuid=args.mse_uuid,
                voxel_spacing=args.voxel_spacing,
                msa_subratio=args.msa_subratio,
                input_msa=args.input_msa,
            )


if __name__ == "__main__":
    run_both_phases_all_datasets()
