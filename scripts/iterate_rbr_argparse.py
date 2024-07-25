"""
Command line interface of running refinement using rocket

rk.refine 
    --path             xxxxx             # Path to the parent folder
    --file_root        xxxxx             # Dataset folder name in the parent folder
    --version          1                 # Bias version of implementation, 1, 2 or 3 or 4 (template)
    --template_pdb     xxxx.pdb          # Name of template pdb file in the file_root
    --iterations       300               # Number of refinement steps
    --lr_add           1e-3              # Learning rate of msa_bias
    --lr_mul           1e-2              # Learning rate of msa_weights
    --weight_decay     None or float     # Weight decay parameters used in AdamW
    --sub_ratio        1.0               # Ratio of reflections for each batch
    --batches          1                 # Number of batches at each step
    --rbr_opt          'lbfgs'           # Using 'lbfgs' or 'adam' in the rbr optimization
    --rbr_lbfgs_lr     150.0             # Learning rate of lbfgs used in RBR
    --align            'B'               # Kabsch to best (B) or initial (I)
    --note             xxxx              # Additional notes used in output name
    --free_flag        'R-free-flags'    # Coloum name for the free flag in mtz file
    --testset_value    0                 # testset value in the freeflag column
    --solvent or --no-solvent            # Turn on the solvent in the llgloss calculation
    --scale   or --no-scale              # Turn on the SFC update_scale in each step
    --added_chain                        # Turn on additional chain in the asu
    --verbose                            # Be verbose during refinement
"""

import argparse
from rocket.refinement import run_refinement, RocketRefinmentConfig


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    parser.add_argument(
        "-p",
        "--path",
        default="/net/cci/alisia/openfold_tests/run_openfold/test_cases",
        help=("Path to the parent folder"),
    )

    # Required arguments
    parser.add_argument(
        "-root",
        "--file_root",
        required=True,
        help=("PDB code or filename root for the dataset"),
    )

    parser.add_argument(
        "-v",
        "--bias_version",
        required=True,
        type=int,
        help=("Bias version to implement (1, 2, 3, 4)"),
    )

    parser.add_argument(
        "-it",
        "--iterations",
        required=True,
        type=int,
        help=("Refinement iterations"),
    )

    # Optional arguments
    parser.add_argument(
        "-temp",
        "--template_pdb",
        default=None,
        help=("Name of template pdb file in the file_root"),
    )

    parser.add_argument("-c", "--cuda_device", type=int, default=0, help="Cuda device")
    parser.add_argument(
        "-solv",
        "--solvent",
        help="Turn on solvent calculation in refinement step",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-s",
        "--sfc_scale",
        help="Turn on SFC scale_update at each epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "-sigmaA",
        "--refine_sigmaA",
        help="Turn on sigmaA refinement at each epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "-lr_a",
        "--additive_learning_rate",
        type=float,
        default=1e-3,
        help=("Learning rate for additive bias. Default 1e-3"),
    )

    parser.add_argument(
        "-lr_m",
        "--multiplicative_learning_rate",
        type=float,
        default=1e-2,
        help=("Learning rate for multiplicative bias. Default 1e-2"),
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help=("Weight decay used in adamW. Default None, use adam"),
    )

    parser.add_argument(
        "-sub_r",
        "--batch_sub_ratio",
        type=float,
        default=1.0,
        help=("Ratio of reflections for each batch. Default 1.0 (no batching)"),
    )

    parser.add_argument(
        "-b",
        "--number_of_batches",
        type=int,
        default=1,
        help=("Number of batches. Default 1 (no batching)"),
    )

    parser.add_argument(
        "--rbr_opt_algorithm",
        type=str,
        default="lbfgs",
        help=("Optimization algorithm used in RBR, lbfgs or adam"),
    )

    parser.add_argument(
        "--rbr_lbfgs_learning_rate",
        type=float,
        default=150.0,
        help=("Learning rate of lbfgs used in RBR"),
    )

    parser.add_argument(
        "-a",
        "--alignment_mode",
        type=str,
        default="B",
        choices=["B", "I"],
        help=("Kabsch to best (B) or initial (I)"),
    )

    parser.add_argument(
        "-n",
        "--note",
        type=str,
        default="",
        help=("Optional additional identified"),
    )

    parser.add_argument(
        "-flag",
        "--free_flag",
        type=str,
        default="R-free-flags",
        help=("Optional additional identified"),
    )

    parser.add_argument(
        "--testset_value",
        type=int,
        default=0,
        help=("Optional additional identified"),
    )

    parser.add_argument(
        "-chain",
        "--additional_chain",
        help="additional chain in asu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--verbose",
        help="Be verbose during refinement",
        action="store_true",
    )

    parser.add_argument(
        "-L2_w",
        "--l2_weight",
        type=float,
        default=0.0,
        help=("Weight for L2 loss"),
    )

    parser.add_argument(
        "-b_thresh",
        "--b_threshold",
        type=float,
        default=10.0,
        help=("B threshold for L2 loss"),
    )

    parser.add_argument(
        "--min_resolution",
        type=float,
        default=None,
        help=("min resolution for llg calculation"),
    )

    parser.add_argument(
        "--max_resolution",
        type=float,
        default=None,
        help=("max resolution for llg calculation"),
    )

    parser.add_argument(
        "--starting_bias",
        type=str,
        default=None,
        help=("initial bias to start with"),
    )

    parser.add_argument(
        "--starting_weights",
        type=str,
        default=None,
        help=("initial weights to start with"),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    args_dict: dict[str, Any] = vars(args)
    config = RocketRefinmentConfig(**args_dict)
    run_refinement(config=config)


if __name__ == "__main__":
    main()
