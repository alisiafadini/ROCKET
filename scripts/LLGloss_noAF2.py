import argparse
import uuid
import copy
import warnings
import torch
import pickle
import numpy as np
from tqdm import tqdm
import rocket
import os
import time
import yaml
from rocket.llg import utils as llg_utils
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from rocket.llg import structurefactors as llg_sf
from openfold.config import model_config


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
        "-it",
        "--iterations",
        required=True,
        type=int,
        help=("Refinement iterations"),
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-3,
        help=("Learning rate for Default 1e-3"),
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
        default=True,
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
    return parser.parse_args()


def llgloss_optimize(
    path: str,
    file_root: str,
    iterations: int,
    learning_rate: float,
    cuda_device: int,
    solvent=True,
    sfc_scale=True,
    refine_sigmaA=True,
    note: str = "",
    free_flag="R-free-flags",
    testset_value=0,
):

    device = "cuda:{}".format(cuda_device)

    # Load external files
    tng_file = "{p}/{r}/{r}-tng_withrfree.mtz".format(p=path, r=file_root)
    input_pdb = "{p}/{r}/{r}-pred-aligned.pdb".format(p=path, r=file_root)

    input_sfc = llg_sf.initial_SFC(
        input_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=free_flag,
        device=device,
        testset_value=testset_value,
        added_chain_HKL=None,
        added_chain_asu=None,
    )

    reference_pos = input_sfc.atom_pos_orth.clone()
    reference_pos.requires_grad = True
    resol_min = min(input_sfc.dHKL)
    resol_max = max(input_sfc.dHKL)

    sfc_rbr = llg_sf.initial_SFC(
        input_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=free_flag,
        device=device,
        testset_value=testset_value,
        added_chain_HKL=None,
        added_chain_asu=None,
    )

    llgloss = rocket.llg.targets.LLGloss(
        input_sfc, tng_file, device, resol_min, resol_max
    )
    llgloss_rbr = rocket.llg.targets.LLGloss(
        sfc_rbr, tng_file, device, resol_min, resol_max
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": reference_pos,
                "lr": learning_rate,
            }
        ]
    )

    progress_bar = tqdm(range(iterations), desc="version LLGnoAF2")

    for iteration in progress_bar:
        optimizer.zero_grad()

        Ecalc, Fc = llgloss.compute_Ecalc(
            reference_pos,
            return_Fc=True,
            update_scales=True,
            added_chain_HKL=None,
            added_chain_asu=None,
        )

        Ecalc_rbr, Fc_rbr = llgloss_rbr.compute_Ecalc(
            reference_pos,
            return_Fc=True,
            solvent=False,
            update_scales=True,
            added_chain_HKL=None,
            added_chain_asu=None,
        )

        if refine_sigmaA is True:

            llgloss.refine_sigmaA_newton(
                Ecalc, n_steps=5, subset="working", smooth_overall_weight=0.0
            )
            llgloss_rbr.refine_sigmaA_newton(
                Ecalc_rbr, n_steps=2, subset="working", smooth_overall_weight=0.0
            )
            # sigmas = llgloss.sigmaAs

        loss = -llgloss(
            reference_pos,
            bin_labels=None,
            num_batch=1,
            sub_ratio=1.0,
            solvent=solvent,
            update_scales=sfc_scale,
            added_chain_HKL=None,
            added_chain_asu=None,
        )
        progress_bar.set_postfix(
            LLG_Estimate=f"{loss:.2f}",
            r_work=f"{llgloss.sfc.r_work.item():.3f}",
            r_free=f"{llgloss.sfc.r_free.item():.3f}",
            memory=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}G",
        )
        loss.backward()
        optimizer.step()


def main():
    args = parse_arguments()
    args_dict = vars(args)
    # print(args_dict)
    llgloss_optimize(**args_dict)


if __name__ == "__main__":
    main()
