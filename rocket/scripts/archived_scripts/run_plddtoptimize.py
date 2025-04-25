import argparse
import glob
import os
import time
import uuid
import warnings

import numpy as np
import torch
from openfold.config import model_config
from SFC_Torch import PDBParser
from tqdm import tqdm

import rocket
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils

PRESET = "model_1"


def int_or_none(value):
    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Must be an integer or 'None'."
        ) from err


def float_or_none(value):
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Must be an float or 'None'."
        ) from err


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
        "--domain_segs",
        type=int,
        nargs="*",
        default=None,
        help=("A list of resid as domain boundaries"),
    )

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

    return parser.parse_args()


def run_plddt_optim(
    working_path: str,
    file_root: str,
    note: str,
    lr_a: float,
    lr_m: float,
    iterations: int,
    init_recycling: int = 20,
    phase2_final_lr: float = 1e-3,
    weight_decay: float | None = 0.0001,
    smooth_stage_epochs: int | None = 50,
    l2_weight: float = 0.0,
    cuda_device: int = 0,
    starting_bias=None,
    starting_weights=None,
    uuid_hex=None,
    domain_segs=None,
):
    device = f"cuda:{cuda_device}"
    input_pdb = f"{working_path}/{file_root}/{file_root}-pred-aligned.pdb"

    run_uuid = uuid.uuid4().hex if uuid_hex is None else uuid_hex

    output_directory_path = f"{working_path}/{file_root}/outputs/{run_uuid}/{note}"
    try:
        os.makedirs(output_directory_path, exist_ok=True)
    except FileExistsError:
        print(
            f"Warning: Directory '{output_directory_path}' already exists. Overwriting..."  # noqa: E501
        )
    print(
        f"System: {file_root}, refinment run ID: {run_uuid!s}, Note: {note}",
        flush=True,
    )
    warnings.filterwarnings("ignore")

    initial_model = PDBParser(input_pdb)
    reference_pos = rk_utils.assert_tensor(
        initial_model.atom_pos, arr_type=torch.float32, device=device
    ).clone()
    init_pos_bfactor = rk_utils.assert_tensor(
        initial_model.atom_b_iso, arr_type=torch.float32, device=device
    ).clone()
    bfactor_weights = rk_utils.weighting_torch(init_pos_bfactor, cutoff2=20.0)

    af_bias = rocket.MSABiasAFv3(model_config(PRESET, train=True), PRESET).to(device)
    af_bias.freeze()  # Free all AF2 parameters to save time

    device_processed_features, feature_key, features_at_it_start = (
        rkrf_utils.init_processed_dict(
            bias_version=3,
            path=working_path,
            file_root=file_root,
            device=device,
            template_pdb=None,
            target_seq=None,
            PRESET=PRESET,
        )
    )

    device_processed_features, optimizer, bias_names = rkrf_utils.init_bias(
        device_processed_features=device_processed_features,
        bias_version=3,
        device=device,
        lr_a=lr_a,
        lr_m=lr_m,
        weight_decay=weight_decay,
        starting_bias=starting_bias,
        starting_weights=starting_weights,
        recombination_bias=None,
    )

    L_plddt_it = []
    all_pldtts = []
    time_by_epoch = []
    memory_by_epoch = []
    best_plddt = 0.0
    progress_bar = tqdm(
        range(iterations),
        desc=f"{file_root}, uuid: {run_uuid[:4]}",
    )

    # Run smooth stage in phase 1
    w_L2 = l2_weight
    early_stopper = rkrf_utils.EarlyStopper(patience=100, min_delta=1.0)

    if smooth_stage_epochs is not None:
        lr_a_initial = lr_a
        lr_m_initial = lr_m
        w_L2_initial = w_L2
        lr_stage1_final = phase2_final_lr

        # Decay rates for each stage
        decay_rate_stage1_add = (lr_stage1_final / lr_a) ** (1 / smooth_stage_epochs)
        decay_rate_stage1_mul = (lr_stage1_final / lr_m) ** (1 / smooth_stage_epochs)

    for iteration in progress_bar:
        start_time = time.time()
        optimizer.zero_grad()

        # Avoid passing through graph a second time
        device_processed_features[feature_key] = features_at_it_start.detach().clone()
        # working_batch = copy.deepcopy(device_processed_features)
        # for bias in bias_names:
        #     working_batch[bias] = device_processed_features[bias].clone()

        # AF pass
        if iteration == 0:
            af2_output, prevs = af_bias(
                device_processed_features,
                [None, None, None],
                num_iters=init_recycling,
                bias=False,
            )
            prevs = [tensor.detach() for tensor in prevs]

            # # MH @ June 19: Fix the iteration 0 for phase 2 running
            # print("config.starting_bias", config.starting_bias)
            # if (config.starting_bias is not None) or (
            #     config.starting_weights is not None
            # ):
            #     deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]
            #     af2_output, __ = af_bias(
            #         device_processed_features,
            #         deep_copied_prevs,
            #         num_iters=1,
            #         bias=True,
            #     )
        deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]
        af2_output, __ = af_bias(
            device_processed_features, deep_copied_prevs, num_iters=1, bias=True
        )

        # pLDDT loss
        L_plddt = -torch.mean(af2_output["plddt"])

        # Position Kabsch Alignment
        aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
            af2_output=af2_output,
            device_processed_features=device_processed_features,
            cra_name=initial_model.cra_name,
            best_pos=reference_pos,
            exclude_res=None,
            domain_segs=domain_segs,
            reference_bfactor=init_pos_bfactor,
        )
        all_pldtts.append(plddts_res)
        L_plddt_it.append(L_plddt.item())

        initial_model.set_positions(rk_utils.assert_numpy(aligned_xyz.detach().clone()))
        initial_model.set_biso(rk_utils.assert_numpy(pseudo_Bs.detach().clone()))
        initial_model.savePDB(f"{output_directory_path!s}/{iteration}_preRBR.pdb")

        if L_plddt_it[-1] < best_plddt:
            best_plddt = L_plddt_it[-1]
            best_msa_bias = (
                device_processed_features["msa_feat_bias"].detach().cpu().clone()
            )
            best_feat_weights = (
                device_processed_features["msa_feat_weights"].detach().cpu().clone()
            )
            best_iter = iteration
            # best_pos = aligned_xyz.detach().clone()

        if w_L2 > 0.0:
            # use
            L2_loss = torch.sum(
                bfactor_weights.unsqueeze(-1) * (aligned_xyz - reference_pos) ** 2
            )  # / conf_best.shape[0]
            loss = w_L2 * L2_loss + L_plddt
            loss.backward()
        else:
            loss = L_plddt
            loss.backward()
            if early_stopper.early_stop(loss.item()):
                break

        # Do smooth in last several iterations of phase 1 instead of beginning of phase 2
        if ("phase1" in note) and (smooth_stage_epochs is not None):
            if iteration > (iterations - smooth_stage_epochs):
                lr_a = lr_a_initial * (decay_rate_stage1_add**iteration)
                lr_m = lr_m_initial * (decay_rate_stage1_mul**iteration)
                w_L2 = w_L2_initial * (1 - (iteration / smooth_stage_epochs))
            # Update the learning rates in the optimizer
            optimizer.param_groups[0]["lr"] = lr_a
            optimizer.param_groups[1]["lr"] = lr_m
            optimizer.step()
        else:
            optimizer.step()

        time_by_epoch.append(time.time() - start_time)
        memory_by_epoch.append(torch.cuda.max_memory_allocated() / 1024**3)
        progress_bar.set_postfix(
            plddt=f"{L_plddt.item():.2f}",
            memory=f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G",
        )

    np.save(
        f"{output_directory_path!s}/mean_it_plddt.npy",
        np.array(L_plddt_it),
    )

    np.save(
        f"{output_directory_path!s}/plddt_res.npy",
        np.array(all_pldtts),
    )

    np.save(
        f"{output_directory_path!s}/time_it.npy",
        rk_utils.assert_numpy(time_by_epoch),
    )

    np.save(
        f"{output_directory_path!s}/memory_it.npy",
        rk_utils.assert_numpy(memory_by_epoch),
    )

    # Save the best msa_bias and feat_weights
    torch.save(
        best_msa_bias,
        f"{output_directory_path!s}/best_msa_bias_{best_iter}.pt",
    )

    torch.save(
        best_feat_weights,
        f"{output_directory_path!s}/best_feat_weights_{best_iter}.pt",
    )

    return run_uuid


def main():
    args = parse_arguments()
    datasets = args.systems

    for file_root in datasets:
        phase1_uuid = run_plddt_optim(
            working_path=args.path,
            file_root=file_root,
            note="phase1" + args.note,
            iterations=100,
            lr_a=args.phase1_add_lr,
            lr_m=args.phase1_mul_lr,
            init_recycling=args.init_recycling,
            weight_decay=0.0001,
            smooth_stage_epochs=args.smooth_stage_epochs,
            l2_weight=args.phase1_w_l2,
            cuda_device=0,
            phase2_final_lr=args.phase2_final_lr,
            starting_bias=None,
            starting_weights=None,
            uuid_hex=None,
            domain_segs=args.domain_segs,
        )

        output_directory_path = f"{args.path}/{file_root}/outputs/{phase1_uuid}"
        phase1_path = glob.glob(f"{output_directory_path}/phase1*/")[0]
        starting_bias_path = glob.glob(os.path.join(phase1_path, "best_msa_bias*.pt"))[
            0
        ]
        starting_weights_path = glob.glob(
            os.path.join(phase1_path, "best_feat_weights*.pt")
        )[0]

        phase2_uuid = run_plddt_optim(
            working_path=args.path,
            file_root=file_root,
            note="phase2" + args.note,
            iterations=500,
            lr_a=args.phase2_final_lr,
            lr_m=args.phase2_final_lr,
            init_recycling=args.init_recycling,
            weight_decay=None,
            smooth_stage_epochs=None,
            l2_weight=0.0,
            cuda_device=0,
            phase2_final_lr=args.phase2_final_lr,
            starting_bias=starting_bias_path,
            starting_weights=starting_weights_path,
            uuid_hex=phase1_uuid,
            domain_segs=args.domain_segs,
        )
