import argparse
import copy
import warnings
import torch
import pickle, json
import numpy as np
from tqdm import tqdm
import rocket
import os
import time
from rocket import MSEloss
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from SFC_Torch import PDBParser
from openfold.config import model_config
from typing import Union

PRESET = "model_1"
EXCLUDING_RES = None

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
        "-target",
        "--target_pdb",
        default=None,
        help=("Name of target pdb file in the file_root"),
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
        "--device",
        type=str,
        default="cuda:0",
        help=("Device to run the model"),
    )

    parser.add_argument(
        "-n",
        "--note",
        type=str,
        default="",
        help=("Optional additional identified"),
    )

    return parser.parse_args()

def mse_optimize(
        path: str,
        file_root: str,
        bias_version: int,
        iterations: int,
        target_pdb: str,
        device: str,
        additive_learning_rate: float,
        multiplicative_learning_rate: float,
        weight_decay: Union[float, None] = 0.0001,
        note: str = "",
    ):

    input_pdb_path = "{p}/{r}/{r}-pred-aligned.pdb".format(p=path, r=file_root)
    target_pdb_path = "{p}/{r}/{t}".format(p=path, r=file_root, t=target_pdb)
    target_pdb_obj = PDBParser(target_pdb_path)
    moving_pdb_obj = PDBParser(input_pdb_path)

    mseloss = MSEloss(target_pdb_obj, moving_pdb_obj, device=device)

    starting_bias = None,
    starting_weights = None,

    if bias_version == 4:
        device_processed_features = rocket.make_processed_dict_from_template(
            template_pdb="{p}/{r}/{t}".format(
                p=path, r=file_root, t=target_pdb
            ),
            config_preset=PRESET,
            device=device,
            msa_dict=None,
        )
    else:
        with open(
            "{p}/{r}/{r}_processed_feats.pickle".format(p=path, r=file_root),
            "rb",
        ) as file:
            # Load the data from the pickle file
            processed_features = pickle.load(file)

        device_processed_features = rk_utils.move_tensors_to_device(
            processed_features, device=device
        )
        # TODO: this still takes up memory in original device?
        del processed_features
    
    # Model initialization
    version_to_class = {
        1: rocket.MSABiasAFv1,
        2: rocket.MSABiasAFv2,
        3: rocket.MSABiasAFv3,
        4: rocket.TemplateBiasAF,
    }
    af_bias = version_to_class[bias_version](
        model_config(PRESET, train=True), PRESET
    ).to(device)
    # af_bias.eval()
    af_bias.freeze()  # Free all AF2 parameters to save time

    # Initiate additive cluster bias
    num_res = device_processed_features["aatype"].shape[0]
    msa_params_bias = torch.zeros((512, num_res, 23), requires_grad=True, device=device)
    device_processed_features["msa_feat_bias"] = msa_params_bias

    # Optimizer settings and initialization
    lr_a = additive_learning_rate
    lr_m = multiplicative_learning_rate

    if bias_version == 4:
        device_processed_features["template_torsion_angles_sin_cos_bias"] = (
            torch.zeros_like(
                device_processed_features["template_torsion_angles_sin_cos"],
                requires_grad=True,
                device=device,
            )
        )
        if weight_decay is None:
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": device_processed_features[
                            "template_torsion_angles_sin_cos_bias"
                        ],
                        "lr": lr_a,
                    },
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": device_processed_features[
                            "template_torsion_angles_sin_cos_bias"
                        ],
                        "lr": lr_a,
                    },
                ],
                weight_decay=weight_decay,
            )
        bias_names = ["template_torsion_angles_sin_cos_bias"]

    elif bias_version == 3:
        if starting_weights is not None:
            device_processed_features["msa_feat_weights"] = (
                torch.load(starting_weights).detach().to(device=device)
            )
        else:
            device_processed_features["msa_feat_weights"] = torch.ones(
                (512, num_res, 23), requires_grad=True, device=device
            )

        if starting_bias is not None:
            device_processed_features["msa_feat_bias"] = (
                torch.load(starting_bias).detach().to(device=device)
            )

        device_processed_features["msa_feat_bias"].requires_grad = True
        device_processed_features["msa_feat_weights"].requires_grad = True

        if weight_decay is None:
            optimizer = torch.optim.Adam(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                    {
                        "params": device_processed_features["msa_feat_weights"],
                        "lr": lr_m,
                    },
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                    {
                        "params": device_processed_features["msa_feat_weights"],
                        "lr": lr_m,
                    },
                ],
                weight_decay=weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif bias_version == 2:
        msa_params_weights = torch.eye(512, dtype=torch.float32, device=device)
        device_processed_features["msa_feat_weights"] = msa_params_weights

        if weight_decay is None:
            optimizer = torch.optim.Adam(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                    {
                        "params": device_processed_features["msa_feat_weights"],
                        "lr": lr_m,
                    },
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                    {
                        "params": device_processed_features["msa_feat_weights"],
                        "lr": lr_m,
                    },
                ],
                weight_decay=weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif bias_version == 1:
        if weight_decay is None:
            optimizer = torch.optim.Adam(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                ]
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                ],
                weight_decay=weight_decay,
            )

        bias_names = ["msa_feat_bias"]
    
    output_directory_path = f"{path}/{file_root}/outputs/MSEoptimze_{note}"
    warnings.filterwarnings("ignore")

    best_loss = float("inf")
    bset_pos = torch.tensor(moving_pdb_obj.atom_pos, dtype=torch.float32, device=device)
    mse_losses_by_epoch = []
    time_by_epoch = []
    memory_by_epoch = []
    all_pldtts = []
    mean_it_plddts = []

    if bias_version == 4:
        features_at_it_start = (
            device_processed_features["template_torsion_angles_sin_cos"][..., 0]
            .detach()
            .clone()
        )
    else:
        features_at_it_start = (
            device_processed_features["msa_feat"][:, :, 25:48, 0].detach().clone()
        )

    progress_bar = tqdm(range(iterations), desc=f"{file_root}, MSE Optimize")
    for iteration in progress_bar:
        start_time = time.time()
        optimizer.zero_grad()
        if iteration == 0:
            try:
                os.makedirs(output_directory_path, exist_ok=True)
            except FileExistsError:
                print(
                    f"Warning: Directory '{output_directory_path}' already exists. Overwriting..."
                )
        
        # Avoid passing through graph a second time
        working_batch = copy.deepcopy(device_processed_features)
        for bias in bias_names:
            working_batch[bias] = device_processed_features[bias].clone()

        
        # AF2 pass
        af2_output = af_bias(working_batch, num_iters=1, bias=True)

        # Position alignment
        xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
            af2_output, device_processed_features, moving_pdb_obj.cra_name
        )

        all_pldtts.append(rk_utils.assert_numpy(af2_output["plddt"]))
        mean_it_plddts.append(rk_utils.assert_numpy(torch.mean(plddts)))

        pseudo_Bs = rk_coordinates.update_bfactors(plddts)
        weights = rk_utils.weighting(rk_utils.assert_numpy(pseudo_Bs))
        aligned_xyz = rk_coordinates.weighted_kabsch(
            xyz_orth_sfc,
            best_pos,
            moving_pdb_obj.cra_name,
            weights=weights,
            exclude_res=EXCLUDING_RES,
        )

        loss = -mseloss(aligned_xyz)

        mse_losses_by_epoch.append(loss.item())
        if mse_losses_by_epoch[-1] < best_loss:
            best_loss = loss.item()
            best_pos = aligned_xyz.detach().clone()
        
        # Save PDB
        moving_pdb_obj.set_biso(rk_utils.assert_numpy(pseudo_Bs))
        moving_pdb_obj.set_positions(rk_utils.assert_numpy(aligned_xyz))
        moving_pdb_obj.savePDB(f"{output_directory_path!s}/{iteration}_postRBR.pdb")

        progress_bar.set_postfix(
            MSEloss=f"{loss.item():.2f}",
            memory=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}G",
        )

        if (iteration % 10 == 0) or (iteration == iterations-1):
            temp_msa_bias = (
                device_processed_features["msa_feat_bias"].detach().cpu().clone()
            )
            temp_feat_weights = (
                device_processed_features["msa_feat_weights"].detach().cpu().clone()
            )

            torch.save(
                temp_msa_bias,
                f"{output_directory_path!s}/add_bias_{iteration}.pt",
            )

            torch.save(
                temp_msa_bias,
                f"{output_directory_path!s}/add_bias_{iteration}.pt",
            )

        loss.backward()
        optimizer.step()
        time_by_epoch.append(time.time() - start_time)
        memory_by_epoch.append(torch.cuda.max_memory_allocated() / 1024**3)


    np.save(
        f"{output_directory_path!s}mse_it.npy",
        np.array(mse_losses_by_epoch),
    )


    np.save(
        f"{output_directory_path!s}/mean_it_plddt.npy",
        np.array(mean_it_plddts),
    )


    np.save(
        f"{output_directory_path!s}/time_it.npy",
        rk_utils.assert_numpy(time_by_epoch),
    )

    np.save(
        f"{output_directory_path!s}/memory_it.npy",
        rk_utils.assert_numpy(memory_by_epoch),
    )

    # Mean plddt per residue (over iterations)
    np.save(
        f"{output_directory_path!s}/mean_plddt_res.npy",
        np.mean(np.array(all_pldtts), axis=0),
    )

    config = {
        "path": path,
        "file_root": file_root,
        "bias_version": bias_version,
        "iterations": iterations,
        "target_pdb": target_pdb,
        "additive_learning_rate": additive_learning_rate,
        "multiplicative_learning_rate": multiplicative_learning_rate,
    }

    with open(f"{output_directory_path!s}/config.json", 'w') as file:
        json.dump(config, file, indent=4)


def main():
    args = parse_arguments()
    args_dict = vars(args)
    mse_optimize(**args_dict)
