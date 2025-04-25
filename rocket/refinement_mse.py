import os
import time
import uuid
import warnings

import numpy as np
import torch
import yaml
from openfold.config import model_config
from pydantic import BaseModel
from SFC_Torch import PDBParser
from tqdm import tqdm

import rocket
from rocket import coordinates as rk_coordinates
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils

PRESET = "model_1"
# THRESH_B = None
EXCLUDING_RES = None


class RocketMSERefinmentConfig(BaseModel):
    path: str
    file_root: str
    bias_version: int
    iterations: int
    template_pdb: str | None = None
    cuda_device: int = (0,)
    num_of_runs: int = 1
    init_recycling: int = (1,)
    additive_learning_rate: float
    multiplicative_learning_rate: float
    weight_decay: float | None = 0.0001  # TODO: should default be 0.0?
    rbr_opt_algorithm: str
    rbr_lbfgs_learning_rate: float
    note: str = ""
    free_flag: str
    testset_value: int
    verbose: bool = False
    starting_bias: str | None = None
    starting_weights: str | None = None
    uuid_hex: str | None = None

    # intercept them upload load/save and cast to string as appropriate
    def to_yaml_file(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            yaml.dump(self.dict(), file)

    @classmethod
    def from_yaml_file(self, file_path: str):
        with open(file_path) as file:
            payload = yaml.safe_load(file)
        return RocketMSERefinmentConfig.model_validate(payload)


def run_refinement_mse(*, config: RocketMSERefinmentConfig) -> str:
    ############ 1. Global settings ############
    # Device
    device = f"cuda:{config.cuda_device}"

    # Using LBFGS or Adam in RBR
    if config.rbr_opt_algorithm == "lbfgs":
        RBR_LBFGS = True
    elif config.rbr_opt_algorithm == "adam":
        RBR_LBFGS = False
    else:
        raise ValueError("rbr_opt only supports lbfgs or adam")

    # Configure input paths
    input_pdb = f"{config.path}/{config.file_root}/{config.file_root}-pred-aligned.pdb"

    # Configure output path
    # Generate uuid for this run
    if config.uuid_hex is None:
        refinement_run_uuid = uuid.uuid4().hex
    else:
        refinement_run_uuid = config.uuid_hex
    output_directory_path = (
        f"{config.path}/{config.file_root}/outputs/{refinement_run_uuid}/{config.note}"
    )
    try:
        os.makedirs(output_directory_path, exist_ok=True)
    except FileExistsError:
        print(
            f"Warning: Directory '{output_directory_path}' already exists. Overwriting..."
        )
    print(
        f"System: {config.file_root}, refinment run ID: {refinement_run_uuid!s}, Note: {config.note}",
        flush=True,
    )
    if not config.verbose:
        warnings.filterwarnings("ignore")

    ############ 2. Initializations ############
    target_pdb = PDBParser(input_pdb)
    reference_pos = torch.tensor(
        target_pdb.atom_pos, dtype=torch.float32, device=device
    )
    mseloss = rocket.MSElossBB(target_pdb, target_pdb, device)

    # CA mask and residue numbers for track
    cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(
        target_pdb.cra_name
    )
    residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]

    # Use initial pos B factor instead of best pos B factor for weighted L2
    init_pos_bfactor = torch.tensor(
        target_pdb.atom_b_iso, dtype=torch.float32, device=device
    )
    bfactor_weights = rk_utils.weighting_torch(init_pos_bfactor, cutoff2=20.0)

    # Model initialization
    version_to_class = {
        1: rocket.MSABiasAFv1,
        2: rocket.MSABiasAFv2,
        3: rocket.MSABiasAFv3,
        4: rocket.TemplateBiasAF,
    }
    af_bias = version_to_class[config.bias_version](
        model_config(PRESET, train=True), PRESET
    ).to(device)
    af_bias.freeze()  # Free all AF2 parameters to save time

    # Optimizer settings and initialization
    lr_a = config.additive_learning_rate
    lr_m = config.multiplicative_learning_rate

    # Initialize best Rfree weights and bias for Phase 1
    best_loss = float("inf")
    best_run = None
    best_msa_bias = None
    best_feat_weights = None
    best_iter = None
    best_pos = reference_pos

    for n in range(config.num_of_runs):
        run_id = rkrf_utils.number_to_letter(n)

        # Initialize the processed dict space
        device_processed_features, feature_key, features_at_it_start = (
            rkrf_utils.init_processed_dict(
                bias_version=config.bias_version,
                path=config.path,
                file_root=config.file_root,
                device=device,
                template_pdb=config.template_pdb,
                PRESET=PRESET,
            )
        )
        # Initialize bias
        device_processed_features, optimizer, bias_names = rkrf_utils.init_bias(
            device_processed_features=device_processed_features,
            bias_version=config.bias_version,
            device=device,
            lr_a=lr_a,
            lr_m=lr_m,
            weight_decay=config.weight_decay,
            starting_bias=config.starting_bias,
            starting_weights=config.starting_weights,
        )

        # List initialization for saving values
        mse_losses = []
        time_by_epoch = []
        memory_by_epoch = []
        all_pldtts = []
        mean_it_plddts = []
        absolute_feats_changes = []

        early_stopper = rkrf_utils.EarlyStopper(patience=50, min_delta=0.1)

        progress_bar = tqdm(
            range(config.iterations),
            desc=f"{config.file_root}, uuid: {refinement_run_uuid[:4]}, run: {run_id}",
        )

        ############ 3. Run Refinement ############
        for iteration in progress_bar:
            start_time = time.time()
            optimizer.zero_grad()

            # Avoid passing through graph a second time
            device_processed_features[feature_key] = (
                features_at_it_start.detach().clone()
            )

            # AF pass
            if iteration == 0:
                af2_output, prevs = af_bias(
                    device_processed_features,
                    [None, None, None],
                    num_iters=config.init_recycling,
                    bias=False,
                )
                prevs = [tensor.detach() for tensor in prevs]

            deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]
            af2_output, __ = af_bias(
                device_processed_features, deep_copied_prevs, num_iters=1, bias=True
            )

            # Position Kabsch Alignment
            aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
                af2_output=af2_output,
                device_processed_features=device_processed_features,
                cra_name=target_pdb.cra_name,
                best_pos=best_pos,
                exclude_res=EXCLUDING_RES,
            )

            all_pldtts.append(plddts_res)
            mean_it_plddts.append(np.mean(plddts_res))

            # MSE loss
            loss = mseloss.forward(aligned_xyz, bfactor_weights)

            mse_losses.append(loss.item())

            # check if current Rfree is the best so far
            if mse_losses[-1] < best_loss:
                best_loss = mse_losses[-1]
                best_msa_bias = (
                    device_processed_features["msa_feat_bias"].detach().cpu().clone()
                )
                best_feat_weights = (
                    device_processed_features["msa_feat_weights"].detach().cpu().clone()
                )
                best_run = run_id
                best_iter = iteration
                best_pos = aligned_xyz.detach().clone()

            # Save postRBR PDB
            mseloss.pdb_obj.atom_b_iso = rk_utils.assert_numpy(pseudo_Bs)
            mseloss.pdb_obj.atom_pos = rk_utils.assert_numpy(aligned_xyz)
            mseloss.pdb_obj.savePDB(
                f"{output_directory_path!s}/{run_id}_{iteration}_aligned.pdb"
            )

            progress_bar.set_postfix(
                MSE=f"{loss.item():.2f}",
                memory=f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G",
            )

            loss.backward()

            if early_stopper.early_stop(loss.item()):
                break

            optimizer.step()

            time_by_epoch.append(time.time() - start_time)
            memory_by_epoch.append(torch.cuda.max_memory_allocated() / 1024**3)

            # Save the absolute difference in mean contribution from each residue channel from previous iteration
            if config.bias_version == 4:
                features_at_step_end = (
                    device_processed_features["template_torsion_angles_sin_cos"][..., 0]
                    .detach()
                    .clone()
                )
                mean_change = torch.mean(
                    torch.abs(features_at_step_end - features_at_it_start[..., 0]),
                    dim=(0, 2, 3),
                )
            else:
                features_at_step_end = (
                    device_processed_features["msa_feat"][:, :, 25:48, 0]
                    .detach()
                    .clone()
                )
                mean_change = torch.mean(
                    torch.abs(
                        features_at_step_end - features_at_it_start[:, :, 25:48, 0]
                    ),
                    dim=(0, 2),
                )
            absolute_feats_changes.append(rk_utils.assert_numpy(mean_change))

        ####### Save data
        # Average plddt per iteration
        np.save(
            f"{output_directory_path!s}/mean_it_plddt_{run_id}.npy",
            np.array(mean_it_plddts),
        )

        # MSE per iteration
        np.save(
            f"{output_directory_path!s}/MSE_it_{run_id}.npy",
            rk_utils.assert_numpy(mse_losses),
        )

        np.save(
            f"{output_directory_path!s}/time_it_{run_id}.npy",
            rk_utils.assert_numpy(time_by_epoch),
        )

        np.save(
            f"{output_directory_path!s}/memory_it_{run_id}.npy",
            rk_utils.assert_numpy(memory_by_epoch),
        )

        # Absolute MSA change per column per iteration
        np.save(
            f"{output_directory_path!s}/MSA_changes_it_{run_id}.npy",
            rk_utils.assert_numpy(absolute_feats_changes),
        )

        # Mean plddt per residue (over iterations)
        np.save(
            f"{output_directory_path!s}/mean_plddt_res_{run_id}.npy",
            np.mean(np.array(all_pldtts), axis=0),
        )

    # Save the best msa_bias and feat_weights
    torch.save(
        best_msa_bias,
        f"{output_directory_path!s}/best_msa_bias_{best_run}_{best_iter}.pt",
    )

    torch.save(
        best_feat_weights,
        f"{output_directory_path!s}/best_feat_weights_{best_run}_{best_iter}.pt",
    )

    config.to_yaml_file(f"{output_directory_path!s}/config.yaml")

    return refinement_run_uuid
