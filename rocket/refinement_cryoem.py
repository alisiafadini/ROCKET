import uuid
import torch
import pickle
import numpy as np
from tqdm import tqdm
import rocket
import os
import time
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from rocket import refinement_utils as rkrf_utils
from openfold.config import model_config

from rocket.cryo import targets as cryo_targets
from rocket.cryo import structurefactors as cryo_sf
from rocket.cryo import utils as cryo_utils
from rocket.refinement_config import RocketRefinmentConfig

from loguru import logger


PRESET = "model_1_ptm"
EXCLUDING_RES = None
N_BINS = 20

def run_cryoem_refinement(config: RocketRefinmentConfig | str) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)
    assert config.datamode == "cryoem", "Make sure to set datamode to 'cryoem'!"
    device = f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu"
    
    target_id = config.file_root
    path = config.path
    # TODO: make sure the follow path match the new pattern
    mtz_file = f"{path}/{target_id}/{target_id}-Edata.mtz"
    input_pdb = f"{path}/{target_id}/{target_id}-pred-aligned.pdb"
    note = config.note
    lr_a = config.additive_learning_rate
    lr_m = config.multiplicative_learning_rate
    bias_version = config.bias_version
    iterations = config.iterations
    num_of_runs = config.num_of_runs

    if config.uuid_hex:
        refinement_run_uuid = config.uuid_hex
    else:
        config.uuid_hex = uuid.uuid4().hex
        refinement_run_uuid = config.uuid_hex
    output_directory_path = f"{path}/{target_id}/outputs/{refinement_run_uuid}/{note}"
    try:
        os.makedirs(output_directory_path, exist_ok=True)
    except FileExistsError:
        logger.info(
            f"Warning: Directory '{output_directory_path}' already exists. Overwriting..."
        )
        logger.info(
            f"System: {target_id}, refinment run ID: {refinement_run_uuid!s}, Note: {note}",
        )

    ############ 2. Initializations ############
    # Do downsampling if specified
    if config.downsample_ratio:
        if config.downsample_ratio == 1:
            logger.info("Downsampling ratio is 1. Skipping downsampling...")
        else:
            logger.info(f"Downsampling in reciprocal space, ratio per axis: {config.downsample_ratio}")
            mtz_file = cryo_utils.downsample_data(mtz_file, config.downsample_ratio)
            
    # Initialize SFC
    cryo_sfc = cryo_sf.initial_cryoSFC(
        input_pdb, mtz_file, "Emean", "PHIEmean", device, N_BINS
    )

    sfc_rbr = cryo_sf.initial_cryoSFC(
        input_pdb, mtz_file, "Emean", "PHIEmean", device, N_BINS
    )

    reference_pos = cryo_sfc.atom_pos_orth.clone()
    cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(
        cryo_sfc.cra_name
    )
    residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]

    # LLG initialization
    cryo_llgloss = cryo_targets.LLGloss(cryo_sfc, mtz_file)
    cryo_llgloss_rbr = cryo_targets.LLGloss(cryo_sfc, mtz_file)


    # Model initialization
    version_to_class = {
        1: rocket.MSABiasAFv1,
        2: rocket.MSABiasAFv2,
        3: rocket.MSABiasAFv3,
        4: rocket.TemplateBiasAF,
    }
    af_bias = version_to_class[bias_version](model_config(PRESET, train=True), PRESET).to(
        device
    )
    af_bias.freeze()

    best_loss = float("inf")
    best_msa_bias = None
    best_feat_weights = None
    best_run = None
    best_iter = None
    best_pos = reference_pos

    for n in range(num_of_runs):

        run_id = rkrf_utils.number_to_letter(n)

        # Initialize the processed dict space
        # TODO: below assumed the processed feature_dict in the folder
        device_processed_features, feature_key, features_at_it_start = (
            rkrf_utils.init_processed_dict(
                bias_version=bias_version,
                path=path,
                file_root=target_id,
                device=device,
                PRESET=PRESET,
            )
        )
        # Initialize bias
        device_processed_features, optimizer, bias_names = rkrf_utils.init_bias(
            device_processed_features=device_processed_features,
            bias_version=bias_version,
            device=device,
            lr_a=lr_a,
            lr_m=lr_m,
        )

        # List initialization for saving values
        sigmas_by_epoch = []
        llg_losses = []
        time_by_epoch = []
        memory_by_epoch = []
        all_pldtts = []
        mean_it_plddts = []

        progress_bar = tqdm(
            range(iterations),
            desc=f"{target_id}, uuid: {refinement_run_uuid[:4]}, run: {run_id}",
        )

        ############ 3. Run Refinement ############
        for iteration in progress_bar:
            start_time = time.time()
            optimizer.zero_grad()

            # Avoid passing through graph a second time
            device_processed_features[feature_key] = features_at_it_start.detach().clone()

            # AF pass
            if iteration == 0:
                af2_output, prevs = af_bias(
                    device_processed_features,
                    [None, None, None],
                    num_iters=3,
                    bias=False,
                )

                prevs = [tensor.detach() for tensor in prevs]

            deep_copied_prevs = [tensor.clone().detach() for tensor in prevs]

            af2_output, __ = af_bias(
                device_processed_features, deep_copied_prevs, num_iters=1, bias=True
            )
            L_plddt = -torch.mean(af2_output["plddt"])

            # Position Kabsch Alignment
            aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
                af2_output=af2_output,
                device_processed_features=device_processed_features,
                cra_name=cryo_sfc.cra_name,
                best_pos=best_pos,
                exclude_res=EXCLUDING_RES,
            )
            # Rigid body refinement (RBR) step
            optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
                aligned_xyz, cryo_llgloss_rbr, sfc_rbr.cra_name, lbfgs=True
            )

            cryo_llgloss.sfc.atom_b_iso = pseudo_Bs.detach()
            all_pldtts.append(plddts_res)
            mean_it_plddts.append(np.mean(plddts_res))

            # Calculate (or refine) sigmaA
            cryo_llgloss.sfc.atom_pos_orth = optimized_xyz.detach().clone()
            cryo_llgloss.sfc.savePDB(
                f"{output_directory_path!s}/{run_id}_{iteration}_preRBR.pdb"
            )

            # LLG loss
            L_llg = -cryo_llgloss(
                optimized_xyz,  # TODO add RBR step
            )

            L_llg = L_llg  # + 30 * L_plddt

            sigmas = cryo_llgloss.sigmaAs
            print("mean sigmaA", np.mean(rk_utils.assert_numpy(sigmas)))
            llg_losses.append(L_llg.clone().item())

            # check if current loss is the best so far
            if llg_losses[-1] < best_loss:
                best_loss = llg_losses[-1]
                best_msa_bias = (
                    device_processed_features["msa_feat_bias"].detach().cpu().clone()
                )
                best_feat_weights = (
                    device_processed_features["msa_feat_weights"].detach().cpu().clone()
                )
                best_run = run_id
                best_iter = iteration
                best_pos = aligned_xyz.detach().clone()  # TODO should be optimized

            progress_bar.set_postfix(
                LLG=f"{L_llg.clone().item():.2f}",
                memory=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}G",
            )

            # Save sigmaA values for further processing
            sigmas_dict = {
                f"sigma_{i + 1}": sigma_value.item() for i, sigma_value in enumerate(sigmas)
            }
            sigmas_by_epoch.append(sigmas_dict)

            L_llg.backward()
            optimizer.step()
            time_by_epoch.append(time.time() - start_time)
            memory_by_epoch.append(torch.cuda.max_memory_allocated() / 1024**3)

        # Average plddt per iteration
        np.save(
            f"{output_directory_path!s}/mean_it_plddt_{run_id}.npy",
            np.array(mean_it_plddts),
        )

        # LLG per iteration
        np.save(
            f"{output_directory_path!s}/LLG_it_{run_id}.npy",
            rk_utils.assert_numpy(llg_losses),
        )

        with open(
            f"{output_directory_path!s}/sigmas_by_epoch_{run_id}.pkl",
            "wb",
        ) as file:
            pickle.dump(sigmas_by_epoch, file)

    torch.save(
        best_msa_bias,
        f"{output_directory_path!s}/best_msa_bias_{best_run}_{best_iter}.pt",
    )

    torch.save(
        best_feat_weights,
        f"{output_directory_path!s}/best_feat_weights_{best_run}_{best_iter}.pt",
    )

    config.to_yaml_file(f"{output_directory_path!s}/config.yaml")

    return config