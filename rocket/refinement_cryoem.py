import glob
import os
import shutil
import time
import uuid

import numpy as np
import torch
from loguru import logger
from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline
from tqdm import tqdm

import rocket
from rocket import coordinates as rk_coordinates
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket.cryo import structurefactors as cryo_sf
from rocket.cryo import targets as cryo_targets
from rocket.cryo import utils as cryo_utils
from rocket.refinement_config import RocketRefinmentConfig

PRESET = "model_1_ptm"
EXCLUDING_RES = None
N_BINS = 20


def run_cryoem_refinement(config: RocketRefinmentConfig | str) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)
    assert config.datamode == "cryoem", "Make sure to set datamode to 'cryoem'!"

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

    target_id = config.file_id
    path = config.path
    mtz_file = f"{path}/ROCKET_inputs/{target_id}-Edata.mtz"
    input_pdb = f"{path}/ROCKET_inputs/{target_id}-pred-aligned.pdb"
    note = config.note
    bias_version = config.bias_version
    iterations = config.iterations
    num_of_runs = config.num_of_runs

    if config.uuid_hex:
        refinement_run_uuid = config.uuid_hex
    else:
        config.paths.uuid_hex = uuid.uuid4().hex[:10]
        refinement_run_uuid = config.uuid_hex
    output_directory_path = f"{path}/ROCKET_outputs/{refinement_run_uuid}/{note}"
    try:
        os.makedirs(output_directory_path, exist_ok=True)
    except FileExistsError:
        logger.info(
            f"Warning: Directory '{output_directory_path}' already exists. Overwriting."
        )
        logger.info(
            f"System: {target_id}, run ID: {refinement_run_uuid!s}, Note: {note}",
        )

    ############ 2. Initializations ############
    # Do downsampling if specified
    if config.downsample_ratio:
        if config.downsample_ratio == 1:
            logger.info("Downsampling ratio is 1. Skipping downsampling...")
        else:
            logger.info(
                f"Downsampling reciprocal space, axis ratio: {config.downsample_ratio}"
            )
            mtz_file = cryo_utils.downsample_data(mtz_file, config.downsample_ratio)

    # Initialize SFC
    cryo_sfc = cryo_sf.initial_cryoSFC(
        input_pdb, mtz_file, "Emean", "PHIEmean", device, N_BINS
    )

    sfc_rbr = cryo_sf.initial_cryoSFC(
        input_pdb, mtz_file, "Emean", "PHIEmean", device, N_BINS
    )

    reference_pos = cryo_sfc.atom_pos_orth.clone()
    target_seq = cryo_sfc._pdb.sequence
    cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(
        cryo_sfc.cra_name
    )

    # Use initial pos B factor instead of best pos B factor for weighted L2
    init_pos_bfactor = cryo_sfc.atom_b_iso.clone()
    bfactor_weights = rk_utils.weighting_torch(init_pos_bfactor, cutoff2=20.0)

    # residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]

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
    af_bias = version_to_class[bias_version](
        model_config(PRESET, train=True), PRESET
    ).to(device)
    af_bias.freeze()

    # Optimizer settings and initialization
    if "phase1" in config.note:
        lr_a = config.additive_learning_rate
        lr_m = config.multiplicative_learning_rate
    elif "phase2" in config.note:
        lr_a = config.phase2_final_lr
        lr_m = config.phase2_final_lr

    best_loss = float("inf")
    best_msa_bias = None
    best_feat_weights = None
    best_run = None
    best_iter = None

    # MH edit @ Nov 8th, 2024: Support to use msa as input
    if config.msa_subratio is not None and config.input_msa is None:
        config.input_msa = "alignments"  # default dir for alignment

    recombination_bias = None
    if config.input_msa is not None:
        fasta_path = [
            f
            for ext in ("*.fa", "*.fasta")
            for f in glob.glob(os.path.join(config.path, ext))
        ][0]
        a3m_path = os.path.join(config.path, config.input_msa)
        if os.path.isfile(a3m_path):
            msa_name, ext = os.path.splitext(os.path.basename(a3m_path))
            alignment_dir = os.path.join(os.path.dirname(a3m_path), "tmp_align")
            os.makedirs(alignment_dir, exist_ok=True)
            shutil.copy(a3m_path, os.path.join(alignment_dir, msa_name + ".a3m"))
            tmp_align = True
        elif os.path.isdir(a3m_path):
            alignment_dir = a3m_path
            tmp_align = False
        data_processor = data_pipeline.DataPipeline(template_featurizer=None)
        feature_dict = rkrf_utils.generate_feature_dict(
            fasta_path,
            alignment_dir,
            data_processor,
        )
        # prepare featuerizer
        afconfig = model_config(PRESET)
        afconfig.data.common.max_recycling_iters = config.init_recycling
        del afconfig.data.common.masked_msa
        afconfig.data.common.resample_msa_in_recycling = False
        feature_processor = feature_pipeline.FeaturePipeline(afconfig.data)
        if tmp_align:
            shutil.rmtree(alignment_dir)

        # MH edits @ Oct 19, 2024, support MSA subsampling at the beginning
        if config.msa_subratio is not None:
            assert config.msa_subratio > 0.0 and config.msa_subratio <= 1.0, (
                "msa_subratio should be None or between 0.0 and 1.0!"
            )
            # Do subsampling of msa, keep the first sequence
            if config.sub_msa_path is None:
                idx = np.arange(feature_dict["msa"].shape[0] - 1) + 1
                sub_idx = np.concatenate((
                    np.array([0]),
                    np.random.choice(
                        idx, size=int(config.msa_subratio * len(idx)), replace=False
                    ),
                ))
                feature_dict["msa"] = feature_dict["msa"][sub_idx]
                feature_dict["deletion_matrix_int"] = feature_dict[
                    "deletion_matrix_int"
                ][sub_idx]
                # Save out the subsampled msa
                np.save(
                    f"{output_directory_path!s}/sub_msa.npy",
                    feature_dict["msa"],
                )
                np.save(
                    f"{output_directory_path!s}/sub_delmat.npy",
                    feature_dict["deletion_matrix_int"],
                )
            else:
                feature_dict["msa"] = np.load(config.sub_msa_path, allow_pickle=True)
                feature_dict["deletion_matrix_int"] = np.load(
                    config.sub_delmat_path, allow_pickle=True
                )
        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode="predict"
        )

        # Edit by MH @ Nov 18, 2024, use bias of fullmsa to realize the cluster msa
        if config.bias_from_fullmsa:
            fullmsa_dir = os.path.join(config.path, "alignments")
            fullmsa_feature_dict = rkrf_utils.generate_feature_dict(
                fasta_path,
                fullmsa_dir,
                data_processor,
            )
            fullmsa_processed_feature_dict = feature_processor.process_features(
                fullmsa_feature_dict, mode="predict"
            )
            fullmsa_profile = fullmsa_processed_feature_dict["msa_feat"][
                :, :, 25:48
            ].clone()
            submsa_profile = processed_feature_dict["msa_feat"][:, :, 25:48].clone()
            processed_feature_dict["msa_feat"][:, :, 25:48] = (
                fullmsa_profile.clone()
            )  # Use full msa's profile as basis for linear space -- higher rank (?)
            recombination_bias = (
                submsa_profile[..., 0] - fullmsa_profile[..., 0]
            )  # Use difference as the initial bias so we can start from desired profile
        elif config.chimera_profile:
            fullmsa_dir = os.path.join(config.path, "alignments")
            fullmsa_feature_dict = rkrf_utils.generate_feature_dict(
                fasta_path,
                fullmsa_dir,
                data_processor,
            )
            fullmsa_processed_feature_dict = feature_processor.process_features(
                fullmsa_feature_dict, mode="predict"
            )
            full_profile = fullmsa_processed_feature_dict["msa_feat"][
                :, :, 25:48
            ].clone()
            sub_profile = processed_feature_dict["msa_feat"][:, :, 25:48].clone()
            processed_feature_dict["msa_feat"][:, :, 25:48] = torch.where(
                sub_profile == 0.0, full_profile.clone(), sub_profile.clone()
            )

        device_processed_features = rk_utils.move_tensors_to_device(
            processed_feature_dict, device=device
        )
        feature_key = "msa_feat"

        if config.msa_feat_init_path is None:
            features_at_it_start = (
                device_processed_features[feature_key].detach().clone()
            )
            np.save(
                f"{output_directory_path!s}/msa_feat_start.npy",
                rk_utils.assert_numpy(features_at_it_start[..., 0]),
            )
        else:
            msa_feat_init_np = np.load(
                glob.glob(config.msa_feat_init_path)[0], allow_pickle=True
            )
            features_at_it_start_np = np.repeat(
                np.expand_dims(msa_feat_init_np, -1), config.init_recycling + 1, -1
            )
            features_at_it_start = torch.tensor(features_at_it_start_np).to(
                device_processed_features[feature_key]
            )
            device_processed_features[feature_key] = (
                features_at_it_start.detach().clone()
            )

    else:
        # Initialize the processed dict space
        device_processed_features, feature_key, features_at_it_start = (
            rkrf_utils.init_processed_dict(
                bias_version=config.bias_version,
                path=config.path,
                device=device,
                template_pdb=config.template_pdb,
                target_seq=target_seq,
                PRESET=PRESET,
            )
        )

    # MH edit @ Oct 2nd, 2024: Support optional template input
    if config.template_pdb is not None:
        device_processed_features_template = rocket.make_processed_dict_from_template(
            config.template_pdb,
            target_seq,
            device=device,
            mask_sidechains_add_cb=True,
            mask_sidechains=True,
            max_recycling_iters=config.init_recycling,
        )
        for key in device_processed_features_template:
            if key.startswith("template_"):
                device_processed_features[key] = device_processed_features_template[key]

    for n in range(num_of_runs):
        run_id = rkrf_utils.number_to_letter(n)
        best_pos = reference_pos

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
            recombination_bias=recombination_bias,
        )

        # List initialization for saving values
        # sigmas_by_epoch = []
        llg_losses = []
        time_by_epoch = []
        memory_by_epoch = []
        all_pldtts = []
        mean_it_plddts = []

        progress_bar = tqdm(
            range(iterations),
            desc=f"{target_id}, uuid: {refinement_run_uuid[:4]}, run: {run_id}",
        )

        # Run smooth stage in phase 1
        if "phase1" in config.note:
            w_L2 = config.l2_weight
        elif "phase2" in config.note:
            w_L2 = 0.0

        ######
        early_stopper = rkrf_utils.EarlyStopper(patience=200, min_delta=10.0)

        #### Phase 1 smooth scheduling ######
        if config.smooth_stage_epochs is not None:
            lr_a_initial = lr_a
            lr_m_initial = lr_m
            w_L2_initial = w_L2
            lr_stage1_final = config.phase2_final_lr
            smooth_stage_epochs = config.smooth_stage_epochs

            # Decay rates for each stage
            decay_rate_stage1_add = (lr_stage1_final / lr_a) ** (
                1 / smooth_stage_epochs
            )
            decay_rate_stage1_mul = (lr_stage1_final / lr_m) ** (
                1 / smooth_stage_epochs
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
            L_plddt = -torch.mean(af2_output["plddt"])

            # Position Kabsch Alignment
            aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
                af2_output=af2_output,
                device_processed_features=device_processed_features,
                cra_name=cryo_sfc.cra_name,
                best_pos=best_pos,
                exclude_res=EXCLUDING_RES,
                domain_segs=config.domain_segs,
                reference_bfactor=init_pos_bfactor,
            )

            cryo_llgloss.sfc.atom_b_iso = pseudo_Bs.detach().clone()
            cryo_llgloss_rbr.sfc.atom_b_iso = pseudo_Bs.detach().clone()
            all_pldtts.append(plddts_res)
            mean_it_plddts.append(np.mean(plddts_res))

            # Save the preRBR structure, for debugging
            cryo_llgloss_rbr.sfc.atom_pos_orth = aligned_xyz.detach().clone()
            cryo_llgloss_rbr.sfc.savePDB(
                f"{output_directory_path!s}/{run_id}_{iteration}_preRBR.pdb"
            )
            if config.sfc_scale:
                cryo_llgloss_rbr.sfc.calc_fprotein()
                cryo_llgloss_rbr.sfc.get_scales_adam(
                    lr=0.01,
                    n_steps=10,
                    sub_ratio=0.7,
                    initialize=False,
                )

            # Rigid body refinement (RBR) step
            optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
                aligned_xyz,
                cryo_llgloss_rbr,
                sfc_rbr.cra_name,
                domain_segs=config.domain_segs,
                lbfgs=RBR_LBFGS,
                lbfgs_lr=config.rbr_lbfgs_learning_rate,
                verbose=config.verbose,
            )

            # Save the postRBR structure
            cryo_llgloss.sfc.atom_pos_orth = optimized_xyz.detach().clone()
            cryo_llgloss.sfc.savePDB(
                f"{output_directory_path!s}/{run_id}_{iteration}_postRBR.pdb"
            )

            # LLG loss
            L_llg = -cryo_llgloss(
                optimized_xyz,
                bin_labels=None,
                num_batch=config.number_of_batches,
                sub_ratio=config.batch_sub_ratio,
                update_scales=config.sfc_scale,
            )

            llg_estimate = L_llg.clone().item() / (
                config.batch_sub_ratio * config.number_of_batches
            )  # + 30 * L_plddt

            llg_losses.append(llg_estimate)

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
                best_pos = optimized_xyz.detach().clone()

            progress_bar.set_postfix(
                LLG=f"{llg_estimate:.2f}",
                memory=f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G",
            )

            #### add an L2 loss to constrain confident atoms ###
            if w_L2 > 0.0:
                # use
                L2_loss = torch.sum(
                    bfactor_weights.unsqueeze(-1) * (optimized_xyz - reference_pos) ** 2
                )  # / conf_best.shape[0]
                loss = L_llg + w_L2 * L2_loss + config.w_plddt * L_plddt
                loss.backward()
            else:
                loss = L_llg + config.w_plddt * L_plddt
                loss.backward()

                if early_stopper.early_stop(loss.item()):
                    break

            # Smooth in last part of phase 1 instead of beginning of phase 2
            if ("phase1" in config.note) and (config.smooth_stage_epochs is not None):
                if iteration > (config.iterations - smooth_stage_epochs):
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
