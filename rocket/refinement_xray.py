import glob
import os
import shutil
import time
import uuid
import warnings

import numpy as np
import torch
from loguru import logger
from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline
from SFC_Torch import SFcalculator
from tqdm import tqdm

import rocket
from rocket import coordinates as rk_coordinates
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from rocket.refinement_config import RocketRefinmentConfig
from rocket.xtal import structurefactors as llg_sf

PRESET = "model_1_ptm"
EXCLUDING_RES = None


def run_xray_refinement(config: RocketRefinmentConfig | str) -> RocketRefinmentConfig:
    if isinstance(config, str):
        config = RocketRefinmentConfig.from_yaml_file(config)
    assert config.datamode == "xray", "Make sure to set datamode to 'xray'!"

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
    tng_file = f"{config.path}/ROCKET_inputs/{config.file_id}-Edata.mtz"
    input_pdb = f"{config.path}/ROCKET_inputs/{config.file_id}-pred-aligned.pdb"
    true_pdb = f"{config.path}/ROCKET_inputs/{config.file_id}_noalts.pdb"

    # Configure output path
    # Generate uuid for this run
    if config.uuid_hex:
        refinement_run_uuid = config.uuid_hex
    else:
        config.paths.uuid_hex = uuid.uuid4().hex[:10]
        refinement_run_uuid = config.uuid_hex
    output_directory_path = (
        f"{config.path}/ROCKET_outputs/{refinement_run_uuid}/{config.note}"
    )
    try:
        os.makedirs(output_directory_path, exist_ok=True)
    except FileExistsError:
        logger.info(
            f"Warning: Directory '{output_directory_path}' already exists. Overwriting."
        )
    logger.info(
        f"System: {config.file_id}, run ID: {refinement_run_uuid!s}, Note: {config.note}",  # noqa: E501
        flush=True,
    )
    if not config.verbose:
        warnings.filterwarnings("ignore")

    # If reference pdb exsits
    REFPDB = bool(os.path.exists(true_pdb))

    ############ 2. Initializations ############

    # Apply resolution cutoff to the reflection file
    if config.min_resolution is not None or config.max_resolution is not None:
        tng_file = rk_utils.apply_resolution_cutoff(
            tng_file,
            min_resolution=config.min_resolution,
            max_resolution=config.max_resolution,
        )

    # If there are additional chain in the system
    if config.additional_chain:
        added_chain_pdb = (
            f"{config.path}/ROCKET_inputs/{config.file_id}_added_chain.pdb"
        )
        if not os.path.exists(added_chain_pdb):
            raise FileNotFoundError(
                f"Additional chain PDB file '{added_chain_pdb}' does not exist!"
            )
        # calculate the structure factors for the additional chain
        sfc_added_chain = SFcalculator(
            added_chain_pdb,
            tng_file,
            expcolumns=["FP", "SIGFP"],
            freeflag=config.free_flag,
            set_experiment=True,
            testset_value=config.testset_value,
            device=device,
        )
        sfc_added_chain.calc_fprotein()
        constant_fp_added_HKL = sfc_added_chain.Fprotein_HKL.clone().detach()
        constant_fp_added_asu = sfc_added_chain.Fprotein_asu.clone().detach()
        del sfc_added_chain

        phitrue_path = f"{config.path}/ROCKET_inputs/{config.file_id}_allchains-phitrue-solvent{config.solvent}.npy"  # noqa: E501
        Etrue_path = f"{config.path}/ROCKET_inputs/{config.file_id}_allchains-Etrue-solvent{config.solvent}.npy"  # noqa: E501

        if os.path.exists(phitrue_path) and os.path.exists(Etrue_path):
            SIGMA_TRUE = True
            phitrue = np.load(phitrue_path)
            Etrue = np.load(Etrue_path)
        else:
            SIGMA_TRUE = False
    else:
        constant_fp_added_HKL = None
        constant_fp_added_asu = None

        phitrue_path = f"{config.path}/ROCKET_inputs/{config.file_id}-phitrue-solvent{config.solvent}.npy"  # noqa: E501
        Etrue_path = f"{config.path}/ROCKET_inputs/{config.file_id}-Etrue-solvent{config.solvent}.npy"  # noqa: E501

        if os.path.exists(phitrue_path) and os.path.exists(Etrue_path):
            SIGMA_TRUE = True
            phitrue = np.load(phitrue_path)
            Etrue = np.load(Etrue_path)
        else:
            SIGMA_TRUE = False

    # Initialize SFC
    sfc = llg_sf.initial_SFC(
        input_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=config.free_flag,
        device=device,
        testset_value=config.testset_value,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
        total_chain_copy=config.total_chain_copy,
        spacing=config.voxel_spacing,
    )
    reference_pos = sfc.atom_pos_orth.clone()
    target_seq = sfc._pdb.sequence
    # CA mask and residue numbers for track
    cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(sfc.cra_name)
    residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]

    # Use initial pos B factor instead of best pos B factor for weighted L2
    init_pos_bfactor = sfc.atom_b_iso.clone()
    bfactor_weights = rk_utils.weighting_torch(init_pos_bfactor, cutoff2=20.0)

    sfc_rbr = llg_sf.initial_SFC(
        input_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=config.free_flag,
        device=device,
        solvent=False,
        testset_value=config.testset_value,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
        total_chain_copy=config.total_chain_copy,
        spacing=config.voxel_spacing,
    )

    if REFPDB:
        # Load true positions
        true_pdb_model = rk_utils.load_pdb(true_pdb)
        true_pos = rk_utils.assert_tensor(true_pdb_model.atom_pos)

    # LLG initialization with resol cut
    llgloss = rkrf_utils.init_llgloss(
        sfc, tng_file, config.min_resolution, config.max_resolution
    )
    llgloss_rbr = rkrf_utils.init_llgloss(
        sfc_rbr, tng_file, config.min_resolution, config.max_resolution
    )

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
    # Run smooth stage in phase 1 instead
    if "phase1" in config.note:
        lr_a = config.additive_learning_rate
        lr_m = config.multiplicative_learning_rate
    elif "phase2" in config.note:
        lr_a = config.phase2_final_lr
        lr_m = config.phase2_final_lr

    # Initialize best Rfree weights and bias for Phase 1
    best_llg = float("inf")
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
            )  # Use the difference as the initial bias, to start from a desired profile
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

    # Write out config used, start the journey
    config.to_yaml_file(f"{output_directory_path!s}/config.yaml")
    for n in range(config.num_of_runs):
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
        mse_losses_by_epoch = []
        rbr_loss_by_epoch = []
        llg_losses = []
        rfree_by_epoch = []
        rwork_by_epoch = []
        time_by_epoch = []
        memory_by_epoch = []
        all_pldtts = []
        mean_it_plddts = []
        absolute_feats_changes = []

        progress_bar = tqdm(
            range(config.iterations),
            desc=f"{config.file_id}, uuid: {refinement_run_uuid[:4]}, run: {run_id}",
        )

        # Prepare an MDTraj trajectory writer so we can append frames
        # to a single trajectory file instead of writing one PDB per iteration.
        # Prefer HDF5 (smaller, single-file) and fall back to a multi-model
        # PDB if HDF5 support isn't available. Import mdtraj lazily so the
        # rest of the code still runs if mdtraj isn't installed.
        traj_writer = None
        md = None
        mdtraj_template = None
        try:
            import mdtraj as md  # type: ignore

            # Prefer HDF5TrajectoryFile, fall back to PDBTrajectoryFile
            try:
                from mdtraj.formats import HDF5TrajectoryFile  # type: ignore
            except Exception:
                HDF5TrajectoryFile = None  # type: ignore
            try:
                from mdtraj.formats import PDBTrajectoryFile  # type: ignore
            except Exception:
                PDBTrajectoryFile = None  # type: ignore

            # Load input topology once (this expects the input PDB to match the
            # atom ordering of the SFC object). MDTraj uses nanometers for
            # coordinates internally, so we'll convert Å -> nm when writing.
            mdtraj_template = md.load_pdb(input_pdb)

            if HDF5TrajectoryFile is not None:
                traj_path_h5 = (
                    f"{output_directory_path!s}/{run_id}_refinement_trajectory.h5"  # noqa: E501
                )
                traj_writer = HDF5TrajectoryFile(traj_path_h5, mode="w")
            elif PDBTrajectoryFile is not None:
                traj_path_pdb = (
                    f"{output_directory_path!s}/{run_id}_refinement_trajectory.pdb"  # noqa: E501
                )
                traj_writer = PDBTrajectoryFile(traj_path_pdb, mode="w")
            else:
                traj_writer = None
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning(
                f"mdtraj not available or failed to initialize writer ({e}); "
                "falling back to one-PDB-per-iteration behavior."
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
                cra_name=sfc.cra_name,
                best_pos=best_pos,
                exclude_res=EXCLUDING_RES,
                domain_segs=config.domain_segs,
                reference_bfactor=init_pos_bfactor,
            )
            llgloss.sfc.atom_b_iso = pseudo_Bs.detach().clone()
            llgloss_rbr.sfc.atom_b_iso = pseudo_Bs.detach().clone()
            all_pldtts.append(plddts_res)
            mean_it_plddts.append(np.mean(plddts_res))

            ##### Residue MSE loss for tracking ######
            if REFPDB:
                total_mse_loss = rk_coordinates.calculate_mse_loss_per_residue(
                    aligned_xyz[calphas_mask], true_pos[calphas_mask], residue_numbers
                )
                mse_losses_by_epoch.append(total_mse_loss)
            ##############################################

            # Calculate (or refine) sigmaA
            if config.refine_sigmaA is True:
                llgloss, llgloss_rbr, Ecalc, Fc = rkrf_utils.update_sigmaA(
                    llgloss=llgloss,
                    llgloss_rbr=llgloss_rbr,
                    aligned_xyz=aligned_xyz,
                    constant_fp_added_HKL=constant_fp_added_HKL,
                    constant_fp_added_asu=constant_fp_added_asu,
                )
            else:
                if SIGMA_TRUE:
                    llgloss, llgloss_rbr = rkrf_utils.sigmaA_from_true(
                        llgloss=llgloss,
                        llgloss_rbr=llgloss_rbr,
                        aligned_xyz=aligned_xyz,
                        Etrue=Etrue,
                        phitrue=phitrue,
                        constant_fp_added_HKL=constant_fp_added_HKL,
                        constant_fp_added_asu=constant_fp_added_asu,
                    )
                else:
                    raise ValueError(
                        "No Etrue or phitrue provided! Can't get the true sigmaA!"
                    )

            # For record
            # if SIGMA_TRUE:
            #     true_sigmas = llg_utils.sigmaA_from_model(
            #         Etrue,
            #         phitrue,
            #         Ecalc,
            #         Fc,
            #         llgloss.sfc.dHKL,
            #         llgloss.bin_labels,
            #     )

            # Update SFC and save
            llgloss.sfc.atom_pos_orth = aligned_xyz.detach().clone()
            # llgloss.sfc.savePDB(
            #    f"{output_directory_path!s}/{run_id}_{iteration}_preRBR.pdb"
            # )

            # Rigid body refinement (RBR) step
            optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
                aligned_xyz,
                llgloss_rbr,
                sfc.cra_name,
                domain_segs=config.domain_segs,
                lbfgs=RBR_LBFGS,
                added_chain_HKL=constant_fp_added_HKL,
                added_chain_asu=constant_fp_added_asu,
                lbfgs_lr=config.rbr_lbfgs_learning_rate,
                verbose=config.verbose,
            )
            rbr_loss_by_epoch.append(loss_track_pose)

            # LLG loss
            L_llg = -llgloss(
                optimized_xyz,
                bin_labels=None,
                num_batch=config.number_of_batches,
                sub_ratio=config.batch_sub_ratio,
                solvent=config.solvent,
                update_scales=config.sfc_scale,
                added_chain_HKL=constant_fp_added_HKL,
                added_chain_asu=constant_fp_added_asu,
            )

            llg_estimate = L_llg.clone().item() / (
                config.batch_sub_ratio * config.number_of_batches
            )
            llg_losses.append(llg_estimate)
            rwork_by_epoch.append(llgloss.sfc.r_work.item())
            rfree_by_epoch.append(llgloss.sfc.r_free.item())

            # check if current Rfree is the best so far
            if llg_losses[-1] < best_llg:
                best_llg = llg_losses[-1]
                best_msa_bias = (
                    device_processed_features["msa_feat_bias"].detach().cpu().clone()
                )
                best_feat_weights = (
                    device_processed_features["msa_feat_weights"].detach().cpu().clone()
                )
                best_run = run_id
                best_iter = iteration
                best_pos = optimized_xyz.detach().clone()
                # best_pos_bfactor = llgloss.sfc.atom_b_iso.detach().clone()

            llgloss.sfc.atom_pos_orth = optimized_xyz
            # Save postRBR frame: either append to a single PDB trajectory using
            # MDTraj if available, or fall back to writing one PDB per iteration.
            try:
                if (
                    traj_writer is not None
                    and md is not None
                    and mdtraj_template is not None
                ):
                    # optimized_xyz is assumed to be in Å; MDTraj expects nm.
                    coords_nm = (
                        optimized_xyz.detach().cpu().numpy().reshape(1, -1, 3) / 10.0
                    )
                    # Create a temporary mdtraj.Trajectory with the same topology
                    # as the input PDB and write it as a frame.
                    traj_frame = md.Trajectory(coords_nm, mdtraj_template.topology)
                    traj_writer.write(traj_frame)
                    # Make a best-effort to flush internal buffers and force
                    # the OS to flush written bytes to disk so previous
                    # frames remain available if the process crashes.
                    import contextlib

                    if hasattr(traj_writer, "flush"):
                        with contextlib.suppress(Exception):
                            traj_writer.flush()
                    # Also fsync the file path we wrote to (h5 preferred)
                    path_to_sync = (
                        traj_path_h5 if os.path.exists(traj_path_h5) else traj_path_pdb
                    )
                    with contextlib.suppress(Exception), open(path_to_sync, "rb") as _f:
                        os.fsync(_f.fileno())
                else:
                    raise RuntimeError("no mdtraj writer")
            except Exception:
                # Fallback: write individual PDB file as before
                pdb_path = f"{output_directory_path!s}/{run_id}_{iteration}_postRBR.pdb"
                llgloss.sfc.savePDB(pdb_path)
                # Ensure the PDB fallback is flushed to disk
                try:
                    with open(pdb_path, "rb") as _f:
                        os.fsync(_f.fileno())
                except Exception:
                    pass

            progress_bar.set_postfix(
                NEG_LLG=f"{llg_estimate:.2f}",
                r_work=f"{llgloss.sfc.r_work.item():.3f}",
                r_free=f"{llgloss.sfc.r_free.item():.3f}",
                memory=f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}G",
            )

            # if config.alignment_mode == "B":
            #     if loss < best_loss:
            #         best_loss = loss

            # Save sigmaA values for further processing
            # sigmas_dict = {
            #     f"sigma_{i + 1}": sigma_value.item()
            #     for i, sigma_value in enumerate(sigmas)
            # }
            # sigmas_by_epoch.append(sigmas_dict)

            # if SIGMA_TRUE:
            #     true_sigmas_dict = {
            #         f"sigma_{i + 1}": sigma_value.item()
            #         for i, sigma_value in enumerate(true_sigmas)
            #     }
            #     true_sigmas_by_epoch.append(true_sigmas_dict)

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

            # Smooth last several iterations of phase 1 instead of beginning of phase 2
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

            # Save the absolute difference in mean contribution
            # from each residue channel from previous iteration
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
        # Close mdtraj writer if used so file is flushed and closed properly
        try:
            if traj_writer is not None:
                traj_writer.close()
        except Exception:
            pass
        # Average plddt per iteration
        np.save(
            f"{output_directory_path!s}/mean_it_plddt_{run_id}.npy",
            np.array(mean_it_plddts),
        )

        # LLG per iteration
        np.save(
            f"{output_directory_path!s}/NEG_LLG_it_{run_id}.npy",
            rk_utils.assert_numpy(llg_losses),
        )

        # MSE loss per iteration
        if REFPDB:
            np.save(
                f"{output_directory_path!s}/MSE_loss_it_{run_id}.npy",
                rk_utils.assert_numpy(mse_losses_by_epoch),
            )

        # R work per iteration
        np.save(
            f"{output_directory_path!s}/rwork_it_{run_id}.npy",
            rk_utils.assert_numpy(rwork_by_epoch),
        )

        # R free per iteration
        np.save(
            f"{output_directory_path!s}/rfree_it_{run_id}.npy",
            rk_utils.assert_numpy(rfree_by_epoch),
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
            f"{output_directory_path!s}/plddt_res_{run_id}.npy",
            np.array(all_pldtts),
        )

        # Iteration sigmaA dictionary
        # with open(
        #     f"{output_directory_path!s}/sigmas_by_epoch_{run_id}.pkl",
        #     "wb",
        # ) as file:
        #     pickle.dump(sigmas_by_epoch, file)

        # if SIGMA_TRUE:
        #     with open(
        #         f"{output_directory_path!s}/true_sigmas_by_epoch_{run_id}.pkl",
        #         "wb",
        #     ) as file:
        #         pickle.dump(true_sigmas_by_epoch, file)

    # Save the best msa_bias and feat_weights
    torch.save(
        best_msa_bias,
        f"{output_directory_path!s}/best_msa_bias_{best_run}_{best_iter}.pt",
    )

    torch.save(
        best_feat_weights,
        f"{output_directory_path!s}/best_feat_weights_{best_run}_{best_iter}.pt",
    )

    # Save best model as a single PDB (preserve input_pdb topology)
    try:
        if best_pos is not None:
            # best_pos is in Å, set it on the llgloss.sfc object and save
            try:
                llgloss.sfc.atom_pos_orth = best_pos
                best_name = (
                    f"{output_directory_path!s}/best_model_{best_run}_{best_iter}.pdb"
                )
                llgloss.sfc.savePDB(best_name)
            except Exception:
                # Fallback: try to write using mdtraj with the input topology
                try:
                    import mdtraj as md  # type: ignore

                    topo = md.load_pdb(input_pdb).topology
                    coords_nm = best_pos.detach().cpu().numpy().reshape(1, -1, 3) / 10.0
                    traj = md.Trajectory(coords_nm, topo)
                    traj.save(
                        f"{output_directory_path!s}/best_model_{best_run}_{best_iter}.pdb"
                    )
                except Exception:
                    logger.warning(
                        "Failed to write best model PDB with both SFC and mdtraj."
                    )
    except NameError:
        # best_pos not defined; nothing to save
        pass

    return config
