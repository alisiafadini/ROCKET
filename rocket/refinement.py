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
from rocket import refinement_utils as rkrf_utils
from rocket.llg import structurefactors as llg_sf
from openfold.config import model_config
from typing import Union, List
from pydantic import BaseModel


PRESET = "model_1"
# THRESH_B = None
EXCLUDING_RES = None


class RocketRefinmentConfig(BaseModel):
    path: str
    file_root: str
    bias_version: int
    iterations: int
    num_of_runs: int = 1
    template_pdb: Union[str, None] = None
    cuda_device: int = (0,)
    init_recycling: int = (1,)
    domain_segs: Union[List[int], None] = None
    solvent: bool
    sfc_scale: bool
    refine_sigmaA: bool
    additive_learning_rate: float
    multiplicative_learning_rate: float
    weight_decay: Union[float, None] = 0.0001  # TODO: should default be 0.0?
    batch_sub_ratio: float
    number_of_batches: int
    rbr_opt_algorithm: str
    rbr_lbfgs_learning_rate: float
    smooth_stage_epochs: Union[int, None] = 50
    phase2_final_lr: float = 1e-3
    note: str = ""
    free_flag: str
    testset_value: int
    additional_chain: bool
    verbose: bool
    l2_weight: float
    w_plddt: float = 0.0
    # b_threshold: float
    min_resolution: Union[float, None] = None
    max_resolution: Union[float, None] = None
    starting_bias: Union[str, None] = None
    starting_weights: Union[str, None] = None
    uuid_hex: Union[str, None] = None
    voxel_spacing: float = 4.5

    # intercept them upload load/save and cast to string as appropriate
    def to_yaml_file(self, file_path: str) -> None:
        with open(file_path, "w") as file:
            yaml.dump(self.dict(), file)

    @classmethod
    def from_yaml_file(self, file_path: str):
        with open(file_path, "r") as file:
            payload = yaml.safe_load(file)
        return RocketRefinmentConfig.model_validate(payload)


def run_refinement(*, config: RocketRefinmentConfig) -> str:

    ############ 1. Global settings ############
    # Device
    device = "cuda:{}".format(config.cuda_device)

    # Using LBFGS or Adam in RBR
    if config.rbr_opt_algorithm == "lbfgs":
        RBR_LBFGS = True
    elif config.rbr_opt_algorithm == "adam":
        RBR_LBFGS = False
    else:
        raise ValueError("rbr_opt only supports lbfgs or adam")

    # Configure input paths
    tng_file = "{p}/{r}/{r}-tng_withrfree.mtz".format(p=config.path, r=config.file_root)
    input_pdb = "{p}/{r}/{r}-pred-aligned.pdb".format(p=config.path, r=config.file_root)
    true_pdb = "{p}/{r}/{r}_noalts.pdb".format(p=config.path, r=config.file_root)

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

    # If reference pdb exsits
    if os.path.exists(true_pdb):
        REFPDB = True
    else:
        REFPDB = False

    # If there are additional chain in the system
    if config.additional_chain:
        constant_fp_added_HKL = torch.load(
            "{p}/{r}/{r}_added_chain_atoms_HKL.pt".format(
                p=config.path, r=config.file_root
            )
        ).to(device=device)
        constant_fp_added_asu = torch.load(
            "{p}/{r}/{r}_added_chain_atoms_asu.pt".format(
                p=config.path, r=config.file_root
            )
        ).to(device=device)

        phitrue_path = "{p}/{r}/{r}_allchains-phitrue-solvent{s}.npy".format(
            p=config.path, r=config.file_root, s=config.solvent
        )
        Etrue_path = "{p}/{r}/{r}_allchains-Etrue-solvent{s}.npy".format(
            p=config.path, r=config.file_root, s=config.solvent
        )

        if os.path.exists(phitrue_path) and os.path.exists(Etrue_path):
            SIGMA_TRUE = True
            phitrue = np.load(phitrue_path)
            Etrue = np.load(Etrue_path)
        else:
            SIGMA_TRUE = False
    else:
        constant_fp_added_HKL = None
        constant_fp_added_asu = None

        phitrue_path = "{p}/{r}/{r}-phitrue-solvent{s}.npy".format(
            p=config.path, r=config.file_root, s=config.solvent
        )
        Etrue_path = "{p}/{r}/{r}-Etrue-solvent{s}.npy".format(
            p=config.path, r=config.file_root, s=config.solvent
        )

        if os.path.exists(phitrue_path) and os.path.exists(Etrue_path):
            SIGMA_TRUE = True
            phitrue = np.load(phitrue_path)
            Etrue = np.load(Etrue_path)
        else:
            SIGMA_TRUE = False

    ############ 2. Initializations ############
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
        spacing=config.voxel_spacing,
    )
    reference_pos = sfc.atom_pos_orth.clone()
    target_seq = sfc._pdb.sequence
    # CA mask and residue numbers for track
    cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(sfc.cra_name)
    residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]
    
    # Use initial pos B factor instead of best pos B factor for weighted L2
    init_pos_bfactor = sfc.atom_b_iso.clone()
    bfactor_weights = rk_utils.weighting_torch(
        init_pos_bfactor, cutoff2=20.0
    )

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
    best_rfree = float("inf")
    best_msa_bias = None
    best_feat_weights = None
    best_run = None
    best_iter = None
    best_pos = reference_pos

    # MH edit @ Oct 2nd, 2024: Support optional template input 
    if config.template_pdb is not None:
        device_processed_features_template = rocket.make_processed_dict_from_template(
            config.template_pdb, 
            target_seq, 
            device=device, 
            mask_sidechains_add_cb=False, 
            mask_sidechains=False, 
            max_recycling_iters=config.init_recycling
        )

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
                target_seq=target_seq,
                PRESET=PRESET,
            )
        )

        # MH edit @ Oct 2nd, 2024: Support optional template input 
        if config.template_pdb is not None:
            for key in device_processed_features_template.keys():
                if key.startswith("template_"):
                    device_processed_features[key] = device_processed_features_template[key]
        
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
        mse_losses_by_epoch = []
        rbr_loss_by_epoch = []
        sigmas_by_epoch = []
        true_sigmas_by_epoch = []
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
            desc=f"{config.file_root}, uuid: {refinement_run_uuid[:4]}, run: {run_id}",
        )
        
        # Run smooth stage in phase 1
        if "phase1" in config.note:
            w_L2 = config.l2_weight
        elif "phase2" in config.note:
            w_L2 = 0.0
            
        ######
        early_stopper = rkrf_utils.EarlyStopper(patience=200, min_delta=0.1)

        #### Phase 1 smooth scheduling ######
        if config.smooth_stage_epochs is not None:
            lr_a_initial = lr_a
            lr_m_initial = lr_m
            w_L2_initial = w_L2
            lr_stage1_final = config.phase2_final_lr
            smooth_stage_epochs = config.smooth_stage_epochs

            # Decay rates for each stage
            decay_rate_stage1_add = (lr_stage1_final / lr_a) ** (1 / smooth_stage_epochs)
            decay_rate_stage1_mul = (lr_stage1_final / lr_m) ** (1 / smooth_stage_epochs)

        ############ 3. Run Refinement ############
        for iteration in progress_bar:
            start_time = time.time()
            optimizer.zero_grad()

            # Avoid passing through graph a second time
            device_processed_features[feature_key] = (
                features_at_it_start.detach().clone()
            )
            # working_batch = copy.deepcopy(device_processed_features)
            # for bias in bias_names:
            #     working_batch[bias] = device_processed_features[bias].clone()

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
            L_plddt = - torch.mean(af2_output["plddt"])

            # Position Kabsch Alignment
            aligned_xyz, plddts_res, pseudo_Bs = rkrf_utils.position_alignment(
                af2_output=af2_output,
                device_processed_features=device_processed_features,
                cra_name=sfc.cra_name,
                best_pos=best_pos,
                exclude_res=EXCLUDING_RES,
                domain_segs=config.domain_segs,
            )
            llgloss.sfc.atom_b_iso = pseudo_Bs.detach()
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
                sigmas = llgloss.sigmaAs
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
            llgloss.sfc.savePDB(
                f"{output_directory_path!s}/{run_id}_{iteration}_preRBR.pdb"
            )

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
            if rfree_by_epoch[-1] < best_rfree:
                best_rfree = rfree_by_epoch[-1]
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
            # Save postRBR PDB
            llgloss.sfc.savePDB(
                f"{output_directory_path!s}/{run_id}_{iteration}_postRBR.pdb"
            )

            progress_bar.set_postfix(
                LLG=f"{llg_estimate:.2f}",
                r_work=f"{llgloss.sfc.r_work.item():.3f}",
                r_free=f"{llgloss.sfc.r_free.item():.3f}",
                memory=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}G",
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

            # Do smooth in last several iterations of phase 1 instead of beginning of phase 2 
            if ("phase1" in config.note) and (config.smooth_stage_epochs is not None):
                if iteration > (config.iterations - smooth_stage_epochs):
                    lr_a = lr_a_initial * (decay_rate_stage1_add**iteration)
                    lr_m = lr_m_initial * (decay_rate_stage1_mul**iteration)
                    w_L2 = w_L2_initial * (
                        1 - (iteration / smooth_stage_epochs)
                    )
                
                # Update the learning rates in the optimizer
                optimizer.param_groups[0]["lr"] = lr_a
                optimizer.param_groups[1]["lr"] = lr_m
                optimizer.step()
            else:
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

        # LLG per iteration
        np.save(
            f"{output_directory_path!s}/LLG_it_{run_id}.npy",
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

    config.to_yaml_file(f"{output_directory_path!s}/config.yaml")

    return refinement_run_uuid
