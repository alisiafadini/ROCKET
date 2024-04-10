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
from pathlib import Path
from typing import Union
from pydantic import BaseModel


PRESET = "model_1"
THRESH_B = None
EXCLUDING_RES = None


class RocketRefinmentConfig(BaseModel):
    path: Path
    file_root: Path
    bias_version: int
    iterations: int
    template_pdb: Union[Path, None]
    cuda_device: int
    solvent: bool
    sfc_scale: bool
    refine_sigmaA: bool
    additive_learning_rate: float
    multiplicative_learning_rate: float
    weight_decay: Union[float, None]  # TODO: should default be 0.0?
    batch_sub_ratio: float
    number_of_batches: int
    rbr_opt_algorithm: str
    rbr_lbfgs_learning_rate: float
    alignment_mode: str
    note: str
    free_flag: str
    testset_value: int
    additional_chain: bool
    verbose: bool
    l2_weight: float
    b_threshold: float
    min_resolution: Union[float, None]
    max_resolution: Union[float, None]
    starting_bias: Union[Path, None]
    starting_weights: Union[Path, None]

    # TODO the Path types are ugly
    # intercept them upload load/save and cast to string as appropriate
    def to_yaml_file(self, file_path: Path) -> None:
        with open(file_path, "w") as file:
            yaml.dump(self.dict(), file)

    @classmethod
    def from_yaml_file(self, file_path: Path):
        with open(file_path, "r") as file:
            payload = yaml.safe_load(file)
        return RocketRefinmentConfig.model_validate(payload)


def run_refinement(*, config: RocketRefinmentConfig) -> uuid.UUID:

    device = "cuda:{}".format(config.cuda_device)

    # Using LBFGS or Adam in RBR
    if config.rbr_opt_algorithm == "lbfgs":
        RBR_LBFGS = True
    elif config.rbr_opt_algorithm == "adam":
        RBR_LBFGS = False
    else:
        raise ValueError("rbr_opt only supports lbfgs or adam")

    # Load external files
    path = config.path  # no!
    # TODO rewrite others paths
    # working_path = config.path / config.file_root
    # tng_file = working_path / f"{config.file_root}-tng_withrfree.mtz"
    tng_file = "{p}/{r}/{r}-tng_withrfree.mtz".format(p=config.path, r=config.file_root)
    input_pdb = "{p}/{r}/{r}-pred-aligned.pdb".format(p=config.path, r=config.file_root)
    true_pdb = "{p}/{r}/{r}_noalts.pdb".format(p=config.path, r=config.file_root)

    if config.additional_chain:
        constant_fp_added_HKL = torch.load(
            "{p}/{r}/{r}_added_chain_atoms_HKL.pt".format(p=path, r=config.file_root)
        ).to(device=device)
        constant_fp_added_asu = torch.load(
            "{p}/{r}/{r}_added_chain_atoms_asu.pt".format(p=path, r=config.file_root)
        ).to(device=device)

        phitrue = np.load(
            "{p}/{r}/{r}_allchains-phitrue-solvent{s}.npy".format(
                p=path, r=config.file_root, s=config.solvent
            )
        )
        Etrue = np.load(
            "{p}/{r}/{r}_allchains-Etrue-solvent{s}.npy".format(
                p=path, r=config.file_root, s=config.solvent
            )
        )
    else:
        constant_fp_added_HKL = None
        constant_fp_added_asu = None
        phitrue = np.load(
            "{p}/{r}/{r}-phitrue-solvent{s}.npy".format(
                p=path, r=config.file_root, s=config.solvent
            )
        )
        Etrue = np.load(
            "{p}/{r}/{r}-Etrue-solvent{s}.npy".format(
                p=path, r=config.file_root, s=config.solvent
            )
        )

    if config.bias_version == 4:
        device_processed_features = rocket.make_processed_dict_from_template(
            template_pdb="{p}/{r}/{t}".format(
                p=path, r=config.file_root, t=config.template_pdb
            ),
            config_preset=PRESET,
            device=device,
            msa_dict=None,
        )
    else:
        with open(
            "{p}/{r}/{r}_processed_feats.pickle".format(p=path, r=config.file_root),
            "rb",
        ) as file:
            # Load the data from the pickle file
            processed_features = pickle.load(file)

        device_processed_features = rk_utils.move_tensors_to_device(
            processed_features, device=device
        )
        # TODO: this still takes up memory in original device?
        del processed_features

    # SFC initialization, only have to do it once
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
    )
    reference_pos = sfc.atom_pos_orth.clone()

    # Load true positions
    sfc_true = llg_sf.initial_SFC(
        true_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=config.free_flag,
        device=device,
        testset_value=config.testset_value,
    )
    true_pos = sfc_true.atom_pos_orth.clone()
    # true_Bs = sfc_true.atom_b_iso.clone()
    # true_cras = sfc_true.cra_name
    del sfc_true

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
    )

    # LLG initialization with resol cut
    # TODO dont duplicate variable names
    if config.min_resolution is None:
        resol_min = min(sfc.dHKL)
    else:
        resol_min = config.min_resolution

    if config.max_resolution is None:
        resol_max = max(sfc.dHKL)
    else:
        resol_max = config.max_resolution

    llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device, resol_min, resol_max)
    llgloss_rbr = rocket.llg.targets.LLGloss(
        sfc_rbr, tng_file, device, resol_min, resol_max
    )
    # llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

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

    # Initiate additive cluster bias
    num_res = device_processed_features["aatype"].shape[0]
    msa_params_bias = torch.zeros((512, num_res, 23), requires_grad=True, device=device)
    device_processed_features["msa_feat_bias"] = msa_params_bias

    # Optimizer settings and initialization
    lr_a = config.additive_learning_rate
    lr_m = config.multiplicative_learning_rate

    if config.bias_version == 4:
        device_processed_features["template_torsion_angles_sin_cos_bias"] = (
            torch.zeros_like(
                device_processed_features["template_torsion_angles_sin_cos"],
                requires_grad=True,
                device=device,
            )
        )
        if config.weight_decay is None:
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
                weight_decay=config.weight_decay,
            )
        bias_names = ["template_torsion_angles_sin_cos_bias"]

    elif config.bias_version == 3:
        if config.starting_weights is not None:
            device_processed_features["msa_feat_weights"] = (
                torch.load(config.starting_weights).detach().to(device=device)
            )
        else:
            device_processed_features["msa_feat_weights"] = torch.ones(
                (512, num_res, 23), requires_grad=True, device=device
            )

        if config.starting_bias is not None:
            device_processed_features["msa_feat_bias"] = (
                torch.load(config.starting_bias).detach().to(device=device)
            )

        device_processed_features["msa_feat_bias"].requires_grad = True
        device_processed_features["msa_feat_weights"].requires_grad = True

        if config.weight_decay is None:
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
                weight_decay=config.weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif config.bias_version == 2:
        msa_params_weights = torch.eye(512, dtype=torch.float32, device=device)
        device_processed_features["msa_feat_weights"] = msa_params_weights

        if config.weight_decay is None:
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
                weight_decay=config.weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif config.bias_version == 1:
        if config.weight_decay is None:
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
                weight_decay=config.weight_decay,
            )

        bias_names = ["msa_feat_bias"]

    refinement_run_uuid = uuid.uuid4()

    output_directory_path = Path(
        f"{path}/{config.file_root}/outputs/{refinement_run_uuid}"
    )

    print(refinement_run_uuid, flush=True)
    if not config.verbose:
        warnings.filterwarnings("ignore")

    # Initialize best variables for alignement
    best_loss = float("inf")
    best_pos = reference_pos

    # Initialize best Rfree weights and bias for Phase 1
    best_rfree = float("inf")
    best_msa_bias = None
    best_feat_weights = None

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

    if config.bias_version == 4:
        features_at_it_start = (
            device_processed_features["template_torsion_angles_sin_cos"][..., 0]
            .detach()
            .clone()
        )
    else:
        features_at_it_start = (
            device_processed_features["msa_feat"][:, :, 25:48, 0].detach().clone()
        )

    progress_bar = tqdm(range(config.iterations), desc=f"version {config.bias_version}")
    loss_weight = config.l2_weight
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
            af2_output, device_processed_features, llgloss.sfc.cra_name
        )

        all_pldtts.append(rk_utils.assert_numpy(af2_output["plddt"]))
        mean_it_plddts.append(rk_utils.assert_numpy(torch.mean(plddts)))

        pseudo_Bs = rk_coordinates.update_bfactors(plddts)
        llgloss.sfc.atom_b_iso = pseudo_Bs.detach()

        aligned_xyz = rk_coordinates.align_positions(
            xyz_orth_sfc,
            best_pos,
            llgloss.sfc.cra_name,
            pseudo_Bs,
            thresh_B=THRESH_B,
            exclude_res=EXCLUDING_RES,
        )

        ##### Residue MSE loss for tracking ######
        # (1) Select CAs
        cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(
            sfc.cra_name
        )

        # (2) Convert residue names to residue numbers
        residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]

        # (3) Calculate total MSE loss
        total_mse_loss = rk_coordinates.calculate_mse_loss_per_residue(
            aligned_xyz[calphas_mask], true_pos[calphas_mask], residue_numbers
        )
        mse_losses_by_epoch.append(total_mse_loss)
        ##############################################

        # Calculate (or refine) sigmaA
        # TODO before or after RBR step?
        Ecalc, Fc = llgloss.compute_Ecalc(
            aligned_xyz,
            return_Fc=True,
            update_scales=True,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
        )

        Ecalc_rbr, Fc_rbr = llgloss_rbr.compute_Ecalc(
            aligned_xyz,
            return_Fc=True,
            solvent=False,
            update_scales=True,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
        )

        if config.refine_sigmaA is True:

            llgloss.refine_sigmaA_newton(
                Ecalc, n_steps=5, subset="working", smooth_overall_weight=0.0
            )
            llgloss_rbr.refine_sigmaA_newton(
                Ecalc_rbr, n_steps=2, subset="working", smooth_overall_weight=0.0
            )
            sigmas = llgloss.sigmaAs

        else:
            sigmas = llg_utils.sigmaA_from_model(
                Etrue,
                phitrue,
                Ecalc,
                Fc,
                llgloss.sfc.dHKL,
                llgloss.bin_labels,
            )
            llgloss.sigmaAs = sigmas

            sigmas_rbr = llg_utils.sigmaA_from_model(
                Etrue,
                phitrue,
                Ecalc_rbr,
                Fc_rbr,
                llgloss.sfc.dHKL,
                llgloss.bin_labels,
            )
            llgloss_rbr.sigmaAs = sigmas_rbr

        true_sigmas = llg_utils.sigmaA_from_model(
            Etrue,
            phitrue,
            Ecalc,
            Fc,
            llgloss.sfc.dHKL,
            llgloss.bin_labels,
        )

        # Update SFC and save
        llgloss.sfc.atom_pos_orth = aligned_xyz
        llgloss.sfc.savePDB(f"{output_directory_path!s}/{iteration}_preRBR.pdb")

        # Rigid body refinement (RBR) step
        optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz,
            llgloss_rbr,
            lbfgs=RBR_LBFGS,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
            lbfgs_lr=config.rbr_lbfgs_learning_rate,
            verbose=config.verbose,
        )
        rbr_loss_by_epoch.append(loss_track_pose)

        # LLG loss
        loss = -llgloss(
            optimized_xyz,
            bin_labels=None,
            num_batch=config.number_of_batches,
            sub_ratio=config.batch_sub_ratio,
            solvent=config.solvent,
            update_scales=config.sfc_scale,
            added_chain_HKL=constant_fp_added_HKL,
            added_chain_asu=constant_fp_added_asu,
        )

        llg_estimate = loss.item() / (config.batch_sub_ratio * config.number_of_batches)
        llg_losses.append(llg_estimate)
        rwork_by_epoch.append(llgloss.sfc.r_work.item())
        rfree_by_epoch.append(llgloss.sfc.r_free.item())

        # Check if current Rfree is the best so far
        if rfree_by_epoch[-1] < best_rfree:
            best_rfree = rfree_by_epoch[-1]
            best_msa_bias = (
                device_processed_features["msa_feat_bias"].detach().cpu().clone()
            )
            best_feat_weights = (
                device_processed_features["msa_feat_weights"].detach().cpu().clone()
            )

        llgloss.sfc.atom_pos_orth = optimized_xyz
        # Save postRBR PDB
        llgloss.sfc.savePDB(f"{output_directory_path!s}/{iteration}_postRBR.pdb")

        progress_bar.set_postfix(
            LLG_Estimate=f"{llg_estimate:.2f}",
            r_work=f"{llgloss.sfc.r_work.item():.3f}",
            r_free=f"{llgloss.sfc.r_free.item():.3f}",
            memory=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}G",
        )

        if config.alignment_mode == "B":
            if loss < best_loss:
                best_loss = loss
                best_pos = optimized_xyz.clone()

        # Save sigmaA values for further processing
        sigmas_dict = {
            f"sigma_{i + 1}": sigma_value.item() for i, sigma_value in enumerate(sigmas)
        }
        sigmas_by_epoch.append(sigmas_dict)

        true_sigmas_dict = {
            f"sigma_{i + 1}": sigma_value.item()
            for i, sigma_value in enumerate(true_sigmas)
        }
        true_sigmas_by_epoch.append(true_sigmas_dict)

        #### add an L2 loss to constrain confident atoms ###
        if loss_weight > 0.0:
            if iteration == 0:
                # L2_ref_pos = xyz_orth_sfc.clone().detach()
                L2_ref_pos = optimized_xyz.detach().clone()
                L2_ref_Bs = llgloss.sfc.atom_b_iso.detach().clone()
                conf_xyz, conf_best = rk_coordinates.select_confident_atoms(
                    optimized_xyz,
                    L2_ref_pos,
                    bfacts=L2_ref_Bs,
                    b_thresh=config.b_threshold,
                )

            else:
                # Avoid passing through graph twice with L2 loss addition
                # L2_ref_pos_copy = L2_ref_pos.clone()
                # L2_ref_Bs_copy = L2_ref_Bs.clone()
                conf_xyz, conf_best = rk_coordinates.select_confident_atoms(
                    optimized_xyz,
                    L2_ref_pos,
                    bfacts=L2_ref_Bs,
                    b_thresh=config.b_threshold,
                )

            L2_loss = torch.sum((conf_xyz - conf_best) ** 2)  # / conf_best.shape[0]
            corrected_loss = loss + loss_weight * L2_loss
            corrected_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        time_by_epoch.append(time.time() - start_time)
        memory_by_epoch.append(torch.cuda.max_memory_allocated() / 1024**3)

        # Save the absolute difference in mean contribution from each residue channel from previous iteration
        if config.bias_version == 4:
            features_at_step_end = (
                working_batch["template_torsion_angles_sin_cos"][..., 0]
                .detach()
                .clone()
            )
            mean_change = torch.mean(
                torch.abs(features_at_step_end - features_at_it_start), dim=(0, 2, 3)
            )
        else:
            features_at_step_end = (
                working_batch["msa_feat"][:, :, 25:48, 0].detach().clone()
            )
            mean_change = torch.mean(
                torch.abs(features_at_step_end - features_at_it_start), dim=(0, 2)
            )
        absolute_feats_changes.append(rk_utils.assert_numpy(mean_change))

    ####### Save data

    # Average plddt per iteration
    np.save(
        f"{output_directory_path!s}/mean_it_plddt.npy",
        np.array(mean_it_plddts),
    )

    # LLG per iteration
    np.save(
        f"{output_directory_path!s}/LLG_it.npy",
        rk_utils.assert_numpy(llg_losses),
    )

    # MSE loss per iteration
    np.save(
        f"{output_directory_path!s}/MSE_loss_it.npy",
        rk_utils.assert_numpy(mse_losses_by_epoch),
    )

    # R work per iteration
    np.save(
        f"{output_directory_path!s}/rwork_it.npy",
        rk_utils.assert_numpy(rwork_by_epoch),
    )

    # R free per iteration
    np.save(
        f"{output_directory_path!s}/rfree_it.npy",
        rk_utils.assert_numpy(rfree_by_epoch),
    )

    np.save(
        f"{output_directory_path!s}/time_it.npy",
        rk_utils.assert_numpy(time_by_epoch),
    )

    np.save(
        f"{output_directory_path!s}/memory_it.npy",
        rk_utils.assert_numpy(memory_by_epoch),
    )

    # Absolute MSA change per column per iteration
    np.save(
        f"{output_directory_path!s}/MSA_changes_it.npy",
        rk_utils.assert_numpy(absolute_feats_changes),
    )

    # Mean plddt per residue (over iterations)
    np.save(
        f"{output_directory_path!s}/mean_plddt_res.npy",
        np.mean(np.array(all_pldtts), axis=0),
    )

    # Iteration sigmaA dictionary
    with open(
        f"{output_directory_path!s}/sigmas_by_epoch.pkl",
        "wb",
    ) as file:
        pickle.dump(sigmas_by_epoch, file)

    with open(
        f"{output_directory_path!s}/true_sigmas_by_epoch.pkl",
        "wb",
    ) as file:
        pickle.dump(true_sigmas_by_epoch, file)

    # Save the best msa_bias and feat_weights
    torch.save(
        best_msa_bias,
        f"{output_directory_path!s}/best_msa_bias.pt",
    )

    torch.save(
        best_feat_weights,
        f"{output_directory_path!s}/best_feat_weights.pt",
    )

    config.to_yaml_file(f"{output_directory_path!s}/config.yaml")

    return refinement_run_uuid
