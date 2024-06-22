import pickle
import torch
import numpy as np
import rocket
from rocket import utils as rk_utils
from rocket import coordinates as rk_coordinates
from rocket.llg import utils as llg_utils

def number_to_letter(n):
    if 0 <= n <= 25:
        return chr(n + 65)
    else:
        return None

def get_current_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]


def init_processed_dict(
        bias_version, 
        path, 
        file_root, 
        device, 
        template_pdb=None, 
        PRESET='model_1'
    ):
    if bias_version == 4:
        device_processed_features = rocket.make_processed_dict_from_template(
            template_pdb="{p}/{r}/{t}".format(
                p=path, r=file_root, t=template_pdb
            ),
            config_preset=PRESET,
            device=device,
            msa_dict=None,
        )
        features_at_it_start = (
            device_processed_features["template_torsion_angles_sin_cos"]
            .detach()
            .clone()
        )
        feature_key = "template_torsion_angles_sin_cos"
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
        features_at_it_start = (
            device_processed_features["msa_feat"].detach().clone()
        )
        feature_key = "msa_feat" 
    return device_processed_features, feature_key, features_at_it_start


def init_llgloss(sfc, tng_file, min_resolution=None, max_resolution=None):
    if min_resolution is None:
        resol_min = min(sfc.dHKL)
    else:
        resol_min = min_resolution

    if max_resolution is None:
        resol_max = max(sfc.dHKL)
    else:
        resol_max = max_resolution
    llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, sfc.device, resol_min, resol_max)
    return llgloss


def init_bias(
        device_processed_features, 
        bias_version, 
        device, 
        lr_a, 
        lr_m, 
        weight_decay=None, 
        starting_bias=None, 
        starting_weights=None
    ):
    num_res = device_processed_features["aatype"].shape[0]
    device_processed_features["msa_feat_bias"] = torch.zeros((512, num_res, 23), requires_grad=True, device=device)

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
                torch.load(starting_weights).detach().to(device=device).requires_grad_(True)
            )
        else:
            device_processed_features["msa_feat_weights"] = torch.ones(
                (512, num_res, 23), requires_grad=True, device=device
            )

        if starting_bias is not None:
            device_processed_features["msa_feat_bias"] = (
                torch.load(starting_bias).detach().to(device=device).requires_grad_(True)
            )

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
        device_processed_features["msa_feat_weights"] = torch.eye(512, dtype=torch.float32, requires_grad=True, device=device)

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
    
    return device_processed_features, optimizer, bias_names


def position_alignment(af2_output, device_processed_features, cra_name, best_pos, exclude_res):
    xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
        af2_output, device_processed_features, cra_name
    )
    plddts_res = rk_utils.assert_numpy(af2_output["plddt"])
    pseudo_Bs = rk_coordinates.update_bfactors(plddts) 
    
    weights = rk_utils.weighting(rk_utils.assert_numpy(pseudo_Bs))
    aligned_xyz = rk_coordinates.weighted_kabsch(
        xyz_orth_sfc,
        best_pos,
        cra_name,
        weights=weights,
        exclude_res=exclude_res,
    )
    return aligned_xyz, plddts_res, pseudo_Bs.detach()


def update_sigmaA(
        llgloss,
        llgloss_rbr,
        aligned_xyz,
        constant_fp_added_HKL=None, 
        constant_fp_added_asu=None,
    ):
    Ecalc, Fc = llgloss.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
    Ecalc_rbr, _ = llgloss_rbr.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        solvent=False,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
    llgloss.refine_sigmaA_newton(
        Ecalc, n_steps=5, subset="working", smooth_overall_weight=0.0
    )
    llgloss_rbr.refine_sigmaA_newton(
        Ecalc_rbr, n_steps=2, subset="working", smooth_overall_weight=0.0
    )
    return llgloss, llgloss_rbr, Ecalc, Fc


def sigmaA_from_true(
        llgloss,
        llgloss_rbr,
        aligned_xyz,
        Etrue,
        phitrue,
        constant_fp_added_HKL=None, 
        constant_fp_added_asu=None,
    ):
    Ecalc, Fc = llgloss.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
    Ecalc_rbr, Fc_rbr = llgloss_rbr.compute_Ecalc(
        aligned_xyz.detach(),
        return_Fc=True,
        solvent=False,
        update_scales=True,
        added_chain_HKL=constant_fp_added_HKL,
        added_chain_asu=constant_fp_added_asu,
    )
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
    return llgloss, llgloss_rbr