"""
Command line interface of running refinement using rocket

rk.refine 
    --path             xxxxx             # Path to the parent folder
    --file_root        xxxxx             # Dataset folder name in the parent folder
    --version          1                 # Bias version of implementation, 1, 2 or 3 or 4 (template)
    --template_pdb     xxxx.pdb          # Name of template pdb file in the file_root
    --iterations       300               # Number of refinement steps
    --lr_add           1e-3              # Learning rate of msa_bias
    --lr_mul           1e-2              # Learning rate of msa_weights
    --weight_decay     None or float     # Weight decay parameters used in AdamW
    --sub_ratio        1.0               # Ratio of reflections for each batch
    --batches          1                 # Number of batches at each step
    --rbr_opt          'lbfgs'           # Using 'lbfgs' or 'adam' in the rbr optimization
    --rbr_lbfgs_lr     150.0             # Learning rate of lbfgs used in RBR
    --align            'B'               # Kabsch to best (B) or initial (I)
    --note             xxxx              # Additional notes used in output name
    --free_flag        'R-free-flags'    # Coloum name for the free flag in mtz file
    --testset_value    0                 # testset value in the freeflag column
    --solvent or --no-solvent            # Turn on the solvent in the llgloss calculation
    --scale   or --no-scale              # Turn on the SFC update_scale in each step
    --added_chain                        # Turn on additional chain in the asu
    --verbose                            # Be verbose during refinement
"""

import copy, warnings
import torch
import pickle
import numpy as np
from tqdm import tqdm
import rocket
import os, time
import argparse
from rocket.llg import utils as llg_utils
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from rocket.llg import structurefactors as llg_sf
from openfold.config import model_config
import pickle

PRESET = "model_1"
THRESH_B = None
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
        "--version",
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
        "-temp",
        "--template_pdb",
        default=None,
        help=("Name of template pdb file in the file_root"),
    )

    parser.add_argument("-c", "--cuda", type=int, default=0, help="Cuda device")
    parser.add_argument(
        "-solv",
        "--solvent",
        help="Turn on solvent calculation in refinement step",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-s",
        "--scale",
        help="Turn on SFC scale_update at each epoch",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "-sigmaA",
        "--sigmaArefine",
        help="Turn on sigmaA refinement at each epoch",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "-lr_a",
        "--lr_add",
        type=float,
        default=1e-3,
        help=("Learning rate for additive bias. Default 1e-3"),
    )

    parser.add_argument(
        "-lr_m",
        "--lr_mul",
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
        "-sub_r",
        "--sub_ratio",
        type=float,
        default=1.0,
        help=("Ratio of reflections for each batch. Default 1.0 (no batching)"),
    )

    parser.add_argument(
        "-b",
        "--batches",
        type=int,
        default=1,
        help=("Number of batches. Default 1 (no batching)"),
    )

    parser.add_argument(
        "--rbr_opt",
        type=str,
        default="lbfgs",
        help=("Optimization algorithm used in RBR, lbfgs or adam"),
    )

    parser.add_argument(
        "--rbr_lbfgs_lr",
        type=float,
        default=150.0,
        help=("Learning rate of lbfgs used in RBR"),
    )

    parser.add_argument(
        "-a",
        "--align",
        type=str,
        default="B",
        choices=["B", "I"],
        help=("Kabsch to best (B) or initial (I)"),
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

    parser.add_argument(
        "-chain",
        "--added_chain",
        help="additional chain in asu",
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--verbose",
        help="Be verbose during refinement",
        action="store_true",
    )

    parser.add_argument(
        "-L2_w",
        "--L2_weight",
        type=float,
        default=0.0,
        help=("Weight for L2 loss"),
    )

    parser.add_argument(
        "-b_thresh",
        "--b_threshold",
        type=float,
        default=10.0,
        help=("B threshold for L2 loss"),
    )

    parser.add_argument(
        "--resol_min",
        type=float,
        default=None,
        help=("min resolution for llg calculation"),
    )

    parser.add_argument(
        "--resol_max",
        type=float,
        default=None,
        help=("max resolution for llg calculation"),
    )

    return parser.parse_args()


def main():
    # Parse commandline arguments
    args = parse_arguments()

    # General settings
    device = "cuda:{}".format(args.cuda)

    # Using LBFGS or Adam in RBR
    if args.rbr_opt == "lbfgs":
        RBR_LBFGS = True
    elif args.rbr_opt == "adam":
        RBR_LBFGS = False
    else:
        raise ValueError("rbr_opt only supports lbfgs or adam")

    # Load external files
    path = args.path
    tng_file = "{p}/{r}/{r}-tng_withrfree.mtz".format(p=path, r=args.file_root)
    input_pdb = "{p}/{r}/{r}-pred-aligned.pdb".format(p=path, r=args.file_root)
    true_pdb = "{p}/{r}/{r}_noalts.pdb".format(p=path, r=args.file_root)

    if args.added_chain:
        constant_fp_added = torch.load(
            "{p}/{r}/{r}_added_chain_atoms.pt".format(p=path, r=args.file_root)
        ).to(device=device)
        phitrue = np.load(
            "{p}/{r}/{r}_allchains-phitrue-solvent{s}.npy".format(
                p=path, r=args.file_root, s=args.solvent
            )
        )
        Etrue = np.load(
            "{p}/{r}/{r}_allchains-Etrue-solvent{s}.npy".format(
                p=path, r=args.file_root, s=args.solvent
            )
        )
    else:
        constant_fp_added = None
        phitrue = np.load(
            "{p}/{r}/{r}-phitrue-solvent{s}.npy".format(
                p=path, r=args.file_root, s=args.solvent
            )
        )
        Etrue = np.load(
            "{p}/{r}/{r}-Etrue-solvent{s}.npy".format(
                p=path, r=args.file_root, s=args.solvent
            )
        )

    if args.version == 4:
        device_processed_features = rocket.make_processed_dict_from_template(
            template_pdb="{p}/{r}/{t}".format(
                p=path, r=args.file_root, t=args.template_pdb
            ),
            config_preset=PRESET,
            device=device,
            msa_dict=None,
        )
    else:
        with open(
            "{p}/{r}/{r}_processed_feats.pickle".format(p=path, r=args.file_root), "rb"
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
        Freelabel=args.free_flag,
        device=device,
        testset_value=args.testset_value,
    )
    reference_pos = sfc.atom_pos_orth.clone()

    # Load true positions
    sfc_true = llg_sf.initial_SFC(
        true_pdb,
        tng_file,
        "FP",
        "SIGFP",
        Freelabel=args.free_flag,
        device=device,
        testset_value=args.testset_value,
    )
    true_pos = sfc_true.atom_pos_orth.clone()
    # true_Bs = sfc_true.atom_b_iso.clone()
    # true_cras = sfc_true.cra_name
    del sfc_true

    # LLG initialization with resol cut
    if args.resol_min is None:
        resol_min = min(sfc.dHKL)
    else:
        resol_min = args.resol_min

    if args.resol_max is None:
        resol_max = max(sfc.dHKL)
    else:
        resol_max = args.resol_max

    llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device, resol_min, resol_max)
    # llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

    # Model initialization
    version_to_class = {
        1: rocket.MSABiasAFv1,
        2: rocket.MSABiasAFv2,
        3: rocket.MSABiasAFv3,
        4: rocket.TemplateBiasAF,
    }
    af_bias = version_to_class[args.version](
        model_config(PRESET, train=True), PRESET
    ).to(device)
    af_bias.freeze()  # Free all AF2 parameters to save time

    # Initiate additive cluster bias
    num_res = device_processed_features["aatype"].shape[0]
    msa_params_bias = torch.zeros((512, num_res, 23), requires_grad=True, device=device)
    device_processed_features["msa_feat_bias"] = msa_params_bias

    # Optimizer settings and initialization
    lr_a = args.lr_add
    lr_m = args.lr_mul

    if args.version == 4:
        device_processed_features["template_torsion_angles_sin_cos_bias"] = (
            torch.zeros_like(
                device_processed_features["template_torsion_angles_sin_cos"],
                requires_grad=True,
                device=device,
            )
        )
        if args.weight_decay is None:
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
                weight_decay=args.weight_decay,
            )
        bias_names = ["template_torsion_angles_sin_cos_bias"]

    elif args.version == 3:
        # Initiate multiplicative cluster bias
        msa_params_weights = torch.ones(
            (512, num_res, 23), requires_grad=True, device=device
        )
        device_processed_features["msa_feat_weights"] = msa_params_weights
        device_processed_features["msa_feat_weights"].requires_grad_(True)

        if args.weight_decay is None:
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
                weight_decay=args.weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif args.version == 2:
        msa_params_weights = torch.eye(512, dtype=torch.float32, device=device)
        device_processed_features["msa_feat_weights"] = msa_params_weights

        if args.weight_decay is None:
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
                weight_decay=args.weight_decay,
            )
        bias_names = ["msa_feat_bias", "msa_feat_weights"]

    elif args.version == 1:
        if args.weight_decay is None:
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
                weight_decay=args.weight_decay,
            )

        bias_names = ["msa_feat_bias"]

    # Run options
    output_name = "{root}_it{it}_v{v}_lr{a}+{m}_wd{wd}_batch{b}_subr{subr}_solv{solv}_scale{scale}_sigmaA{sigma}_resol_{resol_min:.2f}_{resol_max:.2f}_rbr{rbr_opt}_{rbr_lbfgs_lr}_ali{align}_L2{weight}+{thresh}_{add}".format(
        root=args.file_root,
        it=args.iterations,
        v=args.version,
        a=args.lr_add,
        m=args.lr_mul,
        wd=args.weight_decay,
        b=args.batches,
        subr=args.sub_ratio,
        solv=args.solvent,
        scale=args.scale,
        sigma=args.sigmaArefine,
        resol_min=resol_min,
        resol_max=resol_max,
        rbr_opt=args.rbr_opt,
        rbr_lbfgs_lr=args.rbr_lbfgs_lr,
        align=args.align,
        weight=args.L2_weight,
        thresh=args.b_threshold,
        add=args.note,
    )

    if not args.verbose:
        warnings.filterwarnings("ignore")

    # Initialize best variables for alignement
    best_loss = float("inf")
    best_pos = reference_pos

    # List initialization for saving values
    mse_losses_by_epoch = []
    rbr_loss_by_epoch = []
    sigmas_by_epoch = []
    llg_losses = []
    rfree_by_epoch = []
    rwork_by_epoch = []
    time_by_epoch = []
    memory_by_epoch = []
    all_pldtts = []
    mean_it_plddts = []
    absolute_feats_changes = []

    if args.version == 4:
        features_at_it_start = (
            device_processed_features["template_torsion_angles_sin_cos"][..., 0]
            .detach()
            .clone()
        )
    else:
        features_at_it_start = (
            device_processed_features["msa_feat"][:, :, 25:48, 0].detach().clone()
        )

    progress_bar = tqdm(range(args.iterations), desc=f"version {args.version}")
    loss_weight = args.L2_weight
    for iteration in progress_bar:
        start_time = time.time()
        optimizer.zero_grad()

        if iteration == 0:
            directory_path = "{path}/{r}/outputs/{out}".format(
                path=path, r=args.file_root, out=output_name
            )
            try:
                os.makedirs(directory_path, exist_ok=True)
            except FileExistsError:
                print(
                    f"Warning: Directory '{directory_path}' already exists. Overwriting..."
                )

        # if iteration == 250:
        #    lr_a = 1e-4
        #    lr_m = 1e-3

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
            added_chain=constant_fp_added,
        )

        if args.sigmaArefine is True:

            llgloss.refine_sigmaA_adam(
                Ecalc, sub_ratio=0.97, lr=0.005, initialize=True, verbose=True
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

        # print(
        #    "######### sigmas after refinement are",
        #    [sigmaA.item() for sigmaA in llgloss.sigmaAs],
        # )

        true_sigmas = llg_utils.sigmaA_from_model(
            Etrue,
            phitrue,
            Ecalc,
            Fc,
            llgloss.sfc.dHKL,
            llgloss.bin_labels,
        )

        # print("######### True sigmas are", [sigmaA.item() for sigmaA in true_sigmas])

        # Update SFC and save
        llgloss.sfc.atom_pos_orth = aligned_xyz
        llgloss.sfc.savePDB(
            "{path}/{r}/outputs/{out}/{it}_preRBR.pdb".format(
                path=path, r=args.file_root, out=output_name, it=iteration
            )
        )

        # Rigid body refinement (RBR) step
        optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz,
            llgloss,
            lbfgs=RBR_LBFGS,
            added_chain=constant_fp_added,
            lbfgs_lr=args.rbr_lbfgs_lr,
            verbose=args.verbose,
        )
        rbr_loss_by_epoch.append(loss_track_pose)

        # LLG loss
        loss = -llgloss(
            optimized_xyz,
            bin_labels=None,
            num_batch=args.batches,
            sub_ratio=args.sub_ratio,
            solvent=args.solvent,
            update_scales=args.scale,
            added_chain=constant_fp_added,
        )

        llg_estimate = loss.item() / (args.sub_ratio * args.batches)
        llg_losses.append(llg_estimate)
        rwork_by_epoch.append(llgloss.sfc.r_work.item())
        rfree_by_epoch.append(llgloss.sfc.r_free.item())

        llgloss.sfc.atom_pos_orth = optimized_xyz
        # Save postRBR PDB
        llgloss.sfc.savePDB(
            "{path}/{r}/outputs/{out}/{it}_postRBR.pdb".format(
                path=path, r=args.file_root, out=output_name, it=iteration
            )
        )

        progress_bar.set_postfix(
            LLG_Estimate=f"{llg_estimate:.2f}",
            r_work=f"{llgloss.sfc.r_work.item():.3f}",
            r_free=f"{llgloss.sfc.r_free.item():.3f}",
            memory=f"{torch.cuda.max_memory_allocated()/1024**3:.1f}G",
        )

        if args.align == "B":
            if loss < best_loss:
                best_loss = loss
                best_pos = optimized_xyz.clone()

        # Save sigmaA values for further processing
        sigmas_dict = {
            f"sigma_{i + 1}": sigma_value for i, sigma_value in enumerate(sigmas)
        }
        sigmas_by_epoch.append(sigmas_dict)

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
                    b_thresh=args.b_threshold,
                )

            else:
                # Avoid passing through graph twice with L2 loss addition
                # L2_ref_pos_copy = L2_ref_pos.clone()
                # L2_ref_Bs_copy = L2_ref_Bs.clone()
                conf_xyz, conf_best = rk_coordinates.select_confident_atoms(
                    optimized_xyz,
                    L2_ref_pos,
                    bfacts=L2_ref_Bs,
                    b_thresh=args.b_threshold,
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
        if args.version == 4:
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
        # print("Mean absolute feats change", torch.mean(mean_change).item())

        if iteration == (args.iterations - 1):
            print("Currently at last iteration {}".format(iteration))
            torch.save(
                device_processed_features["msa_feat_bias"],
                "{path}/{r}/outputs/{out}/add_bias_final.pt".format(
                    path=path,
                    r=args.file_root,
                    out=output_name,
                ),
            )
            torch.save(
                device_processed_features["msa_feat_weights"],
                "{path}/{r}/outputs/{out}/mul_bias_final.pt".format(
                    path=path,
                    r=args.file_root,
                    out=output_name,
                ),
            )

    ####### Save data

    # Average plddt per iteration
    np.save(
        "{path}/{r}/outputs/{out}/mean_it_plddt.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        np.array(mean_it_plddts),
    )

    # LLG per iteration
    np.save(
        "{path}/{r}/outputs/{out}/LLG_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(llg_losses),
    )

    # MSE loss per iteration
    np.save(
        "{path}/{r}/outputs/{out}/MSE_loss_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(mse_losses_by_epoch),
    )

    # R work per iteration
    np.save(
        "{path}/{r}/outputs/{out}/rwork_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(rwork_by_epoch),
    )

    # R free per iteration
    np.save(
        "{path}/{r}/outputs/{out}/rfree_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(rfree_by_epoch),
    )

    np.save(
        "{path}/{r}/outputs/{out}/time_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(time_by_epoch),
    )

    np.save(
        "{path}/{r}/outputs/{out}/memory_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(memory_by_epoch),
    )

    # Absolute MSA change per column per iteration
    np.save(
        "{path}/{r}/outputs/{out}/MSA_changes_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(absolute_feats_changes),
    )

    # Mean plddt per residue (over iterations)
    np.save(
        "{path}/{r}/outputs/{out}/mean_plddt_res.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        np.mean(np.array(all_pldtts), axis=0),
    )

    # Iteration sigmaA dictionary
    with open(
        "{path}/{r}/outputs/{out}/sigmas_by_epoch.pkl".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        "wb",
    ) as file:
        pickle.dump(sigmas_by_epoch, file)


if __name__ == "__main__":
    main()
