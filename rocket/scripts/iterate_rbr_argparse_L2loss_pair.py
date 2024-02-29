import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
import rocket
import os
import argparse
from rocket.llg import utils as llg_utils
from rocket import coordinates as rk_coordinates
from rocket import utils as rk_utils
from rocket.llg import structurefactors as llg_sf
from openfold.config import model_config
import pickle


def parse_arguments():
    """Parse commandline arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    # Required arguments
    parser.add_argument(
        "-p",
        "--path",
        default="/net/cci/alisia/openfold_tests/run_openfold/test_cases",
        help=("Path to the parent folder"),
    )
    
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
        help=("Bias version to implement (1, 2, 3)"),
    )

    parser.add_argument(
        "-it",
        "--iterations",
        required=True,
        type=int,
        help=("Refinement iterations"),
    )

    # Optional arguments

    parser.add_argument("-c", "--cuda", type=int, default=0, help="Cuda device")
    parser.add_argument(
        "-solv",
        "--solvent",
        help="Solvent calculation in refinement step. Default False.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-s",
        "--scale",
        help="Update scales at each epoch",
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
        default=30.0,
        help=("B threshold for L2 loss"),
    )

    return parser.parse_args()


def main():
    # Parse commandline arguments
    args = parse_arguments()

    # General settings
    preset = "model_1"
    device = "cuda:{}".format(args.cuda)

    # Using LBFGS in RBR or not
    RBR_LBFGS = True

    # Load external files
    path = args.path
    tng_file = "{p}/{r}/{r}-tng_withrfree.mtz".format(p=path, r=args.file_root)
    input_pdb = "{p}/{r}/{r}-pred-aligned.pdb".format(p=path, r=args.file_root)
    true_pdb = "{p}/{r}/{r}_noalts.pdb".format(p=path, r=args.file_root)

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

    if args.added_chain:
        constant_fp_added = torch.load(
            "{p}/{r}/{r}_added_chain_atoms.pt".format(p=path, r=args.file_root)
        ).to(device=device)

    else:
        constant_fp_added = None

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
        input_pdb, tng_file, "FP", "SIGFP", Freelabel=args.free_flag, device=device
    )
    reference_pos = sfc.atom_pos_orth.clone()

    # Load true positions
    sfc_true = llg_sf.initial_SFC(
        true_pdb, tng_file, "FP", "SIGFP", Freelabel=args.free_flag, device=device
    )
    true_pos = sfc_true.atom_pos_orth.clone()
    # true_Bs = sfc_true.atom_b_iso.clone()
    # true_cras = sfc_true.cra_name
    del sfc_true

    # LLG initialization
    llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

    # Model initialization
    version_to_class = {
        1: rocket.MSABiasAFv1,
        2: rocket.MSABiasAFv2,
        3: rocket.MSABiasAFv3,
        5: rocket.MSABiasAFv3,
    }
    af_bias = version_to_class[args.version](
        model_config(preset, train=True), preset
    ).to(device)
    af_bias.freeze()  # Free all AF2 parameters to save time

    # Initiate additive cluster bias
    num_res = device_processed_features["aatype"].shape[0]
    # msa_params_bias = torch.zeros((512, num_res, 23), requires_grad=True, device=device)
    # device_processed_features["msa_feat_bias"] = msa_params_bias

    # Optimizer settings and initialization
    lr_a = args.lr_add
    lr_m = args.lr_mul

    if args.version == 3:
        # Initiate multiplicative cluster bias
        msa_params_weights = torch.ones(
            (512, num_res, 23), requires_grad=True, device=device
        )
        device_processed_features["msa_feat_weights"] = msa_params_weights
        device_processed_features["msa_feat_weights"].requires_grad_(True)

        optimizer = torch.optim.Adam(
            [
                {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                {"params": device_processed_features["msa_feat_weights"], "lr": lr_m},
            ]
        )
    elif args.version == 2:
        msa_params_weights = torch.eye(512, dtype=torch.float32, device=device)
        device_processed_features["msa_feat_weights"] = msa_params_weights

        optimizer = torch.optim.Adam(
            [
                {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
                {"params": device_processed_features["msa_feat_weights"], "lr": lr_m},
            ]
        )

    ################
    elif args.version == 5:
        # device_processed_features["msa_feat_weights"] = torch.load("/net/cci/alisia/openfold_tests/run_openfold/test_cases/3hak/outputs/3hak_it50_v5_lr0.2+5.0_batch1_subr1.0_solvTrue_scaleTrue_Ib50_w1/mul_bias_final.pt").to(device=device)
        # device_processed_features["msa_feat_bias"] = torch.load("/net/cci/alisia/openfold_tests/run_openfold/test_cases/3hak/outputs/3hak_it50_v5_lr0.2+5.0_batch1_subr1.0_solvTrue_scaleTrue_Ib50_w1/add_bias_final.pt").to(device=device)

        pair_params_bias = torch.zeros(
            (num_res, num_res, 128, 23), requires_grad=True, device=device
        )
        device_processed_features["pair_bias_feat"] = pair_params_bias
        optimizer = torch.optim.Adam(
            [
                {"params": device_processed_features["pair_bias_feat"], "lr": lr_a},
            ]
        )
        bias_names = ["pair_bias_feat"]

    #####################

    else:
        optimizer = torch.optim.Adam(
            [
                {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
            ]
        )

    # Run options
    output_name = "{root}_it{it}_v{v}_lr{a}+{m}_batch{b}_subr{subr}_solv{solv}_scale{scale}_{align}{add}".format(
        root=args.file_root,
        it=args.iterations,
        v=args.version,
        a=args.lr_add,
        m=args.lr_mul,
        b=args.batches,
        subr=args.sub_ratio,
        solv=args.solvent,
        scale=args.scale,
        align=args.align,
        add=args.note,
    )

    # Initialize best variables for alignement
    best_loss = float("inf")
    best_pos = reference_pos
    best_Bs = None

    # List initialization for saving values
    mse_losses_by_epoch = []
    rbr_loss_by_epoch = []
    sigmas_by_epoch = []
    llg_losses = []
    all_pldtts = []
    mean_it_plddts = []
    absolute_msa_changes = []
    corrected_losses = []

    for iteration in tqdm(range(args.iterations)):
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

        # Avoid passing through graph a second time
        feats_copy = copy.deepcopy(device_processed_features)
        feats_copy["pair_bias_feat"] = device_processed_features[
            "pair_bias_feat"
        ].clone()
        # feats_copy["msa_feat_bias"] = device_processed_features["msa_feat_bias"].clone()
        # feats_copy["msa_feat_weights"] = device_processed_features[
        #    "msa_feat_weights"
        # ].clone()

        best_pos_copy = best_pos.clone()

        msa_at_it_start = torch.mean(feats_copy["msa_feat"][:, :, 25:48, 0], dim=(0, 2))

        # AF2 pass
        af2_output = af_bias(feats_copy, num_iters=1, bias=False, pair_bias=True)

        # Position alignment
        xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
            af2_output, device_processed_features, llgloss.sfc.cra_name
        )

        all_pldtts.append(rk_utils.assert_numpy(af2_output["plddt"]))
        mean_it_plddts.append(rk_utils.assert_numpy(torch.mean(plddts)))

        pseudo_Bs = rk_coordinates.update_bfactors(plddts)
        llgloss.sfc.atom_b_iso = pseudo_Bs.detach()

        aligned_xyz = rk_coordinates.align_positions(
            xyz_orth_sfc, best_pos_copy, llgloss.sfc.cra_name, pseudo_Bs
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

        sigmas = llg_utils.sigmaA_from_model(
            Etrue,
            phitrue,
            Ecalc,
            Fc,
            llgloss.sfc.dHKL,
            llgloss.bin_labels,
        )
        llgloss.sigmaAs = sigmas

        # Update SFC and save
        llgloss.sfc.atom_pos_orth = aligned_xyz
        llgloss.sfc.savePDB(
            "{path}/{r}/outputs/{out}/{it}_preRBR.pdb".format(
                path=path, r=args.file_root, out=output_name, it=iteration
            )
        )

        # Rigid body refinement (RBR) step
        optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
            aligned_xyz, llgloss, lbfgs=RBR_LBFGS, added_chain=constant_fp_added
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

        llgloss.sfc.atom_pos_orth = optimized_xyz
        # Save postRBR PDB
        llgloss.sfc.savePDB(
            "{path}/{r}/outputs/{out}/{it}_postRBR.pdb".format(
                path=path, r=args.file_root, out=output_name, it=iteration
            )
        )

        # TODO Rwork/Rfree?
        print("Loss", loss.item())
        print("LLG Estimate", llg_estimate)

        if args.align == "B":
            if loss < best_loss:
                # if iteration == 0:
                print("UPDATING BEST")
                best_pos = optimized_xyz.clone().detach()
                best_loss = loss.clone().item()
                best_Bs = llgloss.sfc.atom_b_iso

        #### add an L2 loss to constrain confident atoms ###
        loss_weight = args.L2_weight
        conf_xyz, conf_best = rk_coordinates.select_confident_atoms(
            optimized_xyz, best_pos_copy, best_Bs, b_thresh=args.b_threshold
        )
        print("CONFIDENT ATOMS", conf_xyz.shape)
        L2_loss = torch.sum((conf_xyz - conf_best) ** 2)  # / conf_best.shape[0]
        print("L2 loss", L2_loss.item())

        ####################################################

        # Save sigmaA values for further processing
        sigmas_dict = {
            f"sigma_{i + 1}": sigma_value for i, sigma_value in enumerate(sigmas)
        }
        sigmas_by_epoch.append(sigmas_dict)

        # Save the absolute difference in mean contribution from each residue channel from previous iteration
        new_mean = torch.mean(feats_copy["msa_feat"][:, :, 25:48, 0], dim=(0, 2))
        mean_change = torch.abs(new_mean - msa_at_it_start)
        absolute_msa_changes.append(rk_utils.assert_numpy(mean_change))
        print("Mean absolute msa change", torch.mean(mean_change).item())

        corrected_loss = loss + loss_weight * L2_loss
        print("corrected loss", corrected_loss.item())
        corrected_losses.append(corrected_loss.item())

        corrected_loss.backward()  # loss.backward()
        optimizer.step()

        if iteration == (args.iterations - 1):
            print("Currently at last iteration {}".format(iteration))
            torch.save(
                device_processed_features["pair_bias_feat"],
                "{path}/{r}/outputs/{out}/pair_bias_final.pt".format(
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

    # Corrected loss per iteration
    np.save(
        "{path}/{r}/outputs/{out}/corrected_loss_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(corrected_losses),
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

    # Absolute MSA change per column per iteration
    np.save(
        "{path}/{r}/outputs/{out}/MSA_changes_it.npy".format(
            path=path,
            r=args.file_root,
            out=output_name,
        ),
        rk_utils.assert_numpy(absolute_msa_changes),
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
