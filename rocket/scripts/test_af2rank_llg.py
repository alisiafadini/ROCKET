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
        "-root",
        "--file_root",
        required=True,
        help=("PDB code or filename root for the dataset"),
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
        "-lr_1",
        "--lr_1",
        type=float,
        default=1e-2,
        help=("Learning rate for additive bias. Default 1e-3"),
    )

    parser.add_argument(
        "-lr_2",
        "--lr_2",
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
        "-chain",
        "--added_chain",
        help="additional chain in asu",
        action=argparse.BooleanOptionalAction,
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
    path = "/net/cci/alisia/openfold_tests/run_openfold/test_cases"
    tng_file = "{p}/{r}/{r}-tng_withrfree.mtz".format(p=path, r=args.file_root)
    input_pdb = "{p}/{r}/{r}-pred-aligned.pdb".format(p=path, r=args.file_root)
    true_pdb = "{p}/{r}/{r}_noalts.pdb".format(p=path, r=args.file_root)

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

    if args.added_chain:
        constant_fp_added = torch.load(
            "{p}/{r}/{r}_added_chain_atoms.pt".format(p=path, r=args.file_root)
        ).to(device=device)

    else:
        constant_fp_added = None

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
    del sfc_true

    # LLG initialization
    llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

    # Model initialization
    af_bias = rocket.TemplateBiasAF(model_config(preset, train=True), preset).to(device)
    af_bias.freeze()  # Free all AF2 parameters to save time
    processed_dict = rocket.make_processed_dict_from_template(
        "{p}/{r}/{r}-pred-aligned.pdb".format(p=path, r=args.file_root), device=device
    )

    # Initiate template bias
    processed_dict["template_torsion_angles_sin_cos_bias"] = torch.zeros_like(
        processed_dict["template_torsion_angles_sin_cos"],
        requires_grad=True,
        device=device,
    )

    processed_dict["template_all_atom_positions_bias"] = torch.zeros_like(
        processed_dict["template_all_atom_positions"], requires_grad=True, device=device
    )

    # Optimizer settings and initialization
    lr_1 = args.lr_1
    lr_2 = args.lr_2
    version = "temp"

    # Run options
    output_name = "{root}_it{it}_v{v}_lr{a}+{m}_batch{b}_subr{subr}_solv{solv}_scale{scale}_{align}{add}".format(
        root=args.file_root,
        it=args.iterations,
        v=version,
        a=args.lr_1,
        m=args.lr_2,
        b=args.batches,
        subr=args.sub_ratio,
        solv=args.solvent,
        scale=args.scale,
        align=args.align,
        add=args.note,
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": processed_dict["template_torsion_angles_sin_cos_bias"],
                "lr": lr_1,
            },
            {"params": processed_dict["template_all_atom_positions_bias"], "lr": lr_2},
        ]
    )

    # Initialize best variables for alignement
    best_loss = float("inf")
    best_pos = reference_pos

    # List initialization for saving values
    mse_losses_by_epoch = []
    rbr_loss_by_epoch = []
    sigmas_by_epoch = []
    llg_losses = []
    all_pldtts = []
    mean_it_plddts = []
    absolute_msa_changes = []

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
        working_batch = copy.deepcopy(processed_dict)
        working_batch["template_torsion_angles_sin_cos_bias"] = processed_dict[
            "template_torsion_angles_sin_cos_bias"
        ].clone()
        working_batch["template_all_atom_positions_bias"] = processed_dict[
            "template_all_atom_positions_bias"
        ].clone()

        # AF2 pass
        af2_output = af_bias(working_batch, num_iters=1, bias=True)

        # Position alignment
        xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
            af2_output, working_batch, llgloss.sfc.cra_name
        )

        all_pldtts.append(rk_utils.assert_numpy(af2_output["plddt"]))
        mean_it_plddts.append(rk_utils.assert_numpy(torch.mean(plddts)))

        pseudo_Bs = rk_coordinates.update_bfactors(plddts)
        llgloss.sfc.atom_b_iso = pseudo_Bs.detach()

        aligned_xyz = rk_coordinates.align_positions(xyz_orth_sfc, best_pos, pseudo_Bs)

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
        optimized_xyz.retain_grad()
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
                best_loss = loss
                best_pos = optimized_xyz.clone()

        # Save sigmaA values for further processing
        sigmas_dict = {
            f"sigma_{i + 1}": sigma_value for i, sigma_value in enumerate(sigmas)
        }
        sigmas_by_epoch.append(sigmas_dict)

        loss.backward()
        optimizer.step()

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
