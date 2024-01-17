import copy
import torch
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import rocket
from rocket.llg import utils as llg_utils
from rocket import coordinates as rk_coordinates
from rocket.llg import structurefactors as llg_sf
from openfold.config import model_config
from torch.optim import lr_scheduler
import time


from torch.utils.tensorboard import SummaryWriter

# Create a TensorBoard writer
tensorboard_writer = SummaryWriter()

# General settings
preset = "model_1"
device = "cuda:0"

# Using LBFGS in RBR or not
RBR_LBFGS = True


# Update SFC scales or not
SCALE_FLAG = True

# Load external files
tng_file = "../../run_openfold/3hak/3hak/3hak-tng_withrfree_hres4A.mtz"

input_pdb = "../../run_openfold/3hak/3hak/phaserpred-aligned.pdb"
true_pdb = "../../run_openfold/3hak/3hak/3hak_noalts.pdb"

phitrue = np.load("../../run_openfold/3hak/3hak/3hak-phitrue-nosolvent-hres4A.npy")
Etrue = np.load("../../run_openfold/3hak/3hak/3hak-Etrue-nosolvent-hres4A.npy")

with open("../../run_openfold/3hak/3hak/3hak_processed_feats.pickle", "rb") as file:
    # Load the data from the pickle file
    processed_features = pickle.load(file)

device_processed_features = rocket.utils.move_tensors_to_device(
    processed_features, device=device
)

del processed_features


# Initiate bias feature

# TODO: replace the hardcoded the dimensions here?
msa_params_bias = torch.zeros((512, 103, 23), requires_grad=True, device=device)
device_processed_features["msa_feat_bias"] = msa_params_bias

# Add linear recombination weights
msa_params_weights = torch.ones((512, 103, 23), requires_grad=True, device=device)

device_processed_features["msa_feat_weights"] = msa_params_weights

device_processed_features["msa_feat_weights"].requires_grad_(True)

# SFC initialization, only have to do it once
sfc = llg_sf.initial_SFC(
    input_pdb, tng_file, "FP", "SIGFP", Freelabel="FreeR_flag", device=device
)
reference_pos = sfc.atom_pos_orth.clone()

# Load true positions
sfc_true = llg_sf.initial_SFC(
    true_pdb, tng_file, "FP", "SIGFP", Freelabel="FreeR_flag", device=device
)
true_pos = sfc_true.atom_pos_orth.clone()
true_Bs = sfc_true.atom_b_iso.clone()
del sfc_true

# sfc.atom_b_iso = true_Bs.to(device)

# LLG initialization
llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

# Model initialization
af_bias = rocket.MSABiasAFv3(model_config(preset, train=True), preset).to(device)
af_bias.freeze()  # Free all AF2 parameters to save time

lr_s = 1e-2  # OG: 0.0001
lr_w = 1e-1
optimizer = torch.optim.Adam(
    [
        {"params": device_processed_features["msa_feat_bias"], "lr": lr_s},
        {"params": device_processed_features["msa_feat_weights"], "lr": lr_w},
    ]
)

num_epochs = 500
num_batch = 5
sub_ratio = 0.6
reg_lambda = 0.0
name = "pseudoBs-v3-nomod-confBs-nosolv-alignbest-sigmanosolv-4Acut"

# Initialize best variables for alignement
best_loss = float("inf")
best_epoch = -1
best_pos = reference_pos
epochs_since_last_improvement = 0

mse_losses_by_epoch = []
sigmas_by_epoch = []
rbr_loss_by_epoch = []
llg_losses = []
biases_by_epoch = []


for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad(set_to_none=True)

    if epoch == 0:
        # tensorboard_writer.add_scalar("Loss", loss.item(), epoch)
        # tensorboard_writer.add_scalar("LLG", llg_estimate, epoch)

        tensorboard_writer = SummaryWriter(
            log_dir="tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}".format(
                epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
            )
        )

    # Avoid passing through graph a second time
    # feats_copy = copy.deepcopy(device_processed_features)
    # feats_copy["msa_feat_bias"] = device_processed_features["msa_feat_bias"].clone()
    # feats_copy["msa_feat_weights"] = device_processed_features[
    #     "msa_feat_weights"
    # ].clone()

    # AF2 pass
    af2_output = af_bias(device_processed_features, num_iters=1, biasMSA=True)

    # position alignment
    xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
        af2_output, device_processed_features, llgloss.sfc.cra_name
    )
    pseudo_Bs = rk_coordinates.update_bfactors(plddts)
    llgloss.sfc.atom_b_iso = pseudo_Bs.detach()

    aligned_xyz = rk_coordinates.align_positions(xyz_orth_sfc, best_pos, pseudo_Bs)
    print("aligned_xyz", aligned_xyz[0:20, :])
    # aligned_xyz = rk_coordinates.align_positions(xyz_orth_sfc, best_pos)

    # Residue MSE loss

    # (1) Select CAs
    cra_calphas_list, calphas_mask = rk_coordinates.select_CA_from_craname(sfc.cra_name)

    # (2) Convert residue names to residue numbers
    residue_numbers = [int(name.split("-")[1]) for name in cra_calphas_list]

    # (3) Calculate total MSE loss
    total_mse_loss = rk_coordinates.calculate_mse_loss_per_residue(
        aligned_xyz[calphas_mask], true_pos[calphas_mask], residue_numbers
    )
    mse_losses_by_epoch.append(total_mse_loss)

    # sigmaA calculation
    Ecalc, Fc = llgloss.compute_Ecalc(aligned_xyz, return_Fc=True, update_scales=False)

    sigmas = llg_utils.sigmaA_from_model(
        Etrue,
        phitrue,
        Ecalc,
        Fc,
        llgloss.sfc.dHKL,
        llgloss.bin_labels,
    )
    llgloss.sigmaAs = sigmas

    llgloss.sfc.atom_pos_orth = aligned_xyz
    llgloss.sfc.savePDB(
        "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/{epoch_it}_pre-rbr.pdb".format(
            epoch=num_epochs,
            lr=lr_s,
            lrw=lr_w,
            b=num_batch,
            r=sub_ratio,
            epoch_it=epoch,
            name=name,
        )
    )

    # Call rigid body refinement
    # optimized_xyz = aligned_xyz
    # loss_track_pose = []
    optimized_xyz, loss_track_pose = rk_coordinates.rigidbody_refine_quat(
        aligned_xyz, llgloss, lbfgs=RBR_LBFGS
    )

    rbr_loss_by_epoch.append(loss_track_pose)

    # LLG loss
    loss = -llgloss(
        optimized_xyz,
        bin_labels=None,
        num_batch=num_batch,
        sub_ratio=sub_ratio,
        solvent=False,
        update_scales=SCALE_FLAG
    )

    L1loss = device_processed_features["msa_feat_weights"].abs().sum()
    total_loss = loss + reg_lambda * L1loss

    llg_estimate = loss.item() / (sub_ratio * num_batch)

    print("Loss", loss.item())
    print("LLG Estimate", llg_estimate)
    print(f"R_work: {llgloss.sfc.r_work.item():<7.3f}", f"R_free: {llgloss.sfc.r_free.item():<7.3f}")

    if loss < best_loss:
        best_loss = loss
        best_epoch = epoch
        best_pos = optimized_xyz.clone()
        # print(f"Best epoch: {best_epoch}, Best loss: {llg_estimate}")

        # Reset the counter when a new best epoch is found
        epochs_since_last_improvement = 0

    else:
        # Increase the counter if no improvement
        epochs_since_last_improvement += 1

    sigmas_dict = {
        f"sigma_{i + 1}": sigma_value for i, sigma_value in enumerate(sigmas)
    }

    llgloss.sfc.atom_pos_orth = optimized_xyz
    sigmas_by_epoch.append(sigmas_dict)

    llgloss.sfc.savePDB(
        "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/{epoch_it}_post-rbr.pdb".format(
            epoch=num_epochs,
            lr=lr_s,
            lrw=lr_w,
            b=num_batch,
            r=sub_ratio,
            epoch_it=epoch,
            name=name,
        )
    )
    
    llg_losses.append(llg_estimate)

    bias = torch.mean(device_processed_features["msa_feat_bias"].abs())
    biases_by_epoch.append(bias.item())

    total_loss.backward()

    # plddt_weights = 2.22 * torch.exp(-(af2_output["plddt"] * 0.01))
    # plddt_weights_mod1 = torch.where(
    #    plddt_weights <= 1, 1 * plddt_weights, plddt_weights
    # )
    # plddt_weights_mod = torch.where(
    #    plddt_weights_mod1 > 1, 5e3 * plddt_weights_mod1, plddt_weights_mod1
    # )
    # expanded_weights = plddt_weights_mod.view(1, -1, 1)

    add_bias_pre_grads = torch.mean(
        torch.mean(device_processed_features["msa_feat_bias"].grad, dim=0), dim=1
    )

    mult_bias_pre_grads = torch.mean(
        torch.mean(device_processed_features["msa_feat_weights"].grad, dim=0), dim=1
    )

    # device_processed_features["msa_feat_bias"].grad = torch.mul(
    #    device_processed_features["msa_feat_bias"].grad, expanded_weights
    # )

    # device_processed_features["msa_feat_weights"].grad = torch.mul(
    #    device_processed_features["msa_feat_weights"].grad, expanded_weights
    # )

    # print("pseudoBs min", min(plddt_weights_mod))
    # print("pseudoBs max", max(plddt_weights_mod))
    print("pLDDT min", min(af2_output["plddt"]))
    print("pLDDT max", max(af2_output["plddt"]))

    add_bias_grads = torch.mean(
        torch.mean(device_processed_features["msa_feat_bias"].grad, dim=0), dim=1
    )

    mult_bias_grads = torch.mean(
        torch.mean(device_processed_features["msa_feat_weights"].grad, dim=0), dim=1
    )

    ###########
    optimizer.step(retain_graph=True)
    # scheduler.step()

# save grad update info

plddt_weights_mod = np.array(0)

np.save(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/add_bias_grads.npy".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    rocket.utils.assert_numpy(add_bias_grads),
)

np.save(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/mult_bias_grads.npy".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    rocket.utils.assert_numpy(mult_bias_grads),
)

np.save(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/add_bias_pre_grads.npy".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    rocket.utils.assert_numpy(add_bias_pre_grads),
)

np.save(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/mult_bias_pre_grads.npy".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    rocket.utils.assert_numpy(mult_bias_pre_grads),
)

np.save(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/plddt_weights.npy".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    rocket.utils.assert_numpy(plddt_weights_mod),
)

np.save(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/plddts.npy".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    rocket.utils.assert_numpy(af2_output["plddt"]),
)

# save image heatmap
mse_losses_array = np.array(mse_losses_by_epoch)

np.save(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/mse_losses_matrix.npy".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    mse_losses_array,
)

with open(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/sigmas_by_epoch.pkl".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    "wb",
) as file:
    pickle.dump(sigmas_by_epoch, file)

with open(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/rbrloss_by_epoch.pkl".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    "wb",
) as file:
    pickle.dump(rbr_loss_by_epoch, file)

with open(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/LLGloss_by_epoch.pkl".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    "wb",
) as file:
    pickle.dump(llg_losses, file)

with open(
    "tensorboard_runs/LLG_msabias_runs/{epoch}it-lr{lr}{lrw}-{b}batch-{r}subr-{name}/biases_by_epoch.pkl".format(
        epoch=num_epochs, lr=lr_s, lrw=lr_w, b=num_batch, r=sub_ratio, name=name
    ),
    "wb",
) as file:
    pickle.dump(biases_by_epoch, file)


# Close the TensorBoard writer
tensorboard_writer.close()
