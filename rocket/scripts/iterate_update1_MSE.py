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


from torch.utils.tensorboard import SummaryWriter

# Create a TensorBoard writer
tensorboard_writer = SummaryWriter()

# General settings
preset = "model_1"
device = "cuda:1"

# Load external files
tng_file = "../../run_openfold/3hak/3hak/3hak-tng_withrfree.mtz"
tng_dict = llg_utils.load_tng_data(tng_file, device=device)

input_pdb = "../../run_openfold/3hak/3hak/phaserpred-aligned.pdb"
true_pdb = "../../run_openfold/3hak/3hak/3hak_noalts.pdb"

phitrue = np.load("../../run_openfold/3hak/3hak/3hak-phitrue.npy")

with open("../../run_openfold/3hak/3hak/3hak_processed_feats.pickle", "rb") as file:
    # Load the data from the pickle file
    processed_features = pickle.load(file)

device_processed_features = rocket.utils.move_tensors_to_device(
    processed_features, device=device
)
del processed_features

with open("../../run_openfold/3hak/3hak/true_Bs.pickle", "rb") as file:
    # Load the data from the pickle file
    true_Bs = pickle.load(file)

# Initiate bias feature

# TODO: replace the hardcoded the dimensions here?
msa_params = torch.zeros((512, 103, 23, 21), requires_grad=True, device=device)
device_processed_features["msa_feat_bias"] = msa_params

# SFC initialization, only have to do it once
sfc = llg_sf.initial_SFC(
    input_pdb, tng_file, "FP", "SIGFP", Freelabel="FreeR_flag", device=device
)
reference_pos = sfc.atom_pos_orth
sfc.atom_b_iso = true_Bs.to(device)

# Load true positions
sfc_true = llg_sf.initial_SFC(
    true_pdb, tng_file, "FP", "SIGFP", Freelabel="FreeR_flag", device=device
)
true_pos = sfc_true.atom_pos_orth
del sfc_true

# LLG initialization
llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

# Model initialization
af_bias = rocket.MSABiasAF(model_config(preset, train=True), preset).to(device)

lr_s = 1e-3  # OG: 0.0001
optimizer = torch.optim.Adam(
    [{"params": device_processed_features["msa_feat_bias"], "lr": lr_s}]
)

num_epochs = 1500
num_batch = 3
sub_ratio = 0.7

mse_losses_by_epoch = []
sigmas_by_epoch = []
biases_by_epoch = []
llg_by_epoch = []
loss_by_epoch = []

loss_function = torch.nn.MSELoss(reduction='sum')


for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    # Avoid passing through graph a second time
    feats_copy = copy.deepcopy(device_processed_features)
    feats_copy["msa_feat_bias"] = device_processed_features["msa_feat_bias"].clone()

    # AF2 pass
    af2_output = af_bias(feats_copy, num_iters=1, biasMSA=True)

    # position alignment
    xyz_orth_sfc = rk_coordinates.extract_allatoms(
        af2_output, device_processed_features
    )
    aligned_xyz = rk_coordinates.align_positions(xyz_orth_sfc, true_pos)

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
    Ecalc, Fc = llgloss.compute_Ecalc(aligned_xyz, return_Fc=True)
    sigmas = llg_utils.sigmaA_from_model(
        rocket.utils.assert_numpy(llgloss.Eobs),
        phitrue,
        Ecalc,
        Fc,
        llgloss.sfc.dHKL,
        llgloss.bin_labels,
    )
    llgloss.sigmaAs = sigmas

    loss = loss_function(aligned_xyz, true_pos)
    loss_by_epoch.append(loss.item())

    # LLG loss
    llg = -llgloss(
        aligned_xyz, bin_labels=None, num_batch=num_batch, sub_ratio=sub_ratio
    )
    llg_estimate = llg.item() / (sub_ratio * num_batch)
    sigmas_dict = {
        f"sigma_{i + 1}": sigma_value for i, sigma_value in enumerate(sigmas)
    }

    llgloss.sfc.atom_pos_orth = aligned_xyz

    bias = torch.mean(device_processed_features["msa_feat_bias"].abs())
    tensorboard_writer.add_scalar("Loss", loss.item(), epoch)
    llg_by_epoch.append(llg_estimate)
    biases_by_epoch.append(bias.item())
    sigmas_by_epoch.append(sigmas_dict)

    tensorboard_writer = SummaryWriter(
        log_dir="tensorboard_runs/MSE_msabias_runs/{epoch}it-{b}batch-{r}subr-trueBs-nodrop-aligntrue".format(
            epoch=num_epochs, lr=lr_s, b=num_batch, r=sub_ratio
        )
    )
    llgloss.sfc.savePDB(
    "tensorboard_runs/MSE_msabias_runs/{epoch}it-{b}batch-{r}subr-trueBs-nodrop-aligntrue/{epoch_it}.pdb".format(
        epoch=num_epochs, lr=lr_s, b=num_batch, r=sub_ratio, epoch_it=epoch
            )
        )


    loss.backward()
    optimizer.step()

# save image heatmap
mse_losses_array = np.array(mse_losses_by_epoch)

np.save(
    "tensorboard_runs/MSE_msabias_runs/{epoch}it-{b}batch-{r}subr-trueBs-nodrop-aligntrue/mse_losses_matrix.npy".format(
        epoch=num_epochs, lr=lr_s, b=num_batch, r=sub_ratio 
    ),
    mse_losses_array,
)

with open('tensorboard_runs/MSE_msabias_runs/{epoch}it-{b}batch-{r}subr-trueBs-nodrop-aligntrue/llg_by_epoch.pkl'.format(
        epoch=num_epochs, lr=lr_s, b=num_batch, r=sub_ratio), 'wb') as file:
    pickle.dump(llg_by_epoch, file)

with open('tensorboard_runs/MSE_msabias_runs/{epoch}it-{b}batch-{r}subr-trueBs-nodrop-aligntrue/loss_by_epoch.pkl'.format(
        epoch=num_epochs, lr=lr_s, b=num_batch, r=sub_ratio), 'wb') as file:
    pickle.dump(loss_by_epoch, file)

with open('tensorboard_runs/MSE_msabias_runs/{epoch}it-{b}batch-{r}subr-trueBs-nodrop-aligntrue/biases_by_epoch.pkl'.format(
        epoch=num_epochs, lr=lr_s, b=num_batch, r=sub_ratio), 'wb') as file:
    pickle.dump(biases_by_epoch, file)


with open('tensorboard_runs/MSE_msabias_runs/{epoch}it-{b}batch-{r}subr-trueBs-nodrop-aligntrue/sigmas_by_epoch.pkl'.format(
        epoch=num_epochs, lr=lr_s, b=num_batch, r=sub_ratio), 'wb') as file:
    pickle.dump(sigmas_by_epoch, file)


# Close the TensorBoard writer
tensorboard_writer.close()
