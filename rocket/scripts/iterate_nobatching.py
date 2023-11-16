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


# General settings
preset = "model_1"
device = "cuda:0"

# Load external files
tng_file = "../../run_openfold/3hak/3hak/3hak-tng.mtz"
tng_dict = llg_utils.load_tng_data(tng_file, device=device)

input_pdb = "../../run_openfold/3hak/3hak/phaserpred-aligned.pdb"

phitrue = np.load("../../run_openfold/3hak/3hak/3hak-phitrue.npy")

with open("../../run_openfold/3hak/3hak/3hak_processed_feats.pickle", "rb") as file:
    # Load the data from the pickle file
    processed_features = pickle.load(file)

# Initiate bias feature

# TODO: replace the hardcoded the dimensions here?
msa_params = torch.zeros((512, 103, 23, 21), requires_grad=True, device=device)
processed_features["msa_feat_bias"] = msa_params

# SFC initialization, only have to do it once
sfc = llg_sf.initial_SFC(input_pdb, tng_file, "FP", "SIGFP")
reference_pos = sfc.atom_pos_orth

# LLG initialization
llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

# Model initialization
af_bias = rocket.MSABiasAF(model_config(preset, train=True), preset).to(device)

lr_s = 1e-4  # OG: 0.0001
optimizer = torch.optim.Adam(
    [{"params": processed_features["msa_feat_bias"], "lr": lr_s}]
)

num_epochs = 10

for epoch in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    # Avoid passing through graph a second time
    feats_copy = copy.deepcopy(processed_features)
    feats_copy["msa_feat_bias"] = processed_features["msa_feat_bias"].clone()

    # AF2 pass
    af2_output = af_bias(feats_copy, num_iters=1, biasMSA=True)

    # position alignment
    xyz_orth_sfc = rk_coordinates.extract_allatoms(af2_output, processed_features)
    aligned_xyz = rk_coordinates.align_positions(xyz_orth_sfc, reference_pos)

    # sigmaA calculation
    Ecalc = llgloss.compute_Ecalc(aligned_xyz)
    sigmas = llg_utils.sigmaA_from_model(
        rocket.utils.assert_numpy(llgloss.Eobs),
        phitrue,
        llgloss.Ecalc,
        llgloss.Fc,
        llgloss.sfc.dHKL,
        llgloss.bin_labels,
    )
    llgloss.sigmaAs = sigmas

    # LLG loss
    loss = -llgloss(aligned_xyz, bin_labels=None, num_batch=3, sub_ratio=0.3)
    print(loss)

    loss.backward()
    optimizer.step()
