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


# General settings
preset = "model_1"
device = "cuda:1"


path = "/net/cci/alisia/openfold_tests/run_openfold/test_cases/7dt2_short/"
tng_file = "{p}7dt2-tng_withrfree.mtz".format(p=path)
input_pdb = "{p}7dt2_short_noalts.pdb".format(p=path)

sfc = llg_sf.initial_SFC(
    input_pdb, tng_file, "FP", "SIGFP", Freelabel="FreeR_flag", device=device
)


llgloss = rocket.llg.targets.LLGloss(sfc, tng_file, device)

af_bias = rocket.MSABiasAFv3(model_config(preset, train=True), preset).to(device)
af_bias.freeze()  # Free all AF2 parameters to save time

with open("{p}/7dt2_short_processed_feats.pickle".format(p=path), "rb") as file:
    # Load the data from the pickle file
    processed_features = pickle.load(file)

    device_processed_features = rk_utils.move_tensors_to_device(
        processed_features, device=device
    )
    # TODO: this still takes up memory in original device?
    del processed_features

num_res = device_processed_features["aatype"].shape[0]
msa_params_bias = torch.zeros((512, num_res, 23), requires_grad=True, device=device)
device_processed_features["msa_feat_bias"] = msa_params_bias

lr_a = 1e-3
lr_m = 1e-2

msa_params_weights = torch.ones((512, num_res, 23), requires_grad=True, device=device)
device_processed_features["msa_feat_weights"] = msa_params_weights
device_processed_features["msa_feat_weights"].requires_grad_(True)

optimizer = torch.optim.Adam(
    [
        {"params": device_processed_features["msa_feat_bias"], "lr": lr_a},
        {"params": device_processed_features["msa_feat_weights"], "lr": lr_m},
    ]
)

for iteration in tqdm(np.arange(10)):
    optimizer.zero_grad()

    feats_copy = copy.deepcopy(device_processed_features)
    feats_copy["msa_feat_bias"] = device_processed_features["msa_feat_bias"].clone()
    feats_copy["msa_feat_weights"] = device_processed_features[
        "msa_feat_weights"
    ].clone()

    # AF2 pass
    af2_output = af_bias(feats_copy, num_iters=1, biasMSA=True)

    # Position alignment
    xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
        af2_output, device_processed_features, llgloss.sfc.cra_name
    )

    Ecalc, Fc = llgloss.compute_Ecalc(xyz_orth_sfc, return_Fc=True, update_scales=False)

    loss = torch.mean(Ecalc)
    print(loss)
    loss.backward()
    optimizer.step()
