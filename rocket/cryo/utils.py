"""
Functions relating to sigmaa calculation and refinement for cryoEM
"""

import torch
from rocket.llg import structurefactors
from rocket import utils
from tqdm import tqdm
import numpy as np


def load_tng_data(tng_file, device=utils.try_gpu()):
    tng = utils.load_mtz(tng_file).dropna()

    # Generate PhaserTNG tensors
    emean = torch.tensor(tng["Emean"].values, device=device)
    phi_emean = torch.tensor(tng["PHIE_mean"].values, device=device)
    dobs = torch.tensor(tng["Dobs"].values, device=device)

    data_dict = {
        "Emean": emean,
        "PHIEmean": phi_emean,
        "Dobs": dobs,
    }

    return data_dict
