"""
Functions relating model structure factor manipulation and normalization
"""

import torch
import numpy as np
from SFC_Torch import SFcalculator


def initial_SFC(pdb_file, mtz_file, Flabel, SigFlabel, Freelabel=None):
    sfcalculator = SFcalculator(
        pdb_file,
        mtz_file,
        expcolumns=[Flabel, SigFlabel],
        freeflag=Freelabel,
        set_experiment=True,
        testset_value=0,
    )

    sfcalculator.inspect_data(verbose=False)
    sfcalculator.calc_fprotein()
    sfcalculator.calc_fsolvent()
    sfcalculator.init_scales(requires_grad=True)
    sfcalculator.calc_ftotal()

    return sfcalculator


def ftotal_amplitudes(Ftotal, dHKL, sort_by_res=True):
    F_mag = torch.abs(Ftotal)
    sorted_indices = np.argsort(dHKL)[::-1]
    if sort_by_res:
        F_mag = F_mag[sorted_indices]
    return F_mag


def ftotal_phis(sfcalculator, F_attr):
    PI_on_180 = 0.017453292519943295
    F_out = getattr(sfcalculator, F_attr)
    F_out_phase = torch.angle(F_out) / PI_on_180
    return F_out_phase


def compute_sigmaA_true(Eobs, phiobs, Ecalc, phicalc, bin_labels):
    # Combine the absolute values and phase difference into sigmaA_true
    sigmaA_true = Eobs * Ecalc * np.cos(phiobs - phicalc)
    data = np.stack((sigmaA_true, bin_labels), axis=1)

    Sigma_trues = []
    for label in np.unique(bin_labels):
        F_in_bin = data[data[:, 1] == label][:, 0]
        bin_mean = np.mean(F_in_bin)
        Sigma_trues.append(bin_mean)

    return Sigma_trues


def calculate_Sigma_atoms(Fs, eps, bin_labels):
    F_over_eps = Fs**2 / eps

    data = torch.stack((F_over_eps, bin_labels), dim=1)

    Sigma = []
    for label in torch.unique(bin_labels):
        F_in_bin = data[data[:, 1] == label][:, 0]
        bin_mean = torch.mean(F_in_bin)
        Sigma.append(bin_mean)

    return torch.stack(Sigma)


def normalize_Fs(Fs, eps, Sigma_atoms, bin_labels):
    # e.g. Ecalc = Fc / (eps*SigmaP)**(-0.5)
    data = torch.stack((Fs, eps, bin_labels), dim=1)

    for label in torch.unique(bin_labels):
        indices = data[:, 2] == label
        data[indices, 0] = Fs[indices] / torch.sqrt(
            eps[indices] * Sigma_atoms[label.item()].item()
        )
    assert round(torch.mean(data[:, 0] ** 2).item()) == 1

    return data[:, 0]
