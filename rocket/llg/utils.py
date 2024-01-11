"""
Functions relating to sigmaa calculation and refinement
"""
import torch
from rocket.llg import structurefactors
from rocket import utils
from tqdm import tqdm
import numpy as np


def load_tng_data(tng_file, device=utils.try_gpu()):
    tng = utils.load_mtz(tng_file).dropna()

    # Generate PhaserTNG tensors
    eps = torch.tensor(tng["EPS"].values, device=device)
    centric = torch.tensor(tng["CENT"].values, device=device).bool()
    dobs = torch.tensor(tng["DOBS"].values, device=device)
    feff = torch.tensor(tng["FEFF"].values, device=device)
    bin_labels = torch.tensor(tng["BIN"].values, device=device)

    sigmaN = structurefactors.calculate_Sigma_atoms(feff, eps, bin_labels)
    Edata = structurefactors.normalize_Fs(feff, eps, sigmaN, bin_labels)

    data_dict = {
        "EDATA": Edata,
        "EPS": eps,
        "CENTRIC": centric,
        "DOBS": dobs,
        "FEFF": feff,
        "BIN_LABELS": bin_labels,
    }

    return data_dict


def llgIa_calculate(sigmaA, dobs, Eeff, Ec):
    # acentric reflections
    bessel_arg = (2 * dobs * sigmaA * Eeff * Ec) / (1 - dobs**2 * sigmaA**2)
    exp_bessel = torch.special.i0e(bessel_arg)

    llg = torch.sum(
        torch.log((1 - dobs**2 * sigmaA**2) ** (-1) * exp_bessel)
        + (
            sigmaA
            * dobs
            * (-dobs * sigmaA * Eeff**2 - dobs * sigmaA * Ec**2 + 2 * Eeff * Ec)
        )
        / (1 - dobs**2 * sigmaA**2)
    )
    return llg


def llgIc_calculate(sigmaA, dobs, Eeff, Ec):
    # centric reflections
    cosh_arg = (sigmaA * dobs * Ec * Eeff) / (1 - dobs**2 * sigmaA**2)
    expo_arg = (sigmaA**2 * dobs**2 * (Eeff**2 + Ec**2)) / (
        2 * dobs**2 * sigmaA**2 - 2
    )

    llg = torch.sum(
        torch.log((1 - dobs**2 * sigmaA**2) ** (-0.5))
        + expo_arg
        + logcosh(cosh_arg)
    )

    return llg


def llgItot_calculate(sigmaA, dobs, Eeff, Ec, centric_tensor):
    # (1) Make a centric and acentric tensor
    acentric_tensor = ~centric_tensor

    # (2) Call respective llg targets with indexed data
    llg_centric = llgIc_calculate(
        sigmaA, dobs[centric_tensor], Eeff[centric_tensor], Ec[centric_tensor]
    )
    llg_acentric = llgIa_calculate(
        sigmaA, dobs[acentric_tensor], Eeff[acentric_tensor], Ec[acentric_tensor]
    )

    return llg_acentric + llg_centric


def llgA_calculate(sigmaA, E, Ec):
    # acentric reflections
    bessel_arg = (2 * sigmaA * E * Ec) / (1 - sigmaA**2)
    exp_bessel = torch.special.i0e(bessel_arg)
    llg = torch.sum(
        torch.log((1 - sigmaA**2) ** (-1) * exp_bessel)
        + (sigmaA * (-sigmaA * E**2 - sigmaA * Ec**2 + 2 * E * Ec))
        / (1 - sigmaA**2)
    )

    return llg


def logcosh(x):
    # s always has real part >= 0

    s = torch.sign(x) * x
    p = torch.exp(-2 * s)
    return s + torch.log1p(p) - torch.log(torch.tensor(2.0))


def llgC_calculate(sigmaA, E, Ec):
    # centric reflections
    cosh_arg = (Ec * sigmaA * E) / (1 - sigmaA**2)
    expo_arg = (sigmaA**2 * Ec**2 + sigmaA**2 * E**2 - 2 * Ec * E * sigmaA) / (
        2 * sigmaA**2 - 2
    )
    cosh_exp = -Ec * E * sigmaA / (1 - sigmaA**2)

    llg = torch.sum(
        torch.log((1 - sigmaA**2) ** (-0.5)) + expo_arg + logcosh(cosh_arg) + cosh_exp
    )

    return llg


def llgTot_calculate(sigmaA, E, Ec, centric_tensor):
    # (1) Make a centric and acentric tensor
    acentric_tensor = ~centric_tensor

    # (2) Call respective llg targets with indexed data
    llg_centric = llgC_calculate(sigmaA, E[centric_tensor], Ec[centric_tensor])
    llg_acentric = llgA_calculate(sigmaA, E[acentric_tensor], Ec[acentric_tensor])

    return llg_acentric + llg_centric


# def refine_sigmaa(
#     unique_labels,
#     bin_labels,
#     Eobs,
#     Ecalc,
#     centric,
#     device=utils.try_gpu(),
#     num_epochs=25,
# ):
#     # Training loop
#     sigma_As = [[] for _ in range(len(unique_labels))]
#     corr_coefs = [torch.tensor(0.0, dtype=torch.float32) for _ in unique_labels]

#     for i, label in tqdm(enumerate(unique_labels)):
#         bin_indices = bin_labels == label
#         bin_Eobs = Eobs[bin_indices]
#         bin_Ecalc = Ecalc[bin_indices]

#         # initialize sigmaA values with correlation coefficient
#         corr_coefs[i] = torch.corrcoef(torch.stack((bin_Eobs, bin_Ecalc), dim=0))[1][0]
#         corr_coefs[i] = torch.clamp(corr_coefs[i], min=0.001, max=0.999)
#         print("correlation coeff", corr_coefs[i])
#         sigma_As[i] = np.sqrt(corr_coefs[i].item())

#         sigma_As[i] = torch.tensor(
#             sigma_As[i],
#             dtype=torch.float32,
#             requires_grad=True,
#             device=device,
#         )

#         # optimizer = torch.optim.RMSprop([sigma_As[i]], lr=6e-4)
#         optimizer = torch.optim.Adam([sigma_As[i]], lr=1e-3)

#         for epoch in range(num_epochs):
#             optimizer.zero_grad()  # Clear gradients
#             sigma_A = sigma_As[i]

#             # Compute LLG expression for the bin
#             llg = llgTot_calculate(
#                 sigma_A, bin_Eobs, bin_Ecalc, centric[bin_indices]
#             )

#             # Minimize the negative LLG (maximize LLG)
#             loss = -llg
#             loss.backward(retain_graph=True)

#             # Update the current sigma_A
#             # optimizers[i].step()
#             optimizer.step()

#             # Enforce SIGMAA bounds
#             sigma_A.data = torch.clamp(sigma_A.data, 0.015, 0.99)

#     return sigma_As


def sigmaA_from_model(E_true, phi_true, E_model, Fcs, dHKL, bin_labels):
    # TODO naming Etrue vs Eobs

    phitrue_rad = np.deg2rad(phi_true)
    phimodel = utils.assert_numpy(
        structurefactors.ftotal_phis(Fcs, dHKL, sort_by_res=True)
    )
    phimodel_rad = np.deg2rad(phimodel)

    sigmaAs = structurefactors.compute_sigmaA_true(
        utils.assert_numpy(E_true),
        phitrue_rad,
        utils.assert_numpy(E_model),
        phimodel_rad,
        utils.assert_numpy(bin_labels),
    )

    return sigmaAs


def compute_sigmaA_true(Eobs, phiobs, Ecalc, phicalc, bin_labels):
    # TODO naming Etrue vs Eobs

    # Combine the absolute values and phase difference into sigmaA_true
    sigmaA_true = Eobs * Ecalc * np.cos(phiobs - phicalc)
    data = np.stack((sigmaA_true, bin_labels), axis=1)

    Sigma_trues = []
    for label in np.unique(bin_labels):
        F_in_bin = data[data[:, 1] == label][:, 0]
        bin_mean = np.mean(F_in_bin)
        Sigma_trues.append(bin_mean)

    return Sigma_trues
