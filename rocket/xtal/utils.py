"""
Functions relating to sigmaa calculation and refinement
"""

import numpy as np
import reciprocalspaceship as rs
import torch

from rocket import utils
from rocket.xtal import structurefactors


def newton_step(tensor_pre_update, gradient, Hessian):
    # updated_tensor = torch.clamp(
    #    tensor_pre_update - derivatives_tensor,
    #    0.015,
    #    0.99,
    # )
    gamma = 1.0
    derivatives_tensor = torch.matmul(torch.linalg.inv(Hessian), gradient)
    updated_tensor = tensor_pre_update - gamma * derivatives_tensor

    # ds_tensor = torch.matmul(torch.linalg.inv(Htotal), dL_total)
    # sigmaAs_new = sigmaAs_tensor - ds_tensor
    #             sigmaAs_new = llg_utils.newton_step(
    #            sigmaAs_tensor, dL_total, torch.linalg.inv(Htotal)
    #        )
    return updated_tensor


def llgIa_firstdev(sigmaA, dobs, Eeff, Ec):
    """
    partial llgIa / partial sigmaA
    """

    bessel_argument = (2 * dobs * Ec * Eeff * sigmaA) / (1 - dobs**2 * sigmaA**2)
    bessel_ratio = Ak_approx(
        torch.tensor(1).to(bessel_argument), bessel_argument
    ).squeeze()

    dLa = (
        2
        * dobs
        * (
            Ec * Eeff * (dobs**2 * sigmaA**2 + 1) * bessel_ratio
            - dobs * sigmaA * (dobs**2 * sigmaA**2 + Ec**2 + Eeff**2 - 1)
        )
        / (dobs**2 * sigmaA**2 - 1) ** 2
    )

    return dLa


def llgIc_firstdev(sigmaA, dobs, Eeff, Ec):
    """
    partial llgIc / partial sigmaA
    """

    tanh_argument = (dobs * Ec * Eeff * sigmaA) / (1 - dobs**2 * sigmaA**2)

    dLc = (
        dobs
        * (
            Ec * Eeff * (dobs**2 * sigmaA**2 + 1) * torch.tanh(tanh_argument)
            - dobs * sigmaA * (dobs**2 * sigmaA**2 + Ec**2 + Eeff**2 - 1)
        )
        / (dobs**2 * sigmaA**2 - 1) ** 2
    )

    return dLc


def llgItot_firstdev(sigmaA, dobs, Eeff, Ec, centric_tensor):
    """
    partial llgI / partial sigmaA
    """
    llgIp_centric = llgIc_firstdev(
        sigmaA, dobs[centric_tensor], Eeff[centric_tensor], Ec[centric_tensor]
    )
    llgIp_acentric = llgIa_firstdev(
        sigmaA, dobs[~centric_tensor], Eeff[~centric_tensor], Ec[~centric_tensor]
    )

    return llgIp_centric.sum() + llgIp_acentric.sum()


def llgIa_seconddev(sigmaA, dobs, Eeff, Ec):
    """
    partial^2 llgIa / partial sigmaA^2
    """

    bessel_argument = (2 * dobs * Ec * Eeff * sigmaA) / (1 - dobs**2 * sigmaA**2)
    bessel_ratio = Ak_approx(
        torch.tensor(1).to(bessel_argument), bessel_argument
    ).squeeze()

    d2La = (
        2
        * dobs
        / (dobs**2 * sigmaA**2 - 1) ** 4
        * (
            dobs
            * (
                Ec**2
                * (
                    3 * dobs**4 * sigmaA**4
                    + 2 * (dobs**2 * Eeff * sigmaA**2 + Eeff) ** 2
                    - 2 * dobs**2 * sigmaA**2
                    - 1
                )
                + (dobs**2 * sigmaA**2 - 1)
                * (dobs**4 * sigmaA**4 + Eeff**2 * (3 * dobs**2 * sigmaA**2 + 1) - 1)
            )
            + Ec
            * Eeff
            * bessel_ratio
            * (
                bessel_ratio * (-2 * dobs * Ec * Eeff * (dobs**2 * sigmaA**2 + 1) ** 2)
                - (
                    (
                        dobs**6 * sigmaA**6
                        + 3 * dobs**4 * sigmaA**4
                        - 5 * dobs**2 * sigmaA**2
                        + 1
                    )
                    / sigmaA
                )
            )
        )
    )

    return d2La


def llgIc_seconddev(sigmaA, dobs, Eeff, Ec):
    """
    partial^2 llgIc / partial sigmaA^2
    """

    tanh_and_sech_argument = (dobs * Ec * Eeff * sigmaA) / (1 - dobs**2 * sigmaA**2)

    term_1 = (
        dobs**4 * sigmaA**4
        + 3 * dobs**2 * sigmaA**2 * (Ec**2 + Eeff**2)
        + Ec**2
        + Eeff**2
        - 1
    )
    term_2 = (
        Ec
        * Eeff
        * (dobs**2 * sigmaA**2 + 1) ** 2
        * torch.cosh(tanh_and_sech_argument) ** -2
    )
    term_3 = (dobs**4 * sigmaA**4 + 2 * dobs**2 * sigmaA**2 - 3) * torch.tanh(
        tanh_and_sech_argument
    )

    d2Lc = (dobs**2 * sigmaA**2 - 1) ** -4 * (
        dobs**2
        * (
            (dobs**2 * sigmaA**2 - 1) * term_1
            + Ec * Eeff * (term_2 - 2 * dobs * sigmaA * term_3)
        )
    )

    return d2Lc


def llgItot_seconddev(sigmaA, dobs, Eeff, Ec, centric_tensor):
    """
    partial^2 llgI / partial sigmaA^2
    """

    llgIpp_centric = llgIc_seconddev(
        sigmaA, dobs[centric_tensor], Eeff[centric_tensor], Ec[centric_tensor]
    )
    llgIpp_acentric = llgIa_seconddev(
        sigmaA, dobs[~centric_tensor], Eeff[~centric_tensor], Ec[~centric_tensor]
    )

    return llgIpp_centric.sum() + llgIpp_acentric.sum()


def interpolate_smooth(sigmaAs_tensor, edge_weights=0.25, total_weight=200.0):
    # Calculate loss for internal values
    internal_loss = (
        (sigmaAs_tensor[1:-1] - (sigmaAs_tensor[:-2] + sigmaAs_tensor[2:]) / 2) ** 2
    ).sum()

    # Extrapolate at the edges
    edge_loss_start = (
        sigmaAs_tensor[0] - 2 * sigmaAs_tensor[1] + sigmaAs_tensor[2]
    ) ** 2
    edge_loss_end = (
        sigmaAs_tensor[-1] - 2 * sigmaAs_tensor[-2] + sigmaAs_tensor[-3]
    ) ** 2

    # Total loss
    total_loss = (
        internal_loss + edge_weights * edge_loss_start + edge_weights * edge_loss_end
    ) / float(sigmaAs_tensor.shape[0])

    return total_weight * total_loss


def llgItot_with_derivatives2sigmaA(
    sigmaA, dobs, Eeff, Ec, centric_tensor, method="autodiff"
):
    """
    sigmaA : torch.Tensor

    method : str, "autodiff" or "analytical"
        calculate derivatives with autodiff or analytical expression
    """
    if method == "autodiff":
        sA = sigmaA.detach().clone().requires_grad_(True)
        l = llgItot_calculate(sA, dobs, Eeff, Ec, centric_tensor)  # noqa: E741
        lp = torch.autograd.grad(l, sA, create_graph=True)[0]
        lpp = torch.autograd.grad(lp, sA, create_graph=True, allow_unused=True)[0]
    elif method == "analytical":
        l = llgItot_calculate(sigmaA, dobs, Eeff, Ec, centric_tensor)  # noqa: E741
        lp = llgItot_firstdev(
            sigmaA, dobs.detach(), Eeff.detach(), Ec.detach(), centric_tensor.detach()
        )
        lpp = llgItot_seconddev(
            sigmaA, dobs.detach(), Eeff.detach(), Ec.detach(), centric_tensor.detach()
        )

    return l.detach().clone(), lp.detach().clone(), lpp.detach().clone()


def Ak_approx(nu, z):
    """
    Approximation of ratio of modified Bessel functions of 1st kind.
    https://arxiv.org/pdf/1902.02603.pdf

    Parameters
    ----------
    nu: tensor, shape (N0,)
            Order of modified Bessel functions of 1st kind.
    z: tensor, shape (N1,)
            Argument of Bessel function. Positive values only.

    Return
    ------
    tensor, shape (N1, N0)
    """
    return 0.5 * (lb_Ak(nu, z) + ub_Ak(nu, z))


def ub_Ak(nu, z):
    """
    Upper-bound for the ratio of modified Bessel functions of 1st kind.
    https://arxiv.org/pdf/1606.02008.pdf (Theorems 5 and 6).

    Return
    ------
    ub: tensor, shape (z.shape[0], nu.shape[0])
            Upper-bound for Ak(nu, z).
    """
    assert torch.all(nu >= 0)
    nu = nu.reshape(1, -1)
    z = z.reshape(-1, 1)

    ub = torch.zeros(z.shape[0], nu.shape[1]).to(z)
    ub[:, nu.reshape(-1) >= 0.5] = torch.min(
        B(0, nu[nu >= 0.5], z), B_tilde(2, nu[nu >= 0.5], z)
    )
    ub[:, nu.reshape(-1) < 0.5] = B_tilde(2, nu[nu < 0.5], z)
    return ub


def lb_Ak(nu, z):
    """
    Lower-bound for the ratio of modified Bessel functions of 1st kind.
    https://arxiv.org/pdf/1606.02008.pdf (Theorems 5 and 6).
    """
    assert torch.all(nu >= 0)
    nu = nu.reshape(1, -1)
    z = z.reshape(-1, 1)
    return B_tilde(0, nu, z)


def B_tilde(alpha, nu, z):
    """
    https://arxiv.org/pdf/1606.02008.pdf
    """
    nu = nu.reshape(1, -1)
    z = z.reshape(-1, 1)
    sigma = nu + float(alpha + 1) / 2.0
    delta_p = nu + 0.5 + sigma / (2 * torch.sqrt(sigma**2 + z**2))
    delta_m = nu - 0.5 - sigma / (2 * torch.sqrt(sigma**2 + z**2))
    return z / (delta_m + torch.sqrt(delta_p**2 + z**2))


def B(alpha, nu, z):
    """
    https://arxiv.org/pdf/1606.02008.pdf
    """
    nu = nu.reshape(1, -1)
    z = z.reshape(-1, 1)
    lamda = nu + float(alpha - 1) / 2.0
    delta = nu - 0.5 + lamda / (2 * torch.sqrt(lamda**2 + z**2))
    return z / (delta + torch.sqrt(delta**2 + z**2))


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
    expo_arg = (sigmaA**2 * dobs**2 * (Eeff**2 + Ec**2)) / (2 * dobs**2 * sigmaA**2 - 2)

    llg = torch.sum(
        torch.log((1 - dobs**2 * sigmaA**2) ** (-0.5)) + expo_arg + logcosh(cosh_arg)
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
        + (sigmaA * (-sigmaA * E**2 - sigmaA * Ec**2 + 2 * E * Ec)) / (1 - sigmaA**2)
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


def find_bin_dHKL(dHKLs, bin_labels):
    unique_bins = bin_labels.unique()

    bin_dHKLs = []

    for bin_label in unique_bins:
        # Find indices corresponding to the current bin
        bin_indices = torch.where(bin_labels == bin_label)[0]
        # Extract distances for the current bin
        bin_distances = dHKLs[bin_indices]
        # Find the largest and lowest distances
        max_distance = torch.max(bin_distances)
        min_distance = torch.min(bin_distances)
        # Calculate the halfway distance
        halfway_distance = (max_distance + min_distance) / 2.0
        # Add the halfway distance for each row in the current bin
        bin_dHKLs.extend([halfway_distance.item()] * len(bin_indices))

    return torch.tensor(bin_dHKLs)


def load_tng_data(tng_file, device=None):
    if device is None:
        device = utils.try_gpu()
    if isinstance(tng_file, str):
        tng = utils.load_mtz(tng_file).dropna()
    elif isinstance(tng_file, rs.DataSet):
        tng = tng_file.dropna()
    # Generate PhaserTNG tensors
    eps = torch.tensor(tng["EPS"].values, device=device)
    centric = torch.tensor(tng["CENT"].values, device=device).bool()
    dobs = torch.tensor(tng["DOBS"].values, device=device)
    feff = torch.tensor(tng["FEFF"].values, device=device)
    bin_labels = torch.tensor(tng["BIN"].values, device=device)
    dHKLs = torch.tensor(tng["dHKL"].values, device=device)
    bin_dHKLs = find_bin_dHKL(dHKLs, bin_labels).to(device=device)

    sigmaN = structurefactors.calculate_Sigma_atoms(feff, eps, bin_labels)
    Edata = structurefactors.normalize_Fs(feff, eps, sigmaN, bin_labels)

    data_dict = {
        "EDATA": Edata,
        "EPS": eps,
        "CENTRIC": centric,
        "DOBS": dobs,
        "FEFF": feff,
        "BIN_LABELS": bin_labels,
        "BIN_dHKLS": bin_dHKLs,
    }

    return data_dict
