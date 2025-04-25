"""
Functions relating to sigmaa calculation and refinement for cryoEM
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import reciprocalspaceship as rs
import torch

from rocket import utils


def downsample_data(mtz_path, downsample_ratio: int):
    """
    Downsample the data in an mtz file by a given ratio
    """
    df = rs.read_mtz(mtz_path)
    # Get the h, k, l values
    hkls = df.get_hkls()
    # Create a boolean mask where all h, k, l values are divisible by the ratio
    mask = (
        (hkls[:, 0] % downsample_ratio == 0)
        & (hkls[:, 1] % downsample_ratio == 0)
        & (hkls[:, 2] % downsample_ratio == 0)
    )
    # Apply the mask to the DataFrame
    downsampled_df = df[mask].copy()

    return downsampled_df


def load_tng_data(tng_file, device=utils.try_gpu()):
    tng = utils.load_mtz(tng_file).dropna()

    # Generate PhaserTNG tensors
    emean = torch.tensor(tng["Emean"].values, device=device)
    phi_emean = torch.tensor(tng["PHIEmean"].values, device=device)
    dobs = torch.tensor(tng["Dobs"].values, device=device)

    data_dict = {
        "Emean": emean,
        "PHIEmean": phi_emean,
        "Dobs": dobs,
    }

    return data_dict


# sigmaA functions


def compute_sigmaA_for_bin(eEsel, Ecalc_sel, dobssel, expectE_phi, Ecalc_phi):
    abseE = np.abs(eEsel)
    absEc = np.abs(Ecalc_sel)
    p1 = np.radians(expectE_phi)
    p2 = np.radians(Ecalc_phi)
    cosdphi = np.cos(p2 - p1)

    sum0 = np.sum(2.0 * dobssel * abseE * absEc * cosdphi)
    sum1 = np.sum(2 * np.square(dobssel) * (1.0 - np.square(abseE) - np.square(absEc)))
    sum2 = np.sum(2 * np.power(dobssel, 3) * abseE * absEc * cosdphi)
    sum3 = np.sum(-2 * np.power(dobssel, 4))

    u1 = -2 * sum2**3 + 9 * sum1 * sum2 * sum3 - 27 * sum0 * sum3**2
    sqrt_arg = u1**2 - 4 * (sum2**2 - 3 * sum1 * sum3) ** 3

    if sqrt_arg < 0:
        raise ValueError("Argument of square root in sigmaA calculation is negative")

    third = 1.0 / 3
    x1 = (u1 + math.sqrt(sqrt_arg)) ** third
    sigmaA = (
        2 * 2.0**third * sum2**2
        - 6 * 2.0**third * sum1 * sum3
        - 2 * sum2 * x1
        + 2.0 ** (2 * third) * x1**2
    ) / (6 * sum3 * x1)
    return max(min(sigmaA, 0.999), 1.0e-6)


def compute_sigmaA_error(dobssel, sigmaA, abseE, absEc, cosdphi, over_sampling_factor):
    denom = np.power((1 - np.square(dobssel) * sigmaA**2), 3)
    sum0 = np.sum(
        (2 * np.square(dobssel) * (1.0 - np.square(abseE) - np.square(absEc))) / denom
    )
    sum1 = np.sum((12 * np.power(dobssel, 3) * abseE * absEc * cosdphi) / denom)
    sum2 = np.sum(
        (-6 * np.power(dobssel, 4) * (np.square(abseE) + np.square(absEc))) / denom
    )
    sum3 = np.sum((4 * np.power(dobssel, 5) * abseE * absEc * cosdphi) / denom)
    sum4 = np.sum((-2 * np.power(dobssel, 6)) / denom)
    d2LLGbydsigmaA = (
        sum0 + sum1 * sigmaA + sum2 * sigmaA**2 + sum3 * sigmaA**3 + sum4 * sigmaA**4
    )
    d2LLGbydsigmaA /= over_sampling_factor
    return 1.0 / math.sqrt(abs(d2LLGbydsigmaA))


def fit_line(xdat, ydat, wdat):
    W = np.sum(wdat)
    Wx = np.sum(wdat * xdat)
    Wy = np.sum(wdat * ydat)
    Wxx = np.sum(wdat * xdat * xdat)
    Wxy = np.sum(wdat * xdat * ydat)
    slope = (W * Wxy - Wx * Wy) / (W * Wxx - Wx * Wx)
    intercept = (Wy * Wxx - Wx * Wxy) / (W * Wxx - Wx * Wx)
    return slope, intercept


def combine_sigmaA(slope, intercept, xdat, ydat, wdat, sigma_linlog):
    linlogsiga = []
    logsiga_combined = []

    for i, x in enumerate(xdat):
        linlog = min(math.log(0.999), (intercept + slope * x))
        linlogsiga.append(linlog)
        sigma_sigmaA = 1.0 / math.sqrt(wdat[i])
        sigma_lnsigmaA = math.exp(-ydat[i]) * sigma_sigmaA
        combined_logsiga = (ydat[i] / sigma_lnsigmaA**2 + linlog / sigma_linlog**2) / (
            1.0 / sigma_lnsigmaA**2 + 1.0 / sigma_linlog**2
        )
        logsiga_combined.append(combined_logsiga)

    return linlogsiga, logsiga_combined


def sigmaA_from_model_in_map(
    expectE_amp,
    expectE_phi,
    dobs,
    Ecalc_amp,
    Ecalc_phi,
    sfc,
    over_sampling_factor,
    dhkl,
    plot=False,
):
    n_bins = sfc.n_bins
    model_sigmaA = np.zeros_like(expectE_amp, dtype=np.float32)
    xdat = []
    ydat = []
    wdat = []

    for i_bin in range(n_bins):
        bin_mask = sfc.bins == i_bin
        if not np.any(bin_mask):
            continue

        eEsel = expectE_amp[bin_mask]
        Ecalc_sel = Ecalc_amp[bin_mask]
        dobssel = dobs[bin_mask]
        cosdphi = np.cos(
            np.radians(Ecalc_phi[bin_mask]) - np.radians(expectE_phi[bin_mask])
        )

        sigmaA = compute_sigmaA_for_bin(
            eEsel, Ecalc_sel, dobssel, expectE_phi[bin_mask], Ecalc_phi[bin_mask]
        )
        sigma_sigmaA = compute_sigmaA_error(
            dobssel,
            sigmaA,
            np.abs(eEsel),
            np.abs(Ecalc_sel),
            cosdphi,
            over_sampling_factor,
        )

        ssqr = np.mean(np.square(1 / dhkl[bin_mask]))
        xdat.append(ssqr)
        ydat.append(math.log(sigmaA))
        wdat.append(abs(1.0 / sigma_sigmaA**2))

    xdat = np.array(xdat)
    ydat = np.array(ydat)
    wdat = np.array(wdat)

    slope, intercept = fit_line(xdat, ydat, wdat)

    linlogsiga, logsiga_combined = combine_sigmaA(
        slope, intercept, xdat, ydat, wdat, math.log(1.5)
    )

    i_bin_used = 0
    for i_bin in range(n_bins):
        bin_mask = sfc.bins == i_bin
        if not np.any(bin_mask):
            continue

        combined_siga = math.exp(logsiga_combined[i_bin_used])
        model_sigmaA[bin_mask] = combined_siga
        i_bin_used += 1

    if plot:
        plot_sigmaA(xdat, ydat, linlogsiga, logsiga_combined)

    return model_sigmaA


def plot_sigmaA(xdat, ydat, linlogsiga, logsiga_combined):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(xdat, ydat, label="ln(sigmaA)")
    ax[0].plot(xdat, linlogsiga, label="line fit")
    ax[0].plot(xdat, logsiga_combined, label="combined")
    ax[0].legend(loc="upper right")
    ax[1].plot(xdat, np.exp(ydat), label="sigmaA")
    ax[1].plot(xdat, np.exp(linlogsiga), label="line fit")
    ax[1].plot(xdat, np.exp(logsiga_combined), label="combined")
    ax[1].legend(loc="lower left")
    plt.show()


def llgcryo_calculate(E_amp, E_phi, Ec_amp, Ec_phi, sigmaA, dobs):
    cos_phi_diff = torch.cos(Ec_phi - E_phi)
    term1 = (
        2 / (1 - dobs**2 * sigmaA**2) * dobs * sigmaA * E_amp * Ec_amp * cos_phi_diff
    )

    term2 = (dobs**2 * sigmaA**2 * (E_amp**2 + Ec_amp**2)) / (1 - dobs**2 * sigmaA**2)
    term3 = torch.log(1 - dobs**2 * sigmaA**2)

    llg_cryo = term1 - term2 - term3

    return llg_cryo
