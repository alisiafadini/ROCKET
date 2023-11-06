"""
LLG targets for xtal data
"""

import torch


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
