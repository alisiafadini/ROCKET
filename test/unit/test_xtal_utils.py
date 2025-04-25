import sys
from unittest.mock import MagicMock

import numpy as np
import torch

from rocket.xtal.utils import (
    Ak_approx,
    compute_sigmaA_true,
    find_bin_dHKL,
    interpolate_smooth,
    lb_Ak,
    llgIa_calculate,
    llgIa_firstdev,
    llgIa_seconddev,
    llgIc_calculate,
    llgIc_firstdev,
    llgIc_seconddev,
    llgItot_calculate,
    llgItot_firstdev,
    llgItot_with_derivatives2sigmaA,
    logcosh,
    newton_step,
    ub_Ak,
)

sys.modules["openfold"] = MagicMock()
sys.modules["reciprocalspaceship"] = MagicMock()
sys.modules["SFC_Torch"] = MagicMock()


def test_newton_step_identity():
    x = torch.tensor([1.0, 2.0])
    grad = torch.tensor([0.1, -0.2])
    Hessian = torch.eye(2)
    result = newton_step(x, grad, Hessian)
    expected = x - grad
    assert torch.allclose(result, expected, atol=1e-6)


def test_newton_step_zero_grad():
    x = torch.tensor([1.0, 2.0])
    grad = torch.zeros(2)
    Hessian = torch.eye(2)
    result = newton_step(x, grad, Hessian)
    assert torch.allclose(result, x)


def test_llgIa_firstdev_simple():
    sA = torch.tensor([0.5])
    d = torch.tensor([1.0])
    Eeff = torch.tensor([1.0])
    Ec = torch.tensor([1.0])
    result = llgIa_firstdev(sA, d, Eeff, Ec)
    assert torch.isfinite(result).all()


def test_llgIc_firstdev_simple():
    sA = torch.tensor([0.5])
    d = torch.tensor([1.0])
    Eeff = torch.tensor([1.0])
    Ec = torch.tensor([1.0])
    result = llgIc_firstdev(sA, d, Eeff, Ec)
    assert torch.isfinite(result).all()


def test_llgItot_firstdev_mix():
    sigmaA = torch.tensor([0.5, 0.6])
    d = torch.tensor([1.0, 1.0])
    Eeff = torch.tensor([1.0, 1.0])
    Ec = torch.tensor([1.0, 1.0])
    centric = torch.tensor([True, False])
    total = llgItot_firstdev(sigmaA, d, Eeff, Ec, centric)
    part_c = llgIc_firstdev(sigmaA, d[centric], Eeff[centric], Ec[centric])
    part_a = llgIa_firstdev(sigmaA, d[~centric], Eeff[~centric], Ec[~centric])
    expected = part_c.sum() + part_a.sum()
    assert torch.allclose(total, expected)


def test_llgIa_seconddev_finite():
    sA = torch.tensor([0.5])
    d = torch.tensor([1.0])
    Eeff = torch.tensor([1.0])
    Ec = torch.tensor([1.0])
    out = llgIa_seconddev(sA, d, Eeff, Ec)
    assert torch.isfinite(out).all()


def test_llgIc_seconddev_finite():
    sA = torch.tensor([0.5])
    d = torch.tensor([1.0])
    Eeff = torch.tensor([1.0])
    Ec = torch.tensor([1.0])
    out = llgIc_seconddev(sA, d, Eeff, Ec)
    assert torch.isfinite(out).all()


def test_interpolate_smooth_linear_zero():
    sigmaAs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = interpolate_smooth(sigmaAs)
    assert out >= 0


def test_llgItot_with_derivatives2sigmaA_matches_autodiff():
    sigmaA = torch.tensor([0.5], requires_grad=True)
    d = torch.tensor([1.0])
    Eeff = torch.tensor([1.0])
    Ec = torch.tensor([1.0])
    centric = torch.tensor([False])
    l1, lp1, lpp1 = llgItot_with_derivatives2sigmaA(
        sigmaA, d, Eeff, Ec, centric, method="autodiff"
    )
    l2, lp2, lpp2 = llgItot_with_derivatives2sigmaA(
        sigmaA, d, Eeff, Ec, centric, method="analytical"
    )
    print((lp1 - lp2).abs())
    assert torch.allclose(l1, l2, atol=1e-5)
    assert torch.allclose(lp1, lp2, atol=1e-2)  # TODO this is too high!!!
    assert torch.allclose(lpp1, lpp2, atol=5e-1)  # TODO this is too high!!!


def test_Ak_approx_bounds():
    nu = torch.tensor([1.0])
    z = torch.tensor([2.0])
    lb = lb_Ak(nu, z)
    ub = ub_Ak(nu, z)
    ak = Ak_approx(nu, z)
    assert torch.all(lb <= ak)
    assert torch.all(ak <= ub)


def test_llgIa_calculate_simple():
    sA = torch.tensor([0.5])
    d = torch.tensor([1.0])
    Eeff = torch.tensor([1.0])
    Ec = torch.tensor([1.0])
    out = llgIa_calculate(sA, d, Eeff, Ec)
    assert torch.isfinite(out).all()


def test_llgIc_calculate_simple():
    sA = torch.tensor([0.5])
    d = torch.tensor([1.0])
    Eeff = torch.tensor([1.0])
    Ec = torch.tensor([1.0])
    out = llgIc_calculate(sA, d, Eeff, Ec)
    assert torch.isfinite(out).all()


def test_llgItot_calculate_simple():
    sA = torch.tensor([0.5, 0.7])
    d = torch.tensor([1.0, 1.0])
    Eeff = torch.tensor([1.0, 1.0])
    Ec = torch.tensor([1.0, 1.0])
    centric = torch.tensor([False, True])
    out = llgItot_calculate(sA, d, Eeff, Ec, centric)
    assert torch.isfinite(out).all()


def test_logcosh_matches_numpy():
    xs = torch.tensor([-2.0, 0.0, 2.0])
    out = logcosh(xs)
    np_out = np.log(np.cosh(xs.numpy()))
    assert np.allclose(out.numpy(), np_out, atol=1e-6)


def test_find_bin_dHKL_simple():
    dHKLs = torch.tensor([1.0, 2.0, 3.0, 4.0])
    bin_labels = torch.tensor([0, 0, 1, 1])
    out = find_bin_dHKL(dHKLs, bin_labels)
    assert out[0] == out[1]
    assert out[2] == out[3]
    # Check value is mean of min and max for each bin
    assert np.isclose(out[0], 1.5)
    assert np.isclose(out[2], 3.5)


def test_compute_sigmaA_true_all_phases_aligned():
    Eobs = np.array([1.0, 2.0])
    phiobs = np.array([0.0, 0.0])
    Ecalc = np.array([1.0, 2.0])
    phicalc = np.array([0.0, 0.0])
    bin_labels = np.array([0, 0])
    out = compute_sigmaA_true(Eobs, phiobs, Ecalc, phicalc, bin_labels)
    assert np.isclose(out[0], 2.5)
