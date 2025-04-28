import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from rocket.cryo import utils as cryo_utils


# Test downsample_data
@pytest.mark.parametrize(
    "downsample_ratio, expected_vals",
    [
        (2, {1, 2, 4}),  # Only rows where all H/K/L divisible by 2
        (1, {1, 2, 3, 4}),  # All rows should pass for ratio 1
    ],
)
def test_downsample_data_masking_and_no_downsampling(downsample_ratio, expected_vals):
    df = pd.DataFrame({
        "H": [0, 2, 3, 4],
        "K": [0, 2, 3, 4],
        "L": [0, 2, 3, 4],
        "VAL": [1, 2, 3, 4],
    })
    df.get_hkls = lambda: df[["H", "K", "L"]].values

    with patch("reciprocalspaceship.read_mtz", return_value=df):
        result = cryo_utils.downsample_data(
            "dummy.mtz", downsample_ratio=downsample_ratio
        )
        assert set(result["VAL"]) == expected_vals
        assert result.shape[0] == len(expected_vals)


# Test compute_sigmaA_for_bin is within clamped values
def test_compute_sigmaA_for_bin_typical():
    # Chosen so cosdphi=1, everything positive, no sqrt_arg issues
    eEsel = np.array([1.0, 2.0])
    Ecalc_sel = np.array([1.0, 2.0])
    dobssel = np.array([1.0, 1.0])
    expectE_phi = np.array([0.0, 0.0])
    Ecalc_phi = np.array([0.0, 0.0])
    result = cryo_utils.compute_sigmaA_for_bin(
        eEsel, Ecalc_sel, dobssel, expectE_phi, Ecalc_phi
    )
    assert 1e-6 <= result <= 0.999


def test_compute_sigmaA_for_bin_zero_division_or_nan():
    # Inputs that lead to denominator zero (all zeros)
    eEsel = np.array([0.0, 0.0])
    Ecalc_sel = np.array([0.0, 0.0])
    dobssel = np.array([0.0, 0.0])
    expectE_phi = np.array([0.0, 0.0])
    Ecalc_phi = np.array([0.0, 0.0])
    with pytest.raises((ZeroDivisionError, ValueError)):
        cryo_utils.compute_sigmaA_for_bin(
            eEsel, Ecalc_sel, dobssel, expectE_phi, Ecalc_phi
        )


# Test compute_sigmaA_error
def test_compute_sigmaA_error_finite():
    dobssel = np.array([0.5, 0.7])
    sigmaA = 0.5
    abseE = np.array([1.0, 2.0])
    absEc = np.array([2.0, 1.0])
    cosdphi = np.array([1.0, 1.0])
    over_sampling_factor = 1.0
    result = cryo_utils.compute_sigmaA_error(
        dobssel, sigmaA, abseE, absEc, cosdphi, over_sampling_factor
    )
    assert isinstance(result, float)
    assert result > 0


# Test fit_line
def test_fit_line_perfect():
    x = np.array([1, 2, 3])
    y = 2 * x + 1
    w = np.ones_like(x)
    slope, intercept = cryo_utils.fit_line(x, y, w)
    assert np.isclose(slope, 2.0)
    assert np.isclose(intercept, 1.0)


# Test combine_sigmaA
def test_combine_sigmaA_shape_and_type():
    slope = 1.0
    intercept = 0.0
    xdat = np.array([0.0, 1.0])
    ydat = np.array([0.1, 0.2])
    wdat = np.array([1.0, 2.0])
    sigma_linlog = 1.0
    linlogsiga, logsiga_combined = cryo_utils.combine_sigmaA(
        slope, intercept, xdat, ydat, wdat, sigma_linlog
    )
    assert len(linlogsiga) == 2
    assert len(logsiga_combined) == 2
    assert all(isinstance(v, float) for v in linlogsiga)
    assert all(isinstance(v, float) for v in logsiga_combined)


# Test llgcryo_calculate (torch-based)
def test_llgcryo_calculate_simple():
    E_amp = torch.tensor([1.0])
    E_phi = torch.tensor([0.0])
    Ec_amp = torch.tensor([1.0])
    Ec_phi = torch.tensor([0.0])
    sigmaA = torch.tensor([0.5])
    dobs = torch.tensor([0.5])
    result = cryo_utils.llgcryo_calculate(E_amp, E_phi, Ec_amp, Ec_phi, sigmaA, dobs)
    assert result.shape == (1,)
    assert torch.isfinite(result).all()


# Test load_tng_data with minimal mock
def test_load_tng_data_keys_and_types():
    df = pd.DataFrame({
        "Emean": [1.0, 2.0],
        "PHIEmean": [0.0, 90.0],
        "Dobs": [1.0, 2.0],
    })
    with (
        patch("rocket.utils.load_mtz", return_value=df),
        patch("rocket.utils.try_gpu", return_value="cpu"),
    ):
        result = cryo_utils.load_tng_data("dummy_file")
        assert set(result.keys()) == {"Emean", "PHIEmean", "Dobs"}
        for k in result:
            assert isinstance(result[k], torch.Tensor)
            assert result[k].shape[0] == 2


# Test sigmaA_from_model_in_map with dummy sfc object
def test_sigmaA_from_model_in_map_dummy():
    class DummySFC:
        def __init__(self):
            self.n_bins = 2
            self.bins = np.array([0, 0, 1, 1])

    expectE_amp = np.array([1.0, 1.0, 2.0, 2.0])
    expectE_phi = np.array([0.0, 0.0, 0.0, 0.0])
    dobs = np.array([1.0, 1.0, 1.0, 1.0])
    Ecalc_amp = np.array([1.0, 1.0, 2.0, 2.0])
    Ecalc_phi = np.array([0.0, 0.0, 0.0, 0.0])
    dhkl = np.array([1.0, 1.0, 2.0, 2.0])
    sfc = DummySFC()
    over_sampling_factor = 1.0
    model_sigmaA = cryo_utils.sigmaA_from_model_in_map(
        expectE_amp,
        expectE_phi,
        dobs,
        Ecalc_amp,
        Ecalc_phi,
        sfc,
        over_sampling_factor,
        dhkl,
        plot=False,
    )
    assert model_sigmaA.shape == expectE_amp.shape
    assert np.all(model_sigmaA > 0)
    assert np.all(model_sigmaA < 1)


# Test combine_sigmaA logic (numerical, edge)
def test_combine_sigmaA_behavior():
    slope, intercept = 0.0, 0.0
    xdat = np.array([0.0])
    ydat = np.array([0.0])
    wdat = np.array([1.0])
    sigma_linlog = 1.0
    linlogsiga, logsiga_combined = cryo_utils.combine_sigmaA(
        slope, intercept, xdat, ydat, wdat, sigma_linlog
    )
    # With all zeros, log(0.999) and logsiga_combined = 0
    assert np.isclose(linlogsiga[0], math.log(0.999))
    assert np.isfinite(logsiga_combined[0])


# Test fit_line for zero slope
def test_fit_line_zero_slope():
    x = np.array([1, 2, 3])
    y = np.array([5, 5, 5])
    w = np.ones_like(x)
    slope, intercept = cryo_utils.fit_line(x, y, w)
    assert np.isclose(slope, 0.0)
    assert np.isclose(intercept, 5.0)
