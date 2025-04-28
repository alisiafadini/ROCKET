import numpy as np
import pytest
import torch

from rocket.xtal.structurefactors import (
    calculate_Sigma_atoms,
    compute_sigmaA_true,
    ftotal_amplitudes,
    ftotal_phis,
    initial_SFC,
    normalize_Fs,
)

# ---------- MOCKS for initial_SFC ----------


class DummySFCalc:
    def __init__(self, *a, **k):
        self.Fprotein_HKL = torch.tensor([1.0])
        self.Fprotein_asu = torch.tensor([1.0])
        self.Fmask_HKL = torch.tensor([1.0])
        self.solventpct = 0.5

    def inspect_data(self, **k):
        pass

    def calc_fprotein(self):
        pass

    def calc_fsolvent(self):
        self.Fmask_HKL = torch.zeros(1)

    def get_scales_adam(self):
        pass


@pytest.fixture(autouse=True)
def patch_SFcalculator(monkeypatch):
    import SFC_Torch

    import rocket.xtal.structurefactors

    monkeypatch.setattr(rocket.xtal.structurefactors, "SFcalculator", DummySFCalc)
    monkeypatch.setattr(SFC_Torch, "SFcalculator", DummySFCalc)
    # Patch SFC.utils.try_gpu
    monkeypatch.setattr("SFC_Torch.utils.try_gpu", lambda: "cpu")


# ---------- TESTS ----------


def test_initial_SFC_device_default():
    result = initial_SFC("pdb", "mtz", "F", "SigF", device=None)
    assert isinstance(result, DummySFCalc)


def test_initial_SFC_added_chain():
    result = initial_SFC(
        "pdb",
        "mtz",
        "F",
        "SigF",
        added_chain_HKL=torch.tensor([2.0]),
        added_chain_asu=torch.tensor([3.0]),
    )
    assert torch.allclose(result.Fprotein_HKL, torch.tensor([3.0]))  # 1 + 2
    assert torch.allclose(result.Fprotein_asu, torch.tensor([4.0]))  # 1 + 3
    # solventpct should update: 1 - (1-0.5)*2 = 0.0
    assert pytest.approx(result.solventpct) == 0.0


def test_initial_SFC_solvent_false():
    result = initial_SFC("pdb", "mtz", "F", "SigF", solvent=False)
    assert torch.all(result.Fmask_HKL == 0)


def test_initial_SFC_custom_device(monkeypatch):
    # Should not call try_gpu if device given
    device = "cuda"
    called = []
    monkeypatch.setattr("SFC_Torch.utils.try_gpu", lambda: called.append(True))
    result = initial_SFC("pdb", "mtz", "F", "SigF", device=device)
    assert result is not None
    assert not called  # try_gpu should not be called


# ----------- ftotal_amplitudes -------------


def test_ftotal_amplitudes_sorted():
    Ftotal = torch.tensor([3 + 4j, 1 + 1j, 0 + 2j])
    dHKL = np.array([2.0, 3.0, 1.0])
    result = ftotal_amplitudes(Ftotal, dHKL, sort_by_res=True)
    expected = torch.abs(Ftotal)[torch.tensor([1, 0, 2])]
    torch.testing.assert_close(result, expected)


def test_ftotal_amplitudes_nosort():
    Ftotal = torch.tensor([3 + 4j, 1 + 1j, 0 + 2j])
    dHKL = np.array([2.0, 3.0, 1.0])
    result = ftotal_amplitudes(Ftotal, dHKL, sort_by_res=False)
    expected = torch.abs(Ftotal)
    torch.testing.assert_close(result, expected)


def test_ftotal_amplitudes_empty():
    Ftotal = torch.tensor([], dtype=torch.cfloat)
    dHKL = np.array([])
    result = ftotal_amplitudes(Ftotal, dHKL)
    assert result.shape == (0,)


# ----------- ftotal_phis -------------


def test_ftotal_phis_sorted():
    Fc = torch.tensor([1 + 0j, 0 + 1j, -1 + 0j])
    dHKL = np.array([1.0, 3.0, 2.0])
    result = ftotal_phis(Fc, dHKL, sort_by_res=True)
    PI_on_180 = 0.017453292519943295
    phases = torch.angle(Fc) / PI_on_180
    sorted_indices = np.argsort(dHKL)[::-1]
    sorted_indices = sorted_indices.copy()  # ensures positive strides
    expected = phases[torch.tensor(sorted_indices)]

    torch.testing.assert_close(result, expected)


def test_ftotal_phis_nosort():
    Fc = torch.tensor([1 + 0j, 0 + 1j])
    dHKL = np.array([1.0, 2.0])
    result = ftotal_phis(Fc, dHKL, sort_by_res=False)
    PI_on_180 = 0.017453292519943295
    expected = torch.angle(Fc) / PI_on_180
    torch.testing.assert_close(result, expected)


def test_ftotal_phis_empty():
    Fc = torch.tensor([], dtype=torch.cfloat)
    dHKL = np.array([])
    result = ftotal_phis(Fc, dHKL)
    assert result.shape == (0,)


# ----------- compute_sigmaA_true -------------


def test_compute_sigmaA_true_single_bin():
    Eobs = np.array([1, 2])
    phiobs = np.array([0, np.pi / 2])
    Ecalc = np.array([1, 2])
    phicalc = np.array([0, np.pi / 2])
    bin_labels = np.array([0, 0])
    result = compute_sigmaA_true(Eobs, phiobs, Ecalc, phicalc, bin_labels)
    # cos(0)=1, so: 1*1*1, 2*2*1 => mean(1,4)=2.5
    assert np.allclose(result, [2.5])


def test_compute_sigmaA_true_multiple_bins():
    Eobs = np.array([1, 2, 2])
    phiobs = np.array([0, np.pi / 2, np.pi])
    Ecalc = np.array([2, 1, 3])
    phicalc = np.array([0, np.pi / 2, 0])
    bin_labels = np.array([0, 1, 0])
    # For bin 0: (1*2*cos(0), 2*3*cos(pi)) = (2, -6) => mean(-2)
    # For bin 1: (2*1*cos(0)) = 2
    result = compute_sigmaA_true(Eobs, phiobs, Ecalc, phicalc, bin_labels)
    assert np.allclose(result, [-2, 2])


def test_compute_sigmaA_true_empty():
    result = compute_sigmaA_true(
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    )
    assert result == []


# ----------- calculate_Sigma_atoms -------------


def test_calculate_Sigma_atoms_simple():
    Fs = torch.tensor([2.0, 4.0])
    eps = torch.tensor([1.0, 2.0])
    bin_labels = torch.tensor([0, 0])
    result = calculate_Sigma_atoms(Fs, eps, bin_labels)
    expected = torch.mean(torch.tensor([4.0 / 1.0, 16.0 / 2.0]))
    torch.testing.assert_close(result, torch.tensor([expected]))


def test_calculate_Sigma_atoms_multiple_bins():
    Fs = torch.tensor([2.0, 4.0, 3.0])
    eps = torch.tensor([1.0, 2.0, 3.0])
    bin_labels = torch.tensor([0, 1, 0])
    # bin 0: [4/1, 9/3] = [4, 3.0], mean=3.5
    # bin 1: [16/2]=8
    result = calculate_Sigma_atoms(Fs, eps, bin_labels)
    torch.testing.assert_close(result, torch.tensor([3.5, 8.0]))


def test_calculate_Sigma_atoms_empty():
    Fs = torch.tensor([], dtype=torch.float32)
    eps = torch.tensor([], dtype=torch.float32)
    bin_labels = torch.tensor([], dtype=torch.int64)
    with pytest.raises(ValueError, match="All input tensors must be non-empty"):
        calculate_Sigma_atoms(Fs, eps, bin_labels)


# ----------- normalize_Fs -------------


def test_normalize_Fs_all_one_bin():
    Fs = torch.tensor([2.0, 4.0])
    eps = torch.tensor([1.0, 2.0])
    Sigma_atoms = torch.tensor([6.0])
    bin_labels = torch.tensor([0, 0])
    E = normalize_Fs(Fs, eps, Sigma_atoms, bin_labels)
    assert round(torch.mean(E**2).item()) == 1


def test_normalize_Fs_raises_on_bad_scale():
    Fs = torch.tensor([1.0, 1.0])
    eps = torch.tensor([1.0, 1.0])
    Sigma_atoms = torch.tensor([0.1])  # Will cause mean(E**2) != 1
    bin_labels = torch.tensor([0, 0])
    with pytest.raises(AssertionError):
        normalize_Fs(Fs, eps, Sigma_atoms, bin_labels)
