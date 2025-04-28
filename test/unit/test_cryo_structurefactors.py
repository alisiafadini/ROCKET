import pytest
import torch


class DummyCryoSFCalc:
    def __init__(self, *a, **k):
        self.Fprotein_HKL = torch.tensor([5.0])
        self.Fo = torch.tensor([2.0])
        self.bins = torch.tensor([0])
        self.SigF = "should be overwritten"
        self.called = []

    def calc_fprotein(self):
        self.called.append("calc_fprotein")

    def calc_Ec(self, x):
        self.called.append(("calc_Ec", x))
        # Return dummy normalized value
        return torch.tensor([1.0])

    def get_scales_adam(self):
        self.called.append("get_scales_adam")


@pytest.fixture(autouse=True)
def patch_cryo_SFcalculator(monkeypatch):
    import SFC_Torch

    import rocket.cryo.structurefactors

    monkeypatch.setattr(rocket.cryo.structurefactors, "SFcalculator", DummyCryoSFCalc)
    monkeypatch.setattr(SFC_Torch, "SFcalculator", DummyCryoSFCalc)


def test_initial_cryoSFC_sets_attributes():
    from rocket.cryo.structurefactors import initial_cryoSFC

    result = initial_cryoSFC(
        "model.pdb", "map.mtz", "Emean", "PHIEmean", device="cpu", n_bins=11
    )

    # Instance of DummyCryoSFCalc
    assert isinstance(result, DummyCryoSFCalc)
    assert result.SigF is None
    assert torch.allclose(result.Fprotein_HKL, torch.tensor([1.0]))
    called = result.called
    # Should be three calls
    assert len(called) == 3
    assert called[0] == "calc_fprotein"
    assert called[2] == "get_scales_adam"
    # The middle call is ('calc_Ec', tensor)
    name, tensor_arg = called[1]
    assert name == "calc_Ec"
    assert torch.allclose(tensor_arg, torch.tensor([5.0]))


def test_initial_cryoSFC_device_and_bins(monkeypatch):
    # Patch to record init args
    recorded_kwargs = {}

    class DummyCryoSFCalcRecord(DummyCryoSFCalc):
        def __init__(self, *a, **k):
            recorded_kwargs.update(k)
            super().__init__(*a, **k)

    import SFC_Torch

    import rocket.cryo.structurefactors

    monkeypatch.setattr(
        rocket.cryo.structurefactors, "SFcalculator", DummyCryoSFCalcRecord
    )
    monkeypatch.setattr(SFC_Torch, "SFcalculator", DummyCryoSFCalcRecord)

    from rocket.cryo.structurefactors import initial_cryoSFC

    _ = initial_cryoSFC("x.pdb", "y.mtz", "E", "PHIE", device="cuda:1", n_bins=3)
    assert recorded_kwargs["device"] == "cuda:1"
    assert recorded_kwargs["n_bins"] == 3
    assert recorded_kwargs["mode"] == "cryoem"
    assert recorded_kwargs["freeflag"] == "None"
    assert recorded_kwargs["expcolumns"] == ["E", "PHIE"]
