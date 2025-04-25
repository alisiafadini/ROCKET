# Mock external dependencies before importing utils
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from rocket.utils import (
    assert_numpy,
    assert_tensor,
    convert_feat_tensors_to_numpy,
    d2q,
    dict_map,
    get_params_path,
    load_mtz,
    load_pdb,
    move_tensors_to_device,
    move_tensors_to_device_inplace,
    plddt2pseudoB_np,
    tree_map,
    try_gpu,
    weighting,
    weighting_torch,
)

sys.modules["openfold"] = MagicMock()
sys.modules["reciprocalspaceship"] = MagicMock()
sys.modules["SFC_Torch"] = MagicMock()


def test_plddt2pseudoB_basic():
    arr = np.array([0, 50, 100])
    out = plddt2pseudoB_np(arr)
    assert out.shape == arr.shape
    assert np.all(out > 0)
    assert np.isfinite(out).all()


def test_plddt2pseudoB_empty():
    arr = np.array([])
    out = plddt2pseudoB_np(arr)
    assert out.shape == arr.shape


def test_weighting_cutoffs():
    b = np.array([10.0, 11.5, 20.0, 30.0, 40.0])
    w = weighting(b)
    assert np.isclose(w[0], 1.0)
    assert np.isclose(w[1], 1.0)
    assert 0.5 < w[2] < 1.0
    assert np.isclose(w[3], 0.5)
    assert w[4] < 0.5


def test_weighting_torch_matches_numpy():
    b = torch.tensor([10.0, 11.5, 20.0, 30.0, 40.0])
    wn = weighting(b.numpy())
    wt = weighting_torch(b).numpy()
    np.testing.assert_allclose(wn, wt, rtol=1e-5)


def test_weighting_scalar():
    w = weighting(10.0)
    assert w == 1.0


def test_dict_map_exclude_keys():
    d = {"a": 1, "msa_feat_bias": 2}

    def fn(x):
        return x + 1

    out = dict_map(fn, d, int)
    assert out["a"] == 2
    assert out["msa_feat_bias"] == 2  # should not be mapped


def test_tree_map_nested():
    tree = {"a": [1, 2], "b": (3, 4)}
    out = tree_map(lambda x: x * 2, tree, int)
    assert out["a"] == [2, 4]
    assert out["b"] == (6, 8)


def test_tree_map_non_supported():
    class Dummy:
        pass

    with pytest.raises(ValueError):
        tree_map(lambda x: x, Dummy(), int)


def test_try_gpu_cpu():
    with patch("torch.cuda.device_count", return_value=0):
        dev = try_gpu()
        assert str(dev) == "cpu"


def test_try_gpu_cuda():
    with patch("torch.cuda.device_count", return_value=2):
        dev = try_gpu(1)
        assert str(dev) == "cuda:1"


def test_move_tensors_to_device():
    d = {"a": torch.zeros(2), "b": 3}
    out = move_tensors_to_device(d, device="cpu")
    assert out["a"].device.type == "cpu"
    assert out["b"] == 3


def test_move_tensors_to_device_inplace():
    d = {"a": torch.zeros(2)}
    move_tensors_to_device_inplace(d, device="cpu")
    assert d["a"].device.type == "cpu"


def test_convert_feat_tensors_to_numpy():
    d = {"a": torch.ones(2), "b": "notensor"}
    out = convert_feat_tensors_to_numpy(d)
    assert isinstance(out["a"], np.ndarray)
    assert out["b"] == "notensor"


def test_assert_numpy_from_tensor():
    arr = torch.ones(2)
    np_arr = assert_numpy(arr)
    assert isinstance(np_arr, np.ndarray)


def test_assert_numpy_from_list():
    lst = [1, 2, 3]
    np_arr = assert_numpy(lst)
    assert isinstance(np_arr, np.ndarray)


def test_assert_tensor_from_np():
    arr = np.ones(2)
    tensor = assert_tensor(arr, device="cpu")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"


def test_assert_tensor_from_list():
    lst = [1, 2, 3]
    tensor = assert_tensor(lst, device="cpu")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"


def test_d2q_scalar_and_array():
    assert np.isclose(d2q(2.0), np.pi)
    arr = np.array([1.0, 2.0])
    out = d2q(arr)
    assert np.allclose(out, 2 * np.pi / arr)


def test_load_mtz():
    with patch("rocket.utils.rs.read_mtz") as mock_read:
        mock_ds = MagicMock()
        mock_read.return_value = mock_ds
        load_mtz("file.mtz")
        mock_read.assert_called_once_with("file.mtz")
        mock_ds.compute_dHKL.assert_called_once_with(inplace=True)


def test_load_pdb():
    with patch("rocket.utils.PDBParser") as mock_pdb:
        load_pdb("file.pdb")
        mock_pdb.assert_called_once_with("file.pdb")


def test_get_params_path(monkeypatch):
    monkeypatch.setenv("OPENFOLD_RESOURCES", "/tmp/resources")
    path = get_params_path()
    assert path == "/tmp/resources/params"


def test_get_params_path_missing(monkeypatch):
    monkeypatch.delenv("OPENFOLD_RESOURCES", raising=False)
    with pytest.raises(ValueError):
        get_params_path()
