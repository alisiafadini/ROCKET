import os
from functools import partial

import numpy as np
import reciprocalspaceship as rs
import torch
from SFC_Torch import PDBParser


def plddt2pseudoB(plddts):
    # Use Tom Terwilliger's formula to convert plddt to Bfactor
    deltas = 1.5 * np.exp(4 * (0.7 - 0.01 * plddts))
    b_factors = (8 * np.pi**2 * deltas**2) / 3
    return b_factors


def weighting(x, cutoff1=11.5, cutoff2=30.0):
    """
    Convert B factor to weights for L2 loss and Kabsch Alignment
    w(B) =
    1.0                                           B <= cutoff1
    1.0 - 0.5*(B-cutoff1)/(cutoff2-cutoff1)       cutoff1 < B <= cutoff2
    0.5 * exp(-sqrt(B-cutoff2))                   cutoff2 < B
    """
    a = np.where(x <= cutoff1, 1.0, 1.0 - 0.5 * (x - cutoff1) / (cutoff2 - cutoff1))
    b = 0.5 * np.exp(-np.sqrt(np.where(x <= cutoff2, 1.0, x - cutoff2)))
    return np.where(a >= 0.5, a, b)


def weighting_torch(x, cutoff1=11.5, cutoff2=30.0):
    """
    pytorch implementation of the weighting function:
    w(B) =
    1.0                                           B <= cutoff1
    1.0 - 0.5*(B-cutoff1)/(cutoff2-cutoff1)       cutoff1 < B <= cutoff2
    0.5 * exp(-sqrt(B-cutoff2))                   cutoff2 < B
    """
    a = torch.where(x <= cutoff1, 1.0, 1.0 - 0.5 * (x - cutoff1) / (cutoff2 - cutoff1))
    b = 0.5 * torch.exp(-torch.sqrt(torch.where(x <= cutoff2, 1.0, x - cutoff2)))
    return torch.where(a >= 0.5, a, b)


def dict_map(fn, dic, leaf_type, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = {"msa_feat_bias", "msa_feat_weights"}

    new_dict = {}
    for k, v in dic.items():
        if k in exclude_keys:
            new_dict[k] = v  # Keep the key-value pair as it is
        elif isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type, exclude_keys)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def move_tensors_to_device_inplace(processed_features, device=None):
    """
    Moves PyTorch tensors in a dictionary to the specified device in-place.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").
    """
    if device is None:
        device = try_gpu()
    # Iterate through the keys and values in the input dictionary
    for key, value in processed_features.items():
        # Check if the value is a PyTorch tensor
        if isinstance(value, torch.Tensor):
            # Move the tensor to the specified device in-place
            processed_features[key] = value.to(device)


def move_tensors_to_device(processed_features, device=None):
    """
    Moves PyTorch tensors in a dictionary to the specified device.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").

    Returns:
        dict: Dictionary with tensors moved to the specified device.
    """
    if device is None:
        device = try_gpu()
    # Create a new dictionary to store processed features, tensors moved to the device
    processed_features_on_device = {}

    # Iterate through the keys and values in the input dictionary
    for key, value in processed_features.items():
        # Check if the value is a PyTorch tensor
        if isinstance(value, torch.Tensor):
            # Move the tensor to the specified device
            device_value = value.clone().to(device)
        # Add the key-value pair to the new dictionary
        processed_features_on_device[key] = device_value

    # Return the new dictionary with tensors moved to the device
    return processed_features_on_device


def convert_feat_tensors_to_numpy(dictionary):
    numpy_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            numpy_dict[key] = value.detach().cpu().numpy()
        else:
            numpy_dict[key] = value
    return numpy_dict


def is_list_or_tuple(x):
    return isinstance(x, list | tuple)


def assert_numpy(x, arr_type=None):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x


def assert_tensor(x, arr_type=None, device="cuda:0"):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    if is_list_or_tuple(x):
        x = np.array(x)
        x = torch.tensor(x, device=device)
    assert isinstance(x, torch.Tensor)
    if arr_type is not None:
        x = x.to(arr_type)
    return x


def d2q(d):
    return 2 * np.pi / d


def load_mtz(mtz: str) -> rs.DataSet:
    dataset = rs.read_mtz(mtz)
    dataset.compute_dHKL(inplace=True)
    return dataset


def load_pdb(pdb: str) -> PDBParser:
    model = PDBParser(pdb)
    return model


def get_params_path():
    resources_path = os.environ.get("OPENFOLD_RESOURCES", None)
    if resources_path is None:
        raise ValueError("Please set OPENFOLD_RESOURCES environment variable")
    params_path = os.path.join(resources_path, "params")
    return params_path
