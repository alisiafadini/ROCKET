import torch
import numpy as np
import reciprocalspaceship as rs
from functools import partial

def dict_map(fn, dic, leaf_type, exclude_keys={"msa_feat_bias", "msa_feat_weights"}):
    if exclude_keys is None:
        exclude_keys = set()

    new_dict = {}
    for k, v in dic.items():
        if k in exclude_keys:
            new_dict[k] = v  # Keep the key-value pair as it is
        elif type(v) is dict:
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


def move_tensors_to_device_inplace(processed_features, device=try_gpu()):
    """
    Moves PyTorch tensors in a dictionary to the specified device in-place.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").
    """
    # Iterate through the keys and values in the input dictionary
    for key, value in processed_features.items():
        # Check if the value is a PyTorch tensor
        if isinstance(value, torch.Tensor):
            # Move the tensor to the specified device in-place
            processed_features[key] = value.to(device)


def move_tensors_to_device(processed_features, device=try_gpu()):
    """
    Moves PyTorch tensors in a dictionary to the specified device.

    Args:
        processed_features (dict): Dictionary containing tensors.
        device (str): Device to move tensors to (e.g., "cuda:0", "cpu").

    Returns:
        dict: Dictionary with tensors moved to the specified device.
    """
    # Create a new dictionary to store processed features with tensors moved to the device
    processed_features_on_device = {}

    # Iterate through the keys and values in the input dictionary
    for key, value in processed_features.items():
        # Check if the value is a PyTorch tensor
        if isinstance(value, torch.Tensor):
            # Move the tensor to the specified device
            value = value.to(device)
        # Add the key-value pair to the new dictionary
        processed_features_on_device[key] = value

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
    return isinstance(x, list) or isinstance(x, tuple)


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


def d2q(d):
    return 2 * np.pi / d


def load_mtz(mtz):
    dataset = rs.read_mtz(mtz)
    dataset.compute_dHKL(inplace=True)

    return dataset