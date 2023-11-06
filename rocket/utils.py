import torch
import numpy as np
import reciprocalspaceship as rs
from rocket import structurefactors


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


def load_tng_data(tng_file, device=try_gpu()):
    tng = load_mtz(tng_file).dropna()

    # Generate PhaserTNG tensors
    eps = torch.tensor(tng["EPS"].values, device=device)
    centric = torch.tensor(tng["CENT"].values, device=device).bool()
    dobs = torch.tensor(tng["DOBS"].values, device=device)
    feff = torch.tensor(tng["FEFF"].values, device=device)
    bin_labels = torch.tensor(tng["BIN"].values, device=device)

    sigmaN = structurefactors.calculate_Sigma_atoms(feff, eps, bin_labels)
    Edata = structurefactors.normalize_Fs(feff, eps, sigmaN, bin_labels)

    data_dict = {
        "EDATA": Edata,
        "EPS": eps,
        "CENTRIC": centric,
        "DOBS": dobs,
        "FEFF": feff,
        "BIN_LABELS": bin_labels,
    }

    return data_dict
