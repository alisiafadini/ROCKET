"""
Functions relating model structure factor manipulation and normalization
"""

import torch


def ftotal_amplitudes(Ftotal, dHKL, sort_by_res=True):
    F_mag = torch.abs(Ftotal)
    dHKL_tensor = torch.from_numpy(dHKL)
    if sort_by_res:
        sorted_indices = torch.argsort(dHKL_tensor, descending=True)
        F_out_mag = F_mag[sorted_indices]
    return F_out_mag
