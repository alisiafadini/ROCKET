"""
Inlcude customized pytorch Dataset implementation
"""

from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, tng_dict, transform=None, target_transform=None):
        self.data = tng_dict
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data["EDATA"])

    def __getitem__(self, idx):
        batch = {key: lst[idx] for key, lst in self.data.items()}
        batch_indices = idx

        return batch, batch_indices
