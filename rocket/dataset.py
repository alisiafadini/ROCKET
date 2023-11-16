"""
Inlcude customized pytorch Dataset implementation
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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


def load_batches(all_data: IndexedDataset, num_batches: int):
    batch_sizes = int(len(all_data["EDATA"]) / num_batches)
    data_loader = DataLoader(all_data, batch_size=batch_sizes, shuffle=True)

    return data_loader


"""
E.g. to access batches:

for batch_data, batch_indices in data_loader:
    print(batch_data["EDATA"].shape)
    print(batch_indices)

"""
