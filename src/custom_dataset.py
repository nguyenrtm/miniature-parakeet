import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def padding(self, min_padding_size=3):
        for idx in range(len(self.data)):
            batch = self.data[idx]

            max_len_in_batch = max(max([x.shape[0] for x in batch]), min_padding_size)

            for i in range(len(batch)):
                batch[i] = F.pad(batch[i], (0, 0, 0, max_len_in_batch - batch[i].shape[0]), "constant", 0)

            self.data[idx] = torch.stack(batch, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label