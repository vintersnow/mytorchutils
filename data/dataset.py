from torch.utils.data import Dataset
from os import path
from .reader import Reader
import json


class JsonLineDataset(Dataset):
    def __init__(self, file, transform=None):
        assert path.isfile(file)
        self.file = file
        self.data = Reader(file)

        self.transform = transform
        self.transformed = [None] * len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transformed[idx]:
            return self.transformed[idx]

        sample = json.loads(self.data[idx])

        if self.transform:
            if isinstance(self.transform, list):
                for c in self.transform:
                    sample = c(sample)
            else:
                sample = self.transform(sample)

        self.transformed[idx] = sample

        return sample
