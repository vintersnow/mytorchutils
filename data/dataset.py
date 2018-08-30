from torch.utils.data import Dataset
from os import path
from .reader import Reader
import json
from abc import abstractmethod


class BaseDataset(Dataset):
    def __init__(self, file, transformer=None, save_trans=True):
        assert path.isfile(file), file
        self.file = file
        self.data = None

        self.transformer = transformer
        self.transformed = [None] * len(self.data)
        self.save_trans = save_trans

    def __len__(self):
        return len(self.data)

    def transorm(self, sample):
        if self.transformer:
            if isinstance(self.transformer, list):
                for c in self.transformer:
                    sample = c(sample)
            else:
                sample = self.transformer(sample)
        return sample

    @abstractmethod
    def prepare(self, idx):
        pass

    def __getitem__(self, idx):
        if self.save_trans and self.transformed[idx]:
            return self.transformed[idx]

        sample = self.prepare(idx)

        sample = self.transorm(sample)
        if self.save_trans:
            self.transformed[idx] = sample

        return sample


class LineDataset(BaseDataset):
    def __init__(self, *args, **keys):
        super(LineDataset, *args, **keys)
        self.data = Reader(self.file)

    def prepare(self, idx):
        return self.data[idx]


class JsonLineDataset(BaseDataset):
    def __init__(self, *args, **keys):
        super(JsonLineDataset, self).__init__(*args, **keys)
        self.data = Reader(self.file)

    def prepare(self, idx):
        return json.loads(self.data[idx])
