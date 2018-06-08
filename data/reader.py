from os import path
import mmap
import pickle
import numpy as np
import json


def utf8len(s):
    return len(s.encode('utf-8'))


class Reader(object):
    def __init__(self, data_file, idx_file=None, protocol='pickle'):
        self.data_file = data_file
        self.idx_file = idx_file if idx_file else data_file + '.idx'

        assert protocol == 'pickle' or protocol == 'json'
        self.protocol = protocol

        if not path.exists(self.idx_file):
            self.create_idx()
        elif path.getmtime(self.idx_file) < path.getmtime(self.data_file):
            self.create_idx()

        self.idx = self.load_idx()

        f = open(self.data_file, 'r+b')
        self.mm = mmap.mmap(f.fileno(), 0)

    def create_idx(self):
        lines = open(self.data_file, 'r').read().split('\n')
        idx = np.asarray([0] + [utf8len(d) + 1 for i, d in enumerate(lines)][:-1])
        idx = np.cumsum(idx)

        if self.protocol == 'pickle':
            with open(self.idx_file, 'wb') as f:
                pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)  # 800ms
        else:
            with open(self.idx_file, 'w') as f:
                json.dump(idx.tolist(), f)  # 1633ms

    def load_idx(self):
        if self.protocol == 'pickle':
            return pickle.load(open(self.idx_file, 'rb'))  # 44ms
        else:
            return json.load(open(self.idx_file, 'r'))  # 110ms

    def __getitem__(self, i):
        idx = self.idx[i]
        self.mm.seek(idx)
        return self.mm.readline()[:-1].decode('utf-8')

    def __len__(self):
        return len(self.idx) - 1
