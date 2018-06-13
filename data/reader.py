from os import path
import mmap
# import pickle
import numpy as np


def utf8len(s):
    '''
    Return byte length of utf-8 string

    Examples
    --------
    >>> utf8len('aaaa')
    4
    >>> utf8len('あああ')
    9
    '''
    return len(s.encode('utf-8'))


class Reader(object):
    '''
    ReadOneline data. Using indexes for faster read.

    '''
    def __init__(self, data_file, idx_file=None):
        self.data_file = data_file
        self.idx_file = idx_file if idx_file else data_file + '.idx'

        if not path.exists(self.idx_file):
            self.create_idx()
        elif path.getmtime(self.idx_file) < path.getmtime(self.data_file):
            self.create_idx()

        self.idx = self.load_idx()

        f = open(self.data_file, 'r+b')
        self.mm = mmap.mmap(f.fileno(), 0)

    def create_idx(self):
        idx = [0]
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                idx.append(utf8len(line))
        idx = np.asarray(idx)
        idx = np.cumsum(idx)

        with open(self.idx_file, 'wb') as f:
            np.save(f, idx)
            # pickle.dump(idx, f, protocol=pickle.HIGHEST_PROTOCOL)  # 800ms

    def load_idx(self):
        return np.load(self.idx_file, 'r')
        # return pickle.load(open(self.idx_file, 'rb'))  # 44ms

    def __getitem__(self, i):
        idx = self.idx[i]
        self.mm.seek(idx)
        return self.mm.readline().decode('utf-8')

    def __len__(self):
        return len(self.idx) - 1


if __name__ == '__main__':
    from tempfile import NamedTemporaryFile
    import os
    t = NamedTemporaryFile()
    with open(t.name, 'w') as f:
        f.write('あああ\n')
        f.write('bbbb\n')
    r = Reader(t.name)
    assert len(r) == 2, len(r)
    assert r[1] == 'bbbb\n', r[2]
    print('ok')
    os.remove(r.idx_file)
