from os import path
import mmap


def utf8len(s):
    return len(s.encode('utf-8'))


class Reader(object):
    '''Allow Random read using index file'''
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
        lines = open(self.data_file, 'r').read().split('\n')
        idx = [utf8len(d) + 1 for d in lines]
        with open(self.idx_file, 'w') as f:
            s = 0
            for i in idx:
                f.write('%d,' % s)
                s += i

    def load_idx(self):
        return [int(i) for i in open(self.idx_file, 'r').read().split(',') if i]

    def __getitem__(self, i):
        idx = self.idx[i]
        self.mm.seek(idx)
        return self.mm.readline()[:-1].decode('utf-8')

    def __len__(self):
        return len(self.idx) - 1


def test(file):
    r = Reader(file)

    print(r[0])
    print(r[1])
    print(r[2])
    print(r[3])
    print(len(r))
    print(r.idx)


if __name__ == '__main__':
    test('test/data/a')
