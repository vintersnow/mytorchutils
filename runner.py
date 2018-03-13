from abc import ABCMeta, abstractmethod
from torch.autograd import Variable


def get_vars(batch, *keys, use_cuda=False):
    def tovar(key):
        if key is None:
            return None
        t = batch[key]
        return Variable(t).cuda() if use_cuda else Variable(t)

    if len(keys) == 1:
        return tovar(keys[0])
    return (tovar(key) for key in keys)


class Runner(metaclass=ABCMeta):
    '''
    run forward step and calculate the loss
    '''

    def __init__(self, criterion):
        self.criterion = criterion

    def getvars(self, batch, *keys):
        use_cuda = self.use_cuda if hasattr(self, 'use_cuda') else False
        return get_vars(batch, *keys, use_cuda=use_cuda)

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def target(self):
        pass

    def loss(self, src, tgt):
        return self.criterion(src, tgt)

    def step(self, batch):
        output = self.run(batch)
        tgt = self.target(batch)
        return self.loss(output, tgt), output, tgt
