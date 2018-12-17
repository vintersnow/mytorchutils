from abc import ABCMeta, abstractmethod
import torch


def get_vars(batch, *keys, device=None, non_blocking=False):
    if not device:
        device = torch.device('cpu')

    def tovar(key):
        if key is None:
            return None
        t = batch[key]
        return t.to(device=device, non_blocking=non_blocking)

    if len(keys) == 1:
        return tovar(keys[0])
    return (tovar(key) for key in keys)


class Runner(metaclass=ABCMeta):
    '''
    run forward step and calculate the loss
    '''

    def __init__(self, criterion, device=torch.device('cpu')):
        self.criterion = criterion
        self.device = device

    def getvars(self, batch, *keys):
        return get_vars(batch, *keys, device=self.device)

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
