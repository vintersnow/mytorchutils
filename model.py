from torch import nn
from os import path, makedirs
import json
from .saver import Saver, load_ckpt
from tensorboardX import SummaryWriter


class Model(nn.Module):
    def __init__(self, model, name, log_root='logs', opt=None, hps=None):
        super(Model, self).__init__()

        self.model = model
        self.name = name
        self.log_root = log_root
        self.log_dir = path.join(log_root, name)
        self.opt = opt

        if not path.isdir(self.log_dir):
            makedirs(self.log_dir)
        config_file = path.join(self.log_dir, 'config.json')
        if hps and not path.isfile(config_file):
            json.dump(
                vars(hps),
                open(config_file, 'w'),
                default=lambda x: x.name,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
                separators=(',', ': '))

        self.saver = None
        self.cont = False
        self.training = True

    def forward(self, *args, **keys):
        return self.model(*args, **keys)

    def restore(self, method='latest'):
        self.cont = True
        return load_ckpt(self, method)

    def save(self, step, loss):
        if self.saver is None:
            self.saver = Saver(self, self.opt, self.cont)

        self.saver.save(step, loss)

    def make_writer(self, comment=''):
        summary_dir = path.join(self.log_dir, 'summary')
        self.writer = SummaryWriter(summary_dir, comment)
        return self.writer

    # 後から追加するの気持ちわるい？
    def addopt(self, opt):
        self.opt = opt
        if self.saver:
            self.saver._opt = opt

    @property
    def device(self):
        p = self.model.parameters()
        return next(p).device
