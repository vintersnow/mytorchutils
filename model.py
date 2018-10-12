from torch import nn
from os import path, makedirs
import json
from .saver import MutliParamSaver
from tensorboardX import SummaryWriter


class Model(nn.Module):
    def __init__(self, model, name, log_root=None, opts={}, hps=None, metric=None, cont=False):
        super(Model, self).__init__()

        self.model = model
        self.name = name
        self.opts = opts
        self.metric = metric

        if log_root is None:
            self.nolog = True
        else:
            self.nolog = False
            self.log_root = log_root
            self.log_dir = path.join(log_root, name)

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

            # print out model info
            with open(path.join(self.log_dir, 'model_arch'), 'w') as f:
                f.write(str(model))

            # Saver
            ckpt_dir = path.join(self.log_dir, 'train')
            keep_all = metric is None
            self.saver = MutliParamSaver(ckpt_dir, cont=cont, metric=metric, keep_all=keep_all)
            self.saver.add_param('model', model)

        self.training = True

    def forward(self, *args, **keys):
        return self.model(*args, **keys)

    def restore(self, method='latest'):
        if self.nolog:
            raise ValueError('No log directory')
        params, step = self.saver.load_ckpt(method)
        self.model.load_state_dict(params['model'])
        params.pop('model', None)

        for key, val in params.items():
            if key in self.opts:
                self.opts[key].load_state_dict(val)
        return step

    def save(self, step, loss):
        if self.nolog:
            raise ValueError('No log directory')

        self.saver.save(step, loss)

    def make_writer(self, comment=''):
        if self.nolog:
            raise ValueError('No log directory')
        summary_dir = path.join(self.log_dir, 'summary')
        self._writer = SummaryWriter(summary_dir, comment)
        return self._writer

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close()

    def add_opt(self, key, opt):
        self.opts[key] = opt
        self.saver.add_param(key, opt)

    @property
    def device(self):
        # 全てのパラメータが同一メモリにあると仮定している
        p = self.model.parameters()
        return next(p).device

    @property
    def writer(self):
        if not hasattr(self, '_writer'):
            return self.make_writer(self)
        return self._writer
