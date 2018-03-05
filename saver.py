import torch
from os import path, makedirs, rename
import itertools
import glob
import re
from .logger import get_logger, INFO

logger = get_logger(__name__, INFO)


class Saver(object):
    def __init__(self, model, opt=None, cont=False):
        '''
        Args:
            cont (bool): Trueなら新しいディレクトリを作らない
        '''
        self._log_dir = path.join(model.log_dir, 'train')
        self._name = model.name
        self._model = model.model
        self._opt = opt

        if cont:
            if path.isdir(self._log_dir):
                return
            else:
                raise ValueError('No dir to continue: %s' % self._log_dir)

        # move dir if already exists
        if path.isdir(self._log_dir):
            for i in itertools.count():
                bk_path = self._log_dir + '.' + str(i)
                if not path.isdir(bk_path):
                    rename(self._log_dir, bk_path)
                    break
        makedirs(self._log_dir)

        with open(path.join(self._log_dir, self._name + '.model'), 'w') as f:
            f.write(str(model))

    def save(self, step, loss):
        file_name = 'step-%d_loss-%.3f.ckpt' % (step, loss)
        file_path = path.join(self._log_dir, file_name)
        torch.save(self._model.state_dict(), file_path)
        if self._opt:
            opt_file_name = 'step-%d_loss-%.3f_opt.ckpt' % (step, loss)
            opt_file_path = path.join(self._log_dir, opt_file_name)
            torch.save(self._opt.state_dict(), opt_file_path)


def load_ckpt(model, method='latest'):
    ckpt_dir = path.join(model.log_dir, 'train')
    model_list = glob.glob(path.join(ckpt_dir, '*[0-9].ckpt'))
    opt_list = glob.glob(path.join(ckpt_dir, '*_opt.ckpt'))

    if len(model_list) == 0:
        raise ValueError('No ckpt file is found in %s' % ckpt_dir)

    def latest(files):
        reg = re.compile(r'.*step-([0-9]+)_.*')
        files = [reg.search(f) for f in files]
        files = [(r.group(0), int(r.group(1))) for r in files if r is not None]
        if len(files) > 0:
            return max(files, key=lambda x: x[1])
        else:
            return None, None

    def best(files):
        reg = re.compile(r'.*step-([0-9]+)_loss-([0-9]+.[0-9]+).*')
        files = [reg.search(f) for f in files]
        files = [(r.group(0), int(r.group(1)), float(r.group(2))) for r in files
                 if r is not None]
        if len(files) > 0:
            return min(files, key=lambda x: x[2])
        else:
            return None, None

    if method == 'latest':
        ckpt_model, model_step = latest(model_list)
        ckpt_opt, opt_step = latest(opt_list)
    elif method == 'best':
        ckpt_model, model_step, model_loss = best(model_list)
        ckpt_opt, opt_step, _ = best(opt_list)
    else:
        raise ValueError('Unknown method: %s' % method)

    model.model.load_state_dict(torch.load(ckpt_model))
    if model.opt is None:
        return model_step, ckpt_model

    if opt_step == model_step:
        model.opt.load_state_dict(torch.load(ckpt_opt))
    else:
        logger.error('No ckpt for opt: path "%s"' % ckpt_dir)
    return model_step, ckpt_model
