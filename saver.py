import torch
from os import path, makedirs, rename, remove
import itertools
import glob
import re
from .logger import get_logger, INFO

logger = get_logger(__name__, INFO)


class Saver(object):
    def __init__(self,
                 model,
                 opt=None,
                 cont=False,
                 metric='lower',
                 keep_all=True):
        '''
        Args:
            cont (bool): Trueなら新しいディレクトリを作らない
        '''
        assert metric == 'lower' or metric == 'higher', metric
        self._log_dir = path.join(model.log_dir, 'train')
        self._name = model.name
        self._model = model.model
        self._opt = opt
        self.metric = metric
        self.keep_all = keep_all

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

    def ckpt_list(self):
        ckpts = glob.glob(path.join(self._log_dir, '*[0-9].ckpt'))
        reg = re.compile(r'.*step-([0-9]+)_loss-([0-9]+.[0-9]+).*')
        files = [reg.search(f) for f in ckpts]
        files = [(r.group(0), int(r.group(1)), float(r.group(2)))
                 for r in files if r is not None]
        return files

    def best(self):
        files = self.ckpt_list()
        mm = max if self.metric == 'higher' else min
        if len(files) > 0:
            return mm(files, key=lambda x: x[2])
        else:
            return None, None, None

    def latest(self):
        files = self.ckpt_list()
        if len(files) > 0:
            return max(files, key=lambda x: x[1])
        else:
            return None, None, None

    def rm_ckpt(self, step):
        ckpts = glob.glob(path.join(self._log_dir, 'step-%d_*.ckpt' % step))
        for file in ckpts:
            if path.isfile(file):
                remove(file)

    def save(self, step, score):
        best_file, best_step, best_score = self.best()
        latest_file, latest_step, _ = self.latest()

        file_name = 'step-%d_loss-%.3f.ckpt' % (step, score)
        file_path = path.join(self._log_dir, file_name)
        torch.save(self._model.state_dict(), file_path)
        if self._opt:
            opt_file_name = 'step-%d_loss-%.3f_opt.ckpt' % (step, score)
            opt_file_path = path.join(self._log_dir, opt_file_name)
            torch.save(self._opt.state_dict(), opt_file_path)

        if not self.keep_all and best_score is not None:
            sign = 1 if self.metric == 'higher' else -1
            if (score - best_score) * sign > 0:
                self.rm_ckpt(step)
            if best_step != latest_step:
                self.rm_ckpt(latest_step)


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

    def best(files, m):
        # TODO: formatを変えれるもしくは、lossではない単語にした方が良い
        reg = re.compile(r'.*step-([0-9]+)_loss-([0-9]+.[0-9]+).*')
        files = [reg.search(f) for f in files]
        files = [(r.group(0), int(r.group(1)), float(r.group(2)))
                 for r in files if r is not None]

        if len(files) > 0:
            return m(files, key=lambda x: x[2])
        else:
            return None, None

    def from_file(file):
        assert file in model_list
        opt_file = file[:len(file) - len('.ckpt')] + '_opt.ckpt'
        assert path.isfile(opt_file)
        reg = re.compile(r'.*step-([0-9]+)_loss-([0-9]+.[0-9]+).*')
        r = reg.search(file)
        step = r.group(1)
        return file, opt_file, step

    if method == 'latest':
        ckpt_model, model_step = latest(model_list)
        ckpt_opt, opt_step = latest(opt_list)
    elif method == 'highest':
        ckpt_model, model_step, model_loss = best(model_list, max)
        ckpt_opt, opt_step, _ = best(opt_list, max)
    elif method == 'lowest':
        ckpt_model, model_step, model_loss = best(model_list, min)
        ckpt_opt, opt_step, _ = best(opt_list, min)
    elif path.isfile(method):
        ckpt_model, ckpt_opt, model_step = from_file(method)
        opt_step = model_step
    else:
        raise ValueError('Unknown method or file not exists: %s' % method)

    model.model.load_state_dict(torch.load(ckpt_model))
    if model.opt is None:
        return model_step, ckpt_model

    if opt_step == model_step:
        model.opt.load_state_dict(torch.load(ckpt_opt))
    else:
        logger.error('No ckpt for opt: path "%s"' % ckpt_dir)
    return model_step, ckpt_model


if __name__ == '__main__':
    from .model import Model
    ly = torch.nn.Linear(10, 10)
    m = Model(ly, 'test', '.')
    m.addopt(torch.optim.SGD(ly.parameters(), 1))
    for i in range(10):
        m.save(i, i)
