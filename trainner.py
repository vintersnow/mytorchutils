from .misc import OneLinePrint
import time
import math

from typing import Dict, Union, Any
from .model import Model
import torch


val_t = Union[int, float]


class Context(object):
    def __init__(self) -> None:
        self.epoch = 0
        self.step = 0
        self._clear()

    def _clear(self) -> None:
        self.scalar: Dict[str, val_t] = {}
        self.log: Dict[str, val_t] = {}

    def addlog(self, key: str, val: val_t) -> None:
        self.log[key] = self.scalar[key] = val


class Runner(object):
    def __init__(self, model: Model, summary_prefix: str) -> None:
        self.ctx = Context()
        self.olp = OneLinePrint(' | ')
        self.model = model
        self.summary_prefix = summary_prefix

    def write_summary(self, step: int) -> None:
        for key, val in self.ctx.scalar.items():
            key = f'{self.summary_prefix}/{key}'
            self.model.writer.add_scalar(key, val, step)

    def log_str(self) -> str:
        items = sorted(self.ctx.log.items(), key=lambda x: x[0])
        return ', '.join([f'{key}: {float(val):.3f}' for key, val in items])

    def run(self,
            train_ld: torch.utils.data.DataLoader,
            val_ld: torch.utils.data.DataLoader,
            init_epoch: int = 0,
            max_epoch: int = 5,
            max_step_each_epoch: int = 0,
            save_model: bool = False,
            summary_step: int = 0,
            clear_log_each_step: bool = True):
        epoch_length = min(max_step_each_epoch, len(train_ld)) if max_step_each_epoch > 0 else len(train_ld)

        for epoch in range(init_epoch, max_epoch):
            self.ctx.epoch = epoch

            self.model.train()
            self.ctx._clear()
            elapsed = []
            for i, batch in enumerate(train_ld):
                if i >= epoch_length:
                    break

                if clear_log_each_step:
                    self.ctx._clear()
                self.ctx.step = step = i + epoch * epoch_length

                start = time.time()
                # forward and loss
                loss = self.train_step(batch)
                elapsed.append(time.time() - start)

                if math.isnan(loss):
                    raise ValueError(f'loss is nan, epoch {epoch}, step {step}')

                with self.olp as w:
                    # logging
                    w('%s epoch %d/%d step %d/%d' %
                      (self.model.name, epoch + 1, max_epoch,
                       (step % epoch_length) + 1, epoch_length))
                    w(self.log_str())

                # write summary
                if summary_step > 0 and step % summary_step == 0:
                    self.ctx.scalar['time'] = sum(elapsed) / len(elapsed)
                    elapsed.clear()
                    self.write_summary(step)

            # # clean up summary
            if summary_step > 0 and step % summary_step != 0:
                self.write_summary(step)

            # validation
            self.ctx._clear()
            self.model.eval()
            val_score = self.val_step(val_ld)
            if summary_step > 0:
                self.write_summary(epoch)
            print(self.olp._mark, self.log_str())

            # save model
            if save_model > 0:
                self.model.save(epoch, val_score)

    def train_step(self, batch: Dict[str, Any]) -> val_t:
        raise NotImplementedError()

    def val_step(self, ld: torch.utils.data.DataLoader) -> val_t:
        raise NotImplementedError()
