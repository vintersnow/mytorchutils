from .misc import OneLinePrint
import time
import math


class Context(object):
    def __init__(self):
        self.epoch = 0
        self.step = 0
        self._clear()

    def _clear(self):
        # reset
        self.scalar = {}
        self.log = {}

    def addlog(self, key, val):
        self.log[key] = self.scalar[key] = val


class Runner(object):
    def __init__(self, model, summary_prefix):
        self.ctx = Context()
        self.olp = OneLinePrint(' | ')
        self.model = model
        self.summary_prefix = summary_prefix

    def write_summary(self, step):
        for key, val in self.ctx.scalar.items():
            key = f'{self.summary_prefix}/{key}'
            self.model.writer.add_scalar(key, val, step)

    def log_str(self):
        items = sorted(self.ctx.log.items(), key=lambda x: x[0])
        return ', '.join([f'{key}: {float(val):.3f}' for key, val in items])

    def run(self,
            train_ld,
            val_ld,
            init_epoch=0,
            max_epoch=5,
            max_step_each_epoch=0,
            save_model=False,
            summary_step=0,
            clear_log_each_step=True):
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

    def train_step(self, batch):
        raise NotImplementedError()

    def val_step(self, batch):
        raise NotImplementedError()
