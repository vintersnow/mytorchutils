from torch import nn
from os import path, makedirs
import json
from .saver import MutliParamSaver
from tensorboardX import SummaryWriter

import torch
from typing import Optional, Dict, Tuple, Any


class Model(object):
    def __init__(
        self,
        model: nn.Module,
        name: str,
        log_root: Optional[str] = None,
        opts: dict = {},
        hps: Optional[Dict] = None,
        metric: Optional[str] = None,
        cont: bool = False,
    ) -> None:
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
            config_file = path.join(self.log_dir, "config.json")
            if hps and not path.isfile(config_file):
                json.dump(
                    vars(hps),
                    open(config_file, "w"),
                    default=lambda x: x.name,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                    separators=(",", ": "),
                )

            # print out model info
            with open(path.join(self.log_dir, "model_arch"), "w") as f:
                f.write(str(model))

            # Saver
            ckpt_dir = path.join(self.log_dir, "train")
            keep_all = metric is None
            self.saver = MutliParamSaver(
                ckpt_dir, cont=cont, metric=metric, keep_all=keep_all
            )
            self.saver.add_param("model", model)

        self.training = True

    def forward(self, *args, **keys):
        return self.model(*args, **keys)

    def restore(self, method: str = "latest") -> Tuple[int, str]:
        if self.nolog:
            raise ValueError("No log directory")
        params, step, dir = self.saver.load_ckpt(method)
        assert path.exists(dir)
        self.model.load_state_dict(params["model"])
        params.pop("model", None)

        for key, val in params.items():
            if key in self.opts:
                self.opts[key].load_state_dict(val)
        return step, dir

    def save(self, step: int, loss: float) -> None:
        if self.nolog:
            raise ValueError("No log directory")

        self.saver.save(step, loss)

    def make_writer(self, comment: str = "") -> SummaryWriter:
        if self.nolog:
            raise ValueError("No log directory")
        summary_dir = path.join(self.log_dir, "summary")
        self._writer = SummaryWriter(summary_dir, comment)
        return self._writer

    def add_opt(self, key: str, opt: torch.optim.Optimizer) -> None:
        self.opts[key] = opt
        self.saver.add_param(key, opt)

    @property
    def device(self) -> torch.device:
        # 全てのパラメータが同一メモリにあると仮定している
        p = self.model.parameters()
        return next(p).device

    @property
    def writer(self) -> SummaryWriter:
        if not hasattr(self, "_writer"):
            return self.make_writer()
        return self._writer

    def __del__(self) -> None:
        if hasattr(self, "_writer"):
            self._writer.close()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.model, attr)
