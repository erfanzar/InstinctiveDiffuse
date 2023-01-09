import os
import typing

from erutils.lightning import TorchBaseModule, pars_model_v2
from erutils.utils import read_yaml


class Generator(TorchBaseModule):
    def __init__(self, config: typing.Union[str, os.PathLike]):
        super(Generator, self).__init__()
        self.config = read_yaml(config)

        self.model, self.save_f = pars_model_v2(self.config['generator'], ['GB'], print_status=True,
                                                sc=self.config['sc'],
                                                imports=['from engine.commons import *'])
        self.model = self.model.to(self.DEVICE)

    def forward_once(self, x):
        x = x.view(len(x), self.config['sc'], self.config['image_size'], self.config['image_size'])
        for m in self.model:
            x = m(x)
        return x

    def forward(self, batch):
        return self.forward_once(batch)
