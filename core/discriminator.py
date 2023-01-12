import os
import typing

from erutils.lightning import TorchBaseModule, pars_model_v2
from erutils.utils import read_yaml


class Discriminator(TorchBaseModule):
    def __init__(self, config: typing.Union[str, os.PathLike]):
        super(Discriminator, self).__init__()
        self.config = read_yaml(config)

        self.model, self.save_f = pars_model_v2(self.config['discriminator'], ['DiscriminatorBlock'], print_status=True,
                                                sc=self.config['image_channels'] + self.config['n_classes'],
                                                imports=['from engine.commons import *'])
        self.model = self.model.to(self.DEVICE)

    def forward_once(self, x):
        for m in self.model:
            x = m(x)
        return x.view(len(x), -1)

    def forward(self, batch):
        return self.forward_once(batch)
