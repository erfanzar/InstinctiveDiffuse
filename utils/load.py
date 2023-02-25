import os
import typing

import torch


# from engine.intels import Discriminator,Generator


def load_model(model: torch.nn.Module, path: typing.Union[str, os.PathLike],
               device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    ldd = torch.load(path, device)['state_dict']
    model.load_state_dict(ldd)
    return model
