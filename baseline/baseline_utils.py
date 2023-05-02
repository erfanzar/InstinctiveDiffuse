import typing
from typing import (Optional)

import torch.cuda


class HyperParameters(object):
    load_local: Optional[bool] = True

    device: Optional[str] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    d_type: typing.Any = torch.float16

    # Fun Bra Its Fun Part xD
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

        device: Optional[str] = kwargs.pop('device', self.device)
        d_type: typing.Any = kwargs.pop('d_type', self.d_type)
        show_details: Optional[bool] = kwargs.pop('show_details', True)
        permission_to_use_ram: Optional[bool] = kwargs.pop('permission_to_use_ram', True)
        permission_to_use_cpu: Optional[bool] = kwargs.pop('permission_to_use_cpu', True)
        permission_to_use_gpu: Optional[bool] = kwargs.pop('permission_to_use_gpu', True)
        nsfw_content: Optional[bool] = kwargs.pop('nsfw_content', False)
        load_model: Optional[bool] = kwargs.pop('load_model', True)
        train_model: Optional[bool] = kwargs.pop('train_model', False)
        load_local: Optional[bool] = kwargs.pop('load_local', False)
        model_path: Optional[bool] = kwargs.pop('model_path', None)
        frozen_weight_use: Optional[bool] = kwargs.pop('frozen_weight_use', True)
        load_format: Optional[str] = kwargs.pop('load_format', '.bin')

        self.__dict__['device'] = device
        self.__dict__['d_type'] = d_type
        self.__dict__['show_details'] = show_details

        self.__dict__['permission_to_use_ram'] = permission_to_use_ram
        self.__dict__['permission_to_use_cpu'] = permission_to_use_cpu
        self.__dict__['permission_to_use_gpu'] = permission_to_use_gpu

        self.__dict__['nsfw_content'] = nsfw_content
        self.__dict__['load_model'] = load_model
        self.__dict__['train_model'] = train_model

        self.__dict__['frozen_weight_use'] = frozen_weight_use
        self.__dict__['load_format'] = load_format

        self.__dict__['load_local'] = load_local
        self.__dict__['model_path'] = model_path
