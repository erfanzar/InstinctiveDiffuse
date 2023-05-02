import typing

import erutils

from modules.models import CGRModel
from .baseline_utils import HyperParameters


def baseline_load(hyper_parameters: HyperParameters, logging: typing.Optional[bool] = True) \
        -> typing.Optional[CGRModel]:
    r"""

    :param logging: to Log what is happening under the hood bool : default have been set to True
    :param hyper_parameters: HyperParameter For model
    >>> default_parameters = HyperParameters()
    >>> md = baseline_load(default_parameters,False)
    :return: Requested Model Will Be Returned
    """
    if logging:
        erutils.fprint('Loading Model With Provided HyperParameters')
    erutils.loggers.show_hyper_parameters(hyper_parameters)
    model_id: typing.Optional[str] = hyper_parameters.model_path
    if logging:
        erutils.fprint(f"Loading Model from : {model_id}")
    model: CGRModel = CGRModel.from_pretrained(
        model_id,
        torch_dtype=hyper_parameters.d_type)
    if logging:
        erutils.fprint('Model Loaded')
        erutils.fprint(f'Moving Model To {hyper_parameters.device}')
    model = model.to(hyper_parameters.device)
    return model
