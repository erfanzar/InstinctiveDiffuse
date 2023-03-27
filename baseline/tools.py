import copy
import os.path
import typing

import erutils
import erutils.lightning

from modules.models import CGRModel

VERSIONS: typing.Optional[list[str]] = [
    'v2', 'v3', 'v4'
]
BLACK_WORDS_LIST: typing.Optional[list[str]] = [
    'matin'
]
MAXIMUM_RES = 880

ALLOWED_SAVE_FORMATS: typing.Optional[list[str]] = [
    'png', 'jpg'
]

TASKS: typing.Optional[list[str]] = [
    'PIL', 'dict', 'cv', 'np', 'save', 'check'
]


# VERSIONS = [
#     'v2', 'v3', 'v4'
# ]
# BLACK_WORDS_LIST = [
#     'matin'
# ]
#
# ALLOWED_SAVE_FORMATS = [
#     'png', 'jpg'
# ]
#
# TASKS = [
#     'PIL', 'dict', 'cv', 'np', 'save', 'check'
# ]


def using_v4(prompt: typing.Optional[str]) -> typing.Optional[str]:
    prompt += ', mdjrny-v4 style'
    return prompt


def using_v3(prompt: typing.Optional[str]) -> typing.Optional[str]:
    prompt += ', mdjrny-v3 style'
    return prompt


def using_v2(prompt: typing.Optional[str]) -> typing.Optional[str]:
    prompt += ', mdjrny-v2 style'
    return prompt


def using_realistic(prompt: typing.Optional[str]) -> typing.Optional[str]:
    prompt += ', realistic style'
    return prompt


def run_save(prompt: typing.Union[str, list[str]], model: typing.Optional[CGRModel]) -> None:
    predictions = model(prompt).images[0]
    erutils.fprint(f'Writing Predictions To PNG format as an image into file [{prompt}.png]')
    predictions.save(f'{prompt}.png')
    return None


def check_prompt(prompt: typing.Optional[str]) -> typing.Union[str, None]:
    """

    :param prompt: prompt to be checked if some words are not available in that
    :return: prompt or None
    """
    _s = prompt.split()
    if _s not in BLACK_WORDS_LIST:
        return prompt
    else:
        erutils.fprint('BLACK WORDS DETECTED', erutils.Cp.RED)
        return None


def v_to_prompt(prompt: typing.Union[str, typing.List[str]], use_check_prompt, use_version, version, use_realistic):
    if use_check_prompt:
        prompt = check_prompt(prompt)
    if prompt is not None:
        if version in VERSIONS:

            if use_version:
                if version == VERSIONS[0]:
                    prompt = using_v2(prompt)
                elif version == VERSIONS[1]:
                    prompt = using_v3(prompt)
                elif version == VERSIONS[2]:
                    prompt = using_v4(prompt)
        if use_realistic:
            prompt = using_realistic(prompt)
    return prompt


def generate(prompt: typing.Union[str, list[str]], model: typing.Optional[CGRModel],
             size: typing.Optional[typing.Tuple] = None, out_dir: typing.Optional[str] = 'out',
             use_version: typing.Optional[bool] = True, version: typing.Optional[str] = 'v4',
             use_realistic: typing.Optional[bool] = False, image_format: typing.Optional[str] = 'png',
             nsfw_allowed: typing.Optional[bool] = False,
             use_check_prompt:
             typing.Optional[bool] = False, task: typing.Optional[str] = 'save'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if task == 'save' and image_format not in ALLOWED_SAVE_FORMATS:
        raise ValueError(
            f'stop execution of command cause provided format not available in formats : \n '
            f'[provided : {image_format}  |  formats :{ALLOWED_SAVE_FORMATS}]'
        )
    if image_format not in ALLOWED_SAVE_FORMATS:
        raise Warning(f'provided format not available in formats : \n '
                      f'[provided : {image_format}  |  formats :{ALLOWED_SAVE_FORMATS}]')
    if use_version and version is None:
        raise ValueError(
            f'use_version has been set to True but not version provided [use_version : {use_version} |'
            f' version : {version}]')
    elif use_version and use_realistic:
        raise ValueError(
            f'you only can use custom versions or realistic s you can"t use both of them'
            f'at last at this version'
        )
    else:
        ...
    if task not in TASKS:
        raise ValueError(f'specified task should be in list of tasks \n {TASKS}')
    org_p = copy.deepcopy(prompt)

    size = (None, None) if size is None else size
    if isinstance(prompt, str):
        prompt = v_to_prompt(prompt, use_realistic=use_realistic, use_version=use_version, version=version,
                             use_check_prompt=use_check_prompt)
    elif isinstance(prompt, list):
        prompt = [v_to_prompt(p, use_realistic=use_realistic, use_version=use_version, version=version,
                              use_check_prompt=use_check_prompt) for p in prompt]
    else:
        raise ValueError('Wrong input for prompt input should be string or a list of strings')

    for i, iva in enumerate(model(prompt=prompt, height=size[0], width=size[1], nsfw_allowed=nsfw_allowed)):
        generated_sample = iva

        if isinstance(iva, int):
            if iva < 50:
                yield iva

    if task == 'PIL':
        return generated_sample.images[0]
    elif task == 'dict':
        return dict(
            image=generated_sample.images,
            content=generated_sample,
            size=None,
            nsfw=False,
        )
    elif task == 'cv':
        return NotImplementedError
    elif task == 'np':
        return NotImplementedError
    elif task == 'check':
        return True
    elif task == 'save':
        try:
            print(generated_sample.nsfw_content_detected)
            if isinstance(org_p, list):
                for v in range(len(generated_sample.images)):
                    generated_sample.images[v].save(f'{out_dir}/{org_p[v]}.{image_format}')
            elif isinstance(org_p, str):
                for v in range(len(generated_sample.images)):
                    generated_sample.images[v].save(f'{out_dir}/{org_p}.{image_format}')
            return True
        except Warning as w:
            erutils.fprint(w)
            return False
    else:
        raise ValueError('selected task is going to be available in future')
