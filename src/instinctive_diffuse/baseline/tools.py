import copy
import os.path
from typing import Optional, List, Tuple, Dict, Literal, Union
from importlib.util import find_spec
from diffusers import StableDiffusionPipeline


def spec_is_available(spec: str):
    res = find_spec(spec)
    if res is not None:
        return True
    else:
        return False


VERSIONS: Optional[list[str]] = [
    "v2", "v3", "v4"
]
BLACK_WORDS_LIST: Optional[list[str]] = []
MAXIMUM_RES = 880

ALLOWED_SAVE_FORMATS: Optional[list[str]] = [
    "png", "jpg"
]

TASKS: Optional[list[str]] = ["PIL", "dict", "cv", "np", "save", "check"]


def generate(
        prompt: Union[str, list[str]],
        model,
        size: Optional[Tuple] = None,
        out_dir: Optional[str] = "out",
        use_version: Optional[bool] = True,
        version: Optional[str] = "v4",
        use_realistic: Optional[bool] = False,
        image_format: Optional[str] = "png",
        nsfw_allowed: Optional[bool] = False,
        task: List[str] = "save",
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if task == "save" and image_format not in ALLOWED_SAVE_FORMATS:
        raise ValueError(
            f"stop execution of command cause provided format not available in formats : \n "
            f"[provided : {image_format}  |  formats :{ALLOWED_SAVE_FORMATS}]"
        )
    if image_format not in ALLOWED_SAVE_FORMATS:
        raise Warning(f"provided format not available in formats : \n "
                      f"[provided : {image_format}  |  formats :{ALLOWED_SAVE_FORMATS}]")
    if use_version and version is None:
        raise ValueError(
            f"use_version has been set to True but not version provided [use_version : {use_version} |"
            f" version : {version}]")
    elif use_version and use_realistic:
        raise ValueError(
            f"you only can use custom versions or realistic s you can't use both of them"
            f"at last at this version"
        )
    else:
        ...
    if task not in TASKS:
        raise ValueError(f"specified task should be in list of tasks \n {TASKS}")
    org_p = copy.deepcopy(prompt)

    size = (None, None) if size is None else size
    kwargs = dict(height=size[0], width=size[1], nsfw_allowed=nsfw_allowed) \
        if not isinstance(model,
                          StableDiffusionPipeline) else dict(
        height=size[0], width=size[1])

    for i, iva in enumerate(model(prompt=prompt, **kwargs)):
        generated_sample = iva

        if isinstance(iva, int):
            if iva < 50:
                yield iva
    if task == "PIL":
        return generated_sample.images[0]
    elif task == "dict":
        return dict(
            image=generated_sample.images,
            content=generated_sample,
            size=None,
            nsfw=False,
        )
    elif task == "cv":
        return NotImplementedError()
    elif task == "np":
        return NotImplementedError()
    elif task == "check":
        return True
    elif task == "save":
        try:
            print(generated_sample.nsfw_content_detected)
            if isinstance(org_p, list):
                for v in range(len(generated_sample.images)):
                    generated_sample.images[v].save(f"{out_dir}/{org_p[v]}.{image_format}")
            elif isinstance(org_p, str):
                for v in range(len(generated_sample.images)):
                    generated_sample.images[v].save(f"{out_dir}/{org_p}.{image_format}")
            return True
        except Warning as w:
            print(w)
            return False
    else:
        raise ValueError("selected task is going to be available in future")


def gradio_generate(
        prompt: Union[str, list[str]],
        model,
        size: Optional[Tuple] = None,
        out_dir: Optional[str] = "out",
        use_version: Optional[bool] = True,
        version: Optional[str] = "v4",
        use_realistic: Optional[bool] = False,
        image_format: Optional[str] = "png",
        nsfw_allowed: Optional[bool] = False,
        use_check_prompt:
        Optional[bool] = False,
        task: Optional[str] = "save",
        use_bar=False
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if task == "save" and image_format not in ALLOWED_SAVE_FORMATS:
        raise ValueError(
            f"stop execution of command cause provided format not available in formats : \n "
            f"[provided : {image_format}  |  formats :{ALLOWED_SAVE_FORMATS}]"
        )
    if image_format not in ALLOWED_SAVE_FORMATS:
        raise Warning(f"provided format not available in formats : \n "
                      f"[provided : {image_format}  |  formats :{ALLOWED_SAVE_FORMATS}]")
    if use_version and version is None:
        raise ValueError(
            f"use_version has been set to True but not version provided [use_version : {use_version} |"
            f" version : {version}]")
    elif use_version and use_realistic:
        raise ValueError(
            f"you only can use custom versions or realistic s you can't use both of them"
            f"at last at this version"
        )
    else:
        ...
    if task not in TASKS:
        raise ValueError(f"specified task should be in list of tasks \n {TASKS}")
    org_p = copy.deepcopy(prompt)

    size = (None, None) if size is None else size
    kwargs = dict(height=size[0], width=size[1], nsfw_allowed=nsfw_allowed) \
        if not isinstance(model,
                          StableDiffusionPipeline) else dict(
        height=size[0], width=size[1])

    generated_sample = model(prompt=prompt, **kwargs)
    if task == "PIL":
        return generated_sample.images[0]
    elif task == "dict":
        return dict(
            image=generated_sample.images,
            content=generated_sample,
            size=None,
            nsfw=False,
        )
    elif task == "cv":
        return NotImplementedError()
    elif task == "np":
        return NotImplementedError()
    elif task == "check":
        return True
    elif task == "save":
        try:
            print(generated_sample.nsfw_content_detected)
            if isinstance(org_p, list):
                for v in range(len(generated_sample.images)):
                    generated_sample.images[v].save(f"{out_dir}/{org_p[v]}.{image_format}")
            elif isinstance(org_p, str):
                for v in range(len(generated_sample.images)):
                    generated_sample.images[v].save(f"{out_dir}/{org_p}.{image_format}")
            return True
        except Warning as w:
            print(w)
            return False
    else:
        raise ValueError("selected task is going to be available in future")
