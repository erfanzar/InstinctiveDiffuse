import argparse
import os
import typing
from typing import Union, List, Optional

import numpy as np
from tqdm.auto import tqdm
import torch

from baseline import generate, MAXIMUM_RES
from modules.models import CGRModel

pars = argparse.ArgumentParser()

pars.add_argument('--model-path', '--model-path', default=r'E:\CGRModel-checkpoints', type=str)
pars.add_argument('--prompts', '--prompts', type=str, nargs='+',
                  default='prompts.txt')
pars.add_argument('--save-dir', '--save-dir', type=str, default='out')
pars.add_argument('--step', '--step', type=int, default=4)
pars.add_argument('--device', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
pars.add_argument('--size', '--size', type=int, default=512)
opt = pars.parse_args()


def config_model(model_path: Union[str, os.PathLike],
                 device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu',
                 nsfw_allowed: Optional[bool] = True):
    # data_type = torch.float32 if device != 'cuda' else torch.float16
    # model = CGRModel.from_pretrained(model_path, torch_dtype=data_type).to(device)
    # if nsfw_allowed:
    #     model.safety_checker.to('cpu')
    #
    # return model
    return None


def main(model: Optional[CGRModel], prompts: Union[str, List[str], os.PathLike], size: Optional[typing.Tuple],
         using_step: Optional[bool] = True, nsfw_allowed: Optional[bool] = True, out_dir: Optional[str] = 'out',
         step: Optional[int] = 2):
    kwargs = dict(use_version=True, version='v4', use_realistic=False, size=size, nsfw_allowed=nsfw_allowed,
                  out_dir=out_dir)

    if isinstance(prompts, str) and not prompts.endswith('.txt'):

        generate(prompt=prompts, model=model, **kwargs)

    elif isinstance(prompts, list) or prompts.endswith('.txt'):

        if prompts.endswith('.txt'):
            us_a: bool = True
            prompts = open(prompts, 'r', encoding='utf8').readlines()
        else:
            us_a: bool = False
        if using_step:
            prompts = np.array(prompts)
            bl = len(prompts) % step if len(prompts) > step else step - len(prompts)
            if us_a:
                bl += 2
            if len(prompts) % step != 0:
                v = np.array(['<BREAK>' for _ in range(bl)])
                prompts = np.concatenate((prompts, v), axis=0)
            prompts = prompts.reshape((len(prompts) // step if len(prompts) // step != 0 else 1, step))
            for i, prp in tqdm(enumerate(prompts)):
                # print(prp)
                if '<BREAK>' in prp:
                    fx = 0
                    for ix, p in enumerate(prp):
                        if p == '<BREAK>':
                            fx = ix
                            break
                    prp = prp[:fx]
                generate(prompt=prp.tolist(), model=model, **kwargs)
        else:
            generate(prompt=prompts, model=model, **kwargs)


if __name__ == "__main__":
    if opt.size > MAXIMUM_RES:
        raise ValueError(
            f'You tried to get image with size {opt.size} but our model currently work at maximum {MAXIMUM_RES} try lower resolution')
    grc = config_model(model_path=r'{}'.format(opt.model_path), nsfw_allowed=True, device=opt.device)
    main(model=grc, step=opt.step,
         prompts=opt.prompts, size=(opt.size, opt.size))
