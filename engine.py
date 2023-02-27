import argparse
import os
import typing
from typing import Union, List, Optional

import numpy as np
import torch

from baseline import generate, MAXIMUM_RES
from modules.models import CGRModel

pars = argparse.ArgumentParser()

pars.add_argument('--model-path', '--model-path', default=r'E:\CGRModel-checkpoints', type=str)
pars.add_argument('--prompts', '--prompts', type=str, nargs='+',
                  default='food cort for fast food restaurant,realistic,detailed,sharp,smooth')

pars.add_argument('--step', '--step', type=int, default=1)
pars.add_argument('--size', '--size', type=int, default=512)
opt = pars.parse_args()


def main(model_path: Union[str, os.PathLike], prompts: Union[str, List[str]], size: Optional[typing.Tuple],
         device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu',
         using_step: Optional[bool] = True,
         step: Optional[int] = 2):
    kwargs = dict(use_version=True, version='v4', use_realistic=False, size=size)
    data_type = torch.float32 if device != 'cuda' else torch.float16
    model = CGRModel.from_pretrained(model_path, torch_dtype=data_type).to(device)
    if isinstance(prompts, str):
        generate(prompt=prompts, model=model, **kwargs)
    elif isinstance(prompts, list):
        if using_step:
            prompts = np.array(prompts)
            bl = len(prompts) % step if len(prompts) > step else step - len(prompts)
            if len(prompts) % step != 0:
                v = np.array(['<BREAK>' for _ in range(bl)])
                prompts = np.concatenate((prompts, v), axis=0)
            prompts = prompts.reshape((len(prompts) // step if len(prompts) // step != 0 else 1, step))
            for i, prp in enumerate(prompts):

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
    main(model_path=r'{}'.format(opt.model_path), step=opt.step,
         device='cpu',
         prompts=opt.prompts, size=(opt.size, opt.size))
