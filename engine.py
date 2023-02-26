import os
from typing import Union, List, Optional
import numpy as np

import torch
import argparse
from baseline import generate
from modules.models import CGRModel

pars = argparse.ArgumentParser()

pars.add_argument('--model-path', '--model-path', default=r'E:\CGRModel-checkpoints', type=str)

opt = pars.parse_args()


def main(model_path: Union[str, os.PathLike], prompts: Union[str, List[str]],
         device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu', using_step: Optional[bool] = True,
         step: Optional[int] = 2):
    kwargs = dict(use_version=True, version='v4', use_realistic=False)
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
                generate(prompt=prp, model=model, **kwargs)
        else:
            generate(prompt=prompts, model=model, **kwargs)


if __name__ == "__main__":
    main(model_path=r'{}'.format(opt.model_path),
         prompts=['welcome to CreativeGan', 'welcomed and beautiful happy woman face,detailed,sharp',
                  'an astronaut riding a horse ,detailed,sharp',
                  'beautiful stars in the galaxy space ,detailed,detailed',
                  'a musician woman on the piano ,detailed,sharp',
                  'portrait of female draconian, intricate, elegant, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha, 8k', ]
         )
