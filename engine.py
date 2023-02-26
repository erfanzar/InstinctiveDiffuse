import os
from typing import Union, List

import torch

from baseline import generate
from modules.models import CGRModel


def main(model_path: Union[str, os.PathLike], prompts: Union[str, List[str]]):
    model = CGRModel.from_pretrained(model_path, torch_dtype=torch.float32)
    # model.to('cuda')
    generate(prompt=prompts, model=model, use_version=True, version='v4', use_realistic=False, )


if __name__ == "__main__":
    main(model_path=r'E:\CGRModel-checkpoints',
         prompts=['welcome to CreativeGan', 'welcomed and beautiful happy face,detailed,sharp'])
