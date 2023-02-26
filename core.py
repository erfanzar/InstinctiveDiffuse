import os
from typing import Union

from modules.models import CGRModel


def main():
    model_path: Union[str, os.PathLike] = r'E:\CGRModel-checkpoints'
    model = CGRModel.from_pretrained(model_path)
    prompt = 'something that everybody love , realistic'
    pred = model(prompt)
    pred.images[0].save('pred.png')
    print(model)


if __name__ == "__main__":
    main()
