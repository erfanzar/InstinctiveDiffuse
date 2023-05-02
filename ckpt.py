import os
from pathlib import Path
import logging
import requests
from tqdm.auto import tqdm
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def downloader(url):
    try:
        filesize = int(requests.head(url).headers["Content-Length"])
    except KeyError:
        filesize = None
    filename = os.path.basename(url)
    with requests.get(url, stream=True) as r, \
            open(filename, "wb") as f, \
            tqdm(
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=filesize,
                colour='red',
                desc=f'Downloading {filename}'
            ) as progress:
        for chunk in r.iter_content(chunk_size=1024):
            datasize = f.write(chunk)
            progress.update(datasize)

main_model_json = 'model_index.json'

struct = {
    'feature_extractor': [
        'preprocessor_config.json'
    ],
    'safety_checker': [
        'config.json',
        'pytorch_model.bin'
    ],
    'scheduler': [
        'scheduler_config.json'
    ],
    'text_encoder': [
        'config.json',
        'pytorch_model.bin'
    ],
    "tokenizer": [
        'vocab.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'merges.txt'
    ],
    'unet': [
        'config.json',
        'diffusion_pytorch_model.bin'
    ],
    'vae': [
        'config.json',
        'diffusion_pytorch_model.bin'
    ]

}

main_directory = 'StableCKPT'
path_main_directory = Path(main_directory)


def get_ckpt(url: str = 'https://aihub.game4study.com/Checkpoints/CGRModel-checkpoints/'):
    if not path_main_directory.exists():
        path_main_directory.mkdir()
    old = os.getcwd()
    os.chdir(path_main_directory)
    downloader(url + main_model_json)
    os.chdir(old)
    for fld, ind in struct.items():
        for id_ in ind:
            fld_path = Path(main_directory + '/' + fld)
            if not fld_path.exists():
                fld_path.mkdir()
            old = os.getcwd()
            os.chdir(fld_path)
            get_url = url + fld + '/' + id_
            logger.info(f'Download {id_} in {fld_path} from {get_url}')
            downloader(get_url)
            os.chdir(old)


if __name__ == "__main__":
    get_ckpt()
