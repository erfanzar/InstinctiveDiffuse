import gradio as gr
from transformers import logging
from baseline import gradio_generate

from diffusers import StableDiffusionPipeline
from typing import Union, Optional, List
import torch
import os
from engine import config_model as cm
import argparse

parse = argparse.ArgumentParser(description='DreamCafe')

parse.add_argument('-m', '--model_path', default='erfanzar/StableGAN')
args = parse.parse_args()
model_name = args.model_path

logger = logging.get_logger(__name__)
logging.set_verbosity_error()
logger.setLevel('ERROR')

AUTH_TOKEN = os.getenv('AUTH_TOKEN', 'NONE')


def get_data_type(spec):
    if spec == 'Float 16':
        return torch.float16
    elif spec == 'BFloat 16':
        return torch.bfloat16
    elif spec == 'Float 32':
        return torch.float32
    elif spec == 'TF32':
        # Not supported YET
        return torch.float32
    else:
        raise ValueError


def get_device(spec):
    if spec == 'CPU':
        return 'cpu'
    if spec == 'CUDA':
        return 'cuda:0'
    if spec == 'TPU':
        return 'xla:0'
    else:
        raise


def config_model(model_path: Union[str, os.PathLike],
                 device: Union[torch.device, str] = 'cuda' if torch.cuda.is_available() else 'cpu',
                 nsfw_allowed: Optional[bool] = True, data_type: torch.dtype = torch.float32):
    ck = dict(use_auth_token=AUTH_TOKEN) if AUTH_TOKEN != "NONE" else dict()
    if not nsfw_allowed:
        if device == 'cuda' or device == 'cpu':
            model_ = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=data_type,
                                                             **ck).to(device)
        elif device == 'auto':
            model_ = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=data_type,
                                                             device_map=device, **ck)
        else:
            raise ValueError
    else:
        model_ = cm(model_path, 'cuda', True, torch.float16)
    return model_


model = config_model(model_path=model_name, device='cuda', nsfw_allowed=False, data_type=torch.float16)


# model = None


# c_generate = partial(generate, use_version=True,
#                      nsfw_allowed=False,
#                      use_check_prompt=False, task='PIL'
#                      )


def run(options, prompt, data_type, device, resolution, generate_noise):
    resolution = resolution if resolution < 880 else 880
    print(f'OPTIONS : {options}\nPROMPT : {prompt}\nDATA TYPE : {data_type}\nDEVICE : {device}\n'
          f'RESOLUTION : {resolution}\nGENERATE NOISE : {generate_noise}')

    options = ' ' + ','.join(o.lower() for o in options)
    prompt += options

    print(f'PROMPT : {prompt}')
    image = gradio_generate(model=model, prompt=prompt, size=(resolution, resolution), use_version=True,
                            nsfw_allowed=False, use_realistic=False,
                            use_check_prompt=False, task='PIL', use_bar=False)

    return '', image


# if __name__ == "__main__":
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            data_type_ = gr.Dropdown(choices=['Float 32', 'BFloat 16', 'Float 16', 'TF32'], value='Float 16',
                                     label='Data Type',
                                     allow_custom_value=False, multiselect=False,
                                     info='The Data type for AI to Generate Image Model will use Float 16 by default'
                                          ' cause many GPUS dont support BFloat 16 and TF32 '
                                          '[This state doesnt make change for users ITs a Loading Option Only]')
            device_ = gr.Dropdown(choices=['TPU', 'CUDA', 'CPU'], value='CUDA', label='Device ',
                                  allow_custom_value=False,
                                  multiselect=False,
                                  info='The Accelerator to be used to Generate image its on GPU or CUDA by default '
                                       '[This state doesnt make change for users ITs a Loading Option Only]',
                                  visible=True)
            options_ = gr.CheckboxGroup(choices=[
                'Real',
                'Realistic',
                'Smooth',
                'Sharp',
                'Realistic,smooth',
                '8k',
                'Detailed',
                'Smart',
                'Version 4',
                'Atheistic',
                'Simplified',
                'Davin-chi',
                'CameraMan',
                'Midjourney Style',
                'HyperRealistic',
                'Octane Render',
                'Cinematic Lighting',
                'Cinematic Quality',
                'Dark Cyan And Light Crimson',
                'DreamLike',
                'Artstation',
                'Volumetric Lighting',
                'AlbumCovers'
            ], info='The Modes that will AI takes in as the Idea to Generate Image From them And its required'
                    ' a lot of playing around to know which Options '
                    'working good with each other or which is good to use', label='Generate Options')
            resolution_ = gr.Slider(label='Resolution', value=512,
                                    maximum=4096, minimum=256, step=8,
                                    info='Resolution to be passed to AI to generate image with that resolution '
                                         'the minimum resolution is 256x256 and the maximum is 4094x4094 which '
                                         'our current servers wont support more than 860x860 images cause of lak of'
                                         ' Compute Unit and GPU Power')
            noise_ = gr.Slider(label='Generate Noise', value=0.0,
                               maximum=1.0, minimum=0.0, step=0.01,
                               info='Generate to be passed to AI in main algorythm '
                                    'this will be ignored no matter what you use '
                                    'cause that will make bad results for users and its'
                                    ' only available in debug mode')

        with gr.Column(scale=4, variant='box'):
            gr.Markdown(
                '# DreamCafe Powered By StableGAN from LucidBrains 🧠\n'
                '## About LucidBrains(AlmubdieunTech)\n'
                'LucidBrains is a platform that makes AI accessible and easy to use for everyone.'
                ' Our mission is to empower individuals and businesses'
                'with the tools they need to harness the power of AI and machine learning,'
                ' without requiring a background in data science or anything we will just build'
                ' what you want for you and help you to have better time and living life with using'
                ' Artificial Intelligence and Pushing Technology Beyond Limits'
            )
            image_class_ = gr.Image(label='Generated Image').style(container=True, height=860, )
            with gr.Row():
                progress_bar = gr.Progress(track_tqdm=True, )
                with gr.Column(scale=4):
                    text_box_ = gr.Textbox(placeholder='A Dragon Flying above clouds', show_label=True, label='Prompt')
                with gr.Column(scale=1):
                    button_ = gr.Button('Generate Image')
                    clean_ = gr.Button('Clean')

    button_.click(fn=run, inputs=[options_, text_box_, data_type_, device_, resolution_, noise_],
                  outputs=[text_box_, image_class_], preprocess=False)
    text_box_.submit(fn=run, inputs=[options_, text_box_, data_type_, device_, resolution_, noise_],
                     outputs=[text_box_, image_class_], preprocess=False)
    clean_.click(fn=lambda _: '', outputs=[text_box_], inputs=[noise_])
demo.queue().launch(share=True, show_tips=False, show_error=True)
