import gradio as gr
import tqdm
from transformers import logging
from diffusers import StableDiffusionPipeline
from typing import Union, Optional, List
import torch
import os
from engine import config_model as cm
from baseline import gradio_generate
import time

AUTH_TOKEN = os.getenv('AUTH_TOKEN', 'NONE')


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


load_model = True
model = config_model(model_path='erfanzar/StableGAN', device='cuda', nsfw_allowed=False,
                     data_type=torch.float16) if load_model else None


def run(options, prompt, resolution, num_samples, pr=gr.Progress()):
    resolution = resolution if resolution < 640 else 640
    options = ','.join(osa.lower() for osa in options)
    options = ' ' + options + ', NFT Collection, NFT, nft'
    prompt += options
    images = []
    for _ in pr.tqdm(range(num_samples)):
        image = gradio_generate(model=model, prompt=prompt, size=(resolution, resolution), use_version=True,
                                nsfw_allowed=False, use_realistic=False,
                                use_check_prompt=False, task='PIL', use_bar=False)
        images.append(image)
        # time.sleep(1)
    return '', images


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            options_ = gr.CheckboxGroup(choices=[
                'Real',
                'Realistic',
                'Smooth',
                'Sharp',
                'Realistic,smooth',
                '8k',
                'Detailed',
                'Simplified',
                'Davin-chi',
                'CameraMan',
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
            resolution_ = gr.Slider(label='Resolution', value=416,
                                    maximum=640, minimum=256, step=8,
                                    info='Resolution to be passed to AI to generate image with that resolution '
                                         'the minimum resolution is 256x256 and the maximum is 640x640 which '
                                         'our current servers wont support more than 640x640 images cause of lak of'
                                         ' Compute Unit and GPU Power')
            collection_sample = gr.Slider(label='Collection Images', value=12,
                                          maximum=40, minimum=1, step=1,
                                          info='Number of Images in Collection currently you can\'t generate '
                                               'collections more than 40 images')

        with gr.Column(scale=4, variant='box'):
            gr.Markdown(
                '# DreamCollection Powered By StableGAN from LucidBrains 🧠\n'
                '## About LucidBrains(AlmubdieunTech)\n'
                'LucidBrains is a platform that makes AI accessible and easy to use for everyone.'
                ' Our mission is to empower individuals and businesses'
                'with the tools they need to harness the power of AI and machine learning,'
                ' without requiring a background in data science or anything we will just build'
                ' what you want for you and help you to have better time and living life with using'
                ' Artificial Intelligence and Pushing Technology Beyond Limits'
            )
            image_class_ = gr.Gallery(label='Generated Collection').style(container=True)
            progress_bar = gr.Progress(track_tqdm=True)
            with gr.Row():
                with gr.Column(scale=4):
                    text_box_ = gr.Textbox(placeholder='Hamster Skeleton', show_label=True, label='Collection Name')
                with gr.Column(scale=1):
                    button_ = gr.Button('Generate Image')
                    stop = gr.Button('Stop')

    inputs = [options_, text_box_, resolution_, collection_sample]
    button_click = button_.click(fn=run, inputs=inputs,
                                 outputs=[text_box_, image_class_], preprocess=False)
    text_box_submit = text_box_.submit(fn=run, inputs=inputs,
                                       outputs=[text_box_, image_class_], preprocess=False)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[button_click, text_box_submit])

demo.queue().launch(share=False, show_tips=False, show_error=True)
