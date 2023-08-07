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


def config_model(model_path: Union[str, os.PathLike], data_type: torch.dtype = torch.float16):
    def get_gpu_memory(num_gpus_req=None):
        gpu_m = []
        dc = torch.cuda.device_count()
        num_gpus = torch.cuda.device_count() if num_gpus_req is None else min(num_gpus_req, dc)

        for gpu_id in range(num_gpus):
            with torch.cuda.device(gpu_id):
                gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_m.append(
                    (gpu_properties.total_memory / (1024 ** 3)) - (torch.cuda.memory_allocated() / (1024 ** 3)))
        return gpu_m

    available_gpu_memory = get_gpu_memory()

    ck = {
        'device_map': 'auto',
        'max_memory': {i: str(int(available_gpu_memory[i] * 0.90)) + "GiB" for i in range(len(available_gpu_memory))},
        'torch_dtype': data_type,
        "use_auth_token": AUTH_TOKEN if AUTH_TOKEN != "NONE" else False
    }
    print(ck)
    print('Loading Stage Two')
    model_ = StableDiffusionPipeline.from_pretrained(model_path,
                                                     **ck)

    return model_


print('Loading Stage One')
model = config_model(model_path=model_name, data_type=torch.float16)


def run(options, prompt, resolution, camera_position):
    resolution = resolution if resolution < 880 else 880
    options = ' ' + ','.join(o.lower() for o in options)
    prompt += options + f'camera is position {camera_position} | {camera_position}'
    image = gradio_generate(model=model, prompt=prompt, size=(resolution, resolution), use_version=True,
                            nsfw_allowed=False, use_realistic=False,
                            use_check_prompt=False, task='PIL', use_bar=False)

    return '', image


# if __name__ == "__main__":
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            camera_pos = gr.Dropdown(
                [
                    'Low-Angle',
                    'High-Angle',
                    'Extreme Low-Angle',
                    'Extreme High-Angle',
                    'Side-Angle',
                    'Birds-Eye-View',
                    'Eye-Level',
                    'Closeup Shot',
                    'Extreme Closeup Shot',
                    'Medium-Full Shot',
                    'Full-Body Shot',
                    'Combining Camera Angle + Shot Type',
                    'Centered View, Low-Angle, Extreme Closeup',
                    'Side View, Low-Angle, Closeup',
                    'High-Angle, Shot From Behind',
                    'High-Angle, Closeup',
                    'Reaction Shot',
                    'Point-of-View Shot',
                    'Extreme Close-up Shot',
                    "Bird's Eye View",
                    'Establishing Shot',
                    'Far Distance',
                    'Cowboy Shot',
                    'American Shot'
                ],
                value='Side-Angle',
                label='Camera Position',
                max_choices=1,
                info='Camera Positioning for Image'
            )
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
            image_class_ = gr.Image(label='Generated Image').style(container=True, height=740, )
            with gr.Row():
                progress_bar = gr.Progress(track_tqdm=True, )
                with gr.Column(scale=4):
                    text_box_ = gr.Textbox(placeholder='A Dragon Flying above clouds', show_label=True, label='Prompt')
                with gr.Column(scale=1):
                    button_ = gr.Button('Generate Image')
                    clean_ = gr.Button('Clean')
                    stop = gr.Button('Stop')
    inputs = [options_, text_box_, resolution_, camera_pos]
    c1 = button_.click(fn=run, inputs=inputs,
                       outputs=[text_box_, image_class_], preprocess=False)
    c2 = text_box_.submit(fn=run, inputs=inputs,
                          outputs=[text_box_, image_class_], preprocess=False)
    stop.click(fn=None, inputs=None, outputs=None, cancels=[c1, c2])
    clean_.click(fn=lambda _: '', outputs=[text_box_], inputs=[noise_])
demo.queue().launch(share=True, show_tips=False, show_error=True)
