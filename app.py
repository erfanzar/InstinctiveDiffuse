import gradio as gr
from transformers import logging
from baseline import generate
from functools import partial

from engine import config_model

logger = logging.get_logger(__name__)
logging.set_verbosity_debug()
logger.setLevel('DEBUG')
model = config_model(model_path='', device='cpu', nsfw_allowed=False)
c_generate = partial(generate, use_version=True,
                     version='v4', nsfw_allowed=False,
                     use_check_prompt=False,
                     )


def run(options, prompt, data_type, device, resolution, generate_noise):
    print(f'OPTIONS : {options}\nPROMPT : {prompt}\nDATA TYPE : {data_type}\nDEVICE : {device}\n'
          f'RESOLUTION : {resolution}\nGENERATE NOISE : {generate_noise}')
    return ''


# if __name__ == "__main__":
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            data_type_ = gr.Dropdown(choices=['Float 32', 'BFloat 16', 'Float 16', 'TF32'], value='Float 16',
                                     label='Data Type',
                                     allow_custom_value=False, multiselect=False,
                                     info='The Data type for AI to Generate Image Model will use Float 16 by default'
                                          ' cause many GPUS dont support BFloat 16 and TF32')
            device_ = gr.Dropdown(choices=['TPU', 'CUDA', 'CPU'], value='CUDA', label='Device ',
                                  allow_custom_value=False,
                                  multiselect=False,
                                  info='The Accelerator to be used to Generate image its on GPU or CUDA by default ')
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
                'CameraMan'
            ], info='The Modes that will AI takes in as the Idea to Generate Image From them And its required'
                    ' a lot of playing around to know which Options '
                    'working good with each other or which is good to use')
            resolution_ = gr.Slider(label='Resolution', value=768,
                                    maximum=4096, minimum=512, step=1,
                                    info='Resolution to be passed to AI to generate image with that resolution '
                                         'the minimum resolution is 512x512 and the maximum is 4094x4094 which '
                                         'our current servers wont support more than 860x860 images cause of lak of'
                                         ' Compute Unit and GPU Power')
            noise_ = gr.Slider(label='Generate Noise', value=0.0,
                               maximum=1.0, minimum=0.0, step=0.01,
                               info='Generate to be passed to AI in main algorythm '
                                    'this will be ignored no matter what you use '
                                    'cause that will make bad results for users and its'
                                    ' only available in debug mode')

        with gr.Column(scale=4):
            image_class_ = gr.AnnotatedImage(label='Generated Image').style(height=640)
            with gr.Row():
                with gr.Column(scale=4):
                    text_box_ = gr.Textbox(placeholder='A Dragon Flying above clouds', show_label=True, label='Prompt')
                with gr.Column(scale=1):
                    button_ = gr.Button('Generate Image')
                    clean_ = gr.Button('Clean')

    button_.click(fn=run, inputs=[options_, text_box_, data_type_, device_, resolution_, noise_], outputs=[text_box_])
    text_box_.submit(fn=run, inputs=[options_, text_box_, data_type_, device_, resolution_, noise_],
                     outputs=[text_box_])
    clean_.click(fn=lambda _: '', outputs=[text_box_], inputs=[noise_])
demo.queue().launch()
