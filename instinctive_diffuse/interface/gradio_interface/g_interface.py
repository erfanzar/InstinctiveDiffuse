import gradio as gr
from gradio.themes.utils import colors

from ...sampler import TextToImageSampler
from ..statics import (
    RESOLUTION_INFO,
    ADDITIONAL_IMAGE_OPTIONS_INFO,
    ADDITIONAL_IMAGE_OPTIONS,
    CAMERA_ANGLE_SUPPORTS_INFO,
    CAMERA_ANGLE_SUPPORTS,
    ABOUT_US_MARKDOWN
)
from typing import Optional, Tuple, List


class GradioUserInterface:
    def __init__(
            self, sampler: TextToImageSampler
    ):
        self.sampler = sampler

    def generate(
            self,
            prompt: str,
            options: Optional[List[str]] = None,
            resolution: Optional[Tuple[int, int]] = None,
            camera_position: Optional[str] = None,
            progress_bar=gr.Progress()
    ):
        if camera_position is None:
            camera_position = ""
        if resolution is None:
            resolution = [512, 512]
        if options is None:
            options = []
        string_options = ""
        for option in options:
            string_options += f", {option}"
        string_options += f", Camera Position {camera_position}"
        model_input_prompt = prompt + string_options
        generated_sample, nsfw_concept = self.sampler.model(
            prompt=model_input_prompt,
            height=resolution[0],
            width=resolution[1],
            progress_bar=progress_bar,
            return_dict=False
        )
        return "", generated_sample

    def _create_generation_components(self):
        with gr.Row():
            with gr.Column(scale=1):
                camera_position = gr.Dropdown(
                    CAMERA_ANGLE_SUPPORTS,
                    value=CAMERA_ANGLE_SUPPORTS[-2],
                    label="Camera Position",
                    max_choices=1,
                    info=CAMERA_ANGLE_SUPPORTS_INFO
                )
                options = gr.CheckboxGroup(
                    choices=ADDITIONAL_IMAGE_OPTIONS,
                    info=ADDITIONAL_IMAGE_OPTIONS_INFO,
                    label="Generate Options"
                )
                resolution = gr.Slider(
                    label="Resolution",
                    value=512,
                    maximum=8192,
                    minimum=256,
                    step=8,
                    info=RESOLUTION_INFO
                )

            with gr.Column(scale=4):
                gr.Markdown(
                    ABOUT_US_MARKDOWN,
                )
                output_image = gr.Image(
                    label="Generated Image",
                    container=True,
                    height=740
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        prompt = gr.Textbox(placeholder="A Dragon Flying above clouds", show_label=True,
                                            label="Prompt")
                    with gr.Column(scale=1):
                        button = gr.Button("Generate Image")
                        clean = gr.Button("Clean")
                        stop = gr.Button("Stop")
        inputs = [
            prompt,
            options,
            resolution,
            camera_position,
        ]
        process_b_num1 = button.click(
            fn=self.generate,
            inputs=inputs,
            outputs=[prompt, output_image],
            preprocess=False
        )
        process_b_num2 = prompt.submit(
            fn=self.generate,
            inputs=inputs,
            outputs=[prompt, output_image],
            preprocess=False
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[process_b_num1, process_b_num2]
        )
        clean.click(fn=lambda: "", outputs=[prompt])

    def create(self):
        with gr.Blocks(
                theme=gr.themes.Soft(
                    primary_hue=colors.orange,
                    secondary_hue=colors.orange,
                ),
                title="InstinctAI",
        ) as block:
            with gr.Tab("Text-to-Image"):
                self._create_generation_components()
        return block
