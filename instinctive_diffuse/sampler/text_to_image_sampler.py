from .sampler_config import SamplerConfig
from diffusers import StableDiffusionPipeline


class TextToImageSampler:
    def __init__(
            self,
            model: StableDiffusionPipeline,
            config: SamplerConfig

    ): ...
