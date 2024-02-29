from typing import Optional
from dataclasses import dataclass


@dataclass
class SamplerConfig:
    dtype: str = "fp16"

    stream_tokens_for_gradio: bool = True
    use_prefix_tokenizer: bool = True  # JAX Only Option
    pre_compile: bool = True  # JAX Only Option

    use_mxn_break_point: bool = True
    max_number_of_gpus: Optional[int] = None
    max_gpu_perc_to_use: float = 0.95
