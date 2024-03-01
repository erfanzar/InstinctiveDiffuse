from typing import Optional
from dataclasses import dataclass

import torch


@dataclass
class SamplerConfig:
    dtype: str = "fp16"

    use_prefix_tokenizer: bool = True  # JAX Only Option
    pre_compile: bool = True  # JAX Only Option

    use_mxn_break_point: bool = True
    max_number_of_gpus: Optional[int] = None
    max_gpu_perc_to_use: float = 0.95

    def __post_init__(self):
        if self.dtype not in ["fp16", "fp32", "bf16"]:
            raise ValueError("unknown dtype has been passed")

    def get_torch_dtype(self):
        if self.dtype == "fp16":
            return torch.float16
        elif self.dtype == "bf16":
            return torch.bfloat16
        elif self.dtype == "fp32":
            return torch.float32
        else:
            raise ValueError("Unknown dtype passed")
