from .sampler_config import SamplerConfig
from .stable_diffusion_pipeline import InstinctStableDiffusionPipeLine
import torch
import os
from typing import Union, Optional


def config_model(model_path: Union[str, os.PathLike], data_type: torch.dtype = torch.float16):
    available_gpu_memory = get_gpu_memory()

    ck = {
        'device_map': 'auto',
        'max_memory': {i: str(int(available_gpu_memory[i] * 0.95)) + "GiB" for i in range(len(available_gpu_memory))},
        'torch_dtype': torch.float16
    }
    print(ck)
    print('Loading Stage Two')

    model_ = InstinctStableDiffusionPipeLine.from_pretrained(model_path, **ck)
    if args.uba:
        model_.scheduler = DPMSolverMultistepScheduler.from_config(model_.scheduler.config)
    return model_


class TextToImageSampler:
    def __init__(
            self,
            model: InstinctStableDiffusionPipeLine,
            config: SamplerConfig

    ):
        self.model = model
        self.config = config

    @classmethod
    def get_load_kwargs(
            cls,
            torch_dtype: torch.dtype = torch.float16,
            num_gpus: Optional[int] = None,
            max_gpu_prc_to_use: float = 0.95,
            device_map: str = "auto"
    ):
        available_gpu_memory = cls.get_gpu_memory(num_gpus=num_gpus)
        return {
            "device_map": device_map,
            "max_memory": {
                i: str(
                    int(available_gpu_memory[i] * max_gpu_prc_to_use)
                ) + "GiB" for i in range(
                    len(available_gpu_memory)
                )
            },
            "torch_dtype": torch_dtype
        }

    @staticmethod
    def get_gpu_memory(num_gpus=None):
        gpu_m = []
        dc = torch.cuda.device_count()
        num_gpus = torch.cuda.device_count() if num_gpus is None else min(num_gpus, dc)

        for gpu_id in range(num_gpus):
            with torch.cuda.device(gpu_id):
                gpu_properties = torch.cuda.get_device_properties(torch.cuda.current_device())
                gpu_m.append(
                    (gpu_properties.total_memory / (1024 ** 3)) - (torch.cuda.memory_allocated() / (1024 ** 3)))
        return gpu_m

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path,
            sampler_config: SamplerConfig,
            **kwargs
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        use_onnx = kwargs.pop("use_onnx", None)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)
        kw = cls.get_load_kwargs(
            torch_dtype=sampler_config.get_torch_dtype(),
            num_gpus=sampler_config.max_number_of_gpus,
            device_map="auto",
            max_gpu_prc_to_use=sampler_config.max_gpu_perc_to_use
        )

        if max_memory is None:
            max_memory = kw["max_memory"]
        if device_map is None:
            device_map = kw["device_map"]
        if torch_dtype is None:
            torch_dtype = kw["torch_dtype"]

        model = InstinctStableDiffusionPipeLine.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            from_flax=from_flax,
            torch_dtype=torch_dtype,
            custom_pipeline=custom_pipeline,
            custom_revision=custom_revision,
            provider=provider,
            sess_options=sess_options,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            low_cpu_mem_usage=low_cpu_mem_usage,
            variant=variant,
            use_safetensors=use_safetensors,
            use_onnx=use_onnx,
            load_connected_pipeline=load_connected_pipeline,
        )

        return cls(
            model=model,
            config=sampler_config
        )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
