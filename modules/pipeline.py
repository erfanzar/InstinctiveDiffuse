# IM using a part of transformers neural networks to make job faster
# and I edited them for my own usage in modules

import importlib
import inspect
import json
import os
from pathlib import PosixPath
from typing import Any, Callable, Dict, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
from diffusers.utils import logging, deprecate, is_transformers_available
from packaging import version
from tqdm import tqdm
from transformers import PreTrainedModel

from .pipeline_utils import FrozenDict

logger = logging.get_logger(__name__)
__version__: Optional[str] = '1.13.1'
LOADABLE_CLASSES: Optional[Dict] = {
    "diffusers": {
        "ModelMixin": ["save_static_model", "from_pretrained"],
        "SchedulerMixin": ["save_static_model", "from_pretrained"],
        "DiffusionPipeline": ["save_static_model", "from_pretrained"],
        "OnnxRuntimeModel": ["save_static_model", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_static_model", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_static_model", "from_pretrained"],
        "PreTrainedModel": ["save_static_model", "from_pretrained"],
        "FeatureExtractionMixin": ["save_static_model", "from_pretrained"],
        "ProcessorMixin": ["save_static_model", "from_pretrained"],
        "ImageProcessingMixin": ["save_static_model", "from_pretrained"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_static_model", "from_pretrained"],
    },
}

LOADABLE_CLASSES_V2: Optional[Dict] = {
    "diffusers": {
        "ModelMixin": ["save_static_model", "load_from_local"],
        "SchedulerMixin": ["save_static_model", "load_from_local"],
        "DiffusionPipeline": ["save_static_model", "load_from_local"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_static_model", "load_from_local"],
        "PreTrainedTokenizerFast": ["save_static_model", "load_from_local"],
        "PreTrainedModel": ["save_static_model", "load_from_local"],
        "ImageProcessingMixin": ["save_static_model", "load_from_local"],
    },
    "onnxruntime.training": {
        "ORTModule": ["save_static_model", "__"],
    },
}

ALL_IMPORTABLE_CLASSES: Optional[Dict] = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


class ConfigMixin:
    config_name = None
    ignore_for_config = []
    has_compatibles = False

    _deprecated_kwargs = []

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        # Special case for `kwargs` used in deprecation warning added to schedulers
        # TODO: remove this when we remove the deprecation warning, and the `kwargs` argument,
        # or solve in a more general way.
        kwargs.pop("kwargs", None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")

        self._internal_dict = FrozenDict(internal_dict)

    def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

    @classmethod
    def from_config(cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs):

        # TODO(Patrick) - make sure to remove the following lines when config=="model_path" is deprecated
        if "pretrained_model_name_or_path" in kwargs:
            config = kwargs.pop("pretrained_model_name_or_path")

        if config is None:
            raise ValueError("Please make sure to provide a config as the first positional argument.")

        if not isinstance(config, dict):
            deprecation_message = "It is deprecated to pass a pretrained model name or path to `from_config`."
            if "Scheduler" in cls.__name__:
                deprecation_message += (
                    f"If you were trying to load a scheduler, please use {cls}.load_static_model(...) instead."
                    " Otherwise, please make sure to pass a configuration dictionary instead. This functionality will"
                    " be removed in v1.0.0."
                )
            elif "Model" in cls.__name__:
                deprecation_message += (
                    f"If you were trying to load a model, please use {cls}.load_config(...) followed by"
                    f" {cls}.from_config(...) instead. Otherwise, please make sure to pass a configuration dictionary"
                    " instead. This functionality will be removed in v1.0.0."
                )
            deprecate("config-passed-as-path", "1.0.0", deprecation_message, standard_warn=False)
            config, kwargs = cls.load_config(pretrained_model_name_or_path=config, return_unused_kwargs=True, **kwargs)

        init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)

        # Allow dtype to be specified on initialization
        if "dtype" in unused_kwargs:
            init_dict["dtype"] = unused_kwargs.pop("dtype")

        # add possible deprecated kwargs
        for deprecated_kwarg in cls._deprecated_kwargs:
            if deprecated_kwarg in unused_kwargs:
                init_dict[deprecated_kwarg] = unused_kwargs.pop(deprecated_kwarg)

        # Return model and optionally state and/or unused_kwargs
        model = cls(**init_dict)

        # make sure to also save config parameters that might be used for compatible classes
        model.register_to_config(**hidden_dict)

        # add hidden kwargs of compatible classes to unused_kwargs
        unused_kwargs = {**unused_kwargs, **hidden_dict}

        if return_unused_kwargs:
            return (model, unused_kwargs)
        else:
            return model

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        deprecation_message = (
            f" The function get_config_dict is deprecated. Please use {cls}.load_config instead. This function will be"
            " removed in version v1.0.0"
        )
        deprecate("get_config_dict", "1.0.0", deprecation_message, standard_warn=False)
        return cls.load_config(*args, **kwargs)

    @classmethod
    def load_config(
            cls, pretrained_model_name_or_path: Union[str, os.PathLike], return_unused_kwargs=False, **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:

        cache_dir = kwargs.pop("cache_dir", './here/')
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", None)

        user_agent = {"file_type": "config"}

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        if cls.config_name is None:
            raise ValueError(
                "`self.config_name` is not defined. Note that one should not load a config from "
                "`ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
            )

        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, cls.config_name)):
                # Load from a PyTorch checkpoint
                config_file = os.path.join(pretrained_model_name_or_path, cls.config_name)
            elif subfolder is not None and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            ):
                config_file = os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            try:
                # Load from URL or cache if already cached
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=cls.config_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision,
                )

            except ValueError:
                print('ValueError cause was nothing found')

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(f"It looks like the config file at '{config_file}' is not a valid JSON file.")

        if return_unused_kwargs:
            return config_dict, kwargs

        return config_dict

    @staticmethod
    def _get_init_keys(cls):
        return set(dict(inspect.signature(cls.__init__).parameters).keys())

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        # 0. Copy origin config dict
        original_dict = {k: v for k, v in config_dict.items()}

        # 1. Retrieve expected config attributes from __init__ signature
        expected_keys = cls._get_init_keys(cls)
        expected_keys.remove("self")
        # remove general kwargs if present in dict
        if "kwargs" in expected_keys:
            expected_keys.remove("kwargs")
        # remove flax internal keys
        if hasattr(cls, "_flax_internal_args"):
            for arg in cls._flax_internal_args:
                expected_keys.remove(arg)

        # 2. Remove attributes that cannot be expected from expected config attributes
        # remove keys to be ignored
        if len(cls.ignore_for_config) > 0:
            expected_keys = expected_keys - set(cls.ignore_for_config)

        # load diffusers library to import compatible and original scheduler
        diffusers_library = importlib.import_module(__name__.split(".")[0])

        if cls.has_compatibles:
            compatible_classes = [c for c in cls._get_compatibles() if not isinstance(c, DummyObject)]
        else:
            compatible_classes = []

        expected_keys_comp_cls = set()
        for c in compatible_classes:
            expected_keys_c = cls._get_init_keys(c)
            expected_keys_comp_cls = expected_keys_comp_cls.union(expected_keys_c)
        expected_keys_comp_cls = expected_keys_comp_cls - cls._get_init_keys(cls)
        config_dict = {k: v for k, v in config_dict.items() if k not in expected_keys_comp_cls}

        # remove attributes from orig class that cannot be expected
        orig_cls_name = config_dict.pop("_class_name", cls.__name__)
        if orig_cls_name != cls.__name__ and hasattr(diffusers_library, orig_cls_name):
            orig_cls = getattr(diffusers_library, orig_cls_name)
            unexpected_keys_from_orig = cls._get_init_keys(orig_cls) - expected_keys
            config_dict = {k: v for k, v in config_dict.items() if k not in unexpected_keys_from_orig}

        # remove private attributes
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}

        # 3. Create keyword arguments that will be passed to __init__ from expected keyword arguments
        init_dict = {}
        for key in expected_keys:
            # if config param is passed to kwarg and is present in config dict
            # it should overwrite existing config dict key
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)

            if key in kwargs:
                # overwrite key
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                # use value from config dict
                init_dict[key] = config_dict.pop(key)

        # 4. Give nice warning if unexpected values have been passed
        if len(config_dict) > 0:
            logger.warning(
                f"The config attributes {config_dict} were passed to {cls.__name__}, "
                "but are not expected and will be ignored. Please verify your "
                f"{cls.config_name} configuration file."
            )

        # 5. Give nice info if config attributes are initiliazed to default because they have not been passed
        passed_keys = set(init_dict.keys())
        if len(expected_keys - passed_keys) > 0:
            logger.info(
                f"{expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
            )

        # 6. Define unused keyword arguments
        unused_kwargs = {**config_dict, **kwargs}

        # 7. Define "hidden" config parameters that were saved for compatible classes
        hidden_config_dict = {k: v for k, v in original_dict.items() if k not in init_dict}

        return init_dict, unused_kwargs, hidden_config_dict

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @property
    def config(self) -> Dict[str, Any]:

        return self._internal_dict

    def to_json_string(self) -> str:

        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = self.__class__.__name__
        config_dict["_diffusers_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, PosixPath):
                value = str(value)
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):

        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class PipeLine(ConfigMixin):
    config_name = "model_index.json"
    _optional_components = []

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        from diffusers import pipelines

        for name, module in kwargs.items():
            # retrieve library
            if module is None:
                register_dict = {name: (None, None)}
            else:
                library = module.__module__.split(".")[0]

                pipeline_dir = module.__module__.split(".")[-2] if len(module.__module__.split(".")) > 2 else None
                path = module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                if library not in LOADABLE_CLASSES or is_pipeline_module:
                    library = pipeline_dir

                # retrieve class_name
                class_name = module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def save_static_model(
            self,
            save_directory: Union[str, os.PathLike],
            safe_serialization: bool = False,
            variant: Optional[str] = None,
    ):

        self.save_config(save_directory)

        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_diffusers_version")
        model_index_dict.pop("_module", None)

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}

        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                library = importlib.import_module(library_name)
                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

    def to(self, torch_device: Optional[Union[str, torch.device]] = None):
        if torch_device is None:
            return self

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, torch.nn.Module):
                module.to(torch_device)
        return self

    @property
    def device(self) -> torch.device:
        r"""
        Returns:
            `torch.device`: The torch device on which the pipeline is located.
        """
        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for name in module_names.keys():
            module = getattr(self, name)
            if isinstance(module, torch.nn.Module):
                return module.device
        return torch.device("cpu")

    @classmethod
    def load_static_model(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):

        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)

        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        from_flax = kwargs.pop("from_flax", False)
        torch_dtype = kwargs.pop("torch_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)
        sess_options = kwargs.pop("sess_options", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        return_cached_folder = kwargs.pop("return_cached_folder", False)
        variant = kwargs.pop("variant", None)

        if not os.path.isdir(pretrained_model_name_or_path):
            config_dict = cls.load_config(
                pretrained_model_name_or_path,
                # cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                # local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )

            # retrieve all folder_names that contain relevant files
            folder_names = [k for k, v in config_dict.items() if isinstance(v, list)]

            if not local_files_only:
                info = model_info(
                    pretrained_model_name_or_path,
                    use_auth_token=use_auth_token,
                    revision=revision,
                )
                model_filenames, variant_filenames = variant_compatible_siblings(info, variant=variant)
                model_folder_names = set([os.path.split(f)[0] for f in model_filenames])

                if revision in DEPRECATED_REVISION_ARGS and version.parse(
                        version.parse(__version__).base_version
                ) >= version.parse("0.15.0"):
                    info = model_info(
                        pretrained_model_name_or_path,
                        use_auth_token=use_auth_token,
                        revision=None,
                    )
                    comp_model_filenames, _ = variant_compatible_siblings(info, variant=revision)
                    comp_model_filenames = [
                        ".".join(f.split(".")[:1] + f.split(".")[2:]) for f in comp_model_filenames
                    ]

                    if set(comp_model_filenames) == set(model_filenames):
                        warnings.warn(
                            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` even though you can load it via `variant=`{revision}`. Loading model variants via `revision='{variant}'` is deprecated and will be removed in diffusers v1. Please use `variant='{revision}'` instead.",
                            FutureWarning,
                        )
                    else:
                        warnings.warn(
                            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have the required variant filenames in the 'main' branch. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {revision} files' so that the correct variant file can be added.",
                            FutureWarning,
                        )

                # all filenames compatible with variant will be added
                allow_patterns = list(model_filenames)

                # allow all patterns from non-model folders
                # this enables downloading schedulers, tokenizers, ...
                allow_patterns += [os.path.join(k, "*") for k in folder_names if k not in model_folder_names]
                # also allow downloading config.jsons with the model
                allow_patterns += [os.path.join(k, "*.json") for k in model_folder_names]

                allow_patterns += [
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                    cls.config_name,
                    CUSTOM_PIPELINE_FILE_NAME,
                ]

                if from_flax:
                    ignore_patterns = ["*.bin", "*.safetensors", ".onnx"]
                elif is_safetensors_available() and is_safetensors_compatible(model_filenames, variant=variant):
                    ignore_patterns = ["*.bin", "*.msgpack"]

                    safetensors_variant_filenames = set([f for f in variant_filenames if f.endswith(".safetensors")])
                    safetensors_model_filenames = set([f for f in model_filenames if f.endswith(".safetensors")])
                    if (
                            len(safetensors_variant_filenames) > 0
                            and safetensors_model_filenames != safetensors_variant_filenames
                    ):
                        logger.warn(
                            f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(safetensors_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(safetensors_model_filenames - safetensors_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                        )

                else:
                    ignore_patterns = ["*.safetensors", "*.msgpack"]

                    bin_variant_filenames = set([f for f in variant_filenames if f.endswith(".bin")])
                    bin_model_filenames = set([f for f in model_filenames if f.endswith(".bin")])
                    if len(bin_variant_filenames) > 0 and bin_model_filenames != bin_variant_filenames:
                        logger.warn(
                            f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(bin_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(bin_model_filenames - bin_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                        )

            else:
                # allow everything since it has to be downloaded anyways
                ignore_patterns = allow_patterns = None

            if cls != DiffusionPipeline:
                requested_pipeline_class = cls.__name__
            else:
                requested_pipeline_class = config_dict.get("_class_name", cls.__name__)
            user_agent = {"pipeline_class": requested_pipeline_class}
            if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
                user_agent["custom_pipeline"] = custom_pipeline

            user_agent = http_user_agent(user_agent)

            # download all allow_patterns
            cached_folder = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )
        else:
            cached_folder = pretrained_model_name_or_path
            config_dict = cls.load_config(cached_folder)

        # retrieve which subfolders should load variants
        model_variants = {}
        if variant is not None:
            for folder in os.listdir(cached_folder):
                folder_path = os.path.join(cached_folder, folder)
                is_folder = os.path.isdir(folder_path) and folder in config_dict
                variant_exists = is_folder and any(path.split(".")[1] == variant for path in os.listdir(folder_path))
                if variant_exists:
                    model_variants[folder] = variant

        # 2. Load the pipeline class, if using custom module then load it from the hub
        # if we load from explicit class, let's use it
        if custom_pipeline is not None:
            if custom_pipeline.endswith(".py"):
                path = Path(custom_pipeline)
                # decompose into folder & file
                file_name = path.name
                custom_pipeline = path.parent.absolute()
            else:
                file_name = CUSTOM_PIPELINE_FILE_NAME

            pipeline_class = get_class_from_dynamic_module(
                custom_pipeline, module_file=file_name, cache_dir=cache_dir, revision=custom_revision
            )
        elif cls != PipeLine:
            pipeline_class = cls
        else:
            diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
            pipeline_class = getattr(diffusers_module, config_dict["_class_name"])

        # To be removed in 1.0.0
        if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
                version.parse(config_dict["_diffusers_version"]).base_version
        ) <= version.parse("0.5.1"):
            from diffusers import StableDiffusionInpaintPipelineLegacy

            pipeline_class = StableDiffusionInpaintPipelineLegacy

        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs
        init_kwargs = {k: init_dict.pop(k) for k in optional_kwargs if k in init_dict}
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Special case: safety_checker must be loaded separately when using `from_flax`

        if low_cpu_mem_usage:
            low_cpu_mem_usage = False

        from diffusers import pipelines
        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            # 3.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            if class_name.startswith("Flax"):
                class_name = class_name[4:]

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                if not is_pipeline_module:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if class_candidate is not None and issubclass(class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}"
                        )
                else:
                    logger.warning(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type"
                    )

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = getattr(pipeline_module, class_name)
                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {c: class_obj for c in importable_classes.keys()}
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)

                class_obj = getattr(library, class_name)
                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

            if loaded_sub_model is None:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                if load_method_name is None:
                    none_module = class_obj.__module__
                    is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER) or none_module.startswith(
                        TRANSFORMERS_DUMMY_MODULES_FOLDER
                    )
                    if is_dummy_path and "dummy" in none_module:
                        # call class_obj for nice error message of missing requirements
                        class_obj()

                    raise ValueError(
                        f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
                        f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
                    )

                load_method = getattr(class_obj, load_method_name)
                loading_kwargs = {}

                if issubclass(class_obj, torch.nn.Module):
                    loading_kwargs["torch_dtype"] = torch_dtype
                import diffusers
                if issubclass(class_obj, diffusers.OnnxRuntimeModel):
                    loading_kwargs["provider"] = provider
                    loading_kwargs["sess_options"] = sess_options

                is_diffusers_model = issubclass(class_obj, diffusers.ModelMixin)

                if is_transformers_available():
                    import transformers
                    transformers_version = version.parse(version.parse(transformers.__version__).base_version)
                else:
                    transformers_version = "N/A"

                is_transformers_model = (
                        is_transformers_available()
                        and issubclass(class_obj, PreTrainedModel)
                        and transformers_version >= version.parse("4.20.0")
                )


                if is_diffusers_model or is_transformers_model:
                    loading_kwargs["device_map"] = device_map
                    loading_kwargs["variant"] = model_variants.pop(name, None)
                    if from_flax:
                        loading_kwargs["from_flax"] = True

                    if (
                            is_transformers_model
                            and loading_kwargs["variant"] is not None
                            and transformers_version < version.parse("4.27.0")
                    ):
                        raise ImportError(
                            f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
                        )
                    elif is_transformers_model and loading_kwargs["variant"] is None:
                        loading_kwargs.pop("variant")

                    if not (from_flax and is_transformers_model):
                        loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
                    else:
                        loading_kwargs["low_cpu_mem_usage"] = False

                if os.path.isdir(os.path.join(cached_folder, name)):
                    loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
                else:

                    loaded_sub_model = load_method(cached_folder, **loading_kwargs)

            init_kwargs[name] = loaded_sub_model

        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 5. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        if return_cached_folder:
            return model, cached_folder
        return model

    @staticmethod
    def _get_signature_keys(obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - set(["self"])
        return expected_modules, optional_parameters

    @property
    def components(self) -> Dict[str, Any]:
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components.keys()} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):

        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):

        self.set_use_memory_efficient_attention_xformers(False)

    def set_use_memory_efficient_attention_xformers(
            self, valid: bool, attention_op: Optional[Callable] = None
    ) -> None:

        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for module_name in module_names:
            module = getattr(self, module_name)
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):

        self.set_attention_slice(slice_size)

    def disable_attention_slicing(self):

        self.enable_attention_slicing(None)

    def set_attention_slice(self, slice_size: Optional[int]):
        module_names, _, _ = self.extract_init_dict(dict(self.config))
        for module_name in module_names:
            module = getattr(self, module_name)
            if isinstance(module, torch.nn.Module) and hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size)
