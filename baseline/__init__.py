from importlib.util import find_spec


def spec_is_available(spec: str):
    res = find_spec(spec)
    if res is not None:
        return True
    else:
        return False


if spec_is_available('tensorflow'):
    try:
        from .baseline import baseline_load
    except ModuleNotFoundError:
        print('Import Error ')
from .baseline_utils import HyperParameters

from .tools import generate, TASKS, VERSIONS, ALLOWED_SAVE_FORMATS, BLACK_WORDS_LIST, MAXIMUM_RES, gradio_generate
