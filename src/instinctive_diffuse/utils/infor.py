from enum import Enum
from typing import List, Optional


class Version(Enum):
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"


class AllowedSaveFormat(Enum):
    PNG = "png"
    JPG = "jpg"


class Task(Enum):
    pil = "PIL"
    dict = "dict"
    cv = "cv"
    np = "np"
    save = "save"
    check = "check"


class Platform(Enum):
    JAX = "JAX"
    TORCH = "TORCH"



