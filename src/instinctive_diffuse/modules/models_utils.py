from enum import Enum


class KarrasDiffusionSchedulers(Enum):
    DDIMScheduler = 1
    DDPMScheduler = 2
    PNDMScheduler = 3
    LMSDiscreteScheduler = 4
    EulerDiscreteScheduler = 5
    HeunDiscreteScheduler = 6
    EulerAncestralDiscreteScheduler = 7
    DPMSolverMultistepScheduler = 8
    DPMSolverSinglestepScheduler = 9
    KDPM2DiscreteScheduler = 10
    KDPM2AncestralDiscreteScheduler = 11
    DEISMultistepScheduler = 12
    UniPCMultistepScheduler = 13
