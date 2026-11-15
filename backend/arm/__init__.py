"""Arm skill server. make_arm() gives FakeArm unless PHANTOM_ARM=real, then RealArm (COM3).

    from backend.arm import make_arm, validate, SKILLS
    arm = make_arm()
    arm.run("grip_bottle", {"pace": 0.8, "hold_s": 5}, on_phase=print)
    arm.run("fetch_bottle", {"pace": 0.6, "wait_s": 3}, on_phase=print)
"""
import os

from .fake import FakeArm
from .skills import OK_RESULTS, PHASES, RESULTS, SKILLS, STYLES, Arm, nominal_duration, validate

__all__ = ["make_arm", "FakeArm", "RealArm", "SKILLS", "STYLES", "PHASES", "RESULTS", "OK_RESULTS", "Arm",
           "validate", "nominal_duration"]


def make_arm(**kwargs):
    if os.environ.get("PHANTOM_ARM") == "real":
        from .real import RealArm
        return RealArm(**kwargs)
    return FakeArm(**kwargs)


def __getattr__(name):
    # RealArm is imported only when asked for, so importing this package never loads the serial code
    if name == "RealArm":
        from .real import RealArm
        return RealArm
    raise AttributeError(name)
