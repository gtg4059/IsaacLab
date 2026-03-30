from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from isaaclab.actuators import ActuatorBaseCfg
from .RFI_PDActuator import RFI_PDActuator


@configclass
class RFI_PDActuatorCfg(ActuatorBaseCfg):
    """Configuration for an ideal PD actuator."""

    class_type: type = RFI_PDActuator

    rfi: tuple[float, float] = MISSING
    """Random force/torque input to the actuator. Defaults to None."""
    
    min_delay: int = 0
    """Minimum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""

    max_delay: int = 0
    """Maximum number of physics time-steps with which the actuator command may be delayed. Defaults to 0."""