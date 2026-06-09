# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for rigid articulated assets."""

import sys, os
from pathlib import Path
import torch
_cudacri_dir = Path(__file__).resolve().parent
_lib_dir = _cudacri_dir / "lib"
sys.path.insert(0, str(_lib_dir))
_torch_lib = Path(torch.__file__).parent / "lib"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_lib_dir}:{_torch_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

from .articulation import Articulation
from .articulation_cfg import ArticulationCfg
from .articulation_data import ArticulationData
