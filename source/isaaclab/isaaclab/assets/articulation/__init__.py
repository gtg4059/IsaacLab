# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for rigid articulated assets."""

import os
import sys
from pathlib import Path

import torch

_cudacri_dir = Path(__file__).resolve().parent
_lib_dir = _cudacri_dir / "lib"
_torch_lib = Path(torch.__file__).parent / "lib"

for path in (_cudacri_dir, _lib_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ["LD_LIBRARY_PATH"] = ":".join(
    p
    for p in (
        str(_lib_dir),
        str(_torch_lib),
        os.environ.get("LD_LIBRARY_PATH", ""),
    )
    if p
)

from sfd_setup import configure_cudacri  # noqa: E402

configure_cudacri(_cudacri_dir)

from .articulation import Articulation
from .articulation_cfg import ArticulationCfg
from .articulation_data import ArticulationData
