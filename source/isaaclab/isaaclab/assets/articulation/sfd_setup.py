"""IsaacLab articulation 등에서 sfd_coreservice 최소 bootstrap.

articulation/__init__.py 맨 위 (ArticulationData import 전) 에 configure_cudacri 호출.

lib/ 번들에는 libcrypto++.so.8, libjsoncpp.so.25 등이 포함된다.
TensorRT: Engine/.../model_fp16.engine 우선, deserialize 실패 시 같은 폴더 model.onnx 로 런타임 빌드.
Isaac Sim: export SAFETICS_TRT_PREFER_ONNX=1 로 engine 건너뛰기 가능.
cmake --build build --target isaaclab_deploy 로 patchelf 적용 lib/·Engine/ 을 IsaacLab articulation 에 배포.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def apply_spike_mitigation_env_early() -> None:
    """Set before first PyTorch CUDA allocation in this process (tail latency)."""
    if os.environ.get("SFD_SPIKE_MITIGATION", "1") in ("0", "false", "FALSE"):
        return
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.9",
    )
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    os.environ.setdefault("SAFETICS_USE_TENSOR_CALC_LOOP", "1")


def configure_cudacri(cudacri_dir: str | Path) -> Path:
    import torch

    apply_spike_mitigation_env_early()

    root = Path(cudacri_dir).resolve()
    lib_dir = root / "lib"
    if not lib_dir.is_dir():
        raise FileNotFoundError(f"CUDACRI lib not found: {lib_dir}")

    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p
        for p in (
            str(lib_dir),
            str(torch_lib),
            os.environ.get("LD_LIBRARY_PATH", ""),
        )
        if p
    )
    return lib_dir
