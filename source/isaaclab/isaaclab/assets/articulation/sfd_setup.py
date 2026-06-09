"""IsaacLab articulation 등에서 sfd_coreservice 최소 bootstrap.

articulation/__init__.py 맨 위 (ArticulationData import 전) 에 넣을 코드:

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

그 다음 모듈 어디서든:

    import sfd_coreservice

lib/ 번들에는 libcrypto++.so.8, libjsoncpp.so.25 등이 포함된다.
TensorRT engine 은 CUDACRI/Engine/Chest_Shape5_Cover_off_{Force,Pressure}/model_fp16.engine 에 둔다.
cmake --build build --target sfd_coreservice 로 lib/·Engine/ 을 갱신한 뒤 IsaacLab articulation 에 복사한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_cudacri(cudacri_dir: str | Path) -> Path:
    import torch

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
