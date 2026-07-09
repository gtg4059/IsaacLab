# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print system hardware and software specifications.

Usage:

```bash
python scripts/tools/print_system_specs.py
```
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import psutil


def _format_bytes(num_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.2f} {unit}" if unit != "B" else f"{num_bytes} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def _print_os_info() -> None:
    _print_section("Operating System")
    print(f"  Hostname     : {platform.node()}")
    print(f"  System       : {platform.system()} {platform.release()}")
    print(f"  Version      : {platform.version()}")
    print(f"  Architecture : {platform.machine()}")
    print(f"  Processor    : {platform.processor() or 'N/A'}")


def _print_cpu_info() -> None:
    _print_section("CPU")
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    print(f"  Physical cores : {physical_cores}")
    print(f"  Logical cores  : {logical_cores}")
    if freq is not None and freq.current > 100:
        print(f"  Frequency      : {freq.current:.0f} MHz (max {freq.max:.0f} MHz)")

    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    print(f"  Model          : {line.split(':', 1)[1].strip()}")
                    break
    except FileNotFoundError:
        pass


def _print_memory_info() -> None:
    _print_section("Memory")
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"  Total RAM      : {_format_bytes(vm.total)}")
    print(f"  Available RAM  : {_format_bytes(vm.available)}")
    print(f"  Used RAM       : {_format_bytes(vm.used)} ({vm.percent:.1f}%)")
    print(f"  Swap total     : {_format_bytes(swap.total)}")
    print(f"  Swap used      : {_format_bytes(swap.used)} ({swap.percent:.1f}%)")


def _print_disk_info() -> None:
    _print_section("Disk")
    root = Path("/")
    usage = shutil.disk_usage(root)
    print(f"  Mount point    : {root.resolve()}")
    print(f"  Total          : {_format_bytes(usage.total)}")
    print(f"  Used           : {_format_bytes(usage.used)} ({usage.used / usage.total * 100:.1f}%)")
    print(f"  Free           : {_format_bytes(usage.free)}")


def _print_gpu_info() -> None:
    _print_section("GPU")
    if shutil.which("nvidia-smi") is None:
        print("  nvidia-smi not found.")
        return

    query_fields = "index,name,memory.total,driver_version,compute_cap"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  Failed to query GPU info: {exc.stderr.strip()}")
        return

    for line in result.stdout.strip().splitlines():
        index, name, memory_total, driver_version, compute_cap = [part.strip() for part in line.split(",")]
        print(f"  GPU {index}")
        print(f"    Name           : {name}")
        print(f"    VRAM           : {memory_total}")
        print(f"    Driver         : {driver_version}")
        print(f"    Compute Cap.   : {compute_cap}")


def _print_python_info() -> None:
    _print_section("Python / CUDA")
    print(f"  Python         : {sys.version.split()[0]} ({sys.executable})")

    try:
        import torch

        print(f"  PyTorch        : {torch.__version__}")
        print(f"  CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version   : {torch.version.cuda}")
            print(f"  cuDNN version  : {torch.backends.cudnn.version()}")
            for device_id in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_id)
                print(f"  CUDA device {device_id} : {props.name} ({_format_bytes(props.total_memory)})")
    except ImportError:
        print("  PyTorch        : not installed")


def print_system_specs() -> None:
    """Collect and print system specifications."""
    print("System Specifications")
    _print_os_info()
    _print_cpu_info()
    _print_memory_info()
    _print_disk_info()
    _print_gpu_info()
    _print_python_info()
    print()


def main() -> None:
    """Entry point."""
    print_system_specs()


if __name__ == "__main__":
    main()
