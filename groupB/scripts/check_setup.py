#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

GROUP_DIR = Path(__file__).resolve().parents[1]
THIRD_PARTY = GROUP_DIR / "third_party"

REPOSITORIES = ["openvla-oft", "LIBERO", "VLA-Trace"]


def status(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def check_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    failed = False

    print(f"Python: {platform.python_version()}")
    python_ok = sys.version_info >= (3, 10)
    print(f"Python >= 3.10: {status(python_ok)}")
    failed |= not python_ok

    for package in ["torch", "numpy", "yaml"]:
        ok = check_import(package)
        print(f"Import {package}: {status(ok)}")
        failed |= not ok

    for repo in REPOSITORIES:
        ok = (THIRD_PARTY / repo / ".git").is_dir()
        print(f"Repository {repo}: {status(ok)}")
        failed |= not ok

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("GPU:")
            print(result.stdout.strip())
        except subprocess.CalledProcessError as exc:
            print(f"nvidia-smi failed: {exc}")
            failed = True
    else:
        print("nvidia-smi: MISSING")
        failed = True

    if check_import("torch"):
        import torch

        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime: {torch.version.cuda}")
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        else:
            failed = True

    print("\nResult:", "PASS" if not failed else "CHECK FAILED ITEMS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
