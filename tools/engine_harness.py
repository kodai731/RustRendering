"""Engine launch helpers shared by the tools/ scripts (standard library only)."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

DOOD_IMAGE = "thyllore-screenshot-harness:local"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def engine_env() -> dict[str, str]:
    env = dict(os.environ)
    candidates = sorted((repo_root() / "vendor" / "onnxruntime").glob("*/lib/libonnxruntime.so"))
    if candidates:
        env.setdefault("ORT_DYLIB_PATH", str(candidates[-1]))
    return env


def resolve_engine_override(override: str) -> Path:
    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    candidate = candidate.resolve()

    if not candidate.is_file():
        raise SystemExit(f"THYLLORE_ENGINE points at a missing file: {candidate}")
    if not os.access(candidate, os.X_OK):
        raise SystemExit(f"THYLLORE_ENGINE is not executable: {candidate}")
    return candidate


def engine_path() -> Path:
    override = os.environ.get("THYLLORE_ENGINE", "").strip()
    if override:
        return resolve_engine_override(override)

    for profile in ("release", "debug"):
        candidate = repo_root() / "target" / profile / "thyllore-animation"
        if candidate.is_file():
            return candidate
    raise SystemExit("engine not built: cargo build --bin thyllore-animation")


def dood_wrap(command: list[str]) -> list[str]:
    root = str(repo_root())
    ort = "vendor/onnxruntime/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so"
    inner = f"ORT_DYLIB_PATH={ort} " + shlex.join(command)
    return [
        "docker", "run", "--rm", "--entrypoint", "bash", "--hostname", "kodai-computer",
        "-v", f"{root}:{root}", "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
        "-v", "/run/user/1000/gdm/Xauthority:/xauth:ro",
        "-e", "XAUTHORITY=/xauth", "-e", "DISPLAY=:1",
        "--device", "/dev/dri", "--group-add", "992", "-w", root,
        DOOD_IMAGE, "-c", inner,
    ]
