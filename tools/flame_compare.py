"""Compare flame analytic vs raymarch rendering pixel-by-pixel.

Runs the engine in batch mode three times (analytic twice for a determinism
mask, raymarch once), diffs the two methods outside the noise mask, and writes
metrics JSON plus an amplified diff heatmap. Works with whatever scene the
engine loads because only --batch-flame-mode differs between runs.

Usage:
    uv run --with pillow --with numpy python3 tools/flame_compare.py \
        [--frames 60] [--steps 128] [--out-dir /tmp/thyllore_screenshots/flame_compare]

Exit code 0 = comparison ran (verdict is in the JSON line on stdout).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

MEAN_DIFF_THRESHOLD_LSB = 2.0
MAX_DIFF_THRESHOLD_LSB = 8.0
HEATMAP_GAIN = 16


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def engine_path() -> Path:
    for profile in ("release", "debug"):
        candidate = repo_root() / "target" / profile / "thyllore-animation"
        if candidate.is_file():
            return candidate
    raise SystemExit("engine not built: cargo build --bin thyllore-animation")


def engine_env() -> dict[str, str]:
    env = dict(os.environ)
    candidates = sorted((repo_root() / "vendor" / "onnxruntime").glob("*/lib/libonnxruntime.so"))
    if candidates:
        env.setdefault("ORT_DYLIB_PATH", str(candidates[-1]))
    return env


def capture(output: Path, mode: str, frames: int, steps: int) -> None:
    command = [
        str(engine_path()),
        "--batch-screenshot", str(output),
        "--batch-frames", str(frames),
        "--batch-flame-mode", mode,
        "--batch-flame-steps", str(steps),
    ]
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=300,
        cwd=str(repo_root()), env=engine_env(),
    )
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    result = json.loads(last_line) if last_line.startswith("{") else {"ok": False, "error": "no JSON"}
    if not result.get("ok"):
        raise SystemExit(f"capture failed ({mode}): {result.get('error')}")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "thyllore_screenshots" / "flame_compare",
    )
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    analytic_a = out / "analytic_a.png"
    analytic_b = out / "analytic_b.png"
    raymarch = out / "raymarch.png"

    capture(analytic_a, "analytic", args.frames, args.steps)
    capture(analytic_b, "analytic", args.frames, args.steps)
    capture(raymarch, "raymarch", args.frames, args.steps)

    image_a = load_rgb(analytic_a)
    image_b = load_rgb(analytic_b)
    image_r = load_rgb(raymarch)

    noise_mask = np.any(image_a != image_b, axis=2)
    diff = np.abs(image_a - image_r).max(axis=2)
    diff[noise_mask] = 0

    changed = diff > 0
    mean_diff = float(diff[changed].mean()) if changed.any() else 0.0
    max_diff = int(diff.max())
    changed_pixels = int(changed.sum())

    heatmap = np.clip(diff.astype(np.int32) * HEATMAP_GAIN, 0, 255).astype(np.uint8)
    heatmap_rgb = np.stack([heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap)], axis=2)
    heatmap_path = out / "diff_heatmap.png"
    Image.fromarray(heatmap_rgb).save(heatmap_path)

    passed = mean_diff <= MEAN_DIFF_THRESHOLD_LSB and max_diff <= MAX_DIFF_THRESHOLD_LSB
    print(json.dumps({
        "ok": True,
        "pass": passed,
        "mean_diff_lsb": round(mean_diff, 3),
        "max_diff_lsb": max_diff,
        "changed_pixels": changed_pixels,
        "noise_masked_pixels": int(noise_mask.sum()),
        "thresholds": {"mean": MEAN_DIFF_THRESHOLD_LSB, "max": MAX_DIFF_THRESHOLD_LSB},
        "heatmap": str(heatmap_path),
        "analytic": str(analytic_a),
        "raymarch": str(raymarch),
    }, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
