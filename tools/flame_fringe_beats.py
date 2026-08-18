"""Beat-table analysis of a fringe_probe patch (see design 20260808).

Reads a fringe_probe patch JSON and reports:
  - the measured fringe wavevector(s) from the FFT of the noise field
  - each mode's screen-space frequency via the local Jacobian dw/d(sample)
  - the beat table: mode pairs whose |f_i - f_j| matches the measured fringe

CLI:
    python3 tools/flame_fringe_beats.py --patch <fringe_patch.json> [--top-pairs 12] [--top-modes 8] [--tol 0.004]

Prints ONE json line on stdout.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


def load_patch(path: str):
    d = json.load(open(path))
    res = d["meta"]["res"]
    rays = d["rays"]
    k = np.array(d["meta"]["mode_k"], dtype=np.float64)
    w = np.array([x["w"] for x in rays], dtype=np.float64).reshape(res, res, 3)
    noise = np.array([x["noise"] for x in rays], dtype=np.float64).reshape(res, res)
    contrib = np.array([x["mode_contrib"] for x in rays], dtype=np.float64).reshape(res, res, -1)
    return res, k, w, noise, contrib


def measured_peaks(noise: np.ndarray, res: int, count: int = 3):
    sub = noise - noise.mean()
    win = np.hanning(res)[:, None] * np.hanning(res)[None, :]
    power = np.abs(np.fft.fftshift(np.fft.fft2(sub * win))) ** 2
    c = res // 2
    power[c - 2:c + 3, c - 2:c + 3] = 0
    peaks = []
    work = power.copy()
    for _ in range(count):
        i = np.unravel_index(np.argmax(work), work.shape)
        fy, fx = (i[0] - c) / res, (i[1] - c) / res
        peaks.append({
            "fx": float(fx), "fy": float(fy),
            "period_samples": float(1.0 / max(math.hypot(fx, fy), 1e-12)),
            "angle_deg": float(math.degrees(math.atan2(fy, fx))),
            "power": float(work[i]),
        })
        work[max(0, i[0] - 3):i[0] + 4, max(0, i[1] - 3):i[1] + 4] = 0
    return peaks


def screen_frequencies(k: np.ndarray, w: np.ndarray, res: int):
    """Mode n's screen frequency in cycles/sample: (k_n . dw/dsample) / 2pi."""
    grad_y, grad_x = np.gradient(w, axis=(0, 1))
    mid = slice(res // 4, 3 * res // 4)
    jx = np.median(grad_x[mid, mid].reshape(-1, 3), axis=0)
    jy = np.median(grad_y[mid, mid].reshape(-1, 3), axis=0)
    return (k @ jx) / (2 * math.pi), (k @ jy) / (2 * math.pi), jx, jy


def beat_table(sx, sy, strength, peak, tol: float, limit: int):
    matches = []
    n = len(sx)
    for i in range(n):
        for j in range(i + 1, n):
            bx, by = sx[i] - sx[j], sy[i] - sy[j]
            for sign in (1, -1):
                err = math.hypot(sign * bx - peak["fx"], sign * by - peak["fy"])
                if err < tol:
                    matches.append({
                        "modes": [int(i), int(j)],
                        "weight": float(strength[i] * strength[j]),
                        "beat": [float(sign * bx), float(sign * by)],
                        "period_samples": float(1.0 / max(math.hypot(bx, by), 1e-12)),
                        "err": float(err),
                    })
                    break
    matches.sort(key=lambda m: -m["weight"])
    return len(matches), matches[:limit]


def build_report(patch_path: str, tol: float, top_pairs: int, top_modes: int) -> dict:
    res, k, w, noise, contrib = load_patch(patch_path)
    peaks = measured_peaks(noise, res)
    sx, sy, jx, jy = screen_frequencies(k, w, res)
    strength = np.sqrt((contrib ** 2).mean(axis=(0, 1)))
    order = np.argsort(-strength)[:top_modes]
    total, pairs = beat_table(sx, sy, strength, peaks[0], tol, top_pairs)
    return {
        "patch": patch_path,
        "res": res,
        "jacobian": {"dw_dx": jx.tolist(), "dw_dy": jy.tolist()},
        "fringe_peaks": peaks,
        "top_modes": [
            {
                "mode": int(n),
                "k_magnitude": float(np.linalg.norm(k[n])),
                "rms_contribution": float(strength[n]),
                "screen_f": [float(sx[n]), float(sy[n])],
                "screen_period_samples": float(1.0 / max(math.hypot(sx[n], sy[n]), 1e-12)),
            }
            for n in order
        ],
        "beat_match_count": total,
        "beat_pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch", required=True)
    parser.add_argument("--tol", type=float, default=0.004)
    parser.add_argument("--top-pairs", type=int, default=12)
    parser.add_argument("--top-modes", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build_report(args.patch, args.tol, args.top_pairs, args.top_modes)))


if __name__ == "__main__":
    main()
