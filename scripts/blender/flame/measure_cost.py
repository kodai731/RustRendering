"""Measure the flame resolve pass cost inside Blender (run via measure_cost.sh).

GPU time per frame = (wall time of N back-to-back renders followed by one
texture.read() sync - wall time of the sync alone) / N. Blender's gpu module has
no timestamp queries, so the readback sync is the only GPU fence available.
"""
import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import bpy
import gpu

KNOWN_FP32_TFLOPS = {
    "GTX 1650": 2.9, "GTX 1660": 5.0, "GTX 1660 Ti": 5.4, "RTX 2060": 6.5, "RTX 3050": 9.1,
    "RTX 3060": 12.7, "RTX 4060": 15.1, "RTX 3070": 20.3, "RTX 4070": 29.1, "RTX 4080": 48.7,
    "RX 6500 XT": 5.8, "RX 6600": 8.9, "RX 7600": 21.5, "RX 6700 XT": 13.2, "RX 7800 XT": 37.3,
    "Apple M1": 2.6, "Apple M2": 3.6, "Apple M3": 4.1, "Apple M4": 4.3,
}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="JSON result path")
    parser.add_argument("--preset", default="campfire")
    parser.add_argument("--resolutions", default="1920x1080,2560x1440,3840x2160")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--camera-distance", type=float, default=6.0)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--peak-tflops", type=float, required=True, help="FP32 TFLOPS of the GPU running this script")
    return parser.parse_args(argv)


def load_wheel():
    wheel_dir = REPO_ROOT / "blender_addon" / "effects" / "flame" / "wheels"
    site_dir = REPO_ROOT / "log" / "blender_flame_probe" / "site"
    site_dir.mkdir(parents=True, exist_ok=True)
    wheel = sorted(wheel_dir.glob("thyllore_effect_core-*.whl"))[0]
    with zipfile.ZipFile(wheel) as zf:
        for entry in zf.namelist():
            if entry.startswith("thyllore_effect_core"):
                zf.extract(entry, str(site_dir))
    sys.path.insert(0, str(site_dir))


def build_camera(distance, width, height):
    from blender_addon.common.coordinates import engine_projection, look_at_view_matrix

    position = (0.0, 1.6, distance)
    target = (0.0, 0.8, 0.0)
    forward = tuple(t - p for t, p in zip(target, position))
    length = math.sqrt(sum(f * f for f in forward))
    forward = tuple(f / length for f in forward)
    view = look_at_view_matrix(position, forward, (0.0, 1.0, 0.0))
    proj = engine_projection(math.radians(39.6), width / height, 0.1)
    return view, proj, position


def scissor_pixels(fx, params, view, proj, width, height):
    from blender_addon.common.coordinates import project_bounds_to_pixel_rect

    rect = project_bounds_to_pixel_rect(fx.flame_bounds_corners(params, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)), view, proj, width, height)
    return 0 if rect is None else rect[2] * rect[3]


def time_frames(renderer, render_args, frame_count):
    tex = renderer.render(*render_args)
    tex.read()
    started = time.perf_counter()
    for _ in range(frame_count):
        tex = renderer.render(*render_args)
    tex.read()
    return time.perf_counter() - started


def measure_resolution(fx, renderer_cls, params, preset, width, height, frames, distance):
    view, proj, camera_pos = build_camera(distance, width, height)
    renderer = renderer_cls()
    render_args = (view, proj, camera_pos, (0.0, 2.0, 2.0), params, 1.5, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), width, height)
    for _ in range(3):
        renderer.render(*render_args)
    sync_only = time_frames(renderer, render_args, 0)
    total = min(time_frames(renderer, render_args, frames) for _ in range(3))
    renderer.release()

    frame_ms = max(total - sync_only, 0.0) / frames * 1000.0
    pixels = scissor_pixels(fx, params, view, proj, width, height)
    return {
        "resolution": f"{width}x{height}",
        "scissor_pixels": pixels,
        "gpu_ms_per_frame": frame_ms,
        "ms_per_mpixel_scissor": frame_ms / (pixels / 1e6) if pixels else None,
        "ms_per_mpixel_frame": frame_ms / (width * height / 1e6),
    }


def required_tflops(frame_ms, peak_tflops, target_fps):
    return frame_ms * peak_tflops * target_fps / 1000.0


def minimum_known_gpu(tflops):
    candidates = sorted(KNOWN_FP32_TFLOPS.items(), key=lambda kv: kv[1])
    return next((name for name, value in candidates if value >= tflops), None)


def main():
    args = parse_args()
    load_wheel()
    import thyllore_effect_core as fx
    from blender_addon.effects.flame.draw_handler import FlameViewportRenderer

    params = fx.flame_preset_params(args.preset)
    results = []
    for token in args.resolutions.split(","):
        width, height = (int(v) for v in token.lower().split("x"))
        entry = measure_resolution(fx, FlameViewportRenderer, params, args.preset, width, height, args.frames, args.camera_distance)
        entry["required_tflops_at_target_fps"] = required_tflops(entry["gpu_ms_per_frame"], args.peak_tflops, args.target_fps)
        entry["minimum_known_gpu"] = minimum_known_gpu(entry["required_tflops_at_target_fps"])
        results.append(entry)
        print(f"COST {entry['resolution']}: {entry['gpu_ms_per_frame']:.2f} ms/frame, scissor {entry['scissor_pixels']} px, "
              f"{entry['ms_per_mpixel_scissor'] or 0:.2f} ms/Mpx(scissor), needs {entry['required_tflops_at_target_fps']:.1f} TFLOPS @ {args.target_fps:g} fps "
              f"(>= {entry['minimum_known_gpu']})", flush=True)

    report = {
        "gpu": gpu.platform.renderer_get(),
        "backend": gpu.platform.backend_type_get(),
        "blender": bpy.app.version_string,
        "preset": args.preset,
        "frames": args.frames,
        "camera_distance": args.camera_distance,
        "peak_tflops": args.peak_tflops,
        "target_fps": args.target_fps,
        "method": "wall time of N renders + read() sync minus sync alone; required TFLOPS = ms * peak_tflops * fps / 1000",
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"COST_DONE {args.out}", flush=True)


main()
sys.exit(0)
