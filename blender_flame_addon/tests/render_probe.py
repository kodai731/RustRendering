import sys
import os
import argparse
import json
import zipfile
import struct
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from blender_flame_addon.flame_shader import build_flame_shader, pack_frame_ubo
from blender_flame_addon.coordinates import engine_view_matrix, engine_projection, orbit_camera, look_at_view_matrix

def _write_npy(pixels, h, w, filepath):
    flat = [v for px in pixels for v in px]
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, 4), }" % (h, w)
    pad = 64 - ((10 + len(hdr) + 1) % 64)
    hdr = hdr + " " * pad + "\n"
    with open(filepath, 'wb') as f:
        f.write(bytes([0x93]) + b"NUMPY" + bytes([1, 0]) + struct.pack("<H", len(hdr)) + hdr.encode("latin1") + struct.pack("<%df" % len(flat), *flat))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--time", type=float, default=1.5)
    parser.add_argument("--engine-dump", type=str, default=None, help="Path to engine JSONL dump file")
    parser.add_argument("--camera", type=str, default=None, help="Camera spec: yaw,pitch,distance,px,py,pz")
    parser.add_argument("--camera-json", type=str, default=None, help="path to wall-probe JSON (takes priority over --camera)")
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args = parser.parse_args(argv)

    wheel_dir = REPO_ROOT / "blender_flame_addon" / "wheels"
    wheels = sorted(wheel_dir.glob("thyllore_effect_core-*.whl"))
    if not wheels:
        print("No thyllore_effect_core wheel found", flush=True)
        sys.exit(1)

    site_dir = REPO_ROOT / "log" / "blender_flame_probe" / "site"
    site_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(wheels[0]) as zf:
        for entry in zf.namelist():
            if entry.startswith("thyllore_effect_core"):
                zf.extract(entry, str(site_dir))

    sys.path.insert(0, str(site_dir))

    import thyllore_effect_core as fx

    glsl_path = str(REPO_ROOT / "blender_flame_addon" / "shaders" / "flame_resolve.glsl")
    bindings_path = str(REPO_ROOT / "blender_flame_addon" / "shaders" / "flame_resolve.bindings.json")

    w, h = args.width, args.height

    if args.camera_json:
        with open(args.camera_json) as f:
            record = json.load(f)
        cam = record["camera"]
        position = tuple(cam["position"])
        forward = tuple(cam["forward"])
        up = tuple(cam["up"])
        view = look_at_view_matrix(position, forward, up)
        if args.width == 512 and args.height == 512:
            w, h = int(record["viewport_size"][0]), int(record["viewport_size"][1])
        proj = engine_projection(math.radians(cam["fov_y_degrees"]), w / h, cam["near_plane"])
        camera_pos = position
    elif args.camera:
        parts = [float(v) for v in args.camera.split(",")]
        yaw_deg, pitch_deg, distance, px, py, pz = parts
        position, forward, up = orbit_camera(yaw_deg, pitch_deg, distance, (px, py, pz))
        view = look_at_view_matrix(position, forward, up)
        proj = engine_projection(math.radians(45), w / h, 0.1)
        camera_pos = position
    else:
        world = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.2],
            [0.0, 0.0, 1.0, 4.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
        view = engine_view_matrix(world)
        proj = engine_projection(math.radians(45), w / h, 0.1)
        camera_pos = (0.0, 1.2, 4.5)

    if args.engine_dump:
        with open(args.engine_dump) as f:
            lines = [line.strip() for line in f if line.strip()]
        record = json.loads(lines[-1])
        time = record["time"]
        position = list(record["position"])
        rotation = list(record["rotation"])
        effect_params = record.get("effect_params", record)
        flame_params = {k: v for k, v in effect_params.items() if k in fx.flame_preset_params("campfire")}
        light_position_world = effect_params.get("light_position_world")
        frame_index = int(record.get("frame_index", 0))
        flame_bytes = fx.pack_flame_ubo(flame_params, time, position, rotation, light_position=light_position_world, frame_index=frame_index)
    else:
        flame_bytes = fx.pack_flame_ubo(fx.flame_preset_params("campfire"), args.time, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

    frame_bytes = pack_frame_ubo(view, proj, camera_pos + (1.0,), (0.0, 3.0, 3.0, 1.0), (1.0, 1.0, 1.0, 1.0))

    shader = build_flame_shader(glsl_path, bindings_path)

    frame_ubo = gpu.types.GPUUniformBuf(frame_bytes)
    flame_ubo = gpu.types.GPUUniformBuf(flame_bytes)

    def make_tex(v):
        buf = gpu.types.Buffer('FLOAT', [1, 1, 4], [[[v, v, v, v]]])
        return gpu.types.GPUTexture((1, 1), format='RGBA32F', data=buf)

    color = gpu.types.GPUTexture((w, h), format='RGBA32F')
    history = gpu.types.GPUTexture((w, h), format='RGBA32F')
    fb = gpu.types.GPUFrameBuffer(color_slots=(color, history))

    batch = batch_for_shader(shader, 'TRIS', {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]})

    with fb.bind():
        shader.bind()
        shader.uniform_block("frame", frame_ubo)
        shader.uniform_block("flame", flame_ubo)
        shader.uniform_sampler("flameHistorySampler", make_tex(0.0))
        shader.uniform_sampler("flameSdfSampler", make_tex(0.5))
        shader.uniform_sampler("sceneDepthSampler", make_tex(0.0))
        shader.uniform_int("push_mode", 0)
        shader.uniform_int("push_stepCount", 0)
        shader.uniform_int("push_debugView", 0)
        batch.draw(shader)

    rows = color.read().to_list()
    pixels = [px for row in rows for px in row]

    img = bpy.data.images.new("flame_probe", w, h, alpha=True, float_buffer=True)
    img.pixels = [v for px in pixels for v in px]
    scene = bpy.context.scene
    s = scene.render.image_settings
    s.file_format = 'OPEN_EXR'
    s.color_mode = 'RGBA'
    s.color_depth = '32'
    s.exr_codec = 'NONE'
    img.save_render(filepath=args.out, scene=scene)
    loaded = bpy.data.images.load(args.out)
    import array
    flat_src = [v for px in pixels for v in px]
    buf = array.array('f', [0.0] * len(flat_src))
    loaded.pixels.foreach_get(buf)
    d = max(abs(p - q) for p, q in zip(flat_src, buf))
    print(f"EXR_ROUNDTRIP max_abs_diff={d:.3e}", flush=True)

    npy_path = args.out.rsplit(".", 1)[0] + ".npy"
    _write_npy(pixels, h, w, npy_path)

    n = sum(1 for px in pixels if px[3] > 0.0)
    m = max(max(px[0], px[1], px[2]) for px in pixels)
    print(f"PROBE alpha_nonzero={n} max_rgb={m:.4f}", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
