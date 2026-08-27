import array
import struct
from .coordinates import (
    blender_camera_to_engine_matrix,
    blender_to_engine_point,
    blender_to_engine_quaternion,
    engine_projection,
    engine_view_matrix,
    z_pass_to_engine_depth,
)
from .draw_handler import FlameViewportRenderer
from .properties import collect_params


def capture_scene_depth(scene, w, h, near):
    import bpy
    try:
        bpy.ops.render.render(write_still=False)
    except Exception:
        return None
    viewer = bpy.data.images.get("Viewer Node")
    if viewer is None or len(viewer.pixels) == 0:
        return None
    if len(viewer.pixels) != w * h * 4:
        return None
    buf = array.array('f', [0.0] * len(viewer.pixels))
    viewer.pixels.foreach_get(buf)
    r_channel = buf[0::4]
    depth_values = [z_pass_to_engine_depth(z, near) for z in r_channel]
    return depth_values


def _write_npy(pixels, h, w, filepath):
    flat = [v for px in pixels for v in px]
    hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, 4), }" % (h, w)
    pad = 64 - ((10 + len(hdr) + 1) % 64)
    hdr = hdr + " " * pad + "\n"
    with open(filepath, "wb") as f:
        f.write(
            bytes([0x93])
            + b"NUMPY"
            + bytes([1, 0])
            + struct.pack("<H", len(hdr))
            + hdr.encode("latin1")
            + struct.pack("<%df" % len(flat), *flat)
        )


def blender_camera_to_engine(cam_matrix_world, angle_y, aspect, clip_start):
    engine_world = blender_camera_to_engine_matrix(cam_matrix_world)
    view = engine_view_matrix(engine_world)
    proj = engine_projection(angle_y, aspect, clip_start)
    camera_pos = (engine_world[0][3], engine_world[1][3], engine_world[2][3])
    return view, proj, camera_pos


def frame_time(frame, frame_start, fps):
    return (frame - frame_start) / fps


def sequence_path(out_dir, obj_name, frame):
    return f"{out_dir}/flame_{obj_name}_{frame:04d}.exr"


def render_flame_sequence(scene, obj, out_dir, frame_start, frame_end, write_npy=False, use_scene_depth=False):
    import bpy
    cam = scene.camera
    rx = int(scene.render.resolution_x * scene.render.resolution_percentage / 100.0)
    ry = int(scene.render.resolution_y * scene.render.resolution_percentage / 100.0)

    renderer = FlameViewportRenderer()
    written = []

    if use_scene_depth:
        from .compositor import setup_flame_compositor
        setup_flame_compositor(scene, obj, out_dir, frame_start, frame_end)

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        cam_matrix_world = [list(row) for row in cam.matrix_world]
        angle_y = cam.data.angle_y
        aspect = rx / ry
        clip_start = cam.data.clip_start

        view, proj, camera_pos = blender_camera_to_engine(cam_matrix_world, angle_y, aspect, clip_start)
        time = frame_time(frame, frame_start, scene.render.fps)
        position = blender_to_engine_point(obj.matrix_world.translation)
        rotation = blender_to_engine_quaternion(obj.matrix_world.to_quaternion())

        light_pos = None
        for o in scene.objects:
            if o.type == "LIGHT":
                light_pos = blender_to_engine_point(o.matrix_world.translation)
                break
        if light_pos is None:
            light_pos = (position[0], position[1] + 2.0, position[2] + 2.0)

        props = obj.thyllore_flame
        cls = type(props)
        params = collect_params(props, cls.PARAM_NAMES)

        depth_values = None
        if use_scene_depth:
            depth_values = capture_scene_depth(scene, rx, ry, clip_start)

        tex = renderer.render(view, proj, camera_pos, light_pos, params, time, position, rotation, rx, ry, depth_values=depth_values)
        rows = tex.read().to_list()
        pixels = [px for row in rows for px in row]

        exr_path = sequence_path(out_dir, obj.name, frame)
        img = bpy.data.images.new("temp", width=rx, height=ry, float_buffer=True)
        img.pixels = [v for px in pixels for v in px]
        s = scene.render.image_settings
        s.file_format = "OPEN_EXR"
        s.color_mode = "RGBA"
        s.color_depth = "32"
        s.exr_codec = "NONE"
        img.save_render(filepath=exr_path, scene=scene)
        bpy.data.images.remove(img)
        written.append(exr_path)

        if write_npy:
            npy_path = exr_path.rsplit(".", 1)[0] + ".npy"
            _write_npy(pixels, ry, rx, npy_path)

    renderer.release()

    if use_scene_depth:
        from .compositor import setup_flame_compositor
        setup_flame_compositor(scene, obj, out_dir, frame_start, frame_end)

    return written
