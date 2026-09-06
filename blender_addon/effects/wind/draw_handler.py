import math
import time
import traceback

from ._common import coordinates
from .wind_shader import build_tonemap_composite_shader, build_wind_shader, pack_frame_ubo, specialization_key
from .viewport_depth import ViewportDepthCapture

VIEWPORT_NEAR = 0.1
ENGINE_EXPOSURE = 1.0
DISPLAY_ENCODE_SRGB = 1.0


def flip_projection_y(proj) -> list:
    return [proj[0], [-v for v in proj[1]], proj[2], proj[3]]


_depth_handle = None
_draw_handle = None
_cached_shaders: dict[tuple, object] = {}
_renderers: dict[str, "WindViewportRenderer"] = {}
_viewport_depth = ViewportDepthCapture()
_scene_depth = None
_composite_shader = None
_draw_failure_reported = False
_draw_diagnostic_reported = False


def _load_shader(specialization):
    key = specialization_key(specialization)
    if key in _cached_shaders:
        return _cached_shaders[key]
    from pathlib import Path

    root = Path(__file__).resolve().parent
    glsl_path = str(root / "shaders" / "wind_resolve.glsl")
    bindings_path = str(root / "shaders" / "wind_resolve.bindings.json")
    started = time.perf_counter()
    shader = build_wind_shader(glsl_path, bindings_path, specialization)
    print(f"[Thyllore Wind] shader built in {time.perf_counter() - started:.2f}s for {key}", flush=True)
    _cached_shaders[key] = shader
    return shader


def blender_window_to_engine_projection(window_matrix, near):
    f_y = window_matrix[1][1]
    f_x = window_matrix[0][0]
    fovy = 2.0 * math.atan(1.0 / f_y)
    aspect = f_y / f_x
    return coordinates.engine_projection(fovy, aspect, near)


def blender_view_to_engine_view(view_matrix):
    inv = coordinates.mat4_inverse(view_matrix)
    if inv is None:
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return identity, (0.0, 0.0, 0.0)
    camera_world_blender = inv
    camera_world_engine = coordinates.blender_camera_to_engine_matrix(camera_world_blender)
    view = coordinates.engine_view_matrix(camera_world_engine)
    camera_pos = (camera_world_engine[0][3], camera_world_engine[1][3], camera_world_engine[2][3])
    return view, camera_pos


class WindViewportRenderer:

    def __init__(self):
        self.shader = None
        self.batch = None
        self.shader_key = None
        self.frame_ubo = None
        self.wind_ubo = None
        self.color = None
        self.fb_color = None
        self._w = 0
        self._h = 0

    def ensure_shader(self, params):
        import thyllore_effect_core as fx
        from gpu_extras.batch import batch_for_shader

        specialization = fx.wind_shader_specialization(params)
        key = specialization_key(specialization)
        if key == self.shader_key:
            return
        self.shader = _load_shader(specialization)
        self.batch = batch_for_shader(self.shader, "TRIS", {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]})
        self.shader_key = key

    def ensure_size(self, w, h):
        if self._w == w and self._h == h:
            return
        self._w = w
        self._h = h
        import gpu
        self.color = gpu.types.GPUTexture((w, h), format="RGBA32F")
        self.fb_color = gpu.types.GPUFrameBuffer(color_slots=(self.color,))

    def clear_color_for_discarded_fragments(self):
        with self.fb_color.bind():
            self.fb_color.clear(color=(0.0, 0.0, 0.0, 0.0))

    def render(self, view, proj, camera_pos, light_pos, params, time, position, rotation, w, h, depth_tex=None, flip_y=True):
        import gpu
        import thyllore_effect_core as fx

        self.ensure_size(w, h)
        self.ensure_shader(params)
        if flip_y:
            proj = flip_projection_y(proj)
        frame_bytes = pack_frame_ubo(view, proj, camera_pos + (1.0,), light_pos + (1.0,), (1.0, 1.0, 1.0, 1.0))
        if self.frame_ubo is None:
            self.frame_ubo = gpu.types.GPUUniformBuf(frame_bytes)
        else:
            self.frame_ubo.update(frame_bytes)
        wind_bytes = fx.pack_wind_ubo(params, time, position, rotation)
        if self.wind_ubo is None:
            self.wind_ubo = gpu.types.GPUUniformBuf(wind_bytes)
        else:
            self.wind_ubo.update(wind_bytes)
        if depth_tex is None:
            return self.color
        self.clear_color_for_discarded_fragments()
        scissor = coordinates.project_bounds_to_pixel_rect(fx.wind_bounds_corners(params, position, rotation), view, proj, w, h)
        if scissor is None:
            return self.color
        with self.fb_color.bind():
            gpu.state.scissor_test_set(True)
            gpu.state.scissor_set(*scissor)
            try:
                self.shader.bind()
                self.shader.uniform_block("frame", self.frame_ubo)
                self.shader.uniform_block("wind", self.wind_ubo)
                self.shader.uniform_sampler("sceneDepthSampler", depth_tex)
                self.batch.draw(self.shader)
            finally:
                gpu.state.scissor_test_set(False)
        return self.color

    def release(self):
        for attr in ("frame_ubo", "wind_ubo", "color", "fb_color"):
            setattr(self, attr, None)


def capture_scene_depth():
    global _scene_depth
    import bpy

    region = bpy.context.region
    window_matrix = list(bpy.context.region_data.window_matrix)
    _scene_depth = _viewport_depth.capture(region.width, region.height, window_matrix, VIEWPORT_NEAR)


def draw_viewport():
    global _draw_failure_reported
    try:
        draw_wind()
    except Exception:
        if not _draw_failure_reported:
            _draw_failure_reported = True
            print("[Thyllore Wind] viewport draw failed:\n" + traceback.format_exc(), flush=True)


def find_light_position(scene):
    for obj in scene.objects:
        if obj.type == "LIGHT":
            return coordinates.blender_to_engine_point(obj.matrix_world.translation)
    return (0.0, 2.0, 2.0)


def find_wind_objects(scene):
    return [obj for obj in scene.objects if hasattr(obj, "thyllore_wind") and obj.thyllore_wind.is_wind]


def draw_wind():
    import bpy

    from .properties import wind_render_params

    context = bpy.context
    region = context.region
    region_data = context.region_data
    view_matrix = list(region_data.view_matrix)
    window_matrix = list(region_data.window_matrix)
    proj = blender_window_to_engine_projection(window_matrix, VIEWPORT_NEAR)
    view, camera_pos = blender_view_to_engine_view(view_matrix)
    w = region.width
    h = region.height

    scene = context.scene
    scene_time = (scene.frame_current - scene.frame_start) / scene.render.fps
    light_pos = find_light_position(scene)
    wind_objects = find_wind_objects(scene)

    last_color = None
    render_started = time.perf_counter()
    for obj in wind_objects:
        renderer = _renderers.setdefault(obj.name, WindViewportRenderer())
        params = wind_render_params(obj.thyllore_wind)
        position = coordinates.blender_to_engine_point(obj.matrix_world.translation)
        rotation = coordinates.blender_to_engine_quaternion(obj.matrix_world.to_quaternion())
        last_color = renderer.render(
            view, proj, camera_pos, light_pos, params, scene_time, position, rotation, w, h, depth_tex=_scene_depth
        )

    report_first_draw(w, h, camera_pos, wind_objects, time.perf_counter() - render_started)

    if last_color is not None:
        composite_tonemapped(last_color, w, h)


def composite_tonemapped(color_tex, w, h):
    global _composite_shader
    import gpu
    from gpu_extras.batch import batch_for_shader

    if _composite_shader is None:
        _composite_shader = build_tonemap_composite_shader()
    batch = batch_for_shader(
        _composite_shader, "TRI_FAN",
        {"pos": [(0, 0), (w, 0), (w, h), (0, h)], "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]},
    )
    previous_blend = gpu.state.blend_get()
    previous_depth_test = gpu.state.depth_test_get()
    try:
        gpu.state.blend_set("ALPHA_PREMULT")
        gpu.state.depth_test_set("NONE")
        _composite_shader.bind()
        _composite_shader.uniform_float("ModelViewProjectionMatrix", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())
        _composite_shader.uniform_float("tonemapParams", (ENGINE_EXPOSURE, DISPLAY_ENCODE_SRGB))
        _composite_shader.uniform_sampler("image", color_tex)
        batch.draw(_composite_shader)
    finally:
        gpu.state.depth_test_set(previous_depth_test)
        gpu.state.blend_set(previous_blend)


def report_first_draw(w, h, camera_pos, wind_objects, render_seconds):
    global _draw_diagnostic_reported
    if _draw_diagnostic_reported:
        return
    _draw_diagnostic_reported = True
    positions = [tuple(round(v, 3) for v in coordinates.blender_to_engine_point(o.matrix_world.translation)) for o in wind_objects]
    print(
        f"[Thyllore Wind] first draw: region={w}x{h} winds={len(wind_objects)} "
        f"camera_engine={tuple(round(v, 3) for v in camera_pos)} wind_engine={positions} render={render_seconds:.2f}s",
        flush=True,
    )


def register_draw_handler():
    global _depth_handle, _draw_handle
    import bpy

    _depth_handle = bpy.types.SpaceView3D.draw_handler_add(capture_scene_depth, (), "WINDOW", "POST_VIEW")
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(draw_viewport, (), "WINDOW", "POST_PIXEL")


def unregister_draw_handler():
    global _depth_handle, _draw_handle, _scene_depth, _composite_shader
    import bpy

    for handle in (_draw_handle, _depth_handle):
        if handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
    _depth_handle = _draw_handle = None
    for renderer in _renderers.values():
        renderer.release()
    _renderers.clear()
    _cached_shaders.clear()
    _viewport_depth.release()
    _scene_depth = None
    _composite_shader = None
