import math
import time
import traceback

from ._common import coordinates
from .water_shader import build_tonemap_composite_shader, build_water_shader, matrix_column_major, pack_frame_ubo
from .viewport_color import ViewportColorCapture
from .viewport_depth import ViewportDepthCapture

VIEWPORT_NEAR = 0.1
ENGINE_EXPOSURE = 1.0
DISPLAY_ENCODE_SRGB = 1.0


def flip_projection_y(proj) -> list:
    return [proj[0], [-v for v in proj[1]], proj[2], proj[3]]


_depth_handle = None
_draw_handle = None
_shader = None
_renderers: dict[str, "WaterViewportRenderer"] = {}
_viewport_depth = ViewportDepthCapture()
_viewport_color = ViewportColorCapture()
_scene_depth = None
_scene_color = None
_scene_color_rect = None
_BLACK_COLOR_TEX = None
SCENE_COLOR_PADDING_RATIO = 0.25
_composite_shader = None
_draw_failure_reported = False
_draw_diagnostic_reported = False
_ZERO_DEPTH_TEX = None


def _load_shader():
    global _shader
    if _shader is not None:
        return _shader
    from pathlib import Path

    root = Path(__file__).resolve().parent
    glsl_path = str(root / "shaders" / "water_torus.glsl")
    bindings_path = str(root / "shaders" / "water_torus.bindings.json")
    started = time.perf_counter()
    _shader = build_water_shader(glsl_path, bindings_path)
    print(f"[Thyllore Water] shader built in {time.perf_counter() - started:.2f}s", flush=True)
    return _shader


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


class WaterViewportRenderer:

    def __init__(self):
        self.frame_ubo = None
        self.water_ubo = None
        self.color = None
        self.depth = None
        self.fb_color = None
        self._w = 0
        self._h = 0

    def ensure_size(self, w, h):
        if self._w == w and self._h == h:
            return
        self._w = w
        self._h = h
        import gpu

        self.color = gpu.types.GPUTexture((w, h), format="RGBA32F")
        self.depth = gpu.types.GPUTexture((w, h), format="DEPTH_COMPONENT32F")
        self.fb_color = gpu.types.GPUFrameBuffer(color_slots=(self.color,), depth_slot=self.depth)

    def clear_color_for_discarded_fragments(self):
        with self.fb_color.bind():
            self.fb_color.clear(color=(0.0, 0.0, 0.0, 0.0), depth=0.0)

    def render(self, view, proj, camera_pos, light_pos, params, time, position, rotation, w, h, scene_color, scene_color_rect, flip_y=True):
        import gpu
        import thyllore_effect_core as fx

        self.ensure_size(w, h)
        self.clear_color_for_discarded_fragments()
        if flip_y:
            proj = flip_projection_y(proj)
        frame_bytes = pack_frame_ubo(view, proj, camera_pos + (1.0,), light_pos + (1.0,), (1.0, 1.0, 1.0, 1.0))
        if self.frame_ubo is None:
            self.frame_ubo = gpu.types.GPUUniformBuf(frame_bytes)
        else:
            self.frame_ubo.update(frame_bytes)
        water_bytes = fx.pack_water_ubo(params, time, position, rotation, matrix_column_major(view), matrix_column_major(proj))
        if self.water_ubo is None:
            self.water_ubo = gpu.types.GPUUniformBuf(water_bytes)
        else:
            self.water_ubo.update(water_bytes)

        scissor = coordinates.project_bounds_to_pixel_rect(fx.water_bounds_corners(params, position, rotation), view, proj, w, h)
        if scissor is None:
            return self.color, self.depth

        with self.fb_color.bind():
            gpu.state.scissor_test_set(True)
            gpu.state.scissor_set(*scissor)
            try:
                shader = _load_shader()
                shader.bind()
                shader.uniform_block("frame", self.frame_ubo)
                shader.uniform_block("water", self.water_ubo)
                shader.uniform_sampler("sceneColorSampler", scene_color)
                shader.uniform_float("sceneColorRect", scene_color_rect)
                gpu.state.depth_test_set("ALWAYS")
                gpu.state.depth_mask_set(True)
                from gpu_extras.batch import batch_for_shader
                batch = batch_for_shader(shader, "TRIS", {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]})
                batch.draw(shader)
            finally:
                gpu.state.depth_mask_set(False)
                gpu.state.depth_test_set("NONE")
                gpu.state.scissor_test_set(False)

        return self.color, self.depth

    def release(self):
        for attr in ("frame_ubo", "water_ubo", "color", "depth", "fb_color"):
            setattr(self, attr, None)


def viewport_frame(context):
    region = context.region
    region_data = context.region_data
    view_matrix = list(region_data.view_matrix)
    window_matrix = list(region_data.window_matrix)
    proj = blender_window_to_engine_projection(window_matrix, VIEWPORT_NEAR)
    view, camera_pos = blender_view_to_engine_view(view_matrix)
    return view, proj, camera_pos, window_matrix, region.width, region.height


def water_screen_rects(water_objects, view, proj, w, h):
    import thyllore_effect_core as fx

    from .properties import water_render_params

    rects = []
    for obj in water_objects:
        params = water_render_params(obj.thyllore_water)
        position = coordinates.blender_to_engine_point(obj.matrix_world.translation)
        rotation = coordinates.blender_to_engine_quaternion(obj.matrix_world.to_quaternion())
        rect = coordinates.project_bounds_to_pixel_rect(fx.water_bounds_corners(params, position, rotation), view, proj, w, h)
        if rect is not None:
            rects.append(rect)
    return rects


def padded_union_rect(rects, w, h, padding_ratio):
    if not rects:
        return None
    min_x = min(r[0] for r in rects)
    min_y = min(r[1] for r in rects)
    max_x = max(r[0] + r[2] for r in rects)
    max_y = max(r[1] + r[3] for r in rects)
    pad = int(max(max_x - min_x, max_y - min_y) * padding_ratio)
    x0 = max(min_x - pad, 0)
    y0 = max(min_y - pad, 0)
    x1 = min(max_x + pad, w)
    y1 = min(max_y + pad, h)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def capture_scene_buffers():
    global _scene_depth, _scene_color, _scene_color_rect
    import bpy

    context = bpy.context
    view, proj, _camera_pos, window_matrix, w, h = viewport_frame(context)
    _scene_depth = _viewport_depth.capture(w, h, window_matrix, VIEWPORT_NEAR)

    rect = padded_union_rect(water_screen_rects(find_water_objects(context.scene), view, proj, w, h), w, h, SCENE_COLOR_PADDING_RATIO)
    _scene_color_rect = rect
    _scene_color = _viewport_color.capture(rect) if rect is not None else None


def scene_color_binding(w, h):
    """Returns the linear scene color texture and the uv remap (origin, inverse size) from full-screen uv into it."""
    global _BLACK_COLOR_TEX
    import gpu

    if _scene_color is not None and _scene_color_rect is not None:
        x, y, rw, rh = _scene_color_rect
        return _scene_color, (x / w, y / h, w / rw, h / rh)
    if _BLACK_COLOR_TEX is None:
        _BLACK_COLOR_TEX = gpu.types.GPUTexture((1, 1), format="RGBA16F", data=gpu.types.Buffer("FLOAT", 4, [0.0, 0.0, 0.0, 1.0]))
    return _BLACK_COLOR_TEX, (0.0, 0.0, 1.0, 1.0)


def draw_viewport():
    global _draw_failure_reported
    try:
        draw_water()
    except Exception:
        if not _draw_failure_reported:
            _draw_failure_reported = True
            print("[Thyllore Water] viewport draw failed:\n" + traceback.format_exc(), flush=True)


def find_light_position(scene):
    for obj in scene.objects:
        if obj.type == "LIGHT":
            return coordinates.blender_to_engine_point(obj.matrix_world.translation)
    return (0.0, 2.0, 2.0)


def find_water_objects(scene):
    return [obj for obj in scene.objects if hasattr(obj, "thyllore_water") and obj.thyllore_water.is_water]


def draw_water():
    import bpy

    from .properties import water_render_params

    context = bpy.context
    view, proj, camera_pos, window_matrix, w, h = viewport_frame(context)

    scene = context.scene
    scene_time = (scene.frame_current - scene.frame_start) / scene.render.fps
    light_pos = find_light_position(scene)
    water_objects = find_water_objects(scene)
    scene_color, scene_color_rect = scene_color_binding(w, h)

    last_color = None
    render_started = time.perf_counter()
    for obj in water_objects:
        renderer = _renderers.setdefault(obj.name, WaterViewportRenderer())
        params = water_render_params(obj.thyllore_water)
        position = coordinates.blender_to_engine_point(obj.matrix_world.translation)
        rotation = coordinates.blender_to_engine_quaternion(obj.matrix_world.to_quaternion())
        last_color, _depth = renderer.render(
            view, proj, camera_pos, light_pos, params, scene_time, position, rotation, w, h, scene_color, scene_color_rect
        )

    report_first_draw(w, h, camera_pos, water_objects, time.perf_counter() - render_started)

    if last_color is not None:
        composite_tonemapped(last_color, _renderers[obj.name].depth, w, h, window_matrix)


def composite_tonemapped(color_tex, depth_tex, w, h, window_matrix, scene_depth=None):
    global _composite_shader, _ZERO_DEPTH_TEX
    import gpu
    from gpu_extras.batch import batch_for_shader

    effective_scene_depth = scene_depth if scene_depth is not None else _scene_depth

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
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        _composite_shader.bind()
        _composite_shader.uniform_float("ModelViewProjectionMatrix", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())
        _composite_shader.uniform_float("tonemapParams", (ENGINE_EXPOSURE, DISPLAY_ENCODE_SRGB))
        _composite_shader.uniform_float("depthParams", (window_matrix[2][2], window_matrix[2][3], VIEWPORT_NEAR))
        _composite_shader.uniform_sampler("image", color_tex)
        _composite_shader.uniform_sampler("waterDepth", depth_tex)
        if effective_scene_depth is not None:
            _composite_shader.uniform_sampler("sceneDepth", effective_scene_depth)
        else:
            if _ZERO_DEPTH_TEX is None:
                _ZERO_DEPTH_TEX = gpu.types.GPUTexture((1, 1), format="R32F", data=gpu.types.Buffer("FLOAT", 1, [0.0]))
            _composite_shader.uniform_sampler("sceneDepth", _ZERO_DEPTH_TEX)
        batch.draw(_composite_shader)
    finally:
        gpu.state.depth_mask_set(False)
        gpu.state.depth_test_set(previous_depth_test)
        gpu.state.blend_set(previous_blend)


def report_first_draw(w, h, camera_pos, water_objects, render_seconds):
    global _draw_diagnostic_reported
    if _draw_diagnostic_reported:
        return
    _draw_diagnostic_reported = True
    positions = [tuple(round(v, 3) for v in coordinates.blender_to_engine_point(o.matrix_world.translation)) for o in water_objects]
    print(
        f"[Thyllore Water] first draw: region={w}x{h} waters={len(water_objects)} "
        f"camera_engine={tuple(round(v, 3) for v in camera_pos)} water_engine={positions} render={render_seconds:.2f}s",
        flush=True,
    )


def register_draw_handler():
    global _depth_handle, _draw_handle
    import bpy

    _depth_handle = bpy.types.SpaceView3D.draw_handler_add(capture_scene_buffers, (), "WINDOW", "POST_VIEW")
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(draw_viewport, (), "WINDOW", "POST_PIXEL")


def unregister_draw_handler():
    global _depth_handle, _draw_handle, _scene_depth, _scene_color, _scene_color_rect, _composite_shader, _shader
    import bpy

    for handle in (_draw_handle, _depth_handle):
        if handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
    _depth_handle = _draw_handle = None
    for renderer in _renderers.values():
        renderer.release()
    _renderers.clear()
    _shader = None
    _viewport_depth.release()
    _viewport_color.release()
    _scene_depth = None
    _scene_color = None
    _scene_color_rect = None
    _composite_shader = None
