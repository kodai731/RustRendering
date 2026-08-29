import math
import time
import traceback

from .coordinates import (
    blender_camera_to_engine_matrix,
    blender_to_engine_point,
    blender_to_engine_quaternion,
    engine_projection,
    engine_view_matrix,
    mat4_inverse,
    project_bounds_to_pixel_rect,
)
from .flame_shader import build_flame_shader, pack_frame_ubo, specialization_key


def flip_projection_y(proj) -> list:
    return [proj[0], [-v for v in proj[1]], proj[2], proj[3]]


_draw_handle = None
_cached_shaders: dict[tuple, object] = {}
_renderers: dict[str, "FlameViewportRenderer"] = {}
_draw_failure_reported = False
_draw_diagnostic_reported = False


def _load_shader(specialization):
    key = specialization_key(specialization)
    if key in _cached_shaders:
        return _cached_shaders[key]
    from pathlib import Path

    root = Path(__file__).resolve().parent
    glsl_path = str(root / "shaders" / "flame_resolve.glsl")
    bindings_path = str(root / "shaders" / "flame_resolve.bindings.json")
    started = time.perf_counter()
    shader = build_flame_shader(glsl_path, bindings_path, specialization)
    print(f"[Thyllore Flame] shader built in {time.perf_counter() - started:.2f}s for {key}", flush=True)
    _cached_shaders[key] = shader
    return shader


def blender_window_to_engine_projection(window_matrix, near):
    f_y = window_matrix[1][1]
    f_x = window_matrix[0][0]
    fovy = 2.0 * math.atan(1.0 / f_y)
    aspect = f_y / f_x
    return engine_projection(fovy, aspect, near)


def blender_view_to_engine_view(view_matrix):
    inv = mat4_inverse(view_matrix)
    if inv is None:
        identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return identity, (0.0, 0.0, 0.0)
    camera_world_blender = inv
    camera_world_engine = blender_camera_to_engine_matrix(camera_world_blender)
    view = engine_view_matrix(camera_world_engine)
    camera_pos = (camera_world_engine[0][3], camera_world_engine[1][3], camera_world_engine[2][3])
    return view, camera_pos


class FlameViewportRenderer:

    def __init__(self):
        self.shader = None
        self.batch = None
        self.shader_key = None
        self.frame_ubo = None
        self.flame_ubo = None
        self.sdf_tex = None
        self.depth_tex = None
        self.history_a = None
        self.history_b = None
        self.color = None
        self.fb_color = None
        self.fb_a = None
        self.fb_b = None
        self.frame_index = 0
        self._w = 0
        self._h = 0

    def ensure_shader(self, params):
        import thyllore_effect_core as fx
        from gpu_extras.batch import batch_for_shader

        specialization = fx.flame_shader_specialization(params)
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
        buf = gpu.types.Buffer("FLOAT", [1, 1, 4], [[[0.5, 0.5, 0.5, 0.5]]])
        self.sdf_tex = gpu.types.GPUTexture((1, 1), format="RGBA32F", data=buf)
        buf = gpu.types.Buffer("FLOAT", [1, 1, 4], [[[0.0, 0.0, 0.0, 0.0]]])
        self.depth_tex = gpu.types.GPUTexture((1, 1), format="RGBA32F", data=buf)
        self.history_a = gpu.types.GPUTexture((w, h), format="RGBA32F")
        self.history_b = gpu.types.GPUTexture((w, h), format="RGBA32F")
        self.color = gpu.types.GPUTexture((w, h), format="RGBA32F")
        self.fb_color = gpu.types.GPUFrameBuffer(color_slots=(self.color,))
        self.fb_a = gpu.types.GPUFrameBuffer(color_slots=(self.color, self.history_a))
        self.fb_b = gpu.types.GPUFrameBuffer(color_slots=(self.color, self.history_b))
        with self.fb_a.bind():
            self.fb_a.clear(color=(0.0, 0.0, 0.0, 0.0))
        with self.fb_b.bind():
            self.fb_b.clear(color=(0.0, 0.0, 0.0, 0.0))

    def clear_color_for_discarded_fragments(self):
        with self.fb_color.bind():
            self.fb_color.clear(color=(0.0, 0.0, 0.0, 0.0))

    def render(self, view, proj, camera_pos, light_pos, params, time, position, rotation, w, h, depth_values=None, flip_y=True):
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
        flame_bytes = fx.pack_flame_ubo(params, time, position, rotation, light_position=light_pos, frame_index=self.frame_index)
        if self.flame_ubo is None:
            self.flame_ubo = gpu.types.GPUUniformBuf(flame_bytes)
        else:
            self.flame_ubo.update(flame_bytes)
        if depth_values is not None:
            buf = gpu.types.Buffer("FLOAT", [w * h], depth_values)
            depth_tex = gpu.types.GPUTexture((w, h), format="R32F", data=buf)
        else:
            depth_tex = self.depth_tex
        cur = self.frame_index % 2
        if cur == 0:
            fb = self.fb_a
            history_tex = self.history_b
        else:
            fb = self.fb_b
            history_tex = self.history_a
        self.clear_color_for_discarded_fragments()
        scissor = project_bounds_to_pixel_rect(fx.flame_bounds_corners(params, position, rotation), view, proj, w, h)
        if scissor is None:
            self.frame_index += 1
            return self.color
        with fb.bind():
            gpu.state.scissor_test_set(True)
            gpu.state.scissor_set(*scissor)
            try:
                self.shader.bind()
                self.shader.uniform_block("frame", self.frame_ubo)
                self.shader.uniform_block("flame", self.flame_ubo)
                self.shader.uniform_sampler("flameHistorySampler", history_tex)
                self.shader.uniform_sampler("flameSdfSampler", self.sdf_tex)
                self.shader.uniform_sampler("sceneDepthSampler", depth_tex)
                self.batch.draw(self.shader)
            finally:
                gpu.state.scissor_test_set(False)
        self.frame_index += 1
        return self.color

    def release(self):
        for attr in (
            "frame_ubo", "flame_ubo", "sdf_tex", "depth_tex",
            "history_a", "history_b", "color", "fb_color", "fb_a", "fb_b",
        ):
            setattr(self, attr, None)


def draw_viewport():
    global _draw_failure_reported
    try:
        draw_flames()
    except Exception:
        if not _draw_failure_reported:
            _draw_failure_reported = True
            print("[Thyllore Flame] viewport draw failed:\n" + traceback.format_exc(), flush=True)


def draw_flames():
    import bpy
    import gpu
    from gpu_extras.batch import batch_for_shader
    from mathutils import Matrix

    from .properties import collect_params

    context = bpy.context
    region = context.region
    region_data = context.region_data
    view_matrix = list(region_data.view_matrix)
    window_matrix = list(region_data.window_matrix)
    near = 0.1
    proj = blender_window_to_engine_projection(window_matrix, near)
    view, camera_pos = blender_view_to_engine_view(view_matrix)
    w = region.width
    h = region.height

    scene = context.scene
    scene_time = (scene.frame_current - scene.frame_start) / scene.render.fps

    light_pos = None
    for obj in scene.objects:
        if obj.type == "LIGHT":
            light_pos = blender_to_engine_point(obj.matrix_world.translation)
            break
    if light_pos is None:
        light_pos = (0.0, 2.0, 2.0)

    flame_objects = []
    for obj in scene.objects:
        if hasattr(obj, "thyllore_flame") and obj.thyllore_flame.is_flame:
            flame_objects.append(obj)

    last_color = None
    render_started = time.perf_counter()
    for obj in flame_objects:
        name = obj.name
        if name not in _renderers:
            _renderers[name] = FlameViewportRenderer()
        renderer = _renderers[name]
        props = obj.thyllore_flame
        cls = type(props)
        params = collect_params(props, cls.PARAM_NAMES)
        position = blender_to_engine_point(obj.matrix_world.translation)
        rotation = blender_to_engine_quaternion(obj.matrix_world.to_quaternion())
        if light_pos is None:
            light_pos = (position[0], position[1] + 2.0, position[2] + 2.0)
        last_color = renderer.render(view, proj, camera_pos, light_pos, params, scene_time, position, rotation, w, h)

    report_first_draw(w, h, camera_pos, flame_objects, time.perf_counter() - render_started)

    if last_color is not None:
        image_shader = gpu.shader.from_builtin("IMAGE")
        batch = batch_for_shader(
            image_shader, "TRI_FAN",
            {"pos": [(0, 0), (w, 0), (w, h), (0, h)], "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)]},
        )
        previous_blend = gpu.state.blend_get()
        previous_depth_test = gpu.state.depth_test_get()
        gpu.matrix.push_projection()
        gpu.matrix.push()
        try:
            gpu.matrix.load_projection_matrix(Matrix(((2.0 / w, 0, 0, -1), (0, 2.0 / h, 0, -1), (0, 0, -1, 0), (0, 0, 0, 1))))
            gpu.matrix.load_identity()
            gpu.state.blend_set("ALPHA_PREMULT")
            gpu.state.depth_test_set("NONE")
            image_shader.bind()
            image_shader.uniform_sampler("image", last_color)
            batch.draw(image_shader)
        finally:
            gpu.state.depth_test_set(previous_depth_test)
            gpu.state.blend_set(previous_blend)
            gpu.matrix.pop()
            gpu.matrix.pop_projection()


def report_first_draw(w, h, camera_pos, flame_objects, render_seconds):
    global _draw_diagnostic_reported
    if _draw_diagnostic_reported:
        return
    _draw_diagnostic_reported = True
    positions = [tuple(round(v, 3) for v in blender_to_engine_point(o.matrix_world.translation)) for o in flame_objects]
    print(
        f"[Thyllore Flame] first draw: region={w}x{h} flames={len(flame_objects)} "
        f"camera_engine={tuple(round(v, 3) for v in camera_pos)} flame_engine={positions} render={render_seconds:.2f}s",
        flush=True,
    )


def register_draw_handler():
    import bpy

    bpy.types.SpaceView3D.draw_handler_add(draw_viewport, (), "WINDOW", "POST_VIEW")


def unregister_draw_handler():
    for renderer in _renderers.values():
        renderer.release()
    _renderers.clear()
    _cached_shaders.clear()
