import traceback

from .water_shader import build_depth_convert_shader

_capture_failure_reported = False


def window_depth_to_engine_depth(d: float, p22: float, p23: float, near: float) -> float:
    """Pure Python forward: window depth d -> engine depth.

    Matches the GLSL in depth_convert_fragment_source():
        zEye = p23 / (2.0 * d - 1.0 + p22)
        engineDepth = (d >= 1.0 || zEye <= 0.0) ? 0.0 : near / zEye
    """
    z_eye = p23 / (2.0 * d - 1.0 + p22)
    if d >= 1.0 or z_eye <= 0.0:
        return 0.0
    return near / z_eye


def engine_depth_to_window_depth(engine_depth: float, p22: float, p23: float, near: float) -> float:
    """Pure Python inverse: engine depth -> window depth d.

    Inverts the forward formula:
        zEye = near / engine_depth
        d = (p23 / zEye + 1 - p22) / 2
    """
    if engine_depth == 0.0:
        return 1.0
    z_eye = near / engine_depth
    return (p23 / z_eye + 1.0 - p22) / 2.0


class ViewportDepthCapture:

    def __init__(self):
        self.shader = None
        self.batch = None
        self.window_depth = None
        self.engine_depth = None
        self.framebuffer = None
        self._w = 0
        self._h = 0

    def ensure_resources(self, w, h):
        import gpu
        from gpu_extras.batch import batch_for_shader

        if self.shader is None:
            self.shader = build_depth_convert_shader()
            self.batch = batch_for_shader(self.shader, "TRIS", {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]})
        if self._w == w and self._h == h:
            return
        self._w = w
        self._h = h
        self.engine_depth = gpu.types.GPUTexture((w, h), format="R32F")
        self.framebuffer = gpu.types.GPUFrameBuffer(color_slots=(self.engine_depth,))

    def capture(self, w, h, window_matrix, near):
        global _capture_failure_reported
        try:
            return self._capture(w, h, window_matrix, near)
        except Exception:
            if not _capture_failure_reported:
                _capture_failure_reported = True
                print("[Thyllore Water] viewport depth capture failed, water draws unoccluded:\n" + traceback.format_exc(), flush=True)
            return None

    def _capture(self, w, h, window_matrix, near):
        import gpu

        if window_matrix[3][2] == 0.0:
            return None
        self.ensure_resources(w, h)
        source = gpu.state.active_framebuffer_get()
        depth_buffer = source.read_depth(0, 0, w, h)
        self.window_depth = gpu.types.GPUTexture((w, h), format="R32F", data=depth_buffer)

        with self.framebuffer.bind():
            self.shader.bind()
            self.shader.uniform_sampler("windowDepth", self.window_depth)
            self.shader.uniform_float("depthParams", (window_matrix[2][2], window_matrix[2][3], near))
            self.batch.draw(self.shader)
        return self.engine_depth

    def release(self):
        for attr in ("shader", "batch", "window_depth", "engine_depth", "framebuffer"):
            setattr(self, attr, None)
        self._w = 0
        self._h = 0
