import traceback

from .wind_shader import build_depth_convert_shader

_capture_failure_reported = False


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
                print("[Thyllore Wind] viewport depth capture failed, wind draws unoccluded:\n" + traceback.format_exc(), flush=True)
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
