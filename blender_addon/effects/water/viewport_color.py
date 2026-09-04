import traceback

from .water_shader import build_color_decode_shader

_capture_failure_reported = False


class ViewportColorCapture:
    """Reads the display-encoded viewport color inside the water's screen rect and decodes it to the linear scene color the water refracts."""

    def __init__(self):
        self.shader = None
        self.batch = None
        self.window_color = None
        self.linear_color = None
        self.framebuffer = None
        self._size = (0, 0)

    def ensure_resources(self, w, h):
        import gpu
        from gpu_extras.batch import batch_for_shader

        if self.shader is None:
            self.shader = build_color_decode_shader()
            self.batch = batch_for_shader(self.shader, "TRIS", {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]})
        if self._size == (w, h):
            return
        self._size = (w, h)
        self.linear_color = gpu.types.GPUTexture((w, h), format="RGBA16F")
        self.framebuffer = gpu.types.GPUFrameBuffer(color_slots=(self.linear_color,))

    def capture(self, rect):
        global _capture_failure_reported
        try:
            return self._capture(rect)
        except Exception:
            if not _capture_failure_reported:
                _capture_failure_reported = True
                print("[Thyllore Water] viewport color capture failed, water refracts black:\n" + traceback.format_exc(), flush=True)
            return None

    def _capture(self, rect):
        import gpu

        x, y, w, h = rect
        self.ensure_resources(w, h)
        source = gpu.state.active_framebuffer_get()
        color_buffer = source.read_color(x, y, w, h, 4, 0, "FLOAT")
        self.window_color = gpu.types.GPUTexture((w, h), format="RGBA16F", data=color_buffer)

        with self.framebuffer.bind():
            self.shader.bind()
            self.shader.uniform_sampler("windowColor", self.window_color)
            self.batch.draw(self.shader)
        return self.linear_color

    def release(self):
        for attr in ("shader", "batch", "window_color", "linear_color", "framebuffer"):
            setattr(self, attr, None)
        self._size = (0, 0)
