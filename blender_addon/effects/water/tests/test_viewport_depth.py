from blender_addon.effects.water.viewport_depth import (
    engine_depth_to_window_depth,
    window_depth_to_engine_depth,
)


def _almost_equal(a, b, tol=1e-5):
    return abs(a - b) < tol


class TestRoundTrip:
    """Round-trip consistency: forward then inverse returns the original d."""

    def test_near_plane(self):
        d = 0.01
        assert _almost_equal(engine_depth_to_window_depth(window_depth_to_engine_depth(d, 1.0, 1.0, 1.0), 1.0, 1.0, 1.0), d)

    def test_mid_range(self):
        d = 0.5
        assert _almost_equal(engine_depth_to_window_depth(window_depth_to_engine_depth(d, 1.0, 1.0, 1.0), 1.0, 1.0, 1.0), d)

    def test_far_range(self):
        d = 0.99
        assert _almost_equal(engine_depth_to_window_depth(window_depth_to_engine_depth(d, 1.0, 1.0, 1.0), 1.0, 1.0, 1.0), d)

    def test_different_near(self):
        d = 0.3
        near = 5.0
        assert _almost_equal(engine_depth_to_window_depth(window_depth_to_engine_depth(d, 1.0, 1.0, near), 1.0, 1.0, near), d)


class TestNumericalMatchGLSL:
    """Verify Python matches the GLSL logic in depth_convert_fragment_source().

    GLSL: zEye = p23 / (2.0 * d - 1.0 + p22);
          engineDepth = (d >= 1.0 || zEye <= 0.0) ? 0.0 : near / zEye;
    """

    def test_case_1(self):
        # d=0.5, p22=1.0, p23=1.0, near=1.0
        # zEye = 1.0 / (2*0.5 - 1 + 1) = 1.0; engineDepth = 1.0 / 1.0 = 1.0
        assert _almost_equal(window_depth_to_engine_depth(0.5, 1.0, 1.0, 1.0), 1.0)

    def test_case_2(self):
        # d=0.7, p22=1.0, p23=1.0, near=1.0
        # zEye = 1.0 / (2*0.7 - 1 + 1) = 1.0 / 1.4; engineDepth = 1.0 / (1.0/1.4) = 1.4
        assert _almost_equal(window_depth_to_engine_depth(0.7, 1.0, 1.0, 1.0), 1.4)

    def test_case_3(self):
        # d=1.2, p22=1.0, p23=1.0, near=1.0
        # d >= 1.0 -> engineDepth = 0.0
        assert _almost_equal(window_depth_to_engine_depth(1.2, 1.0, 1.0, 1.0), 0.0)
