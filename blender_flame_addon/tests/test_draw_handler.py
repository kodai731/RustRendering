import math

from blender_flame_addon.coordinates import (
    blender_to_engine_point,
    engine_projection,
)
from blender_flame_addon.draw_handler import (
    blender_view_to_engine_view,
    blender_window_to_engine_projection,
)


def _almost_equal(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_blender_window_to_engine_projection_matches():
    f = 1.0 / math.tan(math.radians(22.5))
    window_matrix = [
        [f / 2, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, -1, -0.2],
        [0, 0, -1, 0],
    ]
    proj = blender_window_to_engine_projection(window_matrix, 0.1)
    expected = engine_projection(math.radians(45), 2.0, 0.1)
    for i in range(4):
        for j in range(4):
            assert _almost_equal(proj[i][j], expected[i][j])


def test_blender_view_to_engine_view_camera_at_origin():
    view = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -4.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    result_view, camera_pos = blender_view_to_engine_view(view)
    expected_pos = blender_to_engine_point((0.0, 0.0, 4.0))
    assert _almost_equal(camera_pos[0], expected_pos[0])
    assert _almost_equal(camera_pos[1], expected_pos[1])
    assert _almost_equal(camera_pos[2], expected_pos[2])
    x = (result_view[0][0] * camera_pos[0] + result_view[0][1] * camera_pos[1] + result_view[0][2] * camera_pos[2] + result_view[0][3])
    y = (result_view[1][0] * camera_pos[0] + result_view[1][1] * camera_pos[1] + result_view[1][2] * camera_pos[2] + result_view[1][3])
    z = (result_view[2][0] * camera_pos[0] + result_view[2][1] * camera_pos[1] + result_view[2][2] * camera_pos[2] + result_view[2][3])
    assert _almost_equal(x, 0.0)
    assert _almost_equal(y, 0.0)
    assert _almost_equal(z, 0.0)
