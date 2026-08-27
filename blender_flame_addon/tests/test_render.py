from __future__ import annotations

import math

from blender_flame_addon.render import frame_time, sequence_path, blender_camera_to_engine
from blender_flame_addon.coordinates import blender_to_engine_point, engine_projection


def test_frame_time():
    assert frame_time(25, 1, 24.0) == 1.0


def test_sequence_path():
    assert sequence_path("/tmp/x", "Flame", 7) == "/tmp/x/flame_Flame_0007.exr"


def test_blender_camera_to_engine():
    world = [
        [1, 0, 0, 0],
        [0, 0, -1, -4],
        [0, 1, 0, 1.2],
        [0, 0, 0, 1],
    ]
    view, proj, camera_pos = blender_camera_to_engine(world, math.radians(45), 4 / 3, 0.1)

    expected_pos = blender_to_engine_point((0, -4, 1.2))
    for a, b in zip(camera_pos, expected_pos):
        assert abs(a - b) < 1e-6, f"camera_pos {camera_pos} != {expected_pos}"

    vt = view[0][3] + camera_pos[0] * view[0][0] + camera_pos[1] * view[0][1] + camera_pos[2] * view[0][2]
    vy = view[1][3] + camera_pos[0] * view[1][0] + camera_pos[1] * view[1][1] + camera_pos[2] * view[1][2]
    vz = view[2][3] + camera_pos[0] * view[2][0] + camera_pos[1] * view[2][1] + camera_pos[2] * view[2][2]
    assert abs(vt) < 1e-6, f"view * camera_pos x = {vt}"
    assert abs(vy) < 1e-6, f"view * camera_pos y = {vy}"
    assert abs(vz) < 1e-6, f"view * camera_pos z = {vz}"

    expected_proj = engine_projection(math.radians(45), 4 / 3, 0.1)
    for r in range(4):
        for c in range(4):
            assert abs(proj[r][c] - expected_proj[r][c]) < 1e-6, f"proj[{r}][{c}] mismatch"

    vt = view[0][0] * 0 + view[0][1] * 1.2 + view[0][2] * 0 + view[0][3]
    vy = view[1][0] * 0 + view[1][1] * 1.2 + view[1][2] * 0 + view[1][3]
    vz = view[2][0] * 0 + view[2][1] * 1.2 + view[2][2] * 0 + view[2][3]
    assert abs(vt) < 1e-6, f"view * (0,1.2,0) x = {vt}"
    assert abs(vy) < 1e-6, f"view * (0,1.2,0) y = {vy}"
    assert abs(vz - (-4)) < 1e-6, f"view * (0,1.2,0) z = {vz}, expected -4"
