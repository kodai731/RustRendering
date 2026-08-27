import math

from blender_flame_addon.coordinates import (
    mat4_inverse,
    blender_to_engine_matrix,
    blender_camera_to_engine_matrix,
    blender_to_engine_point,
    blender_to_engine_quaternion,
    engine_projection,
    engine_view_matrix,
    look_at_view_matrix,
    orbit_camera,
    z_pass_to_engine_depth,
)


def _mat4_mul(a, b):
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _mat4_transform_point(m, p):
    x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3]
    y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3]
    z = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3]
    return (x, y, z)


def _identity():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _quaternion_to_matrix(q):
    w, x, y, z = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), 0.0],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _almost_equal(a, b, tol=1e-6):
    return abs(a - b) < tol

def test_point_transform():
    p = blender_to_engine_point((0.0, 0.0, 1.0))
    assert _almost_equal(p[0], 0.0)
    assert _almost_equal(p[1], 1.0)
    assert _almost_equal(p[2], 0.0)


def test_translation_transform():
    T = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    m = blender_to_engine_matrix(T)
    assert _almost_equal(m[0][3], 0.0)
    assert _almost_equal(m[1][3], 1.0)
    assert _almost_equal(m[2][3], 0.0)


def test_identity_stays_identity():
    m = blender_to_engine_matrix(_identity())
    identity = _identity()
    for i in range(4):
        for j in range(4):
            assert _almost_equal(m[i][j], identity[i][j])


def test_quaternion_matches_matrix():
    q_blender = (0.70710678, 0.0, 0.70710678, 0.0)
    q_engine = blender_to_engine_quaternion(q_blender)
    m_blender = _quaternion_to_matrix(q_blender)
    m_engine = blender_to_engine_matrix(m_blender)
    m_from_q = _quaternion_to_matrix(q_engine)
    for i in range(4):
        for j in range(4):
            assert _almost_equal(m_engine[i][j], m_from_q[i][j])


def test_projection_values():
    proj = engine_projection(math.radians(45.0), 1.0, 0.1)
    f = 1.0 / math.tan(math.radians(45.0) / 2.0)
    assert _almost_equal(proj[0][0], f)
    assert _almost_equal(proj[1][1], -f)
    assert _almost_equal(proj[2][3], 0.1)
    assert _almost_equal(proj[3][2], -1.0)
    assert _almost_equal(proj[2][2], 0.0)
    assert _almost_equal(proj[3][3], 0.0)


def test_depth_near():
    assert _almost_equal(z_pass_to_engine_depth(0.1, 0.1), 1.0)


def test_depth_far():
    d = z_pass_to_engine_depth(1e10, 0.1)
    assert d < 1e-9


def test_depth_zero():
    assert z_pass_to_engine_depth(0.0, 0.1) == 0.0


def test_view_world_identity():
    world = [
        [1.0, 0.0, 0.0, 5.0],
        [0.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 1.0, -2.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    view = engine_view_matrix(world)
    product = _mat4_mul(view, world)
    identity = _identity()
    for i in range(4):
        for j in range(4):
            assert _almost_equal(product[i][j], identity[i][j])


def test_orbit_camera_zero():
    position, forward, up = orbit_camera(0, 0, 4, (0, 0, 0))
    assert _almost_equal(position[0], 0.0)
    assert _almost_equal(position[1], 0.0)
    assert _almost_equal(position[2], 4.0)
    assert _almost_equal(forward[0], 0.0)
    assert _almost_equal(forward[1], 0.0)
    assert _almost_equal(forward[2], -1.0)


def test_look_at_view_matrix_inverse_translation():
    view = look_at_view_matrix((0, 1.2, 4.5), (0, 0, -1), (0, 1, 0))
    translation = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.2],
        [0.0, 0.0, 1.0, 4.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    inv_translation = mat4_inverse(translation)
    for i in range(4):
        for j in range(4):
            assert _almost_equal(view[i][j], inv_translation[i][j])


def test_orbit_camera_view_transforms_position_to_origin():
    position, forward, up = orbit_camera(30, -5, 4.0, (0, 0.8, 0))
    view = look_at_view_matrix(position, forward, up)
    transformed = _mat4_transform_point(view, position)
    assert _almost_equal(transformed[0], 0.0)
    assert _almost_equal(transformed[1], 0.0)
    assert _almost_equal(transformed[2], 0.0)


def test_blender_camera_to_engine_matrix():
    m = [
        [1, 0, 0, 0],
        [0, 0, -1, -4],
        [0, 1, 0, 1.2],
        [0, 0, 0, 1],
    ]
    engine_world = blender_camera_to_engine_matrix(m)
    forward_x = engine_world[0][0] * 0 + engine_world[0][1] * 0 + engine_world[0][2] * -1
    forward_y = engine_world[1][0] * 0 + engine_world[1][1] * 0 + engine_world[1][2] * -1
    forward_z = engine_world[2][0] * 0 + engine_world[2][1] * 0 + engine_world[2][2] * -1
    assert _almost_equal(forward_x, 0.0)
    assert _almost_equal(forward_y, 0.0)
    assert _almost_equal(forward_z, -1.0)
    up_x = engine_world[0][0] * 0 + engine_world[0][1] * 1 + engine_world[0][2] * 0
    up_y = engine_world[1][0] * 0 + engine_world[1][1] * 1 + engine_world[1][2] * 0
    up_z = engine_world[2][0] * 0 + engine_world[2][1] * 1 + engine_world[2][2] * 0
    assert _almost_equal(up_x, 0.0)
    assert _almost_equal(up_y, 1.0)
    assert _almost_equal(up_z, 0.0)
