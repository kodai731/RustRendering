import math


def _mat4_mul(a, b):
    result = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _mat4_inverse(m):
    aug = [[0.0] * 8 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            aug[i][j] = m[i][j]
        aug[i][i + 4] = 1.0
    for col in range(4):
        max_row = col
        max_val = abs(aug[col][col])
        for row in range(col + 1, 4):
            if abs(aug[row][col]) > max_val:
                max_val = abs(aug[row][col])
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            return None
        inv_pivot = 1.0 / pivot
        for j in range(8):
            aug[col][j] *= inv_pivot
        for row in range(4):
            if row == col:
                continue
            factor = aug[row][col]
            for j in range(8):
                aug[row][j] -= factor * aug[col][j]
    return [[aug[i][j + 4] for j in range(4)] for i in range(4)]


def _mat4_row_major_identity():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_transform_point(m, p):
    x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3]
    y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3]
    z = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3]
    return (x, y, z)




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




C = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

C_INV = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def blender_to_engine_matrix(m):
    return _mat4_mul(C, _mat4_mul(m, C_INV))


def blender_to_engine_quaternion(q):
    w, x, y, z = q
    return (w, x, z, -y)


def blender_to_engine_point(p):
    return (p[0], p[2], -p[1])


def engine_projection(fovy_rad, aspect, near):
    f = 1.0 / math.tan(fovy_rad / 2.0)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, 0.0, near],
        [0.0, 0.0, -1.0, 0.0],
    ]


def z_pass_to_engine_depth(z_eye, near):
    if not math.isfinite(z_eye) or z_eye <= 0.0:
        return 0.0
    return near / z_eye


def engine_view_matrix(camera_world_engine):
    inv = _mat4_inverse(camera_world_engine)
    if inv is None:
        return _mat4_row_major_identity()
    return inv


def orbit_camera(yaw_deg, pitch_deg, distance, pivot):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    sp = math.sin(pitch)
    cp = math.cos(pitch)
    backward = (cp * sy, sp, cp * cy)
    position = (pivot[0] + backward[0] * distance,
                pivot[1] + backward[1] * distance,
                pivot[2] + backward[2] * distance)
    direction = (-backward[0], -backward[1], -backward[2])
    right_x = direction[1] * 0.0 - 1.0 * direction[2]
    right_y = direction[2] * 0.0 - direction[0] * 0.0
    right_z = direction[0] * 1.0 - direction[1] * 0.0
    right_len = math.sqrt(right_x * right_x + right_y * right_y + right_z * right_z)
    right = (right_x / right_len, right_y / right_len, right_z / right_len)
    up_x = right[1] * direction[2] - right[2] * direction[1]
    up_y = right[2] * direction[0] - right[0] * direction[2]
    up_z = right[0] * direction[1] - right[1] * direction[0]
    up_len = math.sqrt(up_x * up_x + up_y * up_y + up_z * up_z)
    up = (up_x / up_len, up_y / up_len, up_z / up_len)
    return (position, direction, up)


def look_at_view_matrix(position, forward, up_in):
    right_x = forward[1] * up_in[2] - forward[2] * up_in[1]
    right_y = forward[2] * up_in[0] - forward[0] * up_in[2]
    right_z = forward[0] * up_in[1] - forward[1] * up_in[0]
    right_len = math.sqrt(right_x * right_x + right_y * right_y + right_z * right_z)
    right = (right_x / right_len, right_y / right_len, right_z / right_len)
    up_x = right[1] * forward[2] - right[2] * forward[1]
    up_y = right[2] * forward[0] - right[0] * forward[2]
    up_z = right[0] * forward[1] - right[1] * forward[0]
    up = (up_x, up_y, up_z)
    return [
        [right[0], right[1], right[2], -(right[0] * position[0] + right[1] * position[1] + right[2] * position[2])],
        [up[0], up[1], up[2], -(up[0] * position[0] + up[1] * position[1] + up[2] * position[2])],
        [-forward[0], -forward[1], -forward[2], forward[0] * position[0] + forward[1] * position[1] + forward[2] * position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
