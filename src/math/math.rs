pub use cgmath::Rad;
pub use cgmath::{point3, Deg, InnerSpace, MetricSpace, Vector2};
pub use cgmath::{prelude::*, Vector3};
pub use cgmath::{vec2, vec3, vec4};
use std::ops::{Deref, Mul};
#[derive(Copy, Clone, Debug)]
pub struct Vec2(cgmath::Vector2<f32>);
impl Default for Vec2 {
    fn default() -> Self {
        Self(cgmath::Vector2::new(0.0, 0.0))
    }
}

impl Deref for Vec2 {
    type Target = cgmath::Vector2<f32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Vec2 {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self(cgmath::Vector2::new(x, y))
    }

    pub fn distance(&self, other: Self) -> f32 {
        cgmath::Vector2::distance(self.0, other.0)
    }

    pub fn new_array(array: [f32; 2]) -> Self {
        Self(cgmath::Vector2::new(array[0], array[1]))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vec3(cgmath::Vector3<f32>);
impl Default for Vec3 {
    fn default() -> Self {
        Self(cgmath::Vector3::new(0.0, 0.0, 0.0))
    }
}
impl Deref for Vec3 {
    type Target = cgmath::Vector3<f32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(cgmath::Vector3::new(x, y, z))
    }

    pub fn new_array(p: [f32; 3]) -> Self {
        Self(cgmath::Vector3::new(p[0], p[1], p[2]))
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}
#[derive(Copy, Clone, Debug)]
pub struct Vec4(cgmath::Vector4<f32>);
impl Default for Vec4 {
    fn default() -> Self {
        Self(cgmath::Vector4::new(0.0, 0.0, 0.0, 0.0))
    }
}
impl Deref for Vec4 {
    type Target = cgmath::Vector4<f32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl PartialEq for Vec4 {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(cgmath::Vector4::new(x, y, z, w))
    }
}

pub type Mat3 = cgmath::Matrix3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

pub fn vec3_from_array(a: [f32; 3]) -> Vector3<f32> {
    vec3(a[0], a[1], a[2])
}

pub fn array3_from_vec(v: Vector3<f32>) -> [f32; 3] {
    [v.x, v.y, v.z]
}

pub fn vec2_from_array(a: [f32; 2]) -> Vector2<f32> {
    vec2(a[0], a[1])
}

pub fn array2_from_vec(v: Vector2<f32>) -> [f32; 2] {
    [v.x, v.y]
}

pub unsafe fn rodrigues(
    rotate: &mut cgmath::Matrix3<f32>,
    c: f32,
    s: f32,
    n: &cgmath::Vector3<f32>,
) -> anyhow::Result<()> {
    let ac = 1.0 - c;
    let xyac = n.x * n.y * ac;
    let yzac = n.y * n.z * ac;
    let zxac = n.x * n.z * ac;
    let xs = n.x * s;
    let ys = n.y * s;
    let zs = n.z * s;
    // rotate = glm::mat3(c + n.x * n.x * ac, n.x * n.y * ac + n.z * s, n.z * n.x * ac - n.y * s,
    //     n.x * n.y * ac - n.z * s, c + n.y * n.y * ac, n.y * n.z * ac + n.x * s,
    //     n.z * n.x * ac + n.y * s, n.y * n.z * ac - n.x * s, c + n.z * n.z * ac);
    *rotate = cgmath::Matrix3::new(
        c + n.x * n.x * ac,
        xyac + zs,
        zxac - ys,
        xyac - zs,
        c + n.y * n.y * ac,
        yzac + xs,
        zxac + ys,
        yzac - xs,
        c + n.z * n.z * ac,
    );
    Ok(())
}

pub unsafe fn view(
    camera_pos: cgmath::Vector3<f32>,
    direction: cgmath::Vector3<f32>,
    up: cgmath::Vector3<f32>,
) -> cgmath::Matrix4<f32> {
    let n_z = cgmath::Vector3::normalize(direction);
    let n_x = cgmath::Vector3::normalize(cgmath::Vector3::cross(up, n_z));
    let n_y = cgmath::Vector3::cross(n_x, n_z);
    let orientation = cgmath::Matrix4::new(
        n_x.x, n_y.x, n_z.x, 0.0, n_x.y, n_y.y, n_z.y, 0.0, n_x.z, n_y.z, n_z.z, 0.0, 0.0, 0.0,
        0.0, 1.0,
    );
    let translate = cgmath::Matrix4::new(
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -camera_pos.x,
        -camera_pos.y,
        -camera_pos.z,
        1.0,
    );
    return orientation * translate;
}
