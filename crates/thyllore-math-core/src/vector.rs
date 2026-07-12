use cgmath::{MetricSpace, Vector2, Vector3};
use std::ops::{Add, AddAssign, Deref, DerefMut, Mul, Neg};

pub trait ToArray2 {
    fn to_array(self) -> [f32; 2];
}

pub trait ToArray3 {
    fn to_array(self) -> [f32; 3];
}

pub trait ToArray4 {
    fn to_array(self) -> [f32; 4];
}

pub trait ToVec2 {
    fn to_vec2(self) -> Vec2;
}

pub trait ToVec3 {
    fn to_vec3(self) -> Vec3;
}

pub trait ToVec4 {
    fn to_vec4(self) -> cgmath::Vector4<f32>;
}

impl ToArray2 for Vector2<f32> {
    fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }
}

impl ToArray3 for Vector3<f32> {
    fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl ToArray4 for cgmath::Vector4<f32> {
    fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

impl ToVec2 for [f32; 2] {
    fn to_vec2(self) -> Vec2 {
        Vec2::new(self[0], self[1])
    }
}

impl ToVec3 for [f32; 3] {
    fn to_vec3(self) -> Vec3 {
        Vec3::new(self[0], self[1], self[2])
    }
}

impl ToVec4 for [f32; 4] {
    fn to_vec4(self) -> cgmath::Vector4<f32> {
        cgmath::Vector4::new(self[0], self[1], self[2], self[3])
    }
}

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

impl Neg for Vec2 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self(cgmath::Vector2::new(x, y))
    }

    pub fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    pub fn distance(&self, other: Self) -> f32 {
        cgmath::Vector2::distance(self.0, other.0)
    }
}

impl ToArray2 for Vec2 {
    fn to_array(self) -> [f32; 2] {
        [self[0], self[1]]
    }
}

impl From<Vec2> for Vector2<f32> {
    fn from(v: Vec2) -> Self {
        v.0
    }
}

impl From<Vector2<f32>> for Vec2 {
    fn from(v: Vector2<f32>) -> Self {
        Self(v)
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

impl DerefMut for Vec3 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl PartialEq for Vec3 {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(cgmath::Vector3::new(x, y, z))
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

impl ToVec3 for Vector3<f32> {
    fn to_vec3(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

impl From<Vec3> for Vector3<f32> {
    fn from(v: Vec3) -> Self {
        v.0
    }
}

impl From<Vector3<f32>> for Vec3 {
    fn from(v: Vector3<f32>) -> Self {
        Self(v)
    }
}

pub type Vector4 = cgmath::Vector4<f32>;

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

impl Neg for Vec4 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Vec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(cgmath::Vector4::new(x, y, z, w))
    }

    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }
}

impl ToVec4 for Vec4 {
    fn to_vec4(self) -> cgmath::Vector4<f32> {
        self.0
    }
}

#[deprecated(note = "use [f32; 2].to_vec2() instead")]
pub fn vec2_from_array(a: [f32; 2]) -> Vec2 {
    a.to_vec2()
}

#[deprecated(note = "use [f32; 3].to_vec3() instead")]
pub fn vec3_from_array(a: [f32; 3]) -> Vec3 {
    a.to_vec3()
}

#[deprecated(note = "use [f32; 4].to_vec4() instead")]
pub fn vec4_from_array(a: [f32; 4]) -> cgmath::Vector4<f32> {
    a.to_vec4()
}

#[deprecated(note = "use Vector2::to_array() instead")]
pub fn array2_from_vec(v: Vector2<f32>) -> [f32; 2] {
    v.to_array()
}

#[deprecated(note = "use Vector3::to_array() instead")]
pub fn array3_from_vec(v: Vector3<f32>) -> [f32; 3] {
    v.to_array()
}

#[deprecated(note = "use Vector4::to_array() instead")]
pub fn array4_from_vec(v: cgmath::Vector4<f32>) -> [f32; 4] {
    v.to_array()
}

pub fn approx_equal_array3(a: &[f32; 3], b: &[f32; 3]) -> bool {
    (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5 && (a[2] - b[2]).abs() < 1e-5
}

pub fn approx_equal_vec4(a: &Vector4, b: &Vector4) -> bool {
    (a.x - b.x).abs() < 1e-3
        && (a.y - b.y).abs() < 1e-3
        && (a.z - b.z).abs() < 1e-3
        && (a.w - b.w).abs() < 1e-3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_to_vec2() {
        let v = [3.5, 4.5].to_vec2();
        assert_eq!(v.x, 3.5);
        assert_eq!(v.y, 4.5);
    }

    #[test]
    fn test_vec2_to_array() {
        let arr = Vector2::new(1.5, 2.5).to_array();
        assert_eq!(arr, [1.5, 2.5]);
    }

    #[test]
    fn test_array_to_vec3() {
        let v = [1.0, 2.0, 3.0].to_vec3();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_to_array() {
        let arr = Vector3::new(1.5, 2.5, 3.5).to_array();
        assert_eq!(arr, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn test_array_to_vec4() {
        let v = [1.0, 2.0, 3.0, 4.0].to_vec4();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4_to_array() {
        let arr = cgmath::Vector4::new(1.5, 2.5, 3.5, 4.5).to_array();
        assert_eq!(arr, [1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_vec2_new() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vec2_default() {
        let v = Vec2::default();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn test_vec2_distance() {
        let v1 = Vec2::new(0.0, 0.0);
        let v2 = Vec2::new(3.0, 4.0);
        let distance = v1.distance(v2);
        assert!((distance - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_vec2_neg() {
        let v = Vec2::new(1.0, -2.0);
        let negated = -v;
        assert_eq!(negated.x, -1.0);
        assert_eq!(negated.y, 2.0);
    }

    #[test]
    fn test_vec3_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_default() {
        let v = Vec3::default();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn test_vec3_add() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let result = v1 + v2;
        assert_eq!(result.x, 5.0);
        assert_eq!(result.y, 7.0);
        assert_eq!(result.z, 9.0);
    }

    #[test]
    fn test_vec3_mul_scalar() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = v * 2.0;
        assert_eq!(result.x, 2.0);
        assert_eq!(result.y, 4.0);
        assert_eq!(result.z, 6.0);
    }

    #[test]
    fn test_vec3_neg() {
        let v = Vec3::new(1.0, -2.0, 3.0);
        let negated = -v;
        assert_eq!(negated.x, -1.0);
        assert_eq!(negated.y, 2.0);
        assert_eq!(negated.z, -3.0);
    }

    #[test]
    fn test_vec3_add_assign() {
        let mut v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        v1 += v2;
        assert_eq!(v1.x, 5.0);
        assert_eq!(v1.y, 7.0);
        assert_eq!(v1.z, 9.0);
    }

    #[test]
    fn test_vec4_new() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4_default() {
        let v = Vec4::default();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
        assert_eq!(v.w, 0.0);
    }

    #[test]
    fn test_vec4_neg() {
        let v = Vec4::new(1.0, -2.0, 3.0, -4.0);
        let negated = -v;
        assert_eq!(negated.x, -1.0);
        assert_eq!(negated.y, 2.0);
        assert_eq!(negated.z, -3.0);
        assert_eq!(negated.w, 4.0);
    }

    #[test]
    fn test_approx_equal_vec4() {
        let a = cgmath::Vector4::new(1.0, 2.0, 3.0, 4.0);
        let b = cgmath::Vector4::new(1.0001, 2.0001, 3.0001, 4.0001);
        assert!(approx_equal_vec4(&a, &b));
    }

    #[test]
    fn test_vec3_equality() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        let v3 = Vec3::new(1.0, 2.0, 3.1);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_vec2_equality() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(1.0, 2.0);
        let v3 = Vec2::new(1.0, 2.1);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_vec4_equality() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v3 = Vec4::new(1.0, 2.0, 3.0, 4.1);
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }
}
