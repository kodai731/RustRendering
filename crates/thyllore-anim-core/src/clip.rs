use cgmath::{Matrix4, Quaternion, Vector3};
use std::collections::HashMap;

use crate::{BoneId, Skeleton, SkeletonId};

pub type AnimationClipId = u32;

#[derive(Clone, Debug, PartialEq)]
pub enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

impl Default for Interpolation {
    fn default() -> Self {
        Self::Linear
    }
}

#[derive(Clone, Debug)]
pub struct Keyframe<T> {
    pub time: f32,
    pub value: T,
    pub interpolation: Interpolation,
    pub in_tangent: Option<T>,
    pub out_tangent: Option<T>,
}

impl<T> Keyframe<T> {
    pub fn new(time: f32, value: T) -> Self {
        Self {
            time,
            value,
            interpolation: Interpolation::Linear,
            in_tangent: None,
            out_tangent: None,
        }
    }

    pub fn with_interpolation(time: f32, value: T, interpolation: Interpolation) -> Self {
        Self {
            time,
            value,
            interpolation,
            in_tangent: None,
            out_tangent: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransformChannel {
    pub translation: Vec<Keyframe<Vector3<f32>>>,
    pub rotation: Vec<Keyframe<Quaternion<f32>>>,
    pub scale: Vec<Keyframe<Vector3<f32>>>,
}

impl TransformChannel {
    pub fn sample_translation(&self, time: f32) -> Option<Vector3<f32>> {
        Self::sample_vec3(&self.translation, time, None)
    }

    pub fn sample_translation_looped(&self, time: f32, duration: f32) -> Option<Vector3<f32>> {
        Self::sample_vec3(&self.translation, time, Some(duration))
    }

    pub fn sample_rotation(&self, time: f32) -> Option<Quaternion<f32>> {
        Self::sample_quat(&self.rotation, time, None)
    }

    pub fn sample_rotation_looped(&self, time: f32, duration: f32) -> Option<Quaternion<f32>> {
        Self::sample_quat(&self.rotation, time, Some(duration))
    }

    pub fn sample_scale(&self, time: f32) -> Option<Vector3<f32>> {
        Self::sample_vec3(&self.scale, time, None)
    }

    pub fn sample_scale_looped(&self, time: f32, duration: f32) -> Option<Vector3<f32>> {
        Self::sample_vec3(&self.scale, time, Some(duration))
    }

    fn sample_vec3(
        keyframes: &[Keyframe<Vector3<f32>>],
        time: f32,
        duration: Option<f32>,
    ) -> Option<Vector3<f32>> {
        if keyframes.is_empty() {
            return None;
        }
        if keyframes.len() == 1 {
            return Some(keyframes[0].value);
        }

        if time <= keyframes[0].time {
            return Some(keyframes[0].value);
        }

        let Some(last_kf) = keyframes.last() else {
            return None;
        };
        if time >= last_kf.time {
            if let Some(dur) = duration {
                if dur > last_kf.time && time < dur {
                    let first_kf = &keyframes[0];
                    let wrap_duration = dur - last_kf.time + first_kf.time;
                    if wrap_duration > 0.0001 {
                        let t = (time - last_kf.time) / wrap_duration;
                        return Some(last_kf.value + (first_kf.value - last_kf.value) * t);
                    }
                }
            }
            return Some(last_kf.value);
        }

        let i = crate::keyframe_search::find_keyframe_segment(keyframes, time);
        let k0 = &keyframes[i];
        let k1 = &keyframes[i + 1];
        let t = (time - k0.time) / (k1.time - k0.time);

        match k0.interpolation {
            Interpolation::Step => Some(k1.value),
            Interpolation::Linear => Some(k0.value + (k1.value - k0.value) * t),
            Interpolation::CubicSpline => Some(hermite_vec3(k0, k1, t)),
        }
    }

    fn sample_quat(
        keyframes: &[Keyframe<Quaternion<f32>>],
        time: f32,
        duration: Option<f32>,
    ) -> Option<Quaternion<f32>> {
        if keyframes.is_empty() {
            return None;
        }
        if keyframes.len() == 1 {
            return Some(keyframes[0].value);
        }

        if time <= keyframes[0].time {
            return Some(keyframes[0].value);
        }

        let Some(last_kf) = keyframes.last() else {
            return None;
        };
        if time >= last_kf.time {
            if let Some(dur) = duration {
                if dur > last_kf.time && time < dur {
                    let first_kf = &keyframes[0];
                    let wrap_duration = dur - last_kf.time + first_kf.time;
                    if wrap_duration > 0.0001 {
                        let t = (time - last_kf.time) / wrap_duration;
                        return Some(slerp(last_kf.value, first_kf.value, t));
                    }
                }
            }
            return Some(last_kf.value);
        }

        let i = crate::keyframe_search::find_keyframe_segment(keyframes, time);
        let k0 = &keyframes[i];
        let k1 = &keyframes[i + 1];
        let t = (time - k0.time) / (k1.time - k0.time);

        match k0.interpolation {
            Interpolation::Step => Some(k1.value),
            Interpolation::Linear => Some(slerp(k0.value, k1.value, t)),
            Interpolation::CubicSpline => Some(normalize_quat(hermite_quat(k0, k1, t))),
        }
    }
}

fn hermite_vec3(k0: &Keyframe<Vector3<f32>>, k1: &Keyframe<Vector3<f32>>, t: f32) -> Vector3<f32> {
    let dt = k1.time - k0.time;
    let t2 = t * t;
    let t3 = t2 * t;

    let b0 = k0.out_tangent.unwrap_or(Vector3::new(0.0, 0.0, 0.0));
    let a1 = k1.in_tangent.unwrap_or(Vector3::new(0.0, 0.0, 0.0));

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = (t3 - 2.0 * t2 + t) * dt;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = (t3 - t2) * dt;

    k0.value * h00 + b0 * h10 + k1.value * h01 + a1 * h11
}

fn hermite_quat(
    k0: &Keyframe<Quaternion<f32>>,
    k1: &Keyframe<Quaternion<f32>>,
    t: f32,
) -> Quaternion<f32> {
    let dt = k1.time - k0.time;
    let t2 = t * t;
    let t3 = t2 * t;

    let zero_q = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    let b0 = k0.out_tangent.unwrap_or(zero_q);
    let a1 = k1.in_tangent.unwrap_or(zero_q);

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = (t3 - 2.0 * t2 + t) * dt;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = (t3 - t2) * dt;

    Quaternion::new(
        k0.value.s * h00 + b0.s * h10 + k1.value.s * h01 + a1.s * h11,
        k0.value.v.x * h00 + b0.v.x * h10 + k1.value.v.x * h01 + a1.v.x * h11,
        k0.value.v.y * h00 + b0.v.y * h10 + k1.value.v.y * h01 + a1.v.y * h11,
        k0.value.v.z * h00 + b0.v.z * h10 + k1.value.v.z * h01 + a1.v.z * h11,
    )
}

pub fn slerp(a: Quaternion<f32>, b: Quaternion<f32>, t: f32) -> Quaternion<f32> {
    let dot = a.s * b.s + a.v.x * b.v.x + a.v.y * b.v.y + a.v.z * b.v.z;

    let (b, dot) = if dot < 0.0 {
        (Quaternion::new(-b.s, -b.v.x, -b.v.y, -b.v.z), -dot)
    } else {
        (b, dot)
    };

    let dot = dot.clamp(-1.0, 1.0);

    if dot > 0.9995 {
        let result = Quaternion::new(
            a.s + t * (b.s - a.s),
            a.v.x + t * (b.v.x - a.v.x),
            a.v.y + t * (b.v.y - a.v.y),
            a.v.z + t * (b.v.z - a.v.z),
        );
        return normalize_quat(result);
    }

    let theta_0 = dot.acos();
    let sin_theta_0 = theta_0.sin();

    let s0 = ((1.0 - t) * theta_0).sin() / sin_theta_0;
    let s1 = (t * theta_0).sin() / sin_theta_0;

    let result = Quaternion::new(
        s0 * a.s + s1 * b.s,
        s0 * a.v.x + s1 * b.v.x,
        s0 * a.v.y + s1 * b.v.y,
        s0 * a.v.z + s1 * b.v.z,
    );
    normalize_quat(result)
}

pub fn normalize_quat(q: Quaternion<f32>) -> Quaternion<f32> {
    let len = (q.s * q.s + q.v.x * q.v.x + q.v.y * q.v.y + q.v.z * q.v.z).sqrt();
    if len > 0.0 {
        Quaternion::new(q.s / len, q.v.x / len, q.v.y / len, q.v.z / len)
    } else {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AnimationClip {
    pub id: AnimationClipId,
    pub name: String,
    pub duration: f32,
    pub channels: HashMap<BoneId, TransformChannel>,
}

impl AnimationClip {
    pub fn new(name: &str) -> Self {
        Self {
            id: 0,
            name: name.to_string(),
            duration: 0.0,
            channels: HashMap::new(),
        }
    }

    pub fn add_channel(&mut self, bone_id: BoneId, channel: TransformChannel) {
        self.channels.insert(bone_id, channel);
    }
}

pub fn compose_transform(
    translation: Vector3<f32>,
    rotation: Quaternion<f32>,
    scale: Vector3<f32>,
) -> Matrix4<f32> {
    let t = Matrix4::from_translation(translation);
    let r = Matrix4::from(rotation);
    let s = Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z);
    t * r * s
}

pub fn decompose_transform(m: &Matrix4<f32>) -> (Vector3<f32>, Quaternion<f32>, Vector3<f32>) {
    let translation = Vector3::new(m[3][0], m[3][1], m[3][2]);

    let sx = (m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]).sqrt();
    let sy = (m[1][0] * m[1][0] + m[1][1] * m[1][1] + m[1][2] * m[1][2]).sqrt();
    let sz = (m[2][0] * m[2][0] + m[2][1] * m[2][1] + m[2][2] * m[2][2]).sqrt();
    let scale = Vector3::new(sx, sy, sz);

    let rot_matrix = cgmath::Matrix3::new(
        m[0][0] / sx.max(0.0001),
        m[0][1] / sx.max(0.0001),
        m[0][2] / sx.max(0.0001),
        m[1][0] / sy.max(0.0001),
        m[1][1] / sy.max(0.0001),
        m[1][2] / sy.max(0.0001),
        m[2][0] / sz.max(0.0001),
        m[2][1] / sz.max(0.0001),
        m[2][2] / sz.max(0.0001),
    );
    let rotation = Quaternion::from(rot_matrix);

    (translation, rotation, scale)
}

#[derive(Clone, Debug, Default)]
pub struct AnimationSystem {
    pub skeletons: Vec<Skeleton>,
    next_skeleton_id: SkeletonId,
}

impl AnimationSystem {
    pub fn new() -> Self {
        Self {
            skeletons: Vec::new(),
            next_skeleton_id: 0,
        }
    }

    pub fn add_skeleton(&mut self, mut skeleton: Skeleton) -> SkeletonId {
        let id = self.next_skeleton_id;
        self.next_skeleton_id += 1;
        skeleton.id = id;
        self.skeletons.push(skeleton);
        id
    }

    pub fn get_skeleton(&self, id: SkeletonId) -> Option<&Skeleton> {
        self.skeletons.iter().find(|s| s.id == id)
    }

    pub fn get_skeleton_mut(&mut self, id: SkeletonId) -> Option<&mut Skeleton> {
        self.skeletons.iter_mut().find(|s| s.id == id)
    }

    pub fn clear(&mut self) {
        self.skeletons.clear();
        self.next_skeleton_id = 0;
    }
}

#[derive(Clone, Debug, Default)]
pub struct MorphTarget {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, Default)]
pub struct MorphAnimation {
    pub key_frame: f32,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct MorphAnimationSystem {
    pub animations: Vec<MorphAnimation>,
    pub targets: Vec<Vec<MorphTarget>>,
    pub base_vertices: Vec<Vec<[f32; 3]>>,
    pub scale_factor: f32,
}

impl Default for MorphAnimationSystem {
    fn default() -> Self {
        Self {
            animations: Vec::new(),
            targets: Vec::new(),
            base_vertices: Vec::new(),
            scale_factor: 1.0,
        }
    }
}

impl MorphAnimationSystem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.animations.is_empty()
    }

    pub fn get_animation_index(&self, time: f32) -> usize {
        if self.animations.is_empty() {
            return 0;
        }

        let start_key_frame = self
            .animations
            .first()
            .expect("animations verified non-empty by early return above")
            .key_frame;
        let end_key_frame = self
            .animations
            .last()
            .expect("animations verified non-empty by early return above")
            .key_frame;
        let period = end_key_frame - start_key_frame;
        let mod_time = time.rem_euclid(period);

        let idx = self.animations.partition_point(|a| a.key_frame < mod_time);
        idx.min(self.animations.len() - 1)
    }

    pub fn clear(&mut self) {
        self.animations.clear();
        self.targets.clear();
        self.base_vertices.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_channel() -> TransformChannel {
        TransformChannel {
            translation: vec![
                Keyframe::new(0.0, Vector3::new(0.0, 0.0, 0.0)),
                Keyframe::new(1.0, Vector3::new(10.0, 0.0, 0.0)),
            ],
            rotation: vec![
                Keyframe::new(0.0, Quaternion::new(1.0, 0.0, 0.0, 0.0)),
                Keyframe::new(1.0, Quaternion::new(1.0, 0.0, 0.0, 0.0)),
            ],
            scale: vec![
                Keyframe::new(0.0, Vector3::new(1.0, 1.0, 1.0)),
                Keyframe::new(1.0, Vector3::new(2.0, 2.0, 2.0)),
            ],
        }
    }

    #[test]
    fn test_linear_translation_midpoint() {
        let ch = make_linear_channel();
        let v = ch.sample_translation(0.5).unwrap();
        assert!((v.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_linear_translation_boundary() {
        let ch = make_linear_channel();
        let v0 = ch.sample_translation(0.0).unwrap();
        let v1 = ch.sample_translation(1.0).unwrap();
        assert!((v0.x - 0.0).abs() < 0.01);
        assert!((v1.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_step_interpolation() {
        let ch = TransformChannel {
            translation: vec![
                Keyframe::with_interpolation(0.0, Vector3::new(0.0, 0.0, 0.0), Interpolation::Step),
                Keyframe::with_interpolation(
                    1.0,
                    Vector3::new(10.0, 0.0, 0.0),
                    Interpolation::Step,
                ),
            ],
            ..Default::default()
        };
        let v = ch.sample_translation(0.5).unwrap();
        assert!((v.x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_hermite_vec3_at_endpoints() {
        let k0 = Keyframe {
            time: 0.0,
            value: Vector3::new(0.0, 0.0, 0.0),
            interpolation: Interpolation::CubicSpline,
            in_tangent: Some(Vector3::new(0.0, 0.0, 0.0)),
            out_tangent: Some(Vector3::new(1.0, 0.0, 0.0)),
        };
        let k1 = Keyframe {
            time: 1.0,
            value: Vector3::new(1.0, 0.0, 0.0),
            interpolation: Interpolation::CubicSpline,
            in_tangent: Some(Vector3::new(1.0, 0.0, 0.0)),
            out_tangent: Some(Vector3::new(0.0, 0.0, 0.0)),
        };

        let v0 = hermite_vec3(&k0, &k1, 0.0);
        assert!((v0.x - 0.0).abs() < 0.001);

        let v1 = hermite_vec3(&k0, &k1, 1.0);
        assert!((v1.x - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_hermite_vec3_midpoint() {
        let k0 = Keyframe {
            time: 0.0,
            value: Vector3::new(0.0, 0.0, 0.0),
            interpolation: Interpolation::CubicSpline,
            in_tangent: Some(Vector3::new(0.0, 0.0, 0.0)),
            out_tangent: Some(Vector3::new(1.0, 0.0, 0.0)),
        };
        let k1 = Keyframe {
            time: 1.0,
            value: Vector3::new(1.0, 0.0, 0.0),
            interpolation: Interpolation::CubicSpline,
            in_tangent: Some(Vector3::new(1.0, 0.0, 0.0)),
            out_tangent: Some(Vector3::new(0.0, 0.0, 0.0)),
        };

        let mid = hermite_vec3(&k0, &k1, 0.5);
        assert!(
            mid.x > 0.0 && mid.x < 1.0,
            "midpoint x={} should be between 0 and 1",
            mid.x
        );
    }

    #[test]
    fn test_cubicspline_channel_sampling() {
        let ch = TransformChannel {
            translation: vec![
                Keyframe {
                    time: 0.0,
                    value: Vector3::new(0.0, 0.0, 0.0),
                    interpolation: Interpolation::CubicSpline,
                    in_tangent: Some(Vector3::new(0.0, 0.0, 0.0)),
                    out_tangent: Some(Vector3::new(3.0, 0.0, 0.0)),
                },
                Keyframe {
                    time: 1.0,
                    value: Vector3::new(3.0, 0.0, 0.0),
                    interpolation: Interpolation::CubicSpline,
                    in_tangent: Some(Vector3::new(3.0, 0.0, 0.0)),
                    out_tangent: Some(Vector3::new(0.0, 0.0, 0.0)),
                },
            ],
            ..Default::default()
        };

        let v = ch.sample_translation(0.5).unwrap();
        assert!(
            (v.x - 1.5).abs() < 0.5,
            "CubicSpline midpoint x={} should be near 1.5",
            v.x
        );
    }
}
