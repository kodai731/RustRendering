use cgmath::Vector3;

use crate::ecs::resource::Camera;
use crate::helm::components::tool_call::{ShotPreset, SpeedPreset};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub pivot: Vector3<f32>,
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}

impl CameraPose {
    pub fn from_camera(camera: &Camera) -> Self {
        Self {
            pivot: camera.pivot,
            yaw: camera.yaw,
            pitch: camera.pitch,
            distance: camera.distance,
        }
    }

    pub fn apply_to(&self, camera: &mut Camera) {
        camera.pivot = self.pivot;
        camera.yaw = self.yaw;
        camera.pitch = self.pitch;
        camera.distance = self.distance;
    }
}

#[derive(Clone, Debug)]
pub struct CameraShotTween {
    pub start: CameraPose,
    pub end: CameraPose,
    pub elapsed: f32,
    pub duration: f32,
}

#[derive(Clone, Debug, Default)]
pub struct CameraShotMotion {
    pub active: Option<CameraShotTween>,
}

fn clamp_pitch(pitch: f32) -> f32 {
    let max_pitch = std::f32::consts::FRAC_PI_2 - 0.001;
    pitch.clamp(-max_pitch, max_pitch)
}

pub fn plan_camera_shot(
    camera: &Camera,
    preset: ShotPreset,
    speed: SpeedPreset,
    target: Option<Vector3<f32>>,
) -> CameraShotTween {
    let start = CameraPose::from_camera(camera);
    let mut end = start;

    match preset {
        ShotPreset::LookAtSelection => {
            if let Some(t) = target {
                end.pivot = t;
            }
        }
        ShotPreset::OrbitAroundSelection => {
            if let Some(t) = target {
                end.pivot = t;
            }
            end.yaw += std::f32::consts::TAU;
        }
        ShotPreset::DollyIn => {
            end.distance *= 0.6;
        }
        ShotPreset::DollyOut => {
            end.distance *= 1.6;
        }
        ShotPreset::CraneUp => {
            end.pitch = clamp_pitch(end.pitch + 0.35);
        }
        ShotPreset::CraneDown => {
            end.pitch = clamp_pitch(end.pitch - 0.35);
        }
    }

    let duration = match speed {
        SpeedPreset::Slow => 2.0,
        SpeedPreset::Normal => 1.0,
        SpeedPreset::Fast => 0.5,
    };

    CameraShotTween {
        start,
        end,
        elapsed: 0.0,
        duration,
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn vector3_lerp(a: Vector3<f32>, b: Vector3<f32>, t: f32) -> Vector3<f32> {
    Vector3::new(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t))
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

pub fn camera_shot_step(camera: &mut Camera, motion: &mut CameraShotMotion, delta_time: f32) {
    let tween = match &mut motion.active {
        Some(t) => t,
        None => return,
    };

    tween.elapsed += delta_time;
    let t = (tween.elapsed / tween.duration).min(1.0);
    let eased = smoothstep(t);

    let pose = CameraPose {
        pivot: vector3_lerp(tween.start.pivot, tween.end.pivot, eased),
        yaw: lerp(tween.start.yaw, tween.end.yaw, eased),
        pitch: lerp(tween.start.pitch, tween.end.pitch, eased),
        distance: lerp(tween.start.distance, tween.end.distance, eased),
    };

    pose.apply_to(camera);

    if t >= 1.0 {
        motion.active = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_camera(pivot: Vector3<f32>, yaw: f32, pitch: f32, distance: f32) -> Camera {
        Camera {
            pivot,
            yaw,
            pitch,
            distance,
            fov_y: cgmath::Deg(45.0),
            near_plane: 0.1,
            initial_pivot: pivot,
            initial_yaw: yaw,
            initial_pitch: pitch,
            initial_distance: distance,
        }
    }

    #[test]
    fn test_dolly_in_distance() {
        let camera = make_camera(Vector3::new(0.0, 0.0, 0.0), 0.0, 0.0, 10.0);
        let tween = plan_camera_shot(&camera, ShotPreset::DollyIn, SpeedPreset::Normal, None);
        assert_eq!(tween.start.distance, 10.0);
        assert!((tween.end.distance - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_crane_up_clamp() {
        let camera = make_camera(Vector3::new(0.0, 0.0, 0.0), 0.0, 1.5, 10.0);
        let tween = plan_camera_shot(&camera, ShotPreset::CraneUp, SpeedPreset::Normal, None);
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.001;
        assert!(tween.end.pitch <= max_pitch);
    }

    #[test]
    fn test_look_at_selection_pivot() {
        let camera = make_camera(Vector3::new(0.0, 0.0, 0.0), 0.0, 0.0, 10.0);
        let target = Vector3::new(1.0, 2.0, 3.0);
        let tween = plan_camera_shot(
            &camera,
            ShotPreset::LookAtSelection,
            SpeedPreset::Normal,
            Some(target),
        );
        assert_eq!(tween.end.pivot, target);
    }

    #[test]
    fn test_step_completion() {
        let mut camera = make_camera(Vector3::new(0.0, 0.0, 0.0), 0.0, 0.0, 10.0);
        let tween = plan_camera_shot(&camera, ShotPreset::DollyIn, SpeedPreset::Normal, None);
        let mut motion = CameraShotMotion {
            active: Some(tween),
        };

        // Step with time >= duration to complete
        camera_shot_step(&mut camera, &mut motion, 1.0);

        assert!(motion.active.is_none());
        let pose = CameraPose::from_camera(&camera);
        assert!((pose.distance - 6.0).abs() < 1e-6);
    }
}
