use crate::ecs::world::Entity;
use cgmath::{Deg, Vector3};
use thyllore_render_core::PhysicalCameraParameters;

#[derive(Clone, Debug)]
pub struct CameraComponent {
    pub fov_y: Deg<f32>,
    pub near_plane: f32,
    pub far_plane: Option<f32>,
    pub physical: PhysicalCameraParameters,
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            fov_y: Deg(45.0),
            near_plane: 0.1,
            far_plane: None,
            physical: Default::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CameraAimTarget {
    pub target: Entity,
    pub up_target: Option<Entity>,
    pub aim_axis: Vector3<f32>,
    pub up_axis: Vector3<f32>,
    pub weight: f32,
}

impl CameraAimTarget {
    /// Initialize with default values for looking at a target.
    /// aim_axis is (0,0,-1) (glTF/Blender camera forward is -Z),
    /// up_axis is (0,1,0), weight is 1.0, up_target is None.
    pub fn look_at(target: Entity) -> Self {
        Self {
            target,
            up_target: None,
            aim_axis: Vector3::new(0.0, 0.0, -1.0),
            up_axis: Vector3::new(0.0, 1.0, 0.0),
            weight: 1.0,
        }
    }
}
