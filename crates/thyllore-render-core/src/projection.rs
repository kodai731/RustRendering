use cgmath::{Matrix4, Vector2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceAttenuation {
    Enabled,
    Disabled,
}

impl DistanceAttenuation {
    pub fn is_enabled(self) -> bool {
        self == Self::Enabled
    }

    pub fn as_int(self) -> i32 {
        if self == Self::Enabled {
            1
        } else {
            0
        }
    }
}

pub struct ProjectionData {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
    pub screen_size: Vector2<f32>,
    pub aspect: f32,
}
