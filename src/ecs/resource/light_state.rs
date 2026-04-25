use cgmath::Vector3;

pub use thyllore_render_core::DistanceAttenuation;

#[derive(Clone, Debug)]
pub struct LightState {
    pub light_position: Vector3<f32>,
    pub shadow_strength: f32,
    pub shadow_normal_offset: f32,
    pub distance_attenuation: DistanceAttenuation,
}

impl Default for LightState {
    fn default() -> Self {
        Self {
            light_position: Vector3::new(1.0, 1.0, 2.0),
            shadow_strength: 1.0,
            shadow_normal_offset: 0.5,
            distance_attenuation: DistanceAttenuation::Disabled,
        }
    }
}
