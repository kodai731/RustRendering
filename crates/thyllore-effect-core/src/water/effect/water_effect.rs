use cgmath::{Matrix4, Quaternion, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct WaterTorusEffect {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub time: f32,
    pub time_scale: f32,
    pub time_offset: f32,
    pub major_radius: f32,
    pub minor_radius: f32,
    pub ior: f32,
    pub absorption: [f32; 3],
    pub flow_longitudinal: f32,
    pub flow_meridional: f32,
    pub wave_amplitude: f32,
    pub wave_frequency: f32,
    pub wave_speed: f32,
    pub reflect_strength: f32,
    pub refract_strength: f32,
    pub tint: [f32; 3],
}

impl Default for WaterTorusEffect {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            time: 0.0,
            time_scale: 1.0,
            time_offset: 0.0,
            major_radius: 1.0,
            minor_radius: 0.3,
            ior: 1.333,
            absorption: [0.35, 0.08, 0.02],
            flow_longitudinal: 0.2,
            flow_meridional: 0.0,
            wave_amplitude: 0.02,
            wave_frequency: 6.0,
            wave_speed: 1.0,
            reflect_strength: 1.0,
            refract_strength: 1.0,
            tint: [0.05, 0.25, 0.35],
        }
    }
}

pub fn build_water_model_matrix(effect: &WaterTorusEffect) -> Matrix4<f32> {
    Matrix4::from_translation(effect.position) * Matrix4::from(effect.rotation)
}

pub fn advance_water_time(effect: &mut WaterTorusEffect, delta_time: f32) {
    effect.time += delta_time.max(0.0);
}
