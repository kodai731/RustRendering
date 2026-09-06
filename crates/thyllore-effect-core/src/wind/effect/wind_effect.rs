use cgmath::{Matrix4, Quaternion, Vector3};

#[derive(Clone, Debug, PartialEq)]
pub struct WindTornadoEffect {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub time: f32,
    pub time_scale: f32,
    pub time_offset: f32,
    pub column_height: f32,
    pub core_radius: f32,
    pub core_strength: f32,
    pub wall_radius_base: f32,
    pub wall_radius_top: f32,
    pub wall_width_q: f32,
    pub wall_strength: f32,
    pub top_fade: f32,
    pub density: f32,
    pub albedo: [f32; 3],
    pub ambient_brightness: f32,
    pub phase_g: f32,
    pub sun_intensity: f32,
    pub rise_initial_height: f32,
    pub rise_duration: f32,
    pub spread_start: f32,
    pub spread_rate: f32,
    pub dissipate_start: f32,
    pub dissipate_time: f32,
    pub ring_height: f32,
    pub ring_radius: f32,
    pub ring_width_q: f32,
    pub ring_strength: f32,
    pub ring_spread_rate: f32,
}

impl Default for WindTornadoEffect {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            time: 0.0,
            time_scale: 1.0,
            time_offset: 0.0,
            column_height: 2.0,
            core_radius: 0.15,
            core_strength: 0.5,
            wall_radius_base: 0.35,
            wall_radius_top: 0.6,
            wall_width_q: 0.08,
            wall_strength: 1.0,
            top_fade: 0.3,
            density: 4.0,
            albedo: [0.9, 0.93, 1.0],
            ambient_brightness: 1.0,
            phase_g: 0.6,
            sun_intensity: 1.0,
            rise_initial_height: 1.0,
            rise_duration: 1.0,
            spread_start: 0.0,
            spread_rate: 0.0,
            dissipate_start: 0.0,
            dissipate_time: 0.0,
            ring_height: 0.5,
            ring_radius: 0.3,
            ring_width_q: 0.05,
            ring_strength: 0.0,
            ring_spread_rate: 0.0,
        }
    }
}

pub fn build_wind_model_matrix(effect: &WindTornadoEffect) -> Matrix4<f32> {
    Matrix4::from_translation(effect.position) * Matrix4::from(effect.rotation)
}
