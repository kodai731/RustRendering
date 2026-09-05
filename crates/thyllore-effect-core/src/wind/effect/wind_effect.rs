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
        }
    }
}

pub fn build_wind_model_matrix(effect: &WindTornadoEffect) -> Matrix4<f32> {
    Matrix4::from_translation(effect.position) * Matrix4::from(effect.rotation)
}
