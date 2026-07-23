#[derive(Clone, Debug)]
pub struct BatchFlameOrbit {
    pub radius: f32,
    pub period_seconds: f32,
    pub initial: Option<cgmath::Vector3<f32>>,
}
