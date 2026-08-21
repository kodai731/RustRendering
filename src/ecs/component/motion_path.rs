#[derive(Clone, Debug)]
pub struct MotionPath {
    pub center: cgmath::Vector3<f32>,
    pub radius: f32,
    pub angular_speed: f32,
    pub phase_offset: f32,
    pub enabled: bool,
}

impl Default for MotionPath {
    fn default() -> Self {
        Self {
            center: cgmath::Vector3::new(0.0, 0.0, 0.0),
            radius: 0.0,
            angular_speed: 0.0,
            phase_offset: 0.0,
            enabled: false,
        }
    }
}

/// Compute the XZ-plane circular position for a given time.
///
/// `center + (cos(phase_offset + angular_speed * time) * radius, 0, sin(...) * radius)`
pub fn motion_path_position(path: &MotionPath, time: f32) -> cgmath::Vector3<f32> {
    let angle = path.phase_offset + path.angular_speed * time;
    let x = path.center.x + angle.cos() * path.radius;
    let z = path.center.z + angle.sin() * path.radius;
    cgmath::Vector3::new(x, path.center.y, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_periodicity_and_radius() {
        let omega = 1.0;
        let path = MotionPath {
            center: cgmath::Vector3::new(0.0, 5.0, 0.0),
            radius: 2.0,
            angular_speed: omega,
            phase_offset: 0.0,
            enabled: true,
        };

        let period = 2.0 * PI / omega;
        let pos_0 = motion_path_position(&path, 0.0);
        let pos_period = motion_path_position(&path, period);

        // Periodicity: t=0 and t=2π/ω should be the same position
        assert!(
            (pos_0.x - pos_period.x).abs() < 1e-6,
            "x mismatch: {} vs {}",
            pos_0.x,
            pos_period.x
        );
        assert!(
            (pos_0.y - pos_period.y).abs() < 1e-6,
            "y mismatch: {} vs {}",
            pos_0.y,
            pos_period.y
        );
        assert!(
            (pos_0.z - pos_period.z).abs() < 1e-6,
            "z mismatch: {} vs {}",
            pos_0.z,
            pos_period.z
        );

        // Radius: distance from center should equal radius
        let dx = pos_0.x - path.center.x;
        let dz = pos_0.z - path.center.z;
        let dist = (dx * dx + dz * dz).sqrt();
        assert!(
            (dist - path.radius).abs() < 1e-6,
            "radius mismatch: {} vs {}",
            dist,
            path.radius
        );
    }
}
