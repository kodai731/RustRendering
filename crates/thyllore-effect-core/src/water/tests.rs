use super::*;
use cgmath::{Deg, Matrix4, Quaternion, Rotation3, SquareMatrix, Vector3};

#[test]
fn test_default_water_effect_values() {
    let effect = WaterTorusEffect::default();
    assert_eq!(effect.position, Vector3::new(0.0, 0.0, 0.0));
    assert_eq!(effect.rotation, Quaternion::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(effect.time, 0.0);
    assert_eq!(effect.time_scale, 1.0);
    assert_eq!(effect.time_offset, 0.0);
    assert_eq!(effect.major_radius, 1.0);
    assert_eq!(effect.minor_radius, 0.3);
    assert_eq!(effect.ior, 1.333);
    assert_eq!(effect.absorption, [0.35, 0.08, 0.02]);
    assert_eq!(effect.flow_longitudinal, 0.2);
    assert_eq!(effect.flow_meridional, 0.0);
    assert_eq!(effect.wave_amplitude, 0.02);
    assert_eq!(effect.wave_frequency, 6.0);
    assert_eq!(effect.wave_speed, 1.0);
    assert_eq!(effect.reflect_strength, 1.0);
    assert_eq!(effect.refract_strength, 1.0);
    assert_eq!(effect.caustic_strength, 0.6);
    assert_eq!(effect.light_intensity, 1.0);
    assert_eq!(effect.highlight_sharpness, 64.0);
    assert_eq!(effect.sky_brightness, 1.0);
    assert_eq!(effect.scatter_strength, 1.0);
    assert_eq!(effect.scatter_anisotropy, 0.0);
    assert_eq!(effect.tint, [0.05, 0.25, 0.35]);
}

#[test]
fn test_advance_water_time() {
    let mut effect = WaterTorusEffect::default();
    advance_water_time(&mut effect, 1.0);
    assert_eq!(effect.time, 1.0);
    advance_water_time(&mut effect, 0.5);
    assert_eq!(effect.time, 1.5);
}

#[test]
fn test_advance_water_time_negative_is_noop() {
    let mut effect = WaterTorusEffect::default();
    effect.time = 1.0;
    advance_water_time(&mut effect, -0.5);
    assert_eq!(effect.time, 1.0);
}

#[test]
fn test_build_water_model_matrix_identity() {
    let effect = WaterTorusEffect::default();
    let matrix = build_water_model_matrix(&effect);
    let expected: Matrix4<f32> = SquareMatrix::identity();
    for i in 0..4 {
        for j in 0..4 {
            assert!(
                (matrix[i][j] - expected[i][j]).abs() < 1e-6,
                "matrix[{}][{}] = {}, expected {}",
                i,
                j,
                matrix[i][j],
                expected[i][j]
            );
        }
    }
}

#[test]
fn test_build_water_model_matrix_translation() {
    let effect = WaterTorusEffect {
        position: Vector3::new(1.0, 2.0, 3.0),
        ..WaterTorusEffect::default()
    };
    let matrix = build_water_model_matrix(&effect);
    assert!((matrix[3][0] - 1.0).abs() < 1e-6);
    assert!((matrix[3][1] - 2.0).abs() < 1e-6);
    assert!((matrix[3][2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_build_water_model_matrix_rotation() {
    let effect = WaterTorusEffect {
        rotation: Quaternion::from_angle_z(Deg(90.0)),
        major_radius: 1.0,
        minor_radius: 1.0,
        ..WaterTorusEffect::default()
    };
    let matrix = build_water_model_matrix(&effect);
    let expected_rotation = Matrix4::from(effect.rotation);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (matrix[i][j] - expected_rotation[i][j]).abs() < 1e-6,
                "rotation matrix[{}][{}] = {}, expected {}",
                i,
                j,
                matrix[i][j],
                expected_rotation[i][j]
            );
        }
    }
}

#[test]
fn test_water_secondary_rays_parse_matches_shader_values() {
    let cases = [
        ("rayquery", WaterSecondaryRays::RayQuery, 0),
        ("screenspace", WaterSecondaryRays::ScreenSpace, 1),
        ("raytracing", WaterSecondaryRays::RayTracingPipeline, 2),
    ];
    for (name, mode, shader_value) in cases {
        assert_eq!(WaterSecondaryRays::parse(name), Some(mode));
        assert_eq!(mode.as_shader_value(), shader_value);
    }
    assert_eq!(WaterSecondaryRays::parse("unknown"), None);
}

#[test]
fn test_water_render_settings_default() {
    let settings = WaterRenderSettings::default();
    assert_eq!(settings.secondary_rays, WaterSecondaryRays::RayQuery);
    assert_eq!(settings.debug_view, 0);
}
