use super::*;
use crate::wind::WindTornadoEffect;
use cgmath::{InnerSpace, Vector3};

const REFERENCE_STEPS: usize = 20000;

fn params() -> WindShellParams {
    WindShellParams::from_effect(&WindTornadoEffect::default())
}

fn midpoint_optical_depth(
    params: &WindShellParams,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
    t_near: f32,
    t_far: f32,
) -> f64 {
    let step = (t_far - t_near) as f64 / REFERENCE_STEPS as f64;
    let mut total = 0.0f64;
    for i in 0..REFERENCE_STEPS {
        let t = t_near as f64 + (i as f64 + 0.5) * step;
        let point = origin + direction * t as f32;
        total += wind_density_at(params, point) as f64 * step;
    }
    total
}

fn assert_closed_form_matches_reference(origin: Vector3<f32>, direction: Vector3<f32>) {
    let params = params();
    let direction = direction.normalize();
    let mut t_near = 0.0;
    let mut t_far = 1e4;
    assert!(
        clamp_ray_to_wind_cone(&params, origin, direction, &mut t_near, &mut t_far),
        "ray must hit the envelope"
    );
    let closed = wind_optical_depth(&params, origin, direction, t_near, t_far) as f64;
    let reference = midpoint_optical_depth(&params, origin, direction, t_near, t_far);
    assert!(
        reference > 1e-3,
        "reference {reference} too small to compare"
    );
    let relative = (closed - reference).abs() / reference;
    assert!(
        relative < 2e-3,
        "closed {closed} vs reference {reference} (rel {relative}) for o={origin:?} d={direction:?}"
    );
}

#[test]
fn horizontal_ray_through_the_wall_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference(
        Vector3::new(-5.0, 0.8, 0.05),
        Vector3::new(1.0, 0.0, 0.0),
    );
}

#[test]
fn oblique_ray_crossing_the_top_fade_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference(
        Vector3::new(-3.0, 0.2, 0.3),
        Vector3::new(1.0, 0.55, -0.1),
    );
}

#[test]
fn ray_through_the_core_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference(
        Vector3::new(-4.0, 1.0, 0.0),
        Vector3::new(1.0, 0.02, 0.0),
    );
}

#[test]
fn ray_starting_inside_the_wall_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference(Vector3::new(0.4, 0.5, 0.0), Vector3::new(0.3, 0.2, 1.0));
}

#[test]
fn near_axial_ray_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference(
        Vector3::new(0.42, -1.0, 0.0),
        Vector3::new(0.001, 1.0, 0.0),
    );
}

#[test]
fn ray_missing_the_envelope_has_no_optical_depth() {
    let params = params();
    let origin = Vector3::new(-5.0, 0.5, 5.0);
    let direction = Vector3::new(1.0, 0.0, 0.0);
    let mut t_near = 0.0;
    let mut t_far = 1e4;
    assert!(!clamp_ray_to_wind_cone(
        &params,
        origin,
        direction,
        &mut t_near,
        &mut t_far
    ));
}

#[test]
fn density_vanishes_outside_the_height_slab_and_the_shell_supports() {
    let params = params();
    assert_eq!(wind_density_at(&params, Vector3::new(0.35, -0.1, 0.0)), 0.0);
    assert_eq!(wind_density_at(&params, Vector3::new(0.35, 2.1, 0.0)), 0.0);
    assert_eq!(wind_density_at(&params, Vector3::new(3.0, 1.0, 0.0)), 0.0);
    assert!(wind_density_at(&params, Vector3::new(0.35, 0.5, 0.0)) > 0.0);
    assert!(wind_density_at(&params, Vector3::new(0.0, 0.5, 0.0)) > 0.0);
}

#[test]
fn knots_are_sorted_and_bracketed() {
    let params = params();
    let origin = Vector3::new(-3.0, 0.2, 0.3);
    let direction = Vector3::new(1.0, 0.55, -0.1).normalize();
    let mut t_near = 0.0;
    let mut t_far = 1e4;
    clamp_ray_to_wind_cone(&params, origin, direction, &mut t_near, &mut t_far);
    let (knots, count) = wind_ray_knots(&params, origin, direction, t_near, t_far);
    assert!(count >= 2);
    assert_eq!(knots[0], t_near);
    assert_eq!(knots[count - 1], t_far);
    for i in 1..count {
        assert!(knots[i - 1] <= knots[i], "knots {:?}", &knots[..count]);
    }
}
