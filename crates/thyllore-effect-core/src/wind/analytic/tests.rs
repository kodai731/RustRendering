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

fn evolving_params() -> WindShellParams {
    let effect = WindTornadoEffect {
        time: 1.0,
        rise_initial_height: 0.3,
        rise_duration: 2.0,
        spread_start: 0.25,
        spread_rate: 0.08,
        dissipate_start: 0.5,
        dissipate_time: 1.5,
        ..WindTornadoEffect::default()
    };
    WindShellParams::from_effect(&effect)
}

fn assert_closed_form_matches_reference(origin: Vector3<f32>, direction: Vector3<f32>) {
    assert_closed_form_matches_reference_for(params(), origin, direction);
}

fn assert_closed_form_matches_reference_for(
    params: WindShellParams,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
) {
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
fn rising_and_spreading_wall_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference_for(
        evolving_params(),
        Vector3::new(-3.0, 0.2, 0.3),
        Vector3::new(1.0, 0.35, -0.1),
    );
    assert_closed_form_matches_reference_for(
        evolving_params(),
        Vector3::new(-5.0, 0.5, 0.05),
        Vector3::new(1.0, 0.0, 0.0),
    );
}

#[test]
fn wall_evolution_moves_the_top_and_dims_the_wall() {
    let evolving = evolving_params();
    let still = params();
    assert!(evolving.h_top < still.h_top);
    assert!(evolving.spread_offset > 0.0);
    assert!(evolving.wall_strength < still.wall_strength);
    assert_eq!(
        wind_density_at(
            &evolving,
            Vector3::new(0.35, evolving.h_top * still.height + 0.1, 0.0)
        ),
        0.0
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

fn ring_params() -> WindShellParams {
    let effect = WindTornadoEffect {
        ring_strength: 1.2,
        ring_radius: 0.9,
        ring_width_q: 0.1,
        ring_height: 0.25,
        ..WindTornadoEffect::default()
    };
    WindShellParams::from_effect(&effect)
}

#[test]
fn horizontal_ray_through_the_ring_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference_for(
        ring_params(),
        Vector3::new(-5.0, 0.15, 0.0),
        Vector3::new(1.0, 0.0, 0.0),
    );
}

#[test]
fn oblique_ray_crossing_the_ring_top_matches_the_midpoint_reference() {
    assert_closed_form_matches_reference_for(
        ring_params(),
        Vector3::new(-3.0, 0.05, 0.2),
        Vector3::new(1.0, 0.12, -0.05),
    );
}

#[test]
fn ring_adds_density_only_while_its_strength_is_positive() {
    let ring = ring_params();
    let inside_ring = Vector3::new(0.9, 0.1, 0.0);
    assert!(wind_density_at(&ring, inside_ring) > wind_density_at(&params(), inside_ring));
    assert_eq!(params().ring_strength, 0.0);
}

#[test]
fn ring_density_vanishes_above_the_ring_height() {
    let ring = ring_params();
    let ring_top_y = ring.ring_height * ring.height;
    let below = Vector3::new(0.9, 0.5 * ring_top_y, 0.0);
    let above = Vector3::new(0.9, ring_top_y + 0.01, 0.0);
    assert!(wind_density_at(&ring, below) > 0.0);
    assert_eq!(wind_density_at(&ring, above), 0.0);
}
