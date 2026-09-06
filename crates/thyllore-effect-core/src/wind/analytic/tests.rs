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

fn streaked_params() -> WindShellParams {
    let effect = WindTornadoEffect {
        time: 1.0,
        circulation: 2.0,
        streak_order: 3.0,
        streak_twist: 4.0,
        streak_amplitude: 0.5,
        ..WindTornadoEffect::default()
    };
    WindShellParams::from_effect(&effect)
}

fn streaked_midpoint_optical_depth(
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
        total += (wind_density_at(params, point) * wind_streak_sigma(params, point)) as f64 * step;
    }
    total
}

#[test]
fn streaked_wall_matches_the_midpoint_reference() {
    let params = streaked_params();
    let origin = Vector3::new(-5.0, 0.5, 0.25);
    let direction = Vector3::new(1.0, 0.1, 0.0).normalize();
    let mut t_near = 0.0;
    let mut t_far = 1e4;
    assert!(
        clamp_ray_to_wind_cone(&params, origin, direction, &mut t_near, &mut t_far),
        "ray must hit the envelope"
    );
    let closed = wind_optical_depth(&params, origin, direction, t_near, t_far) as f64;
    let reference = streaked_midpoint_optical_depth(&params, origin, direction, t_near, t_far);
    assert!(
        reference > 1e-3,
        "reference {reference} too small to compare"
    );
    let relative = (closed - reference).abs() / reference;
    assert!(
        relative < 5e-2,
        "closed {closed} vs reference {reference} (rel {relative})"
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

fn total_mass(params: &WindShellParams, q_max: f32) -> f64 {
    let n_h = 64;
    let n_q = 512;
    let h_top = params.h_top * params.height;
    let dh = h_top / n_h as f32;
    let dq = q_max / n_q as f32;
    let mut mass: f64 = 0.0;
    for j in 0..n_h {
        let h = (j as f32 + 0.5) * dh;
        for i in 0..n_q {
            let q = (i as f32 + 0.5) * dq;
            let r = q.sqrt();
            let rho = wind_density_at(params, Vector3::new(r, h, 0.0));
            mass += rho as f64;
        }
    }
    mass * (dq * dh) as f64 * std::f64::consts::PI
}

fn envelope_q_max(params: &WindShellParams) -> f32 {
    let wall_top_r = params.wall_radius_base + params.wall_radius_slope;
    let wall_max_r =
        (wall_top_r * wall_top_r + params.spread_offset).sqrt() + params.wall_width_q.sqrt();
    let ring_max_r = params.ring_bounds_radius();
    let core_max_r = params.core_radius_sq.sqrt();
    let max_r = wall_max_r.max(ring_max_r).max(core_max_r);
    (max_r + 0.1) * (max_r + 0.1)
}

#[test]
fn mass_is_conserved_under_wall_spread() {
    let effect = WindTornadoEffect {
        time: 0.0,
        rise_initial_height: 1.0,
        rise_duration: 0.0,
        spread_start: 0.5,
        spread_rate: 0.1,
        dissipate_start: 0.0,
        dissipate_time: 0.0,
        ..WindTornadoEffect::default()
    };

    let times = [0.3, 1.5, 5.0];

    let q_max = {
        let mut e = effect.clone();
        e.time = *times.last().unwrap();
        let p = WindShellParams::from_effect(&e);
        envelope_q_max(&p)
    };

    let masses: Vec<f64> = times
        .iter()
        .map(|&t| {
            let mut e = effect.clone();
            e.time = t;
            let p = WindShellParams::from_effect(&e);
            total_mass(&p, q_max)
        })
        .collect();

    assert!(masses[0] > 1e-3, "reference mass too small: {:?}", masses);

    for i in 1..times.len() {
        let rel_err = (masses[i] - masses[0]).abs() / masses[0];
        assert!(
            rel_err < 1e-3,
            "wall spread mass not conserved: t[0]={} mass={:.6}, t[{}]={} mass={:.6}, rel_err={:.6}",
            times[0], masses[0], i, times[i], masses[i], rel_err
        );
    }
}

#[test]
fn mass_is_conserved_under_ring_spread() {
    let effect = WindTornadoEffect {
        time: 0.0,
        rise_initial_height: 1.0,
        rise_duration: 0.0,
        spread_start: 0.5,
        spread_rate: 0.0,
        ring_strength: 1.2,
        ring_radius: 0.9,
        ring_width_q: 0.1,
        ring_height: 0.25,
        ring_spread_rate: 0.08,
        dissipate_start: 0.0,
        dissipate_time: 0.0,
        ..WindTornadoEffect::default()
    };

    let times = [0.3, 1.5, 5.0];

    let q_max = {
        let mut e = effect.clone();
        e.time = *times.last().unwrap();
        let p = WindShellParams::from_effect(&e);
        envelope_q_max(&p)
    };

    let masses: Vec<f64> = times
        .iter()
        .map(|&t| {
            let mut e = effect.clone();
            e.time = t;
            let p = WindShellParams::from_effect(&e);
            total_mass(&p, q_max)
        })
        .collect();

    assert!(masses[0] > 1e-3, "reference mass too small: {:?}", masses);

    for i in 1..times.len() {
        let rel_err = (masses[i] - masses[0]).abs() / masses[0];
        assert!(
            rel_err < 1e-3,
            "ring spread mass not conserved: t[0]={} mass={:.6}, t[{}]={} mass={:.6}, rel_err={:.6}",
            times[0], masses[0], i, times[i], masses[i], rel_err
        );
    }
}
