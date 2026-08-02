//! CPU probe of the wall-plate regime: view-ray grid vs the envelope mirror
//! (no warp/erosion/displacement; cylinder/SDF use the axis-centered approximation).

use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};

use crate::flame::{
    build_flame_inverse_model_matrix, build_flame_model_matrix, flame_bounding_radius, FlameEffect,
};
use crate::flame_radial::{
    build_height_series, evaluate_ring_smooth_density, ring_support_span, FlameRadialTaper,
};

pub const WALL_PROBE_GRID_COLS: usize = 9;
pub const WALL_PROBE_GRID_ROWS: usize = 5;
const DENSITY_SAMPLES: usize = 32;
const TANGENTIAL_GRAZING_DEG: f32 = 15.0;
const SATURATED_RAY_FRACTION: f32 = 0.5;

#[derive(Clone, Copy, Debug)]
pub struct WallProbeView {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_radians: f32,
    pub viewport_size_px: [f32; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WallProbeRay {
    pub ndc: [f32; 2],
    pub hit: bool,
    pub t_enter: f32,
    pub t_exit: f32,
    pub chord: f32,
    pub chord_world: f32,
    pub noise_cells: f32,
    pub grazing_deg: f32,
    pub density_mean: f32,
    pub density_max: f32,
    pub saturated_fraction: f32,
    pub tau: f32,
    pub pixels_per_cell: f32,
}

#[derive(Clone, Debug)]
pub struct WallProbeReport {
    pub camera_local: [f32; 3],
    pub camera_density: f32,
    pub camera_inside_support: bool,
    pub emitter_approximated: bool,
    pub rays: Vec<WallProbeRay>,
    pub hit_fraction: f32,
    pub tangential_hit_fraction: f32,
    pub saturated_hit_fraction: f32,
    pub median_noise_cells: f32,
    pub median_pixels_per_cell: f32,
    pub mean_tau: f32,
}

fn transform_point(matrix: &Matrix4<f32>, point: Vector3<f32>) -> Vector3<f32> {
    (matrix * Vector4::new(point.x, point.y, point.z, 1.0)).truncate()
}

fn transform_vector(matrix: &Matrix4<f32>, vector: Vector3<f32>) -> Vector3<f32> {
    (matrix * Vector4::new(vector.x, vector.y, vector.z, 0.0)).truncate()
}

fn slab_interval(origin_y: f32, direction_y: f32, t_max: f32) -> Option<(f32, f32)> {
    if direction_y.abs() < 1e-6 {
        return (0.0..=1.0).contains(&origin_y).then_some((0.0, t_max));
    }
    let to_floor = (0.0 - origin_y) / direction_y;
    let to_ceiling = (1.0 - origin_y) / direction_y;
    let lo = to_floor.min(to_ceiling).max(0.0);
    let hi = to_floor.max(to_ceiling).min(t_max);
    (hi > lo).then_some((lo, hi))
}

fn median(values: &mut Vec<f32>) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

pub fn probe_flame_wall(effect: &FlameEffect, view: &WallProbeView) -> WallProbeReport {
    let inverse_model = build_flame_inverse_model_matrix(effect);
    let model = build_flame_model_matrix(effect);
    let height_series = build_height_series(&effect.coefficients.height);
    let taper = FlameRadialTaper::from_effect(effect);
    let ring_major = if effect.emitter_kind == 1 {
        effect.ring_major_radius / flame_bounding_radius(effect).max(1e-3)
    } else {
        0.0
    };
    let wiggle_trim = 1.0 + effect.contour_wiggle_amp;
    let density_at = |p: Vector3<f32>| {
        evaluate_ring_smooth_density(
            [p.x, p.y, p.z],
            &height_series,
            taper,
            effect.radial_sharpness,
            ring_major,
            1.0,
        )
    };

    let camera_world = Vector3::from(view.position);
    let camera_local = transform_point(&inverse_model, camera_world);
    let camera_density = density_at(camera_local);

    let tan_half = (view.fov_y_radians * 0.5).tan();
    let aspect = (view.viewport_size_px[0] / view.viewport_size_px[1].max(1.0)).max(1e-3);
    let forward = Vector3::from(view.forward);
    let right = Vector3::from(view.right);
    let up = Vector3::from(view.up);
    let t_max = camera_local.magnitude() + 4.0;

    let mut rays = Vec::with_capacity(WALL_PROBE_GRID_COLS * WALL_PROBE_GRID_ROWS);
    for row in 0..WALL_PROBE_GRID_ROWS {
        for col in 0..WALL_PROBE_GRID_COLS {
            let ndc = [
                (col as f32 + 0.5) / WALL_PROBE_GRID_COLS as f32 * 2.0 - 1.0,
                (row as f32 + 0.5) / WALL_PROBE_GRID_ROWS as f32 * 2.0 - 1.0,
            ];
            let direction_world =
                (forward + right * ndc[0] * tan_half * aspect + up * ndc[1] * tan_half).normalize();
            let direction_local = transform_vector(&inverse_model, direction_world).normalize();

            let mut ray = WallProbeRay {
                ndc,
                ..Default::default()
            };
            let span = slab_interval(camera_local.y, direction_local.y, t_max).and_then(
                |(slab_lo, slab_hi)| {
                    ring_support_span(
                        [camera_local.x, camera_local.y, camera_local.z],
                        [direction_local.x, direction_local.y, direction_local.z],
                        slab_lo,
                        slab_hi,
                        ring_major,
                        taper,
                        effect.radial_sharpness,
                        wiggle_trim,
                    )
                },
            );
            if let Some((t_enter, t_exit)) = span {
                let chord = t_exit - t_enter;
                let world_step = transform_vector(&model, direction_local).magnitude();
                let dt = chord / DENSITY_SAMPLES as f32;
                let mut density_sum = 0.0;
                let mut density_max = 0.0f32;
                let mut saturated = 0usize;
                for i in 0..DENSITY_SAMPLES {
                    let t = t_enter + (i as f32 + 0.5) * dt;
                    let density = density_at(camera_local + direction_local * t);
                    density_sum += density;
                    density_max = density_max.max(density);
                    if density >= effect.edge_high {
                        saturated += 1;
                    }
                }

                let boundary_point = if t_enter > 1e-4 {
                    camera_local + direction_local * t_enter
                } else {
                    camera_local + direction_local * t_exit
                };
                let radial = Vector3::new(boundary_point.x, 0.0, boundary_point.z);
                let grazing_deg = if radial.magnitude() < 1e-4 {
                    90.0
                } else {
                    direction_local
                        .dot(radial.normalize())
                        .abs()
                        .clamp(0.0, 1.0)
                        .asin()
                        .to_degrees()
                };

                let mid_world = transform_point(
                    &model,
                    camera_local + direction_local * (0.5 * (t_enter + t_exit)),
                );
                let mid_distance = (mid_world - camera_world).magnitude().max(1e-3);
                let pixels_per_world =
                    view.viewport_size_px[1] / (2.0 * mid_distance * tan_half.max(1e-4));
                let cell_world = flame_bounding_radius(effect) / effect.noise_frequency.max(1e-3);

                ray.hit = true;
                ray.t_enter = t_enter;
                ray.t_exit = t_exit;
                ray.chord = chord;
                ray.chord_world = chord * world_step;
                ray.noise_cells = chord * effect.noise_frequency;
                ray.grazing_deg = grazing_deg;
                ray.density_mean = density_sum / DENSITY_SAMPLES as f32;
                ray.density_max = density_max;
                ray.saturated_fraction = saturated as f32 / DENSITY_SAMPLES as f32;
                ray.tau = effect.sigma_t * density_sum * dt * world_step;
                ray.pixels_per_cell = pixels_per_world * cell_world;
            }
            rays.push(ray);
        }
    }

    let hits: Vec<&WallProbeRay> = rays.iter().filter(|ray| ray.hit).collect();
    let hit_count = hits.len().max(1) as f32;
    let tangential = hits
        .iter()
        .filter(|ray| ray.grazing_deg < TANGENTIAL_GRAZING_DEG)
        .count() as f32;
    let saturated = hits
        .iter()
        .filter(|ray| ray.saturated_fraction >= SATURATED_RAY_FRACTION)
        .count() as f32;
    let mut cells: Vec<f32> = hits.iter().map(|ray| ray.noise_cells).collect();
    let mut pixels: Vec<f32> = hits.iter().map(|ray| ray.pixels_per_cell).collect();
    let mean_tau = hits.iter().map(|ray| ray.tau).sum::<f32>() / hit_count;

    WallProbeReport {
        camera_local: [camera_local.x, camera_local.y, camera_local.z],
        camera_density,
        camera_inside_support: camera_density > 0.0,
        emitter_approximated: effect.emitter_kind != 1,
        hit_fraction: hits.len() as f32 / rays.len() as f32,
        tangential_hit_fraction: tangential / hit_count,
        saturated_hit_fraction: saturated / hit_count,
        median_noise_cells: median(&mut cells),
        median_pixels_per_cell: median(&mut pixels),
        mean_tau,
        rays,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_effect() -> FlameEffect {
        FlameEffect {
            emitter_kind: 1,
            ring_major_radius: 1.5,
            radius: 0.6,
            height: 1.6,
            contour_wiggle_amp: 0.0,
            ..FlameEffect::default()
        }
    }

    fn view_at(position: [f32; 3], forward: [f32; 3]) -> WallProbeView {
        let forward = Vector3::from(forward).normalize();
        let world_up = Vector3::new(0.0, 1.0, 0.0);
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();
        WallProbeView {
            position,
            forward: forward.into(),
            right: right.into(),
            up: up.into(),
            fov_y_radians: 45.0f32.to_radians(),
            viewport_size_px: [1600.0, 900.0],
        }
    }

    fn center_ray(report: &WallProbeReport) -> WallProbeRay {
        report.rays[(WALL_PROBE_GRID_ROWS / 2) * WALL_PROBE_GRID_COLS + WALL_PROBE_GRID_COLS / 2]
    }

    #[test]
    fn transversal_ray_crosses_wall_with_short_chord() {
        let effect = ring_effect();
        let report = probe_flame_wall(&effect, &view_at([0.0, 0.4, 0.0], [1.0, 0.0, 0.0]));
        let ray = center_ray(&report);
        assert!(ray.hit);
        assert!((ray.noise_cells - ray.chord * effect.noise_frequency).abs() < 1e-4);
        assert!(
            ray.grazing_deg > 45.0,
            "wall-crossing ray must be transversal, got {}",
            ray.grazing_deg
        );
    }

    #[test]
    fn tangential_ray_along_wall_has_longer_chord_and_smaller_grazing() {
        let effect = ring_effect();
        let transversal = center_ray(&probe_flame_wall(
            &effect,
            &view_at([0.0, 0.4, 0.0], [1.0, 0.0, 0.0]),
        ));
        let tangential = center_ray(&probe_flame_wall(
            &effect,
            &view_at([1.5, 0.4, -1.2], [0.0, 0.0, 1.0]),
        ));
        assert!(transversal.hit && tangential.hit);
        assert!(tangential.chord > transversal.chord);
        assert!(tangential.grazing_deg < transversal.grazing_deg);
        assert!(tangential.tau > transversal.tau);
    }

    #[test]
    fn camera_inside_wall_reports_inside_support() {
        let effect = ring_effect();
        let report = probe_flame_wall(&effect, &view_at([1.5, 0.3, 0.0], [0.0, 0.0, 1.0]));
        assert!(report.camera_inside_support);
        assert!(report.camera_density > 0.0);
    }

    #[test]
    fn rays_away_from_flame_miss() {
        let effect = ring_effect();
        let report = probe_flame_wall(&effect, &view_at([0.0, 5.0, 0.0], [0.05, 1.0, 0.0]));
        assert_eq!(report.hit_fraction, 0.0);
        assert!(!report.rays.iter().any(|ray| ray.hit));
    }

    #[test]
    fn cylinder_emitter_is_marked_approximated_and_hits() {
        let effect = FlameEffect {
            contour_wiggle_amp: 0.0,
            ..FlameEffect::default()
        };
        let report = probe_flame_wall(&effect, &view_at([2.0, 0.5, 0.0], [-1.0, 0.0, 0.0]));
        assert!(report.emitter_approximated);
        assert!(report.hit_fraction > 0.0);
    }
}
