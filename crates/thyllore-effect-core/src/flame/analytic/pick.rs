use cgmath::{Matrix4, Vector3, Vector4};

use crate::flame::{flame_bounding_radius, FlameEffect};
use crate::flame_shell::{flame_shell_outer_radius, flame_shell_support_scale};

/// Axis-aligned bounds of the shell proxy in flame-local units, widened by the wind bend so a
/// leaning flame stays enclosed. Shared by the render-pass scissor and by click picking: both
/// ask the same question, so the bend and taper only have to be tracked in one place.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlameLocalBounds {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

/// Emitter-dependent proxy widening for an effect (ring tubes reach past the
/// cylinder support). See `flame_shell_support_scale`.
pub fn flame_support_scale(effect: &FlameEffect) -> f32 {
    let ring_major_norm = if effect.emitter_kind == 1 {
        effect.ring_major_radius / flame_bounding_radius(effect).max(1e-6)
    } else {
        0.0
    };
    flame_shell_support_scale(effect.emitter_kind, ring_major_norm, effect.support_margin)
}

pub fn flame_bend_offset(effect: &FlameEffect) -> [f32; 2] {
    [
        effect.wind_direction.x * effect.bend_amount,
        effect.wind_direction.y * effect.bend_amount,
    ]
}

pub fn flame_local_bounds(
    bend_offset: [f32; 2],
    support_scale: f32,
    support_margin: f32,
) -> FlameLocalBounds {
    let radius = flame_shell_outer_radius(0.0, support_scale, support_margin)
        .max(flame_shell_outer_radius(1.0, support_scale, support_margin));

    FlameLocalBounds {
        min: Vector3::new(
            -radius + bend_offset[0].min(0.0),
            0.0,
            -radius + bend_offset[1].min(0.0),
        ),
        max: Vector3::new(
            radius + bend_offset[0].max(0.0),
            1.0,
            radius + bend_offset[1].max(0.0),
        ),
    }
}

pub fn flame_local_bounds_corners(bounds: &FlameLocalBounds) -> [Vector3<f32>; 8] {
    let mut corners = [Vector3::new(0.0, 0.0, 0.0); 8];
    for (index, corner) in corners.iter_mut().enumerate() {
        corner.x = if index & 1 == 0 {
            bounds.min.x
        } else {
            bounds.max.x
        };
        corner.y = if index & 2 == 0 {
            bounds.min.y
        } else {
            bounds.max.y
        };
        corner.z = if index & 4 == 0 {
            bounds.min.z
        } else {
            bounds.max.z
        };
    }
    corners
}

/// Distance along the ray at which it enters the proxy, or `None` when it misses.
///
/// The ray must already be in flame-local space and its direction must be transformed without
/// renormalizing, so the returned parameter is still measured in world units.
pub fn intersect_flame_bounds(
    bounds: &FlameLocalBounds,
    origin: Vector3<f32>,
    direction: Vector3<f32>,
) -> Option<f32> {
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;

    for axis in 0..3 {
        let (origin_axis, direction_axis) = (origin[axis], direction[axis]);
        let (min_axis, max_axis) = (bounds.min[axis], bounds.max[axis]);

        if direction_axis.abs() < 1e-8 {
            if origin_axis < min_axis || origin_axis > max_axis {
                return None;
            }
            continue;
        }

        let t_min = (min_axis - origin_axis) / direction_axis;
        let t_max = (max_axis - origin_axis) / direction_axis;
        t_enter = t_enter.max(t_min.min(t_max));
        t_exit = t_exit.min(t_min.max(t_max));
    }

    if t_enter > t_exit || t_exit < 0.0 {
        return None;
    }
    Some(t_enter.max(0.0))
}

/// Same test taken in world space: the ray is pulled into flame-local space by the inverse model
/// matrix, which keeps a rotated or scaled flame from being tested against an inflated box.
pub fn intersect_flame_proxy(
    effect: &FlameEffect,
    inverse_model: &Matrix4<f32>,
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
) -> Option<f32> {
    let local_origin = inverse_model * Vector4::new(ray_origin.x, ray_origin.y, ray_origin.z, 1.0);
    let local_direction =
        inverse_model * Vector4::new(ray_direction.x, ray_direction.y, ray_direction.z, 0.0);

    intersect_flame_bounds(
        &flame_local_bounds(
            flame_bend_offset(effect),
            flame_support_scale(effect),
            effect.support_margin,
        ),
        local_origin.truncate(),
        local_direction.truncate(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::SquareMatrix;

    fn unit_bounds() -> FlameLocalBounds {
        flame_local_bounds([0.0, 0.0], 1.0, 1.0)
    }

    #[test]
    fn ray_through_the_middle_enters_at_the_near_face() {
        let bounds = unit_bounds();
        let hit = intersect_flame_bounds(
            &bounds,
            Vector3::new(0.0, 0.5, -10.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(hit, Some(10.0 + bounds.min.z));
    }

    #[test]
    fn ray_beside_the_proxy_misses() {
        let hit = intersect_flame_bounds(
            &unit_bounds(),
            Vector3::new(5.0, 0.5, -10.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(hit, None);
    }

    #[test]
    fn ray_above_the_proxy_misses() {
        let hit = intersect_flame_bounds(
            &unit_bounds(),
            Vector3::new(0.0, 2.0, -10.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(hit, None);
    }

    #[test]
    fn ray_pointing_away_misses() {
        let hit = intersect_flame_bounds(
            &unit_bounds(),
            Vector3::new(0.0, 0.5, -10.0),
            Vector3::new(0.0, 0.0, -1.0),
        );
        assert_eq!(hit, None);
    }

    #[test]
    fn origin_inside_the_proxy_enters_at_zero() {
        let hit = intersect_flame_bounds(
            &unit_bounds(),
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(hit, Some(0.0));
    }

    #[test]
    fn bend_widens_the_bounds_only_towards_the_lean() {
        let straight = unit_bounds();
        let bent = flame_local_bounds([0.4, 0.0], 1.0, 1.0);

        assert!(bent.max.x > straight.max.x);
        assert_eq!(bent.min.x, straight.min.x);
        assert_eq!(bent.min.z, straight.min.z);
    }

    #[test]
    fn corners_span_the_bounds() {
        let bounds = unit_bounds();
        let corners = flame_local_bounds_corners(&bounds);

        assert!(corners.contains(&bounds.min));
        assert!(corners.contains(&bounds.max));
    }

    #[test]
    fn world_space_test_matches_the_local_one_under_identity() {
        let effect = FlameEffect {
            bend_amount: 0.0,
            ..FlameEffect::default()
        };
        let hit = intersect_flame_proxy(
            &effect,
            &Matrix4::identity(),
            Vector3::new(0.0, 0.5, -10.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(hit, Some(10.0 + unit_bounds().min.z));
    }

    #[test]
    fn a_translated_flame_is_hit_where_it_stands() {
        let effect = FlameEffect {
            bend_amount: 0.0,
            ..FlameEffect::default()
        };
        let model = Matrix4::from_translation(Vector3::new(4.0, 0.0, 0.0));
        let inverse_model = model.invert().unwrap();

        let missed = intersect_flame_proxy(
            &effect,
            &inverse_model,
            Vector3::new(0.0, 0.5, -10.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        let hit = intersect_flame_proxy(
            &effect,
            &inverse_model,
            Vector3::new(4.0, 0.5, -10.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        assert_eq!(missed, None);
        assert!(hit.is_some());
    }

    #[test]
    fn scaling_keeps_the_parameter_in_world_units() {
        let effect = FlameEffect {
            bend_amount: 0.0,
            ..FlameEffect::default()
        };
        let model = Matrix4::from_nonuniform_scale(2.0, 1.0, 2.0);
        let inverse_model = model.invert().unwrap();

        let hit = intersect_flame_proxy(
            &effect,
            &inverse_model,
            Vector3::new(0.0, 0.5, -10.0),
            Vector3::new(0.0, 0.0, 1.0),
        )
        .unwrap();

        assert!((hit - (10.0 + 2.0 * unit_bounds().min.z)).abs() < 1e-5);
    }
}
