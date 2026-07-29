use cgmath::Vector3;

// Verification-only mirror of shaders/flameShellGeometry.geom.
// Constants and vertex ordering must stay identical to the geometry shader;
// the winding/closure unit tests are the machine check for that contract.
pub const FLAME_SHELL_RING_SEGMENTS: usize = 8;
pub const FLAME_SHELL_STACKS: usize = 8;
pub const FLAME_SHELL_BASE_RADIUS: f32 = 0.5;
pub const FLAME_SHELL_TAPER_TIP_SCALE: f32 = 1.0;
pub const FLAME_SHELL_CIRCUMSCRIBE: f32 = 1.0823922; // 1/cos(pi/8): circumscribe octagon over unit cylinder
const R: f32 = FLAME_SHELL_BASE_RADIUS;
const QUAD_CORNERS: [[f32; 3]; 4] = [[-R, 0.0, -R], [R, 0.0, -R], [R, 0.0, R], [-R, 0.0, R]];

/// Multiplier on the shell's base half-extent at a normalized height. Includes the
/// circumscribe factor, so a cone built from it encloses the rasterized octagon.
pub fn flame_shell_radius_scale(height01: f32) -> f32 {
    FLAME_SHELL_CIRCUMSCRIBE * (1.0 + (FLAME_SHELL_TAPER_TIP_SCALE - 1.0) * height01)
}

/// Outer radius of the shell in flame-local units. Linear in height, so callers that
/// need a bound over the whole shell take the maximum of the two endpoints.
pub fn flame_shell_outer_radius(height01: f32) -> f32 {
    FLAME_SHELL_BASE_RADIUS * flame_shell_radius_scale(height01)
}

pub fn generate_flame_shell_triangles(
    wind: [f32; 2],
    bend_amount: f32,
    bend_power: f32,
) -> Vec<[Vector3<f32>; 3]> {
    let corners: Vec<Vector3<f32>> = QUAD_CORNERS
        .iter()
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();
    let center = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25;
    let radius_x = distance(corners[1], corners[0]) * 0.5;
    let radius_z = distance(corners[3], corners[0]) * 0.5;

    let mut triangles = Vec::new();
    for stack in 0..FLAME_SHELL_STACKS {
        append_wall_band_triangles(
            &mut triangles,
            center,
            radius_x,
            radius_z,
            stack,
            wind,
            bend_amount,
            bend_power,
        );
    }
    append_cap_triangles(
        &mut triangles,
        center,
        radius_x,
        radius_z,
        wind,
        bend_amount,
        bend_power,
    );
    triangles
}

fn compute_ring_position(
    center: Vector3<f32>,
    radius_x: f32,
    radius_z: f32,
    segment: usize,
    stack: usize,
    wind: [f32; 2],
    bend_amount: f32,
    bend_power: f32,
) -> Vector3<f32> {
    let height01 = stack as f32 / FLAME_SHELL_STACKS as f32;
    let radius_scale = flame_shell_radius_scale(height01);
    let angle = std::f32::consts::TAU * segment as f32 / FLAME_SHELL_RING_SEGMENTS as f32;
    let mut pos = center
        + Vector3::new(
            angle.cos() * radius_x * radius_scale,
            height01,
            angle.sin() * radius_z * radius_scale,
        );
    // Wind bend deformation (horizontal-only)
    pos.x += wind[0] * bend_amount * height01.powf(bend_power);
    pos.z += wind[1] * bend_amount * height01.powf(bend_power);
    pos
}

fn append_wall_band_triangles(
    triangles: &mut Vec<[Vector3<f32>; 3]>,
    center: Vector3<f32>,
    radius_x: f32,
    radius_z: f32,
    stack: usize,
    wind: [f32; 2],
    bend_amount: f32,
    bend_power: f32,
) {
    let mut strip = Vec::new();
    for i in 0..=FLAME_SHELL_RING_SEGMENTS {
        let segment = i % FLAME_SHELL_RING_SEGMENTS;
        strip.push(compute_ring_position(
            center,
            radius_x,
            radius_z,
            segment,
            stack,
            wind,
            bend_amount,
            bend_power,
        ));
        strip.push(compute_ring_position(
            center,
            radius_x,
            radius_z,
            segment,
            stack + 1,
            wind,
            bend_amount,
            bend_power,
        ));
    }

    for j in 0..strip.len() - 2 {
        if j % 2 == 0 {
            triangles.push([strip[j], strip[j + 1], strip[j + 2]]);
        } else {
            triangles.push([strip[j + 1], strip[j], strip[j + 2]]);
        }
    }
}

fn append_cap_triangles(
    triangles: &mut Vec<[Vector3<f32>; 3]>,
    center: Vector3<f32>,
    radius_x: f32,
    radius_z: f32,
    wind: [f32; 2],
    bend_amount: f32,
    bend_power: f32,
) {
    let top_center = center + Vector3::new(0.0, 1.0, 0.0);
    for i in 0..FLAME_SHELL_RING_SEGMENTS {
        let next = (i + 1) % FLAME_SHELL_RING_SEGMENTS;
        triangles.push([
            center,
            compute_ring_position(
                center,
                radius_x,
                radius_z,
                i,
                0,
                wind,
                bend_amount,
                bend_power,
            ),
            compute_ring_position(
                center,
                radius_x,
                radius_z,
                next,
                0,
                wind,
                bend_amount,
                bend_power,
            ),
        ]);
    }
    for i in 0..FLAME_SHELL_RING_SEGMENTS {
        let next = (i + 1) % FLAME_SHELL_RING_SEGMENTS;
        triangles.push([
            top_center,
            compute_ring_position(
                center,
                radius_x,
                radius_z,
                next,
                FLAME_SHELL_STACKS,
                wind,
                bend_amount,
                bend_power,
            ),
            compute_ring_position(
                center,
                radius_x,
                radius_z,
                i,
                FLAME_SHELL_STACKS,
                wind,
                bend_amount,
                bend_power,
            ),
        ]);
    }
}

fn distance(a: Vector3<f32>, b: Vector3<f32>) -> f32 {
    let d = a - b;
    (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_math_core::{
        compute_winding_number, ray_to_triangle_intersection, verify_shell_is_closed,
    };

    #[test]
    fn test_flame_shell_winding_is_one_inside() {
        let shell = generate_flame_shell_triangles([0.0, 0.0], 0.0, 1.0);
        for probe in [
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.1, 0.1, 0.1),
            Vector3::new(0.0, 0.95, 0.0),
        ] {
            let winding = compute_winding_number(&shell, probe);
            assert!(
                (winding - 1.0).abs() < 1e-4,
                "probe {probe:?}: winding = {winding}"
            );
        }
    }

    #[test]
    fn test_flame_shell_winding_is_zero_outside() {
        let shell = generate_flame_shell_triangles([0.0, 0.0], 0.0, 1.0);
        // Off-axis probe near the tip: derived from the taper so it stays outside
        // whatever radius the shell profile is set to.
        let beyond_taper = flame_shell_outer_radius(0.95) * 1.5 / std::f32::consts::SQRT_2;
        for probe in [
            Vector3::new(flame_shell_outer_radius(0.0) * 2.0, 0.5, 0.0),
            Vector3::new(0.0, -0.5, 0.0),
            Vector3::new(0.0, 1.5, 0.0),
            Vector3::new(beyond_taper, 0.95, beyond_taper),
        ] {
            let winding = compute_winding_number(&shell, probe);
            assert!(winding.abs() < 1e-4, "probe {probe:?}: winding = {winding}");
        }
    }

    #[test]
    fn test_flame_shell_is_closed() {
        let shell = generate_flame_shell_triangles([0.0, 0.0], 0.0, 1.0);
        let probes = [
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.2, 0.2, 0.1),
            Vector3::new(3.0, 0.5, 0.0),
        ];
        assert!(verify_shell_is_closed(&shell, &probes));
    }

    #[test]
    fn test_flame_shell_rays_have_even_hit_count() {
        let shell = generate_flame_shell_triangles([0.0, 0.0], 0.0, 1.0);
        let rays = [
            (Vector3::new(-2.0, 0.5, 0.03), Vector3::new(1.0, 0.0, 0.0)),
            (Vector3::new(0.03, -2.0, 0.05), Vector3::new(0.0, 1.0, 0.0)),
            (Vector3::new(-2.0, 0.2, -1.9), Vector3::new(0.7, 0.1, 0.7)),
        ];
        for (origin, direction) in rays {
            let mut hits: Vec<f32> = shell
                .iter()
                .filter_map(|t| ray_to_triangle_intersection(origin, direction, t[0], t[1], t[2]))
                .filter(|t| *t > 0.0)
                .collect();
            hits.sort_by(|a, b| a.partial_cmp(b).expect("hit distances are finite"));
            assert!(
                hits.len() % 2 == 0,
                "ray {origin:?} -> {direction:?}: {} hits at {hits:?}",
                hits.len()
            );
        }
    }

    #[test]
    fn test_flame_shell_bent_winding_is_closed() {
        let shell = generate_flame_shell_triangles([1.0, 0.0], 0.6, 1.7);
        // Inside probe near the bent axis (shifted right by wind)
        let winding_inside = compute_winding_number(&shell, Vector3::new(0.3, 0.5, 0.0));
        assert!(
            (winding_inside - 1.0).abs() < 1e-4,
            "bent shell inside probe: winding = {winding_inside}"
        );
        // Outside probe far from the bent shape
        let winding_outside = compute_winding_number(&shell, Vector3::new(2.0, 0.5, 0.0));
        assert!(
            winding_outside.abs() < 1e-4,
            "bent shell outside probe: winding = {winding_outside}"
        );
    }

    #[test]
    fn test_flame_shell_bend_position() {
        // Verify compute_ring_position composes the radial scale and the bend offset:
        //   pos.xz = cos/sin(angle) * radius * CIRCUMSCRIBE * (1 + (TIP - 1) * h)
        //          + wind * bend_amount * h^bend_power
        // center=(0,0,0), radiusX=1, radiusZ=1, wind=(1,1), bend_amount=1, bend_power=2,
        // stack=4 (h=0.5), segment=0 (angle=0).
        let center = Vector3::new(0.0, 0.0, 0.0);
        let radius_x = 1.0;
        let radius_z = 1.0;
        let segment = 0;
        let stack = 4; // height01 = 4/8 = 0.5
        let wind: [f32; 2] = [1.0, 1.0];
        let bend_amount = 1.0;
        let bend_power = 2.0;

        let height01 = 0.5;
        let radial =
            FLAME_SHELL_CIRCUMSCRIBE * (1.0 + (FLAME_SHELL_TAPER_TIP_SCALE - 1.0) * height01);
        let bend = height01.powf(bend_power);
        let expected_x = radial + bend;
        let expected_z = bend;

        let pos = compute_ring_position(
            center,
            radius_x,
            radius_z,
            segment,
            stack,
            wind,
            bend_amount,
            bend_power,
        );

        assert!(
            (pos.x - expected_x).abs() < 1e-5,
            "bend position x: expected {expected_x}, got {}",
            pos.x
        );
        assert!(
            (pos.y - height01).abs() < 1e-5,
            "bend position y: expected {height01}, got {}",
            pos.y
        );
        assert!(
            (pos.z - expected_z).abs() < 1e-5,
            "bend position z: expected {expected_z}, got {}",
            pos.z
        );
    }
}
