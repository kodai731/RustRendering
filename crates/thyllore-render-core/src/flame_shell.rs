use cgmath::Vector3;

// Verification-only mirror of shaders/flameShellGeometry.geom.
// Constants and vertex ordering must stay identical to the geometry shader;
// the winding/closure unit tests are the machine check for that contract.
pub const FLAME_SHELL_RING_SEGMENTS: usize = 8;
pub const FLAME_SHELL_STACKS: usize = 3;
pub const FLAME_SHELL_TAPER_TIP_SCALE: f32 = 0.25;

const QUAD_CORNERS: [[f32; 3]; 4] = [
    [-0.5, 0.0, -0.5],
    [0.5, 0.0, -0.5],
    [0.5, 0.0, 0.5],
    [-0.5, 0.0, 0.5],
];

pub fn generate_flame_shell_triangles() -> Vec<[Vector3<f32>; 3]> {
    let corners: Vec<Vector3<f32>> = QUAD_CORNERS
        .iter()
        .map(|c| Vector3::new(c[0], c[1], c[2]))
        .collect();
    let center = (corners[0] + corners[1] + corners[2] + corners[3]) * 0.25;
    let radius_x = distance(corners[1], corners[0]) * 0.5;
    let radius_z = distance(corners[3], corners[0]) * 0.5;

    let mut triangles = Vec::new();
    for stack in 0..FLAME_SHELL_STACKS {
        append_wall_band_triangles(&mut triangles, center, radius_x, radius_z, stack);
    }
    append_cap_triangles(&mut triangles, center, radius_x, radius_z);
    triangles
}

fn compute_ring_position(
    center: Vector3<f32>,
    radius_x: f32,
    radius_z: f32,
    segment: usize,
    stack: usize,
) -> Vector3<f32> {
    let height01 = stack as f32 / FLAME_SHELL_STACKS as f32;
    let taper = 1.0 + (FLAME_SHELL_TAPER_TIP_SCALE - 1.0) * height01;
    let angle = std::f32::consts::TAU * segment as f32 / FLAME_SHELL_RING_SEGMENTS as f32;
    center
        + Vector3::new(
            angle.cos() * radius_x * taper,
            height01,
            angle.sin() * radius_z * taper,
        )
}

fn append_wall_band_triangles(
    triangles: &mut Vec<[Vector3<f32>; 3]>,
    center: Vector3<f32>,
    radius_x: f32,
    radius_z: f32,
    stack: usize,
) {
    let mut strip = Vec::new();
    for i in 0..=FLAME_SHELL_RING_SEGMENTS {
        let segment = i % FLAME_SHELL_RING_SEGMENTS;
        strip.push(compute_ring_position(
            center, radius_x, radius_z, segment, stack,
        ));
        strip.push(compute_ring_position(
            center,
            radius_x,
            radius_z,
            segment,
            stack + 1,
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
) {
    let top_center = center + Vector3::new(0.0, 1.0, 0.0);
    for i in 0..FLAME_SHELL_RING_SEGMENTS {
        let next = (i + 1) % FLAME_SHELL_RING_SEGMENTS;
        triangles.push([
            center,
            compute_ring_position(center, radius_x, radius_z, i, 0),
            compute_ring_position(center, radius_x, radius_z, next, 0),
        ]);
    }
    for i in 0..FLAME_SHELL_RING_SEGMENTS {
        let next = (i + 1) % FLAME_SHELL_RING_SEGMENTS;
        triangles.push([
            top_center,
            compute_ring_position(center, radius_x, radius_z, next, FLAME_SHELL_STACKS),
            compute_ring_position(center, radius_x, radius_z, i, FLAME_SHELL_STACKS),
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
        let shell = generate_flame_shell_triangles();
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
        let shell = generate_flame_shell_triangles();
        for probe in [
            Vector3::new(2.0, 0.5, 0.0),
            Vector3::new(0.0, -0.5, 0.0),
            Vector3::new(0.0, 1.5, 0.0),
            Vector3::new(0.4, 0.95, 0.4),
        ] {
            let winding = compute_winding_number(&shell, probe);
            assert!(winding.abs() < 1e-4, "probe {probe:?}: winding = {winding}");
        }
    }

    #[test]
    fn test_flame_shell_is_closed() {
        let shell = generate_flame_shell_triangles();
        let probes = [
            Vector3::new(0.0, 0.5, 0.0),
            Vector3::new(0.2, 0.2, 0.1),
            Vector3::new(3.0, 0.5, 0.0),
        ];
        assert!(verify_shell_is_closed(&shell, &probes));
    }

    #[test]
    fn test_flame_shell_rays_have_even_hit_count() {
        let shell = generate_flame_shell_triangles();
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
}
