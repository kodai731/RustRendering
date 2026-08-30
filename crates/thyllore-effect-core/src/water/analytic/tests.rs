use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3};

use super::pick::pick_torus;
use super::project::{project_to_torus, water_surface_point};
use super::torus_intersect::{intersect_torus, torus_implicit};

fn random_in_unit_sphere() -> Vector3<f32> {
    let mut x: f32 = 0.0;
    let mut y: f32 = 0.0;
    let mut z: f32 = 0.0;
    while x * x + y * y + z * z >= 1.0 {
        x = fastrand::f32() * 2.0 - 1.0;
        y = fastrand::f32() * 2.0 - 1.0;
        z = fastrand::f32() * 2.0 - 1.0;
    }
    Vector3::new(x, y, z)
}

fn random_direction() -> Vector3<f32> {
    let theta = fastrand::f32() * 2.0 * std::f32::consts::PI;
    let cos_phi = fastrand::f32() * 2.0 - 1.0;
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
    Vector3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin()).normalize()
}

#[test]
fn test_intersect_torus_residuals_and_root_counts() {
    let mut fallback_count: usize = 0;
    let mut total_rays: usize = 0;

    for _ in 0..10000 {
        let ratio = fastrand::f32() * 18.0 + 2.0;
        let minor_radius = fastrand::f32() * 0.5 + 0.1;
        let major_radius = ratio * minor_radius;
        let r_hat = minor_radius / major_radius;
        let is_nearby = fastrand::usize(..4) == 0;
        let origin = if is_nearby {
            let scale = fastrand::f32() * (major_radius + minor_radius) * 1.5;
            random_in_unit_sphere() * scale
        } else {
            let scale = fastrand::f32() * (major_radius + minor_radius) * 5.0
                + (major_radius + minor_radius);
            random_in_unit_sphere() * scale
        };

        let dir = random_direction();

        let hits = intersect_torus(origin, dir, major_radius, minor_radius);
        total_rays += 1;

        if hits.fallback_used {
            fallback_count += 1;
        }

        assert!(
            hits.count == 0 || hits.count == 2 || hits.count == 4,
            "count={} for origin={:?} dir={:?} major_radius={} minor_radius={}",
            hits.count,
            origin,
            dir,
            major_radius,
            minor_radius
        );

        for i in 0..hits.count as usize {
            let t = hits.roots[i];
            let p = origin + dir * t;
            let p_norm = Vector3::new(p.x / major_radius, p.y / major_radius, p.z / major_radius);
            let residual = torus_implicit(p_norm, r_hat);
            assert!(
                residual.abs() < 1e-5,
                "root[{}] residual={:.6} t={} origin={:?} dir={:?} major_radius={} minor_radius={}",
                i,
                residual,
                t,
                origin,
                dir,
                major_radius,
                minor_radius
            );
        }
    }

    let fallback_rate = fallback_count as f32 / total_rays as f32;
    eprintln!(
        "W2: {} rays, fallback count={}, rate={:.4}",
        total_rays, fallback_count, fallback_rate
    );
}

#[test]
fn test_project_to_torus_residual_and_identity() {
    for _ in 0..10000 {
        let ratio = fastrand::f32() * 18.0 + 2.0;
        let minor_radius = fastrand::f32() * 0.5 + 0.1;
        let major_radius = ratio * minor_radius;
        let r_hat = minor_radius / major_radius;

        let p = random_in_unit_sphere() * (major_radius + minor_radius) * 2.0;

        let proj = project_to_torus(p, major_radius, minor_radius);

        let p_norm = Vector3::new(
            proj.point.x / major_radius,
            proj.point.y / major_radius,
            proj.point.z / major_radius,
        );
        let residual = torus_implicit(p_norm, r_hat);
        assert!(
            residual.abs() < 1e-6,
            "projection residual={:.8} p={:?} proj={:?} major_radius={} minor_radius={}",
            residual,
            p,
            proj.point,
            major_radius,
            minor_radius
        );

        let surface_point = water_surface_point(proj.u, proj.v, major_radius, minor_radius);
        let diff = (surface_point - proj.point).magnitude();
        assert!(
            diff < 1e-5 * major_radius.max(1.0),
            "identity diff={:.8} p={:?} proj={:?} surface={:?} u={} v={}",
            diff,
            p,
            proj.point,
            surface_point,
            proj.u,
            proj.v
        );
    }
}

#[test]
fn test_pick_torus_identity_transform() {
    let major_radius = 5.0;
    let minor_radius = 1.0;
    let model = Matrix4::identity();
    let inverse_model = model.invert().unwrap();

    let ray_origin = Vector3::new(0.0, 0.0, -10.0);
    let ray_dir = Vector3::new(0.0, 0.0, 1.0);

    let hit = pick_torus(
        ray_origin,
        ray_dir,
        model,
        inverse_model,
        major_radius,
        minor_radius,
    );
    assert!(hit.is_some());

    let t = hit.unwrap();
    let expected_t = 10.0 - (major_radius + minor_radius);
    assert!(
        (t - expected_t).abs() < 1e-4,
        "pick hit={:.4} expected={:.4}",
        t,
        expected_t
    );
}

#[test]
fn test_water_local_bounds() {
    let major_radius = 5.0;
    let minor_radius = 1.0;
    let (min, max) = super::torus_intersect::water_local_bounds(major_radius, minor_radius);
    assert_eq!(min, Vector3::new(-6.0, -1.0, -6.0));
    assert_eq!(max, Vector3::new(6.0, 1.0, 6.0));
}

#[test]
fn test_water_local_bounds_corners() {
    let major_radius = 5.0;
    let minor_radius = 1.0;
    let corners = super::torus_intersect::water_local_bounds_corners(major_radius, minor_radius);
    assert_eq!(corners.len(), 8);
    for corner in &corners {
        assert!(corner.x >= -6.0 && corner.x <= 6.0);
        assert!(corner.y >= -1.0 && corner.y <= 1.0);
        assert!(corner.z >= -6.0 && corner.z <= 6.0);
    }
}

#[test]
fn test_torus_gradient() {
    let p = Vector3::new(6.0, 0.0, 0.0);
    let r_hat = 0.2;
    let grad = super::torus_intersect::torus_gradient(p, r_hat);
    assert!(grad.magnitude() > 1e-6);
}

#[test]
fn test_bounding_sphere_reject() {
    let major_radius = 5.0;
    let minor_radius = 1.0;
    let origin = Vector3::new(100.0, 100.0, 100.0);
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let hits = intersect_torus(origin, dir, major_radius, minor_radius);
    assert_eq!(hits.count, 0);
}

#[test]
fn test_roots_are_sorted() {
    let major_radius = 5.0;
    let minor_radius = 1.0;
    let origin = Vector3::new(0.0, 0.0, -10.0);
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let hits = intersect_torus(origin, dir, major_radius, minor_radius);
    assert_eq!(hits.count, 4);
    for i in 1..hits.count as usize {
        assert!(
            hits.roots[i - 1] <= hits.roots[i],
            "roots not sorted: {:?}",
            hits.roots
        );
    }
    let expected = [4.0f32, 6.0, 14.0, 16.0];
    for (index, want) in expected.iter().enumerate() {
        assert!(
            (hits.roots[index] - want).abs() < 1e-3,
            "root[{}]={} expected={}",
            index,
            hits.roots[index],
            want
        );
    }
}

#[test]
fn test_roots_are_positive() {
    let major_radius = 5.0;
    let minor_radius = 1.0;
    let origin = Vector3::new(0.0, 0.0, -10.0);
    let dir = Vector3::new(0.0, 0.0, 1.0);
    let hits = intersect_torus(origin, dir, major_radius, minor_radius);
    for i in 0..hits.count as usize {
        assert!(hits.roots[i] > 0.0, "root[{}] = {}", i, hits.roots[i]);
    }
}

#[test]
fn test_water_surface_normal_is_normalized() {
    for _ in 0..100 {
        let u = fastrand::f32() * 2.0 * std::f32::consts::PI;
        let v = fastrand::f32() * 2.0 * std::f32::consts::PI;
        let n = super::project::water_surface_normal(u, v);
        assert!(
            (n.magnitude() - 1.0).abs() < 1e-6,
            "normal magnitude={:.8}",
            n.magnitude()
        );
    }
}
