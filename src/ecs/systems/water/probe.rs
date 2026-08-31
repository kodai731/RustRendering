use cgmath::{InnerSpace, Matrix, Matrix4, SquareMatrix, Vector3};
use serde::Serialize;

/// Compute inverse(proj * view) using f64 precision to minimize fp32 rounding error.
/// Returns the result as Matrix4<f32>.
pub fn inverse_view_proj_f64(proj: Matrix4<f32>, view: Matrix4<f32>) -> Matrix4<f32> {
    let p: Matrix4<f64> = proj.cast().unwrap();
    let v: Matrix4<f64> = view.cast().unwrap();
    let inv = (p * v)
        .invert()
        .expect("view-proj matrix must be invertible");
    inv.cast().unwrap()
}

/// Which root to compare: nearest (roots[0]) or exit (roots[1]).
#[derive(Debug, Clone, Copy)]
pub enum ProbeRoot {
    Nearest,
    Exit,
}

/// Report from comparing GLSL torus intersection roots against Rust analytic solver.
#[derive(Debug, Clone, Serialize)]
pub struct WaterProbeReport {
    /// Number of pixels with hitCount > 0 (compared).
    pub pixels: usize,
    /// Pixels where root count differs between GLSL and Rust.
    pub count_mismatch: usize,
    /// Which root was compared ("nearest" or "exit").
    pub root: String,
    /// Relative error stats: |t_rust - t_glsl| / R.
    pub max_rel: f32,
    pub mean_rel: f32,
    /// Median relative error (50th percentile).
    pub p50_rel: f32,
    /// 99th percentile relative error.
    pub p99_rel: f32,
    /// Fraction of compared pixels where rel > 1e-4 / R.
    pub frac_over_1e_4: f32,
    /// Fraction of compared pixels where rel > 1e-3 / R.
    pub frac_over_1e_3: f32,
    /// Fraction of compared pixels where GLSL used fallback.
    pub glsl_fallback_rate: f32,
    /// Fraction of compared pixels where Rust solver used fallback.
    pub rust_fallback_rate: f32,
    pub width: u32,
    pub height: u32,
}

/// Pure comparison: iterate over image pixels, reconstruct rays, solve in Rust, compare with GLSL.
///
/// `image_data` is a flat f32 RGBA array of size `[width * height * 4]`.
/// Each pixel's color comes from the shader's debugView 3/4 output:
///   `vec4(hi, mid, lo, marker)`
/// where `t = hi + (mid + lo) / 1024.0` is the high-precision root time in world units,
/// and `marker = -(float(hitCount) + (fallbackUsed ? 10.0 : 0.0))`.
///
/// Water pixels are marked by A < 0 (negative). Background pixels have A >= 0 and are skipped.
/// hitCount is extracted from |marker|: if |marker| >= 10, fallback was used and hitCount = 1;
/// otherwise hitCount = |marker|.
///
/// `inv_view_proj` = inverse(proj * view) — same as the shader's `inverse(frame.proj * frame.view)`.
/// `inverse_model` = water torus inverse model matrix — transforms ray to local space.
/// `torus_radius` = major radius (water.radii.x) — normalizes local-space coordinates.
/// `camera_pos` = camera world position — used to construct the ray direction.
/// `minor_over_major` = minor_radius / major_radius (radii.y / radii.x) — the rHat value from GLSL.
/// `which` = which root to compare: Nearest (roots[0]) or Exit (roots[1]).
pub fn compute_water_probe_report(
    image_data: &[f32],
    width: u32,
    height: u32,
    inv_view_proj: Matrix4<f32>,
    inverse_model: Matrix4<f32>,
    torus_radius: f32,
    camera_pos: Vector3<f32>,
    minor_over_major: f32,
    which: ProbeRoot,
) -> WaterProbeReport {
    let mut pixels: usize = 0;
    let mut count_mismatch: usize = 0;
    let mut glsl_fallback_count: usize = 0;
    let mut rust_fallback_count: usize = 0;
    let mut rel_errors: Vec<f32> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize * 4;
            if idx + 3 >= image_data.len() {
                continue;
            }

            // Skip background pixels: water pixels have A < 0
            let marker = image_data[idx + 3];
            if marker >= 0.0 {
                continue;
            }

            // Decode high-precision t from hi, mid, lo
            let hi = image_data[idx];
            let mid = image_data[idx + 1];
            let lo = image_data[idx + 2];
            let t_glsl = hi + (mid + lo) / 1024.0;

            // Extract hitCount and fallbackUsed from marker
            let abs_marker = marker.abs();
            let is_fallback = abs_marker >= 10.0;
            let hit_count: u8 = if is_fallback {
                1
            } else {
                abs_marker.round() as u8
            };

            pixels += 1;

            if is_fallback {
                glsl_fallback_count += 1;
            }

            // Reconstruct ray from NDC (same as shader's reconstructRayDirection)
            let ndc_x = (x as f32 + 0.5) / width as f32 * 2.0 - 1.0;
            let ndc_y = (y as f32 + 0.5) / height as f32 * 2.0 - 1.0;

            // invViewProj * vec4(ndc, DEPTH_NEAR=1.0, 1.0)
            let world = inv_view_proj * cgmath::vec4(ndc_x, ndc_y, 1.0, 1.0);
            let world_pos = Vector3::new(world.x / world.w, world.y / world.w, world.z / world.w);

            // Ray direction from camera position to world position
            let ray_dir = (world_pos - camera_pos).normalize();

            // Transform to local space: origin w=1, dir w=0 (same as shader)
            let p_local_origin =
                inverse_model * cgmath::vec4(camera_pos.x, camera_pos.y, camera_pos.z, 1.0);
            let p_local =
                Vector3::new(p_local_origin.x, p_local_origin.y, p_local_origin.z) / torus_radius;

            let d_local_raw = inverse_model * cgmath::vec4(ray_dir.x, ray_dir.y, ray_dir.z, 0.0);
            let d_local = Vector3::new(d_local_raw.x, d_local_raw.y, d_local_raw.z).normalize();

            // Solve in Rust using thyllore_effect_core's analytic solver
            let rust_hits = thyllore_effect_core::water::analytic::intersect_torus(
                p_local,
                d_local,
                1.0,
                minor_over_major,
            );

            // Compare root count
            if rust_hits.count != hit_count {
                count_mismatch += 1;
                continue;
            }

            if rust_hits.fallback_used {
                rust_fallback_count += 1;
            }

            // Compare the correct root based on `which`
            let root_index = match which {
                ProbeRoot::Nearest => 0,
                ProbeRoot::Exit => 1,
            };

            // For Exit root, skip when hit_count < 2 (only one root exists)
            if root_index >= rust_hits.count as usize {
                continue;
            }

            let t_rust = rust_hits.roots[root_index] * torus_radius;
            let rel_err = (t_rust - t_glsl).abs() / torus_radius;
            rel_errors.push(rel_err);
        }
    }

    let count = pixels as f32;
    let n = rel_errors.len();

    // Compute statistics from sorted relative errors
    let mut max_rel: f32 = 0.0;
    let mut mean_rel: f32 = 0.0;
    let mut p50_rel: f32 = 0.0;
    let mut p99_rel: f32 = 0.0;
    let mut frac_over_1e_4: f32 = 0.0;
    let mut frac_over_1e_3: f32 = 0.0;

    if n > 0 {
        rel_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        max_rel = *rel_errors.last().unwrap();
        mean_rel = rel_errors.iter().sum::<f32>() / n as f32;

        // Percentiles: use nearest-rank method
        let p50_idx = ((0.5 * n as f32).ceil() as usize).min(n - 1);
        p50_rel = rel_errors[p50_idx];

        let p99_idx = ((0.99 * n as f32).ceil() as usize).min(n - 1);
        p99_rel = rel_errors[p99_idx];

        // Fraction over thresholds (relative to torus_radius, so threshold is already normalized)
        let mut over_1e_4: usize = 0;
        let mut over_1e_3: usize = 0;
        for &err in &rel_errors {
            if err > 1e-4 {
                over_1e_4 += 1;
            }
            if err > 1e-3 {
                over_1e_3 += 1;
            }
        }
        frac_over_1e_4 = over_1e_4 as f32 / n as f32;
        frac_over_1e_3 = over_1e_3 as f32 / n as f32;
    }

    let root_str = match which {
        ProbeRoot::Nearest => "nearest",
        ProbeRoot::Exit => "exit",
    };

    WaterProbeReport {
        pixels,
        count_mismatch,
        root: root_str.to_string(),
        max_rel,
        mean_rel,
        p50_rel,
        p99_rel,
        frac_over_1e_4,
        frac_over_1e_3,
        glsl_fallback_rate: if count > 0.0 {
            glsl_fallback_count as f32 / count
        } else {
            0.0
        },
        rust_fallback_rate: if count > 0.0 {
            rust_fallback_count as f32 / count
        } else {
            0.0
        },
        width,
        height,
    }
}
