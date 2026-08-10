use crate::flame::FlameEffect;
use thyllore_math_core::{evaluate_chebyshev, ChebyshevSeries};
use thyllore_texture_fit_core::{adjacent_correlation, fit_color, fit_color_ramp, fit_envelope_profile, fit_radius_profile, fit_turbulence_and_tilt, preprocess, projection_residual, FlameTexturePrep};

/// Abel projection of a biweight density row.
///
/// 3D density: ε(r) = amplitude · (1 − r² / Sr²)² inside the support radius Sr = S(sharpness) · R
/// Line integral along sightline z: ∫ ε(√(y²+z²)) dz
/// Closed form: amplitude · (16/15) · Sr · (1 − y² / Sr²)^{5/2}, zero for |y| >= Sr.
///
/// `profile_radius` is R, `y` is the transverse coordinate on the projection plane.
pub fn project_row(amplitude: f32, profile_radius: f32, sharpness: f32, y: f32) -> f32 {
    let support_radius =
        crate::flame_radial::flame_radial_support_radius(sharpness) * profile_radius;
    let inside = (1.0 - y * y / (support_radius * support_radius)).max(0.0);
    amplitude * (16.0 / 15.0) * support_radius * inside * inside * inside.sqrt()
}

/// Project the flame effect to a 2D silhouette profile.
///
/// For each height `h` in `heights`, compute F(h) from the Chebyshev height envelope and
/// R(h) from the radial Gaussian scale, then evaluate `project_row` for each column `y`.
///
/// Base subtraction is omitted for simplicity — only the relative shape of the projection matrix is used.
pub fn project_profile(effect: &FlameEffect, heights: &[f32], columns: &[f32]) -> Vec<Vec<f32>> {
    let height = ChebyshevSeries::new(
        effect
            .coefficients
            .height
            .iter()
            .flatten()
            .copied()
            .collect(),
        (0.0, 1.0),
    );
    let taper = crate::flame_radial::FlameRadialTaper::from_effect(effect);
    let sharpness = effect.radial_sharpness;

    heights
        .iter()
        .map(|&h| {
            let f_h = evaluate_chebyshev(&height, h);
            let r_h = crate::flame_radial::flame_radial_radius_scale(h, taper);
            columns
                .iter()
                .map(|&y| project_row(f_h, r_h, sharpness, y))
                .collect()
        })
        .collect()
}

pub struct FlameTextureFit {
    pub envelope_peak: f32,
    pub envelope_base: f32,
    pub envelope_tail: f32,
    pub radius: f32,
    pub radius_tip_ratio: f32,
    pub taper_power: f32,
    pub temperature_base_k: f32,
    pub temperature_tip_k: f32,
    pub noise_amplitude: f32,
    pub contour_wiggle_amp: f32,
    pub noise_frequency: f32,
    pub wind_x: f32,
    pub wind_z: f32,
    pub bend_amount: f32,
    pub use_blackbody: bool,
    pub color_bands: [[f32; 3]; 3],
    pub suggested_instances: usize,
    pub envelope_profile: [f32; 33],
    pub radius_profile: [f32; 33],
    pub color_ramp: [[f32; 3]; 8],
}

/// Fit silhouette parameters by minimizing projection residual via coordinate descent.
pub fn fit_silhouette(prep: &FlameTexturePrep, initial: &FlameEffect) -> [f32; 6] {
    let heights: Vec<f32> = (0..64).map(|i| 1.0 - i as f32 / 63.0).collect();
    let columns: Vec<f32> = (0..33).map(|j| (j as f32 / 32.0) * 3.0 * crate::flame_radial::flame_radial_support_radius(initial.radial_sharpness)).collect();

    let mut params: [f32; 6] = [
        initial.envelope_peak,
        initial.envelope_base,
        initial.envelope_tail,
        initial.radius,
        initial.radius_tip_ratio,
        initial.taper_power,
    ];

    let mut best_residual = {
        let mut eff = initial.clone();
        apply_params(&mut eff, &params);
        projection_residual(&project_profile(&eff, &heights, &columns), &prep.sym)
    };

    // Coordinate descent: optimize each parameter by golden-section search
    // Note: radius (index 3) is excluded from the candidate grid and remains fixed at
    // initial.radius. Scale is unobservable from normalized symbols — the projection residual
    // is scale-invariant, so optimizing radius would be meaningless.
    for _iter in 0..5 {
        let param_ranges: &[(f32, f32)] = &[
            (0.01, 2.0), // envelope_peak
            (0.01, 2.0), // envelope_base
            (0.01, 5.0), // envelope_tail
            (0.0, 0.0),  // radius (fixed — see note above)
            (0.01, 1.0), // radius_tip_ratio
            (0.1, 5.0),  // taper_power
        ];

        for i in 0..6 {
            // Skip radius (index 3) — it is fixed at initial.radius
            if i == 3 {
                continue;
            }
            let (lo, hi) = param_ranges[i];
            let best_val =
                golden_section_search(&params, i, lo, hi, initial, &heights, &columns, &prep.sym);
            // Only accept if it improves the residual
            let mut p = params;
            p[i] = best_val;
            let mut eff = initial.clone();
            apply_params(&mut eff, &p);
            let new_residual =
                projection_residual(&project_profile(&eff, &heights, &columns), &prep.sym);
            if new_residual < best_residual {
                params[i] = best_val;
                best_residual = new_residual;
            }
        }
    }

    params
}

fn apply_params(effect: &mut FlameEffect, params: &[f32; 6]) {
    effect.envelope_peak = params[0];
    effect.envelope_base = params[1];
    effect.envelope_tail = params[2];
    effect.radius = params[3];
    effect.radius_tip_ratio = params[4];
    effect.taper_power = params[5];
    crate::flame::refresh_flame_coefficients(effect);
}

fn golden_section_search(
    params: &[f32; 6],
    dim: usize,
    lo: f32,
    hi: f32,
    initial: &FlameEffect,
    heights: &[f32],
    columns: &[f32],
    target: &[Vec<f32>],
) -> f32 {
    let tol = 1e-4;
    let gr = (5.0_f32.sqrt() - 1.0) / 2.0; // golden ratio conjugate ~0.618

    let mut a = lo;
    let mut b = hi;
    let mut c = a + (1.0 - gr) * (b - a);
    let mut d = a + gr * (b - a);

    for _ in 0..30 {
        if (b - a).abs() < tol {
            break;
        }

        let fc = eval_residual(params, dim, c, initial, heights, columns, target);
        let fd = eval_residual(params, dim, d, initial, heights, columns, target);

        if fc < fd {
            b = d;
            d = c;
            c = a + (1.0 - gr) * (b - a);
        } else {
            a = c;
            c = d;
            d = a + gr * (b - a);
        }
    }

    // Return the best parameter value (midpoint of final bracket)
    (a + b) / 2.0
}

fn eval_residual(
    params: &[f32; 6],
    dim: usize,
    value: f32,
    initial: &FlameEffect,
    heights: &[f32],
    columns: &[f32],
    target: &[Vec<f32>],
) -> f32 {
    let mut p = *params;
    p[dim] = value;
    let mut eff = initial.clone();
    apply_params(&mut eff, &p);
    projection_residual(&project_profile(&eff, heights, columns), target)
}

/// Compute the aspect ratio of an effect's projected profile.
///
/// Projects the effect using `project_profile` (heights 64 points 0..1, columns 33 points -1.5..1.5).
/// The "maximum row width" is computed as: for each row, find the widest span of columns where
/// the value is >= 15% of the maximum value in the entire projected profile. The maximum such
/// width across all rows is taken. The width is measured in world units (number of columns * column spacing).
/// Returns `max_row_width / height` where height = 1.0.
fn projected_aspect(effect: &FlameEffect) -> f32 {
    let heights: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
    let columns: Vec<f32> = (0..33).map(|i| -1.5 + i as f32 * 3.0 / 32.0).collect();

    let profile = project_profile(effect, &heights, &columns);

    // Find the maximum value in the projected profile
    let max_val = profile
        .iter()
        .flatten()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    if max_val <= 0.0 {
        return 1.0;
    }

    let threshold = max_val * 0.15;
    let col_spacing = columns[1] - columns[0]; // 3.0 / 32.0

    // For each row, find the widest span of columns where value >= threshold
    let mut max_row_width = 0.0f32;
    for row in &profile {
        let mut first_above = None;
        let mut last_above = None;
        for (j, &v) in row.iter().enumerate() {
            if v >= threshold {
                if first_above.is_none() {
                    first_above = Some(j);
                }
                last_above = Some(j);
            }
        }
        if let (Some(first), Some(last)) = (first_above, last_above) {
            let width = (last - first) as f32 * col_spacing;
            if width > max_row_width {
                max_row_width = width;
            }
        }
    }

    // height = 1.0 (heights span 0..1)
    max_row_width / 1.0
}
/// Fit flame texture parameters from an image.
pub fn fit_flame_texture(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
    initial: &FlameEffect,
) -> Option<FlameTextureFit> {
    let prep = preprocess(pixels, width, height)?;

    let silhouette = fit_silhouette(&prep, initial);
    let turbulence = fit_turbulence_and_tilt(&prep);
    let (use_blackbody, temperature_base_k, temperature_tip_k, color_bands) =
        fit_color(pixels, width, height);

    // Compute radius from aspect ratio: initial.radius * (prep.aspect_ratio / model_aspect).clamp(0.2, 3.0)
    let model_aspect = projected_aspect(initial);
    let radius = initial.radius * (prep.aspect_ratio / model_aspect).clamp(0.2, 3.0);

    // Compute envelope and radius profiles
    let radius_profile = fit_radius_profile(&prep);
    let envelope_profile = fit_envelope_profile(&prep, &radius_profile);

    Some(FlameTextureFit {
        envelope_peak: silhouette[0],
        envelope_base: silhouette[1],
        envelope_tail: silhouette[2],
        radius,
        radius_tip_ratio: silhouette[4],
        taper_power: silhouette[5],
        temperature_base_k,
        temperature_tip_k,
        noise_amplitude: turbulence[0],
        contour_wiggle_amp: turbulence[1],
        noise_frequency: turbulence[2],
        wind_x: turbulence[3],
        wind_z: turbulence[4],
        bend_amount: turbulence[5],
        use_blackbody,
        color_bands,
        suggested_instances: prep.branch_count.clamp(1, 4),
        envelope_profile,
        radius_profile,
        color_ramp: fit_color_ramp(&prep),
    })
}


/// Compute the radius profile from silhouette data.
/// For each sample i (0..=32), h = i/32, row r = round((1.0-h)*63.0) in symmetry matrix.
/// Half-width w = max column j where sym[r][j] >= 0.15 * (max of all sym).
/// Normalize by radius[0] (if radius[0] < 5% of total max, use total max). Clamp to [0.05, 4.0].


#[derive(Clone, Copy)]
pub struct TextureFitGroups {
    pub silhouette: bool,
    pub color: bool,
    pub turbulence: bool,
    pub tilt: bool,
}

impl Default for TextureFitGroups {
    fn default() -> Self {
        Self {
            silhouette: true,
            color: true,
            turbulence: true,
            tilt: true,
        }
    }
}

/// Maximum allowed drop between adjacent envelope entries (slope limiter).
const ENVELOPE_MAX_DROP_SLOPE: f32 = 1.2 / 32.0;

/// Finalize the fit envelope after silhouette group application.
///
/// This is called at the end of `apply_texture_fit`'s silhouette group, before
/// `refresh_flame_coefficients`. It performs three steps on the height envelope F[i] (i=0..=32):
/// 1. Visibility Rescue: if the visible portion of the envelope (weighted by capfade) is too low,
///    scale it up to match the target visibility.
/// 2. Transition Shell Softening: apply a slope limiter so no adjacent pair drops more than
///    ENVELOPE_MAX_DROP_SLOPE.
/// 3. Clamp and write back: clamp all entries to [0,1] and write to baked_envelope only if
///    the change exceeds 1e-3.
pub fn finalize_fit_envelope(effect: &mut crate::flame::FlameEffect) {
    let profile = crate::flame::profile_from_effect(effect);
    let height_falloff = &profile.height_falloff;

    // Evaluate F[i] for i=0..=32 with h = i/32
    let mut f: [f32; 33] = [0.0; 33];
    let mut f_old: [f32; 33] = [0.0; 33];
    for i in 0..=32 {
        let h = i as f64 / 32.0;
        let v = height_falloff(h) as f32;
        f[i] = v;
        f_old[i] = v;
    }

    // Visibility Rescue. The runtime cap fade follows the displaced tip (P4:
    // only columns lifted past y=1 fade), so the envelope itself is the
    // visibility — no fixed-h capfade in the model.
    let mut vis_max = 0.0f32;
    for value in f.iter().take(33) {
        vis_max = vis_max.max(*value);
    }
    let target = effect.edge_high + 0.12;
    if vis_max < target {
        let scale = (target / vis_max.max(1e-4)).min(20.0);
        for i in 0..=32 {
            f[i] *= scale;
        }
    }

    // Transition Shell Softening: slope limiter
    for i in 1..=32 {
        f[i] = f[i].max(f[i - 1] - ENVELOPE_MAX_DROP_SLOPE);
    }

    // Clamp to [0, 1]
    let mut max_change = 0.0f32;
    for i in 0..=32 {
        f[i] = f[i].clamp(0.0, 1.0);
        max_change = max_change.max((f[i] - f_old[i]).abs());
    }

    // Write back only if change exceeds 1e-3
    if max_change > 1e-3 {
        effect.baked_envelope = Some(f);
        effect.baked_blend = 1.0;
    }
}

pub fn apply_texture_fit(
    effect: &mut crate::flame::FlameEffect,
    fit: &FlameTextureFit,
    groups: TextureFitGroups,
    blend: f32,
    profile: bool,
) {
    let blend = blend.clamp(0.0, 1.0);
    if blend == 0.0 {
        return;
    }

    // Silhouette
    if groups.silhouette {
        effect.envelope_peak =
            effect.envelope_peak + (fit.envelope_peak - effect.envelope_peak) * blend;
        effect.envelope_base =
            effect.envelope_base + (fit.envelope_base - effect.envelope_base) * blend;
        effect.envelope_tail =
            effect.envelope_tail + (fit.envelope_tail - effect.envelope_tail) * blend;
        effect.radius_tip_ratio =
            effect.radius_tip_ratio + (fit.radius_tip_ratio - effect.radius_tip_ratio) * blend;
        effect.taper_power = effect.taper_power + (fit.taper_power - effect.taper_power) * blend;

        // Baked envelope and radius: set based on profile flag
        if profile {
            effect.baked_envelope = Some(fit.envelope_profile);
            let max_radius = (1.5
                / crate::flame_radial::flame_radial_support_radius(effect.radial_sharpness))
            .min(2.0);
            effect.baked_radius = Some(fit.radius_profile.map(|v| v.clamp(0.05, max_radius)));
            effect.baked_blend = blend;
        } else {
            effect.baked_envelope = None;
            effect.baked_radius = None;
            effect.baked_blend = 0.0;
            }

        finalize_fit_envelope(effect);
    }

    // Color
    if groups.color {
        // Baked color ramp: set based on profile flag
        if profile {
            effect.baked_color = Some(fit.color_ramp);
        } else {
            effect.baked_color = None;
        }

        if fit.use_blackbody {
            if blend >= 0.5 {
                effect.use_blackbody = true;
            }
            effect.temperature_base_k = effect.temperature_base_k
                + (fit.temperature_base_k - effect.temperature_base_k) * blend;
            effect.temperature_tip_k = effect.temperature_tip_k
                + (fit.temperature_tip_k - effect.temperature_tip_k) * blend;
        } else {
            if blend >= 0.5 {
                effect.use_blackbody = false;
            }
            effect.color_base[0] =
                effect.color_base[0] + (fit.color_bands[0][0] - effect.color_base[0]) * blend;
            effect.color_base[1] =
                effect.color_base[1] + (fit.color_bands[0][1] - effect.color_base[1]) * blend;
            effect.color_base[2] =
                effect.color_base[2] + (fit.color_bands[0][2] - effect.color_base[2]) * blend;
            effect.color_tip[0] =
                effect.color_tip[0] + (fit.color_bands[2][0] - effect.color_tip[0]) * blend;
            effect.color_tip[1] =
                effect.color_tip[1] + (fit.color_bands[2][1] - effect.color_tip[1]) * blend;
            effect.color_tip[2] =
                effect.color_tip[2] + (fit.color_bands[2][2] - effect.color_tip[2]) * blend;
        }
    }

    // Turbulence
    if groups.turbulence {
        effect.noise_amplitude =
            effect.noise_amplitude + (fit.noise_amplitude - effect.noise_amplitude) * blend;
        effect.noise_frequency =
            effect.noise_frequency + (fit.noise_frequency - effect.noise_frequency) * blend;
        effect.contour_wiggle_amp = effect.contour_wiggle_amp
            + (fit.contour_wiggle_amp - effect.contour_wiggle_amp) * blend;
    }

    // Tilt
    if groups.tilt {
        effect.wind_direction.x =
            effect.wind_direction.x + (fit.wind_x - effect.wind_direction.x) * blend;
        effect.wind_direction.y =
            effect.wind_direction.y + (fit.wind_z - effect.wind_direction.y) * blend;
        effect.bend_amount = effect.bend_amount + (fit.bend_amount - effect.bend_amount) * blend;
    }

    crate::flame::refresh_flame_coefficients(effect);
}

#[cfg(test)]
mod tests {
  use super::*;

    /// Build a FlameTexturePrep from a boolean mask (64x64).
    /// sym[r][j] = 1.0 if column j is within the mask width at row r, else 0.0.
    fn build_prep_from_mask(mask: [bool; 64 * 64]) -> FlameTexturePrep {
        let mut sym = Vec::with_capacity(64);
        for r in 0..64 {
            let mut row = vec![0.0f32; 33];
            // Find leftmost and rightmost true columns in this mask row
            let mut left = None;
            let mut right = None;
            for c in 0..64 {
                if mask[r * 64 + c] {
                    if left.is_none() {
                        left = Some(c);
                    }
                    right = Some(c);
                }
            }
            if let (Some(l), Some(right)) = (left, right) {
                let centroid = (l as f32 + right as f32) * 0.5;
                let half_width = (right - l) as f32 * 0.5;
                // Map columns j=0..32 to dx = (j/32)*2*half_width
                for j in 0..33 {
                    let dx = (j as f32 / 32.0) * 2.0 * half_width;
                    let x_plus = (centroid + dx).round() as usize;
                    let x_minus = (centroid - dx).round() as usize;
                    let v_plus = if x_plus < 64 && mask[r * 64 + x_plus] {
                        1.0
                    } else {
                        0.0
                    };
                    let v_minus = if x_minus < 64 && mask[r * 64 + x_minus] {
                        1.0
                    } else {
                        0.0
                    };
                    row[j] = (v_plus + v_minus) * 0.5;
                }
            }
            sym.push(row);
        }
        FlameTexturePrep {
            sym,
            residual_rms: 0.0,
            axis_slope: 0.0,
            boundary_wiggle_amp: 0.0,
            residual_corr: 1.0,
            branch_count: 1,
            aspect_ratio: 1.0,
            row_chroma: vec![[1.0, 1.0, 1.0]; 64],
        }
    }
    #[test]
    fn test_apply_blend_zero_unchanged() {
        let mut effect = crate::flame::FlameEffect::default();
        let original = effect.clone();
        let fit = FlameTextureFit {
            envelope_peak: 2.0,
            envelope_base: 0.5,
            envelope_tail: 0.1,
            radius: 1.0,
            radius_tip_ratio: 0.3,
            taper_power: 4.0,
            color_bands: [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            temperature_base_k: 3000.0,
            temperature_tip_k: 1500.0,
            use_blackbody: true,
            noise_amplitude: 0.8,
            contour_wiggle_amp: 0.6,
            noise_frequency: 5.0,
            wind_x: 1.0,
            wind_z: 0.5,
            bend_amount: 0.9,
            suggested_instances: 1,
            envelope_profile: [0.0; 33],
            radius_profile: [0.0; 33],
            color_ramp: [[0.0; 3]; 8],
        };
        apply_texture_fit(&mut effect, &fit, TextureFitGroups::default(), 0.0, false);
        assert_eq!(effect, original);
    }

    #[test]
    fn test_apply_blend_one_silhouette_only() {
        let mut effect = crate::flame::FlameEffect::default();
        let original_color_base = effect.color_base;
        let original_color_tip = effect.color_tip;
        let original_noise_amplitude = effect.noise_amplitude;
        let original_noise_frequency = effect.noise_frequency;
        let original_contour_wiggle_amp = effect.contour_wiggle_amp;
        let original_wind_direction = effect.wind_direction;
        let original_bend_amount = effect.bend_amount;

        let fit = FlameTextureFit {
            envelope_peak: 2.0,
            envelope_base: 0.5,
            envelope_tail: 0.1,
            radius: 1.0,
            radius_tip_ratio: 0.3,
            taper_power: 4.0,
            color_bands: [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            temperature_base_k: 3000.0,
            temperature_tip_k: 1500.0,
            use_blackbody: true,
            noise_amplitude: 0.8,
            contour_wiggle_amp: 0.6,
            noise_frequency: 5.0,
            wind_x: 1.0,
            wind_z: 0.5,
            suggested_instances: 1,
            bend_amount: 0.0,
            envelope_profile: [0.0; 33],
            radius_profile: [0.0; 33],
            color_ramp: [[0.0; 3]; 8],
        };
        apply_texture_fit(
            &mut effect,
            &fit,
            TextureFitGroups {
                silhouette: true,
                color: false,
                turbulence: false,
                tilt: false,
            },
            1.0,
            false,
        );

        // Silhouette fields should match fit values
        assert!((effect.envelope_peak - fit.envelope_peak).abs() < 1e-6);
        assert!((effect.envelope_base - fit.envelope_base).abs() < 1e-6);
        assert!((effect.envelope_tail - fit.envelope_tail).abs() < 1e-6);
        assert!((effect.radius_tip_ratio - fit.radius_tip_ratio).abs() < 1e-6);
        assert!((effect.taper_power - fit.taper_power).abs() < 1e-6);

        // Color fields should be unchanged
        assert_eq!(effect.color_base, original_color_base);
        assert_eq!(effect.color_tip, original_color_tip);

        // Turbulence fields should be unchanged
        assert!((effect.noise_amplitude - original_noise_amplitude).abs() < 1e-6);
        assert!((effect.noise_frequency - original_noise_frequency).abs() < 1e-6);
        assert!((effect.contour_wiggle_amp - original_contour_wiggle_amp).abs() < 1e-6);

        // Tilt fields should be unchanged
        assert_eq!(effect.wind_direction, original_wind_direction);
        assert!((effect.bend_amount - original_bend_amount).abs() < 1e-6);
    }

    #[test]
    fn test_apply_blend_half_envelope_peak_midpoint() {
        let mut effect = crate::flame::FlameEffect::default();
        let current_peak = effect.envelope_peak;

        let fit = FlameTextureFit {
            envelope_peak: 2.0,
            envelope_base: 0.5,
            envelope_tail: 0.1,
            radius: 1.0,
            radius_tip_ratio: 0.3,
            taper_power: 4.0,
            color_bands: [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            temperature_base_k: 3000.0,
            temperature_tip_k: 1500.0,
            use_blackbody: false,
            noise_amplitude: 0.8,
            contour_wiggle_amp: 0.6,
            noise_frequency: 5.0,
            wind_x: 1.0,
            wind_z: 0.5,
            bend_amount: 0.9,
            suggested_instances: 1,
            envelope_profile: [0.0; 33],
            radius_profile: [0.0; 33],
            color_ramp: [[0.0; 3]; 8],
        };
        apply_texture_fit(
            &mut effect,
            &fit,
            TextureFitGroups {
                silhouette: true,
                color: false,
                turbulence: false,
                tilt: false,
            },
            0.5,
            false,
        );

        let expected = (current_peak + fit.envelope_peak) / 2.0;
        assert!(
            (effect.envelope_peak - expected).abs() < 1e-6,
            "envelope_peak {} != expected {}",
            effect.envelope_peak,
            expected
        );
    }

    #[test]
    fn test_project_row_vs_numerical_integration() {
        let cases: &[(f32, f32, f32, f32)] = &[
            (1.0, 1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0, 0.5),
            (1.0, 1.0, 1.0, 1.0),
            (2.0, 0.5, 2.0, 0.3),
            (0.5, 2.0, 0.5, 1.5),
            (1.0, 1.0, 3.0, 0.8),
            (3.0, 0.3, 1.5, 0.0),
        ];

        for &(amplitude, radius, sharpness, y) in cases {
            let closed = project_row(amplitude, radius, sharpness, y);

            // Numerical integration: ∫ amplitude * (1 - (y^2+z^2)/Sr^2)^2 dz over the support,
            // Sr = support_radius(sharpness) * R
            let support_radius =
                crate::flame_radial::flame_radial_support_radius(sharpness) * radius;
            let z_max = support_radius;
            let steps = 200000;
            let dz = 2.0 * z_max / steps as f32;
            let mut numerical = 0.0;
            for i in 0..steps {
                let z = -z_max + (i as f32 + 0.5) * dz;
                let r_squared = y * y + z * z;
                let inside = (1.0 - r_squared / (support_radius * support_radius)).max(0.0);
                numerical += amplitude * inside * inside * dz;
            }

            let rel_error = (closed - numerical).abs() / numerical.abs().max(1e-10);
            assert!(
                rel_error < 1e-4 || (closed - numerical).abs() < 1e-6,
                "project_row vs numerical: amplitude={}, radius={}, sharpness={}, y={}, closed={}, numerical={}, rel_error={}",
                amplitude, radius, sharpness, y, closed, numerical, rel_error
            );
        }
    }

    #[test]
    fn test_project_profile_dimensions_and_monotonicity() {
        let effect = FlameEffect::default();
        let heights: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
        let columns: [f32; 7] = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];

        let profile = project_profile(&effect, &heights, &columns);

        // Check dimensions
        assert_eq!(profile.len(), heights.len());
        for row in &profile {
            assert_eq!(row.len(), columns.len());
        }

        // Check monotonicity at y=0: the center column (y=0) should be the maximum in each row
        let center_idx = columns.iter().position(|&c| c == 0.0).unwrap();
        for row in &profile {
            let center_val = row[center_idx].abs();
            for &val in row {
                assert!(
                    val.abs() <= center_val + 1e-6,
                    "y=0 should be maximum: center={}, value={}",
                    center_val,
                    val
                );
            }
        }

        // Check that values decrease as |y| increases (monotonic decay away from center)
        for row in &profile {
            for i in 0..center_idx {
                let left = row[center_idx - 1 - i];
                let right = row[center_idx + 1 + i];
                // Both sides should be <= the value closer to center
                if i > 0 {
                    let prev_left = row[center_idx - i];
                    let prev_right = row[center_idx + i];
                    assert!(
                        left <= prev_left + 1e-6,
                        "Left side not monotonically decreasing"
                    );
                    assert!(
                        right <= prev_right + 1e-6,
                        "Right side not monotonically decreasing"
                    );
                }
            }
        }
    }

    #[test]
    fn test_projection_residual_identity() {
        let matrix: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let residual = projection_residual(&matrix, &matrix);
        assert!(
            (residual - 0.0).abs() < 1e-9,
            "Identity residual should be 0, got {}",
            residual
        );
    }

    #[test]
    fn test_projection_residual_scale_invariance() {
        let matrix: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let scaled: Vec<Vec<f32>> = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
        let residual = projection_residual(&matrix, &scaled);
        assert!(
            (residual - 0.0).abs() < 1e-9,
            "Scale-invariant residual should be 0, got {}",
            residual
        );
    }

    #[test]
    fn test_projection_residual_non_identity() {
        let matrix_a: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let matrix_b: Vec<Vec<f32>> = vec![vec![4.0, 3.0], vec![2.0, 1.0]];
        let residual = projection_residual(&matrix_a, &matrix_b);
        assert!(
            residual > 0.0,
            "Non-identity residual should be > 0, got {}",
            residual
        );
    }

    #[test]
    fn test_fit_silhouette_improves_over_initial() {
        // Build a synthetic image from project_profile with envelope_peak and radius_tip_ratio
        // shifted from initial (these are observable parameters, unlike radius which is
        // unobservable from normalized symbols).
        let mut target_effect = FlameEffect::default();
        target_effect.envelope_peak = 0.45;
        target_effect.radius_tip_ratio = 0.60;
        crate::flame::refresh_flame_coefficients(&mut target_effect);

        let heights: Vec<f32> = (0..64).map(|i| 1.0 - i as f32 / 63.0).collect();
        let columns: Vec<f32> = (0..33).map(|j| (j as f32 / 32.0) * 3.0 * crate::flame_radial::flame_radial_support_radius(target_effect.radial_sharpness)).collect();

        let profile = project_profile(&target_effect, &heights, &columns);

        // Build a fake prep from the profile (sym = profile, residual_rms = 0, etc.)
        let prep = FlameTexturePrep {
            sym: profile.clone(),
            residual_rms: 0.0,
            axis_slope: 0.0,
            boundary_wiggle_amp: 0.0,
            residual_corr: 1.0,
            branch_count: 1,
            aspect_ratio: 1.0,
            row_chroma: vec![[1.0, 1.0, 1.0]; 64],
        };

        let initial = FlameEffect::default();
        let initial_residual =
            projection_residual(&project_profile(&initial, &heights, &columns), &prep.sym);

        let fitted = fit_silhouette(&prep, &initial);

        // Build effect with fitted parameters
        let mut fitted_effect = initial.clone();
        fitted_effect.envelope_peak = fitted[0];
        fitted_effect.envelope_base = fitted[1];
        fitted_effect.envelope_tail = fitted[2];
        fitted_effect.radius = fitted[3];
        fitted_effect.radius_tip_ratio = fitted[4];
        fitted_effect.taper_power = fitted[5];
        crate::flame::refresh_flame_coefficients(&mut fitted_effect);

        let fitted_residual = projection_residual(
            &project_profile(&fitted_effect, &heights, &columns),
            &prep.sym,
        );

        assert!(
            fitted_residual < initial_residual,
            "fit_silhouette should improve: initial_residual={}, fitted_residual={}",
            initial_residual,
            fitted_residual
        );
    }

    #[test]
    fn test_fit_flame_texture_edge_cases() {
        let initial = FlameEffect::default();

        // All black image
        let width = 64;
        let height = 64;
        let black_pixels: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; width * height];
        let result = fit_flame_texture(&black_pixels, width, height, &initial);
        assert!(result.is_none(), "All-black image should return None");

        // All white image (no flame structure)
        let white_pixels: Vec<[f32; 3]> = vec![[1.0, 1.0, 1.0]; width * height];
        let result = fit_flame_texture(&white_pixels, width, height, &initial);
        // Should be None (mask covers everything, no flame structure) or finite values

        // Uniform noise image
        let mut rng = 0.5;
        let noise_pixels: Vec<[f32; 3]> = (0..width * height)
            .map(|_| {
                rng = (rng * 16807.0 + 1.0) % 2147483647.0;
                let v = rng / 2147483647.0;
                [v, v, v]
            })
            .collect();
        let result = fit_flame_texture(&noise_pixels, width, height, &initial);
        // Should be None or finite values (no NaN)
        if let Some(ref fit) = result {
            assert!(fit.envelope_peak.is_finite());
            assert!(fit.radius.is_finite());
            assert!(fit.noise_amplitude.is_finite());
        }
    }

    #[test]
    fn test_fit_color_orange_blackbody() {
        // Orange-ish synthetic image: linear RGB ~ (0.8, 0.35, 0.05)
        let width = 64;
        let height = 64;
        let orange_pixels: Vec<[f32; 3]> = vec![[0.8, 0.35, 0.05]; width * height];

        let (use_blackbody, temperature_base_k, temperature_tip_k, band_colors) =
            fit_color(&orange_pixels, width, height);

        assert!(use_blackbody, "Orange image should use blackbody");
        assert!(
            temperature_base_k >= 1000.0 && temperature_base_k <= 8000.0,
            "Base CCT {} should be in 1000..8000",
            temperature_base_k
        );
        assert!(
            temperature_tip_k >= 1000.0 && temperature_tip_k <= 8000.0,
            "Tip CCT {} should be in 1000..8000",
            temperature_tip_k
        );
        // Band colors should be approximately the input color
        for band in &band_colors {
            assert!(
                (band[0] - 0.8).abs() < 0.01,
                "Band R {} should be ~0.8",
                band[0]
            );
            assert!(
                (band[1] - 0.35).abs() < 0.01,
                "Band G {} should be ~0.35",
                band[1]
            );
        }
    }

    #[test]
    fn test_preprocess_mask_at_far_left_edge() {
        // Synthetic image where the flame mask is at the far left edge (centroid x ≈ 2).
        // This exercises the code path where centroid - dx can be negative, causing
        // usize wrap-around in (centroid - dx).round() as usize.
        let width = 64;
        let height = 64;
        let mut pixels: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; width * height];

        // Draw a flame-like shape at the far left edge (columns 1-5)
        for y in 10..=50 {
            for x in 1..=5 {
                let intensity = (5 - x) as f32 * 0.2;
                pixels[y * width + x] = [intensity, intensity * 0.8, intensity * 0.4];
            }
        }

        // Should not panic with "attempt to subtract with overflow"
        let result = preprocess(&pixels, width, height);
        assert!(
            result.is_some(),
            "preprocess should succeed for left-edge mask"
        );
        let prep = result.unwrap();
        assert_eq!(prep.sym.len(), 64, "sym should have 64 rows");
        for (i, row) in prep.sym.iter().enumerate() {
            assert_eq!(row.len(), 33, "sym row {} should have 33 columns", i);
        }
    }

    #[test]
    fn test_preprocess_extremely_flat_mask() {
        // Synthetic image where the flame mask is only 1 row high.
        // This exercises the code path where n_rows == 1 and row_t * 63.0 rounds to 64,
        // causing an index-out-of-bounds on the sym array (len 64).
        let width = 64;
        let height = 64;
        let mut pixels: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; width * height];

        // Draw a flame-like shape on only row 30 (1 row high)
        for x in 20..=40 {
            let intensity = 1.0 - (x as f32 - 30.0).abs() / 10.0;
            pixels[30 * width + x] = [intensity, intensity * 0.8, intensity * 0.4];
        }

        // Should not panic with "index out of bounds: the len is 64 but the index is 64"
        let result = preprocess(&pixels, width, height);
        assert!(result.is_some(), "preprocess should succeed for flat mask");
        let prep = result.unwrap();
        assert_eq!(prep.sym.len(), 64, "sym should have 64 rows");
        for (i, row) in prep.sym.iter().enumerate() {
            assert_eq!(row.len(), 33, "sym row {} should have 33 columns", i);
        }
    }

    #[test]
    fn test_preprocess_very_tall_mask() {
        // Synthetic image with a vertically long mask (height 200px, ~150 rows of flame).
        // This exercises the code path where n (mask row count) exceeds 64,
        // which would cause index-out-of-bounds on sym[i][j] without the clamped sym_row fix.
        let width = 64;
        let height = 200;
        let mut pixels: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; width * height];

        // Draw a flame-like shape spanning rows 5 to 155 (~150 rows, exceeding 64)
        for r in 5..=155 {
            let t = (r - 5) as f32 / 150.0; // 0 at top, 1 at bottom
            let center = width as f32 * 0.5;
            let spread = 10.0 + t * 15.0; // wider at bottom
            for x in 0..width {
                let dx = (x as f32 - center).abs();
                if dx < spread {
                    let intensity = 1.0 - dx / spread;
                    pixels[r * width + x] = [intensity, intensity * 0.8, intensity * 0.4];
                }
            }
        }

        // Should not panic with "index out of bounds: the len is 64 but the index is 64"
        let result = preprocess(&pixels, width, height);
        assert!(
            result.is_some(),
            "preprocess should succeed for very tall mask"
        );
    }

    #[test]
    fn test_forward_fit_beats_axis_profile_on_envelope_peak() {
        // Create an effect with envelope_peak = 0.45
        let mut effect = FlameEffect::default();
        effect.envelope_peak = 0.45;
        crate::flame::refresh_flame_coefficients(&mut effect);

        // Project this effect using project_profile to create a synthetic brightness matrix
        let heights: Vec<f32> = (0..64).map(|i| 1.0 - i as f32 / 63.0).collect();
        let columns: Vec<f32> = (0..33).map(|j| (j as f32 / 32.0) * 3.0 * crate::flame_radial::flame_radial_support_radius(effect.radial_sharpness)).collect();
        let profile = project_profile(&effect, &heights, &columns);

        // (a) Axis Profile Method: Take the brightness column at y=0, and pass it to fit_envelope_from_profile
        let center_col_idx = columns
            .iter()
            .position(|&c| (c - 0.0).abs() < 1e-6)
            .unwrap();
        let axis_profile: Vec<f32> = profile.iter().map(|row| row[center_col_idx]).collect();
        let axis_result = crate::flame_fit::fit_envelope_from_profile(
            &axis_profile,
            effect.radius_tip_ratio,
            effect.taper_power,
        );
        let axis_peak = match axis_result {
            Some((p, _, _)) => p,
            None => 0.0,
        };

        // (b) Forward Method: Use FlameTexturePrep with the projection matrix as sym (and residual 0)
        let prep = FlameTexturePrep {
            sym: profile.clone(),
            residual_rms: 0.0,
            axis_slope: 0.0,
            boundary_wiggle_amp: 0.0,
            residual_corr: 1.0,
            branch_count: 1,
            aspect_ratio: 1.0,
            row_chroma: vec![[1.0, 1.0, 1.0]; 64],
        };
        let initial = FlameEffect::default();
        let fitted = fit_silhouette(&prep, &initial);
        let forward_peak = fitted[0];

        // Assert that |forward_peak - 0.45| <= |axis_peak - 0.45|
        let true_peak = 0.45;
        let forward_error = (forward_peak - true_peak).abs();
        let axis_error = (axis_peak - true_peak).abs();
        assert!(
            forward_error <= axis_error,
            "Forward method error {:.6} should be <= axis profile method error {:.6}",
            forward_error,
            axis_error
        );
   }

    #[test]
    fn test_envelope_profile_body_values_ge_05_for_tip_thin_mask() {
        // Build a mask that is thin only at the tip: full width (32) for rows 1-63,
        // but very narrow (2) at row 0 (the tip).
        let mut mask = [false; 64 * 64];
        for r in 1..64 {
            for c in 0..32 {
                mask[r * 64 + c] = true;
            }
        }
        // Tip row: only 2 columns wide (very thin)
        mask[0 * 64 + 15] = true;
        mask[0 * 64 + 16] = true;
      let prep = build_prep_from_mask(mask);
        let radius_profile = fit_radius_profile(&prep);

       // The tip's radius_profile should be small (< 0.15), so the old code would
        // compute a huge envelope value there, making max_val huge and crushing
        // all body values below 0.5. After the fix, max_val is computed only from
        // entries where radius_profile[i] >= 0.15, so body values should be >= 0.5.
        let envelope = fit_envelope_profile(&prep, &radius_profile);

        // Check that body entries (i >= 1, which correspond to rows near the base)
        // have envelope values >= 0.5
        for i in 1..=32 {
            assert!(
                envelope[i] >= 0.5,
                "envelope[{}] = {:.4} < 0.5 (radius_profile[{}] = {:.4})",
                i,
                envelope[i],
                i,
                radius_profile[i]
            );
        }

        // Also verify that all entries are clamped to <= 1.0
        for i in 0..=32 {
            assert!(
                envelope[i] <= 1.0,
                "envelope[{}] = {:.4} > 1.0 (not clamped)",
                i,
                envelope[i]
            );
        }
    }

    #[test]
    fn test_radius_profile_empty_base_uses_width_unit() {
        // Build a mask where the base row (row 63) is empty, so radius[0] = 0.
        // The old code compared radius[0] < 0.05 * total_max where total_max is
        // luminance (~1.0), so 0 < 0.05 was true and it used total_max (luminance)
        // as the normalizer — a dimension mismatch.
        // After the fix, it uses radius_max (width unit), so norm = radius_max.
        let mut mask = [false; 64 * 64];
        // Fill rows 0-62 with width 16 (columns 8-23)
        for r in 0..63 {
            for c in 8..24 {
                mask[r * 64 + c] = true;
            }
        }
        // Row 63 (base) is empty

      let prep = build_prep_from_mask(mask);
        let radius_profile = fit_radius_profile(&prep);

      // radius[0] corresponds to the base row (row 63), which is empty -> width 0.
        // radius_max should be ~16 (the width of filled rows).
        // norm = radius_max (since radius[0] = 0 < 0.05 * radius_max).
        // So radius[0] / norm = 0/radius_max = 0, clamped to 0.05 (minimum).
        // The filled rows should be ~16/16 = 1.0.
        assert!(
            (radius_profile[0] - 0.05).abs() < 0.001,
            "radius_profile[0] (empty base) should be ~0.05 (clamped minimum), got {:.4}",
            radius_profile[0]
        );
        // The filled rows should normalize to ~1.0 as well
        for i in 1..=32 {
            assert!(
                (radius_profile[i] - 1.0).abs() < 0.1,
                "radius_profile[{}] = {:.4}, expected ~1.0",
                i,
                radius_profile[i]
            );
       }
    }

    #[test]
    fn test_finalize_visibility_recovery() {
        // Test 1: A baked envelope crushed to 0.2 by the fit should recover to vis_max >= 0.45
        let mut effect = crate::flame::FlameEffect::default();
        // Set up a baked envelope where all entries are crushed to 0.2
        let mut crushed: [f32; 33] = [0.0; 33];
        for i in 0..=32 {
            crushed[i] = 0.2;
        }
        effect.baked_envelope = Some(crushed);
        effect.baked_blend = 1.0;
        // Default edge_high is 0.33, so target = 0.33 + 0.12 = 0.45

        finalize_fit_envelope(&mut effect);

        // After finalize, compute vis_max from the resulting baked envelope
        let profile = crate::flame::profile_from_effect(&effect);
        let height_falloff = &profile.height_falloff;
        let mut vis_max = 0.0f32;
        for i in 0..=32 {
            let h = i as f64 / 32.0;
            vis_max = vis_max.max(height_falloff(h) as f32);
        }
        assert!(
            vis_max >= 0.45,
            "vis_max = {:.4}, expected >= 0.45 after visibility recovery",
            vis_max
        );
    }

    #[test]
    fn test_finalize_transition_shell_softening() {
        // Test 2: For a default parametric envelope (peak 0.25, base 0.05, tail 1.25),
        // the max drop between adjacent entries after softening should be <= 1.25/32
        let mut effect = crate::flame::FlameEffect::default();
        // Default values: envelope_peak=0.25, envelope_base=0.05, envelope_tail=1.25
        // No baked envelope, so it uses the parametric form
        assert!(effect.baked_envelope.is_none(), "expected no baked envelope");

        finalize_fit_envelope(&mut effect);

        // After finalize, check that the max drop between adjacent entries is <= 1.25/32
        let profile = crate::flame::profile_from_effect(&effect);
        let height_falloff = &profile.height_falloff;
        let mut max_drop = 0.0f32;
        for i in 1..=32 {
            let h_prev = (i - 1) as f64 / 32.0;
            let h_curr = i as f64 / 32.0;
            let v_prev = height_falloff(h_prev) as f32;
            let v_curr = height_falloff(h_curr) as f32;
            let drop = v_prev - v_curr;
            if drop > 0.0 {
                max_drop = max_drop.max(drop);
            }
        }
        let allowed = 1.25 / 32.0;
        assert!(
            max_drop <= allowed,
            "max drop = {:.6}, expected <= {:.6} (1.25/32)",
            max_drop,
            allowed
        );
    }

    #[test]
    fn test_finalize_idempotency() {
        // Test 3: Calling finalize twice should result in a change < 1e-3 (idempotency)
        let mut effect = crate::flame::FlameEffect::default();
        // Set up a baked envelope that will be modified by finalize
        let mut envelope: [f32; 33] = [0.0; 33];
        for i in 0..=32 {
            let h = i as f32 / 32.0;
            // A steep envelope that will be softened
            envelope[i] = (1.0 - h * h).max(0.0);
        }
        effect.baked_envelope = Some(envelope);
        effect.baked_blend = 1.0;

        // First call
        finalize_fit_envelope(&mut effect);
        let first = effect.baked_envelope.unwrap();

        // Second call
        finalize_fit_envelope(&mut effect);
        let second = effect.baked_envelope.unwrap();

        // Compute max change between first and second
        let mut max_change = 0.0f32;
        for i in 0..=32 {
            max_change = max_change.max((first[i] - second[i]).abs());
        }
        assert!(
            max_change < 1e-3,
            "idempotency: max change between first and second finalize = {:.6}, expected < 1e-3",
            max_change
        );
    }

    #[test]
    fn test_fit_silhouette_trapezoid_not_degenerate() {
        // Synthetic "bright trapezoid mask": wide at bottom, narrow at top
        let mut mask = [false; 64 * 64];
        for r in 0..64 {
            // h = 1.0 - r/63.0 (row 0 = top/h=1, row 63 = bottom/h=0)
            let h = 1.0 - r as f32 / 63.0;
            // width decreases toward the top: at bottom (h=0) half_width ~16, at top (h=1) half_width ~3.2
            let half_width = 16.0 * (1.0 - h * 0.8);
            let center = 31.5;
            for c in 0..64 {
                if (c as f32 - center).abs() < half_width {
                    mask[r * 64 + c] = true;
                }
            }
        }
        let prep = build_prep_from_mask(mask);
        let initial = FlameEffect::default();
        let fit = fit_silhouette(&prep, &initial);
        let radius_tip_ratio = fit[3];
        let taper_power = fit[5];
        assert!(
            radius_tip_ratio > 0.05,
            "radius_tip_ratio {:.4} should be > 0.05 (not degenerate)",
            radius_tip_ratio
        );
        assert!(
            taper_power > 0.15,
            "taper_power {:.4} should be > 0.15 (not degenerate)",
            taper_power
        );
    }
}
