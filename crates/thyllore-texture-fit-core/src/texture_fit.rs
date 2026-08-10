pub struct FlameTexturePrep {
    pub sym: Vec<Vec<f32>>,
    pub residual_rms: f32,
    pub axis_slope: f32,
    pub boundary_wiggle_amp: f32,
    pub residual_corr: f32,
    pub branch_count: usize,
    pub aspect_ratio: f32,
    pub row_chroma: Vec<[f32; 3]>,
}

/// RMS residual between two projection matrices after normalizing each by its maximum value.
///
/// If a matrix's max ≤ 0, it is treated as a zero matrix.
pub fn projection_residual(model: &[Vec<f32>], target: &[Vec<f32>]) -> f32 {
    let model_max = model.iter().flatten().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    let target_max = target.iter().flatten().fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));

    let model_scale = if model_max <= 0.0 { 1.0 } else { 1.0 / model_max };
    let target_scale = if target_max <= 0.0 { 1.0 } else { 1.0 / target_max };

    let mut sum_sq = 0.0;
    let mut count = 0usize;

    for (row_m, row_t) in model.iter().zip(target.iter()) {
        for (&v_m, &v_t) in row_m.iter().zip(row_t.iter()) {
            let diff = v_m * model_scale - v_t * target_scale;
            sum_sq += diff * diff;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f32).sqrt()
}

fn adjacent_correlation(grid: &[f32], cols: usize) -> f32 {
    let mut count = 0usize;
    for i in 0..grid.len() {
        let col = i % cols;
        if col != cols - 1 {
            count += 1;
        }
    }
    if count < 2 {
        return 1.0;
    }

    let mut sum_left = 0.0;
    let mut sum_right = 0.0;
    for (i, &v) in grid.iter().enumerate() {
        let col = i % cols;
        if col == cols - 1 {
            continue;
        }
        sum_left += v;
        sum_right += grid[i + 1];
    }
    let mean_left = sum_left / count as f32;
    let mean_right = sum_right / count as f32;

    let mut cov = 0.0;
    let mut var_l = 0.0;
    let mut var_r = 0.0;
    for (i, &v) in grid.iter().enumerate() {
        let col = i % cols;
        if col == cols - 1 {
            continue;
        }
        let right = grid[i + 1];
        cov += (v - mean_left) * (right - mean_right);
        var_l += (v - mean_left) * (v - mean_left);
        var_r += (right - mean_right) * (right - mean_right);
    }
    if var_l < 1e-12 || var_r < 1e-12 {
        return 1.0;
    }
    let corr = cov / (var_l * var_r).sqrt();
    corr.clamp(-1.0, 1.0)
}

/// Compute the color ramp from silhouette data.
/// For each sample i (0..8), h = (i + 0.5) / 8.0, row r = round((1.0 - h) * 63.0).min(63),
/// value is prep.row_chroma[r].
pub fn fit_color_ramp(prep: &FlameTexturePrep) -> [[f32; 3]; 8] {
    let mut ramp = [[0.0f32; 3]; 8];
    for i in 0..8 {
        let h = (i as f32 + 0.5) / 8.0;
        let r = ((1.0 - h) * 63.0).round() as usize;
        let r = r.min(63);
        ramp[i] = prep.row_chroma[r];
    }
    ramp
}

/// Compute the envelope profile from silhouette data using Abel inverse correction.
/// For each sample i (0..=32), h = i/32, row r = round((1.0-h)*63.0).
/// peak[i] = max of sym[r] row. a[i] = peak[i] / max(radius_profile[i], 0.05).
/// Normalize by max to 1.0.
pub fn fit_envelope_profile(prep: &FlameTexturePrep, radius_profile: &[f32; 33]) -> [f32; 33] {
    let sym = &prep.sym;
    let height = sym.len();

    let mut envelope = [0.0f32; 33];
    for i in 0..=32 {
        let h = i as f32 / 32.0;
        let r = ((1.0 - h) * 63.0).round() as usize;
        let r = r.min(height - 1);
        let row = &sym[r];

        // row_peak = max of sym[r] row
        let mut row_peak = 0.0f32;
        for &v in row {
            if v > row_peak {
                row_peak = v;
            }
        }

        // lum = luminance of row_chroma[r]
        let lum = 0.2126 * prep.row_chroma[r][0]
            + 0.7152 * prep.row_chroma[r][1]
            + 0.0722 * prep.row_chroma[r][2];

        // a[i] = row_peak / max(lum, 1e-3) / max(radius_profile[i], 0.05)
        envelope[i] = row_peak / lum.max(1e-3) / radius_profile[i].max(0.05);
    }

    // Normalize by max to 1.0 (only consider entries where radius_profile[i] >= 0.15)
    let mut max_val = 0.0f32;
    for (i, &v) in envelope.iter().enumerate() {
        if radius_profile[i] >= 0.15 && v > max_val {
            max_val = v;
        }
    }
    if max_val > 0.0 {
        for v in &mut envelope {
            *v /= max_val;
        }
    }

    // Clamp all entries to 1.0
    for v in &mut envelope {
        *v = (*v).min(1.0);
    }

    envelope
}
