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
    let model_max = model
        .iter()
        .flatten()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));
    let target_max = target
        .iter()
        .flatten()
        .fold(f32::NEG_INFINITY, |acc, &v| acc.max(v));

    let model_scale = if model_max <= 0.0 {
        1.0
    } else {
        1.0 / model_max
    };
    let target_scale = if target_max <= 0.0 {
        1.0
    } else {
        1.0 / target_max
    };

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

pub fn adjacent_correlation(grid: &[f32], cols: usize) -> f32 {
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

pub fn preprocess(pixels: &[[f32; 3]], width: usize, height: usize) -> Option<FlameTexturePrep> {
    // Luminance field
    let mut lum = Vec::with_capacity(width * height);
    for p in pixels {
        lum.push(super::luminance(*p));
    }

    // Max luminance and threshold
    let max_lum = lum.iter().fold(0.0f32, |acc, &v| acc.max(v));
    if max_lum < 1e-6 {
        return None;
    }
    let threshold = max_lum * 0.15;

    // Mask
    let mask = super::flame_mask(&lum, width, height, threshold);
    let mask_count = mask.iter().filter(|&&m| m).count();
    let min_mask = (width * height) as f32 * 0.001;
    if (mask_count as f32) < min_mask {
        return None;
    }

    // Row range [row_min, row_max] of the mask
    let mut row_min = height;
    let mut row_max = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            let r = i / width;
            if r < row_min {
                row_min = r;
            }
            if r > row_max {
                row_max = r;
            }
        }
    }

    // Row-wise centroid x and width for each row in [row_min, row_max]
    let mut centroids: Vec<f32> = Vec::new();
    let mut widths: Vec<f32> = Vec::new();
    for r in row_min..=row_max {
        let mut sum_x = 0.0;
        let mut count = 0usize;
        let mut min_x = width;
        let mut max_x = 0usize;
        for x in 0..width {
            if mask[r * width + x] {
                sum_x += x as f32;
                count += 1;
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
            }
        }
        let xc = if count > 0 {
            sum_x / count as f32
        } else {
            width as f32 * 0.5
        };
        let w = (max_x.saturating_sub(min_x)) as f32;
        centroids.push(xc);
        widths.push(w);
    }

    // Max width for sample range
    let max_width = widths.iter().fold(0.0f32, |acc, &v| acc.max(v));
    let hw = max_width * 0.75;
    let n_rows = row_max.saturating_sub(row_min) + 1;
    // Build sym: 64 rows x 33 columns
    let mut sym = Vec::with_capacity(64);
    let mut row_chroma: Vec<[f32; 3]> = Vec::with_capacity(64);
    for i in 0..64 {
        let t = i as f32 / 63.0;
        let src_r = (row_min + (t * n_rows as f32).round() as usize).min(row_max);
        let centroid = centroids[src_r - row_min];
        let mut row = Vec::with_capacity(33);
        for j in 0..33 {
            let dx = (j as f32 / 32.0) * 2.0 * hw;
            let x_plus = (centroid + dx).round() as f32;
            let x_minus = (centroid - dx).round() as f32;
            let x_plus_clamped = x_plus.max(0.0).min((width.saturating_sub(1)) as f32) as usize;
            let x_minus_clamped = x_minus.max(0.0).min((width.saturating_sub(1)) as f32) as usize;
            let v_plus = lum[src_r * width + x_plus_clamped];
            let v_minus = lum[src_r * width + x_minus_clamped];
            row.push((v_plus + v_minus) * 0.5);
        }
        sym.push(row);

        // Compute row_chroma for this row: average color of pixels where luminance >= 50% of row max
        let mut max_lum_row = 0.0f32;
        for x in 0..width {
            let idx = src_r * width + x;
            if !mask[idx] {
                continue;
            }
            let p = &pixels[idx];
            let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            if l > max_lum_row {
                max_lum_row = l;
            }
        }
        let threshold_chroma = 0.5 * max_lum_row;
        let mut sum_r = 0.0f32;
        let mut sum_g = 0.0f32;
        let mut sum_b = 0.0f32;
        let mut count_chroma = 0usize;
        for x in 0..width {
            let idx = src_r * width + x;
            if !mask[idx] {
                continue;
            }
            let p = &pixels[idx];
            let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
            if l >= threshold_chroma {
                sum_r += p[0];
                sum_g += p[1];
                sum_b += p[2];
                count_chroma += 1;
            }
        }
        let chroma = if count_chroma > 0 {
            let avg_r = sum_r / count_chroma as f32;
            let avg_g = sum_g / count_chroma as f32;
            let avg_b = sum_b / count_chroma as f32;
            let max_c = avg_r.max(avg_g).max(avg_b);
            if max_c < 1e-6 {
                [1.0, 1.0, 1.0]
            } else {
                [avg_r / max_c, avg_g / max_c, avg_b / max_c]
            }
        } else {
            [1.0, 1.0, 1.0]
        };
        row_chroma.push(chroma);
    }

    // residual_rms: RMS of (I - sym_value) for mask pixels / max_lum
    let mut sum_sq = 0.0;
    let mut count = 0usize;
    for (i, &m) in mask.iter().enumerate() {
        if !m {
            continue;
        }
        let r = i / width;
        let x = i % width;
        if r < row_min || r > row_max {
            continue;
        }
        let centroid = centroids[r - row_min];
        let dx = (x as f32 - centroid).abs();
        let col_t = dx / hw * 0.5;
        let col_idx = (col_t * 32.0).round() as usize;
        let col_idx = col_idx.min(32);
        let row_t = (r - row_min) as f32 / n_rows as f32;
        let sym_row_idx = (row_t * 63.0).round() as usize;
        let sym_row_idx = sym_row_idx.min(63);
        let diff = lum[i] - sym[sym_row_idx][col_idx];
        sum_sq += diff * diff;
        count += 1;
    }
    let residual_rms = if count > 0 && max_lum > 0.0 {
        (sum_sq / count as f32).sqrt() / max_lum
    } else {
        0.0
    };

    // axis_slope: least squares gradient of row centroids over row indices, normalized by width
    let n = centroids.len();
    if n < 2 {
        // branch_count: upper half of mask, count continuous true segments (width >= 3px) per row
        let upper_end = row_min + (row_max - row_min) / 2;
        let mut branch_count = 1usize;
        for r in row_min..=upper_end {
            let mut seg_count = 0usize;
            let mut run_len = 0usize;
            for x in 0..width {
                if mask[r * width + x] {
                    run_len += 1;
                } else {
                    if run_len >= 3 {
                        seg_count += 1;
                    }
                    run_len = 0;
                }
            }
            if run_len >= 3 {
                seg_count += 1;
            }
            if seg_count > branch_count {
                branch_count = seg_count;
            }
        }
        // aspect_ratio: max row width / number of rows
        let n_rows = row_max.saturating_sub(row_min) + 1;
        let aspect_ratio = if n_rows == 0 {
            1.0
        } else {
            max_width / n_rows as f32
        };
        return Some(FlameTexturePrep {
            sym,
            residual_rms,
            axis_slope: 0.0,
            boundary_wiggle_amp: 0.0,
            residual_corr: 1.0,
            branch_count,
            aspect_ratio,
            row_chroma,
        });
    }
    let mean_r = (n as f32 - 1.0) / 2.0;
    let mean_c = centroids.iter().sum::<f32>() / n as f32;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &c) in centroids.iter().enumerate() {
        let dr = i as f32 - mean_r;
        num += dr * (c - mean_c);
        den += dr * dr;
    }
    let axis_slope = if den > 0.0 {
        num / den / width as f32
    } else {
        0.0
    };

    // boundary_wiggle_amp: RMS of adjacent differences of row widths / average width
    let avg_width = widths.iter().sum::<f32>() / n as f32;
    let mut wiggle_sum = 0.0;
    for i in 1..n {
        let diff = (widths[i] - widths[i - 1]).abs();
        wiggle_sum += diff * diff;
    }
    let boundary_wiggle_amp = if n > 1 && avg_width > 0.0 {
        (wiggle_sum / (n - 1) as f32).sqrt() / avg_width
    } else {
        0.0
    };

    // residual_corr: horizontal adjacent correlation of the residual grid
    let mut residuals = Vec::with_capacity(n * 33);
    for i in 0..n {
        let r = row_min + i;
        let centroid = centroids[i];
        for j in 0..33 {
            let dx = (j as f32 / 32.0) * 2.0 * hw;
            let x_plus = (centroid + dx).round() as f32;
            let x_minus = (centroid - dx).round() as f32;
            let x_plus_clamped = x_plus.max(0.0).min((width.saturating_sub(1)) as f32) as usize;
            let x_minus_clamped = x_minus.max(0.0).min((width.saturating_sub(1)) as f32) as usize;
            let v_plus = lum[r * width + x_plus_clamped];
            let v_minus = lum[r * width + x_minus_clamped];
            let i_val = (v_plus + v_minus) * 0.5;
            let sym_row = if n > 1 {
                ((i as f32 / (n - 1) as f32) * 63.0).round() as usize
            } else {
                0
            }
            .min(63);
            let sym_val = sym[sym_row][j];
            residuals.push(i_val - sym_val);
        }
    }
    let residual_corr = adjacent_correlation(&residuals, 33);

    // branch_count: upper half of mask, count continuous true segments (width >= 3px) per row
    let upper_end = row_min + (row_max - row_min) / 2;
    let mut branch_count = 1usize;
    for r in row_min..=upper_end {
        let mut seg_count = 0usize;
        let mut run_len = 0usize;
        for x in 0..width {
            if mask[r * width + x] {
                run_len += 1;
            } else {
                if run_len >= 3 {
                    seg_count += 1;
                }
                run_len = 0;
            }
        }
        if run_len >= 3 {
            seg_count += 1;
        }
        if seg_count > branch_count {
            branch_count = seg_count;
        }
    }

    // aspect_ratio: max row width / number of rows
    let n_rows = row_max.saturating_sub(row_min) + 1;
    let aspect_ratio = if n_rows == 0 {
        1.0
    } else {
        max_width / n_rows as f32
    };

    Some(FlameTexturePrep {
        sym,
        residual_rms,
        axis_slope,
        boundary_wiggle_amp,
        residual_corr,
        branch_count,
        aspect_ratio,
        row_chroma,
    })
}

pub fn fit_turbulence_and_tilt(prep: &FlameTexturePrep) -> [f32; 6] {
    [
        0.2 + 0.6 * (prep.residual_rms * 4.0).clamp(0.0, 1.0),
        (prep.boundary_wiggle_amp * 3.0).clamp(0.0, 0.6),
        2.0 + 10.0 * (1.0 - prep.residual_corr).clamp(0.0, 1.0),
        prep.axis_slope.clamp(-1.0, 1.0),
        0.0,
        (prep.axis_slope.abs() * 1.5).clamp(0.0, 1.0),
    ]
}

pub fn fit_color(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
) -> (bool, f32, f32, [[f32; 3]; 3]) {
    let mut luminances = Vec::with_capacity(width * height);
    for p in pixels {
        luminances.push(super::luminance(*p));
    }

    let mut sorted_luminances = luminances.clone();
    sorted_luminances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let threshold = match super::percentile(&sorted_luminances, 0.7) {
        Some(t) => t,
        None => 0.0,
    };

    let mut band_sums = [[0.0f32; 3]; 3];
    let mut band_counts = [0usize; 3];

    for y in 0..height {
        for x in 0..width {
            let lum = luminances[y * width + x];
            if lum >= threshold {
                let band_idx = if y < height / 3 {
                    0
                } else if y < 2 * height / 3 {
                    1
                } else {
                    2
                };
                for i in 0..3 {
                    band_sums[band_idx][i] += pixels[y * width + x][i];
                }
                band_counts[band_idx] += 1;
            }
        }
    }

    let mut colors = [[0.0f32; 3]; 3];
    for i in 0..3 {
        if band_counts[i] > 0 {
            for j in 0..3 {
                colors[i][j] = band_sums[i][j] / band_counts[i] as f32;
            }
        }
    }

    let bottom_cct = super::chromaticity_xy(colors[0]).and_then(super::mccamy_cct);
    let top_cct = super::chromaticity_xy(colors[2]).and_then(super::mccamy_cct);

    let valid = bottom_cct.is_some()
        && top_cct.is_some()
        && bottom_cct.unwrap() >= 1000.0
        && bottom_cct.unwrap() <= 8000.0
        && top_cct.unwrap() >= 1000.0
        && top_cct.unwrap() <= 8000.0;

    if valid {
        (true, bottom_cct.unwrap(), top_cct.unwrap(), colors)
    } else {
        (false, 0.0, 0.0, colors)
    }
}

pub fn fit_radius_profile(prep: &FlameTexturePrep) -> [f32; 33] {
    let sym = &prep.sym;
    let height = sym.len();

    // Find max of all symmetry values
    let mut total_max = 0.0f32;
    for row in sym {
        for &v in row {
            if v > total_max {
                total_max = v;
            }
        }
    }

    // Compute radius[i] for i = 0..=32
    let mut radius = [0.0f32; 33];
    for i in 0..=32 {
        let h = i as f32 / 32.0;
        let r = ((1.0 - h) * 63.0).round() as usize;
        let r = r.min(height - 1);
        let row = &sym[r];

        // Find row peak (max of current row)
        let mut row_peak = 0.0f32;
        for &v in row {
            if v > row_peak {
                row_peak = v;
            }
        }

        // If row peak < 5% of total max, w = 0
        if row_peak < 0.05 * total_max {
            radius[i] = 0.0;
            continue;
        }

        let threshold = 0.15 * row_peak;

        // Find last column j where row[j] >= threshold, then fractional intersection
        let mut last_j: Option<usize> = None;
        for (j, &v) in row.iter().enumerate() {
            if v >= threshold {
                last_j = Some(j);
            }
        }

        let w = match last_j {
            Some(j) => {
                if j < row.len() - 1 {
                    // Fractional intersection: j + (row[j] - threshold) / max(row[j] - row[j+1], 1e-9)
                    let frac = (row[j] - threshold) / (row[j] - row[j + 1]).abs().max(1e-9);
                    j as f32 + frac
                } else {
                    // j is the last column
                    j as f32
                }
            }
            None => 0.0,
        };
        radius[i] = w;
    }

    // Normalize by radius[0], but if radius[0] < 5% of radius_max, use radius_max
    let radius_max: f32 = radius.iter().copied().fold(0.0f32, f32::max);
    let norm = if radius[0] < 0.05 * radius_max {
        radius_max
    } else {
        radius[0]
    };

    if norm > 0.0 {
        for i in 0..=32 {
            radius[i] = (radius[i] / norm).clamp(0.05, 4.0);
        }
    } else {
        // If total_max is 0, just set all to 1.0
        for i in 0..=32 {
            radius[i] = 1.0;
        }
    }

    radius
}
