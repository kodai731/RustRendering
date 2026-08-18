//! Sequence descriptors extractor for flame animation frames.
//!
//! Computes normalized shape coordinates (r0, neck_y, pool_bulge_y, zeta) from a sequence of
//! luminance frames, and provides helper functions to resample zeta values into fixed-point arrays.

use super::profile_fit::{flame_mask, percentile};

/// Metadata about the normalization coordinates found in the flame sequence.
#[derive(serde::Serialize)]
pub struct SequenceMeta {
    pub r0_px: f32,
    pub zeta_max: f32,
    pub neck_y: usize,
    pub pool_bulge_y: usize,
    pub frame_count: usize,
    pub fps: f32,
}

/// Descriptors extracted from a flame animation sequence.
#[derive(serde::Serialize)]
pub struct SequenceDescriptors {
    pub f1_width: Vec<f32>,
    pub f2_rough: f32,
    pub f2_lambda_over_r0: f32,
    pub f2_low_band_ratio: f32,
    pub f3_flicker: Vec<f32>,
    pub f4_meander_rms: f32,
    pub f4_freq_hz: f32,
    pub f7_components_mean: f32,
    pub f7_base_disconnected_ratio: f32,
    pub meta: SequenceMeta,
}

/// Time-averaged luminance across all frames, with 3x3 box blur applied.
fn time_averaged_luminance(frames: &[Vec<f32>], width: usize, height: usize) -> Vec<f32> {
    let n = width * height;
    let mut avg = vec![0.0f32; n];
    let count = frames.len() as f32;
    for frame in frames {
        for (a, &v) in avg.iter_mut().zip(frame.iter()) {
            *a += v;
        }
    }
    for a in &mut avg {
        *a /= count;
    }
    box_blur(&avg, width, height)
}

/// 3x3 box blur: each pixel is replaced by the mean of itself and its 8 neighbors (clamped at edges).
fn box_blur(data: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut out = data.to_vec();
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    if ny < height && nx < width {
                        sum += data[ny * width + nx];
                        count += 1;
                    }
                }
            }
            out[y * width + x] = sum / count as f32;
        }
    }
    out
}

/// Row mass: sum of time-averaged luminance per row.
#[allow(dead_code)]
fn row_mass(time_avg: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut masses = Vec::with_capacity(height);
    for row in 0..height {
        let start = row * width;
        let end = start + width;
        let sum: f32 = time_avg[start..end].iter().sum();
        masses.push(sum);
    }
    masses
}

/// Half-width of a row from its flame mask.
/// Returns the half-width in pixels (distance from center to edge), or 0.0 if no true pixels.
fn row_half_width(mask: &[bool], width: usize, row: usize) -> f32 {
    let start = row * width;
    let end = start + width;
    let row_slice = &mask[start..end];
    let first = row_slice.iter().position(|&b| b);
    let last = row_slice.iter().rposition(|&b| b);
    match (first, last) {
        (Some(f), Some(l)) => {
            let center = (f + l) as f32 * 0.5;
            (l as f32 - center).abs()
        }
        _ => 0.0,
    }
}

/// Compute normalization coordinates from frames.
/// Returns (r0, neck_y, pool_bulge_y, zeta_max) in pixels.
fn compute_normalization_coordinates(
    frames: &[Vec<f32>],
    width: usize,
    height: usize,
) -> (f32, usize, usize, f32) {
    let time_avg = time_averaged_luminance(frames, width, height);

    // Mask at 0.3 * percentile99 of time-averaged luminance
    let mut sorted_time_avg = time_avg.clone();
    sorted_time_avg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mask_threshold = 0.3 * percentile(&sorted_time_avg, 0.99).unwrap_or(0.0);
    let mask = flame_mask(&time_avg, width, height, mask_threshold);

    let mut half_widths = vec![0.0f32; height];
    for row in 0..height {
        half_widths[row] = row_half_width(&mask, width, row);
    }

    // Solid rows by mask pixel count (geometric): saturation-robust, unlike a
    // luminance-mass floor which lets a bright base starve the dimmer column.
    let mask_counts: Vec<usize> = (0..height)
        .map(|row| {
            mask[row * width..(row + 1) * width]
                .iter()
                .filter(|&&b| b)
                .count()
        })
        .collect();
    // Floor from the 75th percentile of positive-count rows (column-width order),
    // not the max row: a wide pool/ground-glow row would otherwise starve the
    // dimmer, narrower column rows out of the solid set.
    let mut positive_counts: Vec<f32> = mask_counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| c as f32)
        .collect();
    positive_counts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let count_ref = percentile(&positive_counts, 0.75).unwrap_or(0.0);
    let count_floor = ((0.15 * count_ref) as usize).max(2);
    let solid_rows: Vec<usize> = (0..height)
        .filter(|&row| mask_counts[row] >= count_floor)
        .collect();

    if solid_rows.is_empty() {
        return (1.0, 0, 0, 1.0);
    }

    // pool_bulge_y: solid row with maximum half-width
    let pool_bulge_y = solid_rows
        .iter()
        .max_by(|&&a, &&b| half_widths[a].partial_cmp(&half_widths[b]).unwrap())
        .copied()
        .unwrap();

    // Band above pool_bulge_y: solid rows where y < pool_bulge_y (higher in image)
    let band_rows: Vec<usize> = solid_rows
        .iter()
        .filter(|&&row| row < pool_bulge_y)
        .copied()
        .collect();

    // r0: median half-width of the band above pool_bulge_y
    let mut band_half_widths: Vec<f32> = band_rows.iter().map(|&r| half_widths[r]).collect();
    band_half_widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r0 = percentile(&band_half_widths, 0.5).unwrap_or(1.0);

    // neck_y: lowest solid row where half-width <= 1.3 * r0 and y < pool_bulge_y
    let neck_threshold = 1.3 * r0;
    let neck_candidates: Vec<usize> = solid_rows
        .iter()
        .filter(|&&row| row < pool_bulge_y && half_widths[row] <= neck_threshold)
        .copied()
        .collect();

    let neck_y = if neck_candidates.is_empty() {
        // Fallback: use the highest solid row above pool_bulge_y
        *band_rows.first().unwrap_or(&0)
    } else {
        // Lowest (largest y value) among candidates
        *neck_candidates.iter().max().unwrap()
    };

    // zeta = (neck_y - y) / r0 for rows above the neck; the observable extent is
    // from the neck up to the topmost solid row (not down to the image bottom).
    let top_solid = *solid_rows.first().unwrap_or(&0);
    let zeta_max = (neck_y.saturating_sub(top_solid)) as f32 / r0;

    (r0, neck_y, pool_bulge_y, zeta_max)
}

/// Resample zeta values in [0, zeta_max] into `n` evenly spaced points.
/// For each output point, compute the weighted average of the input values at nearby zeta positions.
fn resample_zeta(values: &[f32], zetas: &[f32], zeta_max: f32, n: usize) -> Vec<f32> {
    if values.is_empty() || zetas.is_empty() || zeta_max <= 1e-9 {
        return vec![0.0; n];
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let target_zeta = (i as f32) * zeta_max / (n - 1) as f32;

        // Find values near this zeta and interpolate
        let mut sum = 0.0f32;
        let mut weight_total = 0.0f32;

        for (j, &z) in zetas.iter().enumerate() {
            if z < 0.0 || z > zeta_max {
                continue;
            }
            let dist = (z - target_zeta).abs();
            // Gaussian-like weighting with sigma = zeta_max / n
            let sigma = zeta_max / n as f32;
            let weight = (-dist * dist / (2.0 * sigma * sigma)).exp();
            sum += weight * values[j];
            weight_total += weight;
        }

        if weight_total > 1e-9 {
            result.push(sum / weight_total);
        } else {
            result.push(0.0);
        }
    }
    result
}

/// Build zeta values for each row: zeta = (neck_y - y) / r0, clamped to [0, zeta_max].
fn row_zetas(height: usize, neck_y: usize, r0: f32, zeta_max: f32) -> Vec<f32> {
    let mut zetas = Vec::with_capacity(height);
    for y in 0..height {
        let z = (neck_y as f32 - y as f32) / r0;
        let clamped = z.max(0.0).min(zeta_max);
        zetas.push(clamped);
    }
    zetas
}

/// Count connected components in a binary mask using 4-neighbor flood fill (stack-based DFS),
/// and compute the ratio of total area belonging to components that do NOT touch the bottom solid row.
///
/// The "bottom solid row" is defined as the last row (largest y) that has at least one true pixel.
/// Components touching any true pixel in that row are considered "base-connected".
fn count_components_and_base_ratio(mask: &[bool], width: usize, height: usize) -> (usize, f32) {
    let total_pixels = width * height;
    let mut visited = vec![false; total_pixels];
    let mut components: Vec<(usize, bool)> = Vec::new(); // (area, touches_base_row)

    // Find the bottom solid row (last row with at least one true pixel)
    let mut bottom_solid_row: Option<usize> = None;
    for y in (0..height).rev() {
        let start = y * width;
        let end = start + width;
        if mask[start..end].iter().any(|&b| b) {
            bottom_solid_row = Some(y);
            break;
        }
    }

    // 4-neighbor flood fill
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !mask[idx] || visited[idx] {
                continue;
            }
            // Start a new component
            let mut area = 0usize;
            let mut touches_base = false;
            let mut stack = vec![idx];
            visited[idx] = true;

            while let Some(p) = stack.pop() {
                area += 1;
                let py = p / width;
                let px = p % width;

                // Check if this pixel is in the bottom solid row
                if let Some(bottom_row) = bottom_solid_row {
                    if py == bottom_row {
                        touches_base = true;
                    }
                }

                // 4 neighbors: up, down, left, right
                if py > 0 {
                    let n = (py - 1) * width + px;
                    if mask[n] && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
                if py < height - 1 {
                    let n = (py + 1) * width + px;
                    if mask[n] && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
                if px > 0 {
                    let n = py * width + (px - 1);
                    if mask[n] && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
                if px < width - 1 {
                    let n = py * width + (px + 1);
                    if mask[n] && !visited[n] {
                        visited[n] = true;
                        stack.push(n);
                    }
                }
            }

            components.push((area, touches_base));
        }
    }

    let component_count = components.len();
    let total_area: usize = components.iter().map(|(a, _)| a).sum();
    if total_area == 0 {
        return (component_count, 0.0);
    }
    let disconnected_area: usize = components
        .iter()
        .filter(|(_, touches_base)| !touches_base)
        .map(|(a, _)| a)
        .sum();
    let ratio = disconnected_area as f32 / total_area as f32;
    (component_count, ratio)
}

/// Gaussian elimination with partial pivoting to solve Ax = b.
fn gaussian_elimination(a: &mut Vec<Vec<f64>>, b: &mut Vec<f64>) -> Vec<f64> {
    let n = a.len();
    for col in 0..n {
        // Partial pivoting: find row with largest absolute value in this column
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            let val = a[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        // Swap rows
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }
        // Eliminate below
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }
    x
}

/// Compute 4th-order polynomial least squares fit coefficients.
/// Returns [c0, c1, c2, c3, c4] for p(x) = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4.
fn polynomial_least_squares(x: &[f32], y: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n < 5 {
        // Not enough points for a 4th order fit; return zeros
        return vec![0.0; 5];
    }

    // Build normal equations: (A^T A) c = A^T y
    // A is the Vandermonde matrix with columns [1, x, x^2, x^3, x^4]
    let order = 5;
    let mut ata = vec![vec![0.0f64; order]; order];
    let mut aty = vec![0.0f64; order];

    for i in 0..n {
        let xi = x[i] as f64;
        let yi = y[i] as f64;
        let mut powers = [1.0f64; 5];
        for k in 1..5 {
            powers[k] = powers[k - 1] * xi;
        }
        for j in 0..order {
            aty[j] += powers[j] * yi;
            for k in 0..order {
                ata[j][k] += powers[j] * powers[k];
            }
        }
    }

    let coeffs = gaussian_elimination(&mut ata, &mut aty);
    coeffs.iter().map(|&c| c as f32).collect()
}

/// Compute DFT of a real signal and find the peak wavelength lambda/r0 and low-band power ratio.
/// Returns (lambda_over_r0, low_band_ratio) where:
/// - lambda_over_r0 = (peak_bin * r0) / n for the dominant frequency bin
/// - low_band_ratio = fraction of total DFT power at wavelengths > 2*r0
fn peak_wavelength_dft(signal: &[f32], r0: f32) -> (f32, f32) {
    let n = signal.len();
    if n < 4 {
        return (0.0, 0.0);
    }

    // Compute DFT magnitudes squared (power spectrum) for bins 1..n/2
    let half_n = n / 2;
    let mut powers = vec![0.0f32; half_n];
    for k in 1..half_n {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        for t in 0..n {
            let angle = 2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            real += signal[t] * angle.cos();
            imag -= signal[t] * angle.sin();
        }
        powers[k] = real * real + imag * imag;
    }

    // Find peak bin (excluding DC)
    let mut peak_bin = 1;
    let mut peak_power = powers[1];
    for k in 2..half_n {
        if powers[k] > peak_power {
            peak_power = powers[k];
            peak_bin = k;
        }
    }

    // lambda = n / peak_bin (in samples), lambda/r0 = n / peak_bin
    let lambda_over_r0 = (n as f32 / peak_bin as f32) / r0;

    // Low-band ratio: fraction of power at wavelengths > 2*r0
    // wavelength in samples = n / k, so n/k > 2*r0 => k < n/(2*r0)
    let threshold_bin = (n as f32 / (2.0 * r0)).floor() as usize;
    let mut total_power = 0.0f32;
    let mut low_band_power = 0.0f32;
    for k in 1..half_n {
        total_power += powers[k];
        if k < threshold_bin {
            low_band_power += powers[k];
        }
    }

    let low_band_ratio = if total_power > 1e-9 {
        low_band_power / total_power
    } else {
        0.0
    };

    (lambda_over_r0, low_band_ratio)
}

/// Compute dominant frequency in Hz from DFT of a real signal.
fn dominant_frequency_hz(signal: &[f32], fps: f32) -> f32 {
    let n = signal.len();
    if n < 2 {
        return 0.0;
    }

    let half_n = n / 2;
    let mut peak_bin = 1;
    let mut peak_power = 0.0f32;

    for k in 1..half_n {
        let mut real = 0.0f32;
        let mut imag = 0.0f32;
        for t in 0..n {
            let angle = 2.0 * std::f32::consts::PI * k as f32 * t as f32 / n as f32;
            real += signal[t] * angle.cos();
            imag -= signal[t] * angle.sin();
        }
        let power = real * real + imag * imag;
        if power > peak_power {
            peak_power = power;
            peak_bin = k;
        }
    }

    // Frequency in Hz = peak_bin * fps / n
    peak_bin as f32 * fps / n as f32
}

/// Extract sequence descriptors from flame animation frames.
///
/// Returns `None` if the frame sequence is empty or too small to compute meaningful descriptors.
pub fn extract_sequence_descriptors(
    frames: &[Vec<f32>],
    width: usize,
    height: usize,
    fps: f32,
) -> Option<SequenceDescriptors> {
    if frames.is_empty() || width == 0 || height == 0 {
        return None;
    }

    let frame_count = frames.len();

    // Compute normalization coordinates from time-averaged luminance
    let (r0, neck_y, pool_bulge_y, zeta_max) =
        compute_normalization_coordinates(frames, width, height);

    // Build per-frame masks and extract half-widths and centroids
    // Compute mask_threshold from 0.3 * percentile99 of blurred time-averaged luminance
    let time_avg = time_averaged_luminance(frames, width, height);
    let mut sorted_time_avg = time_avg.clone();
    sorted_time_avg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mask_threshold = 0.3 * percentile(&sorted_time_avg, 0.99).unwrap_or(0.0);

    // Per-frame half-width arrays (indexed by row)
    let mut frame_half_widths: Vec<Vec<f32>> = Vec::with_capacity(frame_count);
    // Per-frame centroid arrays (indexed by row)
    let mut frame_centroids: Vec<Vec<f32>> = Vec::with_capacity(frame_count);

    for frame in frames {
        let blurred_frame = box_blur(frame, width, height);
        let mask = flame_mask(&blurred_frame, width, height, mask_threshold);
        let mut half_widths = vec![0.0f32; height];
        let mut centroids = vec![0.0f32; height];
        for row in 0..height {
            half_widths[row] = row_half_width(&mask, width, row);
            // Centroid: mean x of true pixels in this row
            let start = row * width;
            let end = start + width;
            let mut sum_x = 0.0f32;
            let mut count = 0usize;
            for (col, &b) in mask[start..end].iter().enumerate() {
                if b {
                    sum_x += col as f32;
                    count += 1;
                }
            }
            centroids[row] = if count > 0 { sum_x / count as f32 } else { 0.0 };
        }
        frame_half_widths.push(half_widths);
        frame_centroids.push(centroids);
    }

    // Build zeta values for each row (clamped to [0, zeta_max] for resampling)
    let zetas = row_zetas(height, neck_y, r0, zeta_max);

    // Build unclamped zeta values for polynomial fit: zeta = (neck_y - y) / r0
    let unclamped_zetas: Vec<f32> = (0..height)
        .map(|y| (neck_y as f32 - y as f32) / r0)
        .collect();

    // f1: time-average half-width / r0, resampled to 48 zeta points
    let mut time_avg_half_widths = vec![0.0f32; height];
    for row in 0..height {
        let sum: f32 = frame_half_widths.iter().map(|fw| fw[row]).sum();
        time_avg_half_widths[row] = sum / frame_count as f32;
    }
    let f1_normalized: Vec<f32> = time_avg_half_widths.iter().map(|&w| w / r0).collect();
    let f1_width = resample_zeta(&f1_normalized, &zetas, zeta_max, 48);

    // f3: time std dev / time average, resampled to 48 zeta points
    let mut f3_values = vec![0.0f32; height];
    for row in 0..height {
        let avg = time_avg_half_widths[row];
        if avg < 1e-9 {
            f3_values[row] = 0.0;
            continue;
        }
        let variance: f32 = frame_half_widths
            .iter()
            .map(|fw| {
                let diff = fw[row] - avg;
                diff * diff
            })
            .sum();
        let std_dev = (variance / frame_count as f32).sqrt();
        f3_values[row] = std_dev / avg;
    }
    let f3_flicker = resample_zeta(&f3_values, &zetas, zeta_max, 48);

    // f2: 4th-order least squares trend residuals of half-widths vs zeta per frame
    // For each frame, fit a 4th order polynomial to half-widths vs zeta and compute residuals
    let mut frame_residuals: Vec<Vec<f32>> = Vec::with_capacity(frame_count);
    for fw in &frame_half_widths {
        // Build x (zeta) and y (half-width) arrays for rows with positive zeta and non-zero half-width
        let mut x_vals: Vec<f32> = Vec::new();
        let mut y_vals: Vec<f32> = Vec::new();
        for row in 0..height {
            if fw[row] > 1e-9 && unclamped_zetas[row] > 1e-6 {
                x_vals.push(unclamped_zetas[row]);
                y_vals.push(fw[row]);
            }
        }
        let coeffs = polynomial_least_squares(&x_vals, &y_vals);
        // Compute residuals: actual - fitted (only for rows with positive zeta and data)
        let mut residuals = vec![0.0f32; height];
        for row in 0..height {
            if fw[row] > 1e-9 && unclamped_zetas[row] > 1e-6 {
                let x = unclamped_zetas[row];
                let fitted = coeffs[0]
                    + coeffs[1] * x
                    + coeffs[2] * x * x
                    + coeffs[3] * x * x * x
                    + coeffs[4] * x * x * x * x;
                residuals[row] = fw[row] - fitted;
            }
        }
        frame_residuals.push(residuals);
    }

    // Time-average of residuals
    let mut time_avg_residuals = vec![0.0f32; height];
    for row in 0..height {
        let sum: f32 = frame_residuals.iter().map(|r| r[row]).sum();
        time_avg_residuals[row] = sum / frame_count as f32;
    }

    // f2_rough: RMS / r0 of the time-average of residuals
    let sum_sq: f32 = time_avg_residuals.iter().map(|&r| r * r).sum();
    let f2_rough = (sum_sq / height as f32).sqrt() / r0;

    // f2_lambda_over_r0 and f2_low_band_ratio from DFT of time-averaged residuals
    let (f2_lambda_over_r0, f2_low_band_ratio) = peak_wavelength_dft(&time_avg_residuals, r0);

    // f4: deviations from per-zeta time average of row centroids
    // Per-zeta time average of centroids
    let mut zeta_avg_centroids = vec![0.0f32; height];
    for row in 0..height {
        let sum: f32 = frame_centroids.iter().map(|c| c[row]).sum();
        zeta_avg_centroids[row] = sum / frame_count as f32;
    }

    // Deviations from per-zeta time average
    let mut total_deviation_sq = 0.0f32;
    let mut deviation_count = 0usize;
    // Also build the zeta-averaged deviation time series for dominant frequency
    let mut zeta_averaged_deviation: Vec<f32> = vec![0.0f32; frame_count];
    for fi in 0..frame_count {
        let mut frame_dev_sum = 0.0f32;
        let mut frame_dev_count = 0usize;
        for row in 0..height {
            let deviation = frame_centroids[fi][row] - zeta_avg_centroids[row];
            total_deviation_sq += deviation * deviation;
            deviation_count += 1;
            frame_dev_sum += deviation.abs();
            frame_dev_count += 1;
        }
        if frame_dev_count > 0 {
            zeta_averaged_deviation[fi] = frame_dev_sum / frame_dev_count as f32;
        }
    }

    // f4_meander_rms: RMS / r0 of deviations from per-zeta time average
    let f4_meander_rms = if deviation_count > 0 {
        (total_deviation_sq / deviation_count as f32).sqrt() / r0
    } else {
        0.0
    };

    // f4_freq_hz: dominant frequency in Hz from DFT of zeta-averaged deviation time series
    let f4_freq_hz = dominant_frequency_hz(&zeta_averaged_deviation, fps);

    // f7: mean component count and mean base-disconnected ratio from flood fill analysis
    let mut total_components = 0usize;
    let mut sum_disconnected_ratio = 0.0f32;
    for frame in frames {
        let blurred_frame = box_blur(frame, width, height);
        let mask = flame_mask(&blurred_frame, width, height, mask_threshold);
        let (count, ratio) = count_components_and_base_ratio(&mask, width, height);
        total_components += count;
        sum_disconnected_ratio += ratio;
    }
    let f7_components_mean = total_components as f32 / frame_count as f32;
    let f7_base_disconnected_ratio = sum_disconnected_ratio / frame_count as f32;

    let meta = SequenceMeta {
        r0_px: r0,
        zeta_max,
        neck_y,
        pool_bulge_y,
        frame_count,
        fps,
    };

    Some(SequenceDescriptors {
        f1_width,
        f2_rough,
        f2_lambda_over_r0,
        f2_low_band_ratio,
        f3_flicker,
        f4_meander_rms,
        f4_freq_hz,
        f7_components_mean,
        f7_base_disconnected_ratio,
        meta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a frame with a solid column (centered) of given half-width at each row.
    /// Each row is filled with 1.0 from (width/2 - half_width) to (width/2 + half_width).
    fn build_frame_with_half_widths(width: usize, height: usize, half_widths: &[f32]) -> Vec<f32> {
        let mut frame = vec![0.0f32; width * height];
        let center = width as f32 * 0.5;
        for row in 0..height {
            let hw = half_widths[row] as usize;
            if hw == 0 {
                continue;
            }
            let start_col = (center - half_widths[row]).max(0.0) as usize;
            let end_col = (center + half_widths[row]).min(width as f32) as usize;
            for col in start_col..end_col {
                frame[row * width + col] = 1.0;
            }
        }
        frame
    }

    #[test]
    fn test_corn_synthesis_identical_frames() {
        // Linear half-width: y-axis linearly increasing, 40 identical frames
        let width = 1024;
        let height = 2048;
        let num_frames = 40;

        // Build half-widths: linear from 40 to 160 (increasing with row)
        let mut half_widths = vec![0.0f32; height];
        for row in 0..height {
            half_widths[row] = 40.0 + 120.0 * row as f32 / (height - 1) as f32;
        }

        // Build identical frames
        let frame = build_frame_with_half_widths(width, height, &half_widths);
        let frames: Vec<Vec<f32>> = vec![frame; num_frames];

        let descriptors = extract_sequence_descriptors(&frames, width, height, 30.0).unwrap();

        // f2_rough should be near zero (identical frames, polynomial fit is exact for linear)
        assert!(
            descriptors.f2_rough < 0.05,
            "f2_rough = {}, expected < 0.05",
            descriptors.f2_rough
        );

        // f4_meander_rms should be near zero (identical frames, no deviation from time average)
        assert!(
            descriptors.f4_meander_rms < 0.01,
            "f4_meander_rms = {}, expected < 0.01",
            descriptors.f4_meander_rms
        );

        // f7_base_disconnected_ratio should be near zero (single connected component touching base)
        assert!(
            descriptors.f7_base_disconnected_ratio < 0.01,
            "f7_base_disconnected_ratio = {}, expected < 0.01",
            descriptors.f7_base_disconnected_ratio
        );
    }

    #[test]
    fn test_column_plus_sin_lobe() {
        // Column + sin lobe: flame-like tapered half-width + sinusoidal modulation
        // The taper (narrow at top, wide at bottom) gives zeta variation so polynomial fit
        // residuals are non-trivial. pool_bulge_y is at the bottom, so band_rows is non-empty.
        let width = 256;
        let height = 128;
        let num_frames = 40;
        let center = width as f32 * 0.5;

        // r0: half-width of the column at the bottom, ~40px
        let r0_px = 40.0f32;
        let amplitude = 0.1 * r0_px; // 0.1 * r0

        let mut frames: Vec<Vec<f32>> = Vec::with_capacity(num_frames);
        for fi in 0..num_frames {
            let center_shift = (fi as f32 / num_frames as f32) * 5.0;
            let mut frame = vec![0.0f32; width * height];
            for row in 0..height {
                // Flame-like taper: half-width increases from 0.5*r0_px at top to r0_px at bottom
                // plus sinusoidal modulation with spatial period 3*r0_px (static, same for all frames)
                let taper = 0.5 + 0.5 * row as f32 / (height - 1) as f32;
                let hw = r0_px * taper
                    + amplitude * (2.0 * std::f32::consts::PI * row as f32 / (3.0 * r0_px)).sin();
                let start_col = (center + center_shift - hw).max(0.0) as usize;
                let end_col = (center + center_shift + hw).min(width as f32) as usize;
                for col in start_col..end_col {
                    frame[row * width + col] = 1.0;
                }
            }
            frames.push(frame);
        }

        let descriptors = extract_sequence_descriptors(&frames, width, height, 30.0).unwrap();

        // f2_lambda_over_r0 should be in [2.0, 4.0] (wavelength of sin modulation)
        assert!(
            descriptors.f2_lambda_over_r0 >= 2.0 && descriptors.f2_lambda_over_r0 <= 4.0,
            "f2_lambda_over_r0 = {}, expected in [2.0, 4.0]",
            descriptors.f2_lambda_over_r0
        );

        // f4_meander_rms should be > 0.005 (centroids oscillate due to sin modulation)
        assert!(
            descriptors.f4_meander_rms > 0.005,
            "f4_meander_rms = {}, expected > 0.005",
            descriptors.f4_meander_rms
        );
    }
}
