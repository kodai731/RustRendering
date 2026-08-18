/// Convert linear RGB to CIE xy chromaticity coordinates via the sRGB D65 matrix.
/// Returns None if X + Y + Z < 1e-6 (near-black).
pub fn chromaticity_xy(rgb_linear: [f32; 3]) -> Option<[f32; 2]> {
    let x = 0.4124 * rgb_linear[0] + 0.3576 * rgb_linear[1] + 0.1805 * rgb_linear[2];
    let y = 0.2126 * rgb_linear[0] + 0.7152 * rgb_linear[1] + 0.0722 * rgb_linear[2];
    let z = 0.0193 * rgb_linear[0] + 0.1192 * rgb_linear[1] + 0.9505 * rgb_linear[2];
    let sum = x + y + z;
    if sum < 1e-6 {
        None
    } else {
        Some([x / sum, y / sum])
    }
}

/// McCamy's approximation of Correlated Color Temperature from xy chromaticity.
/// Returns None if the denominator is ~0 or the result falls outside 500..=10000 K.
pub fn mccamy_cct(xy: [f32; 2]) -> Option<f32> {
    let denom = 0.1858 - xy[1];
    if denom.abs() < 1e-6 {
        return None;
    }
    let n = (xy[0] - 0.3320) / denom;
    let cct = 449.0 * n.powi(3) + 3525.0 * n.powi(2) + 6823.3 * n + 5520.33;
    if cct >= 500.0 && cct <= 10000.0 {
        Some(cct)
    } else {
        None
    }
}
