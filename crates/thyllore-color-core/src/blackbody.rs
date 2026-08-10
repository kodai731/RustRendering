/// Approximate blackbody (Planckian locus) color for a given temperature in Kelvin.
/// Uses a polynomial approximation valid for 800K-3000K, returning clamped linear RGB [0,1].
pub fn blackbody_rgb(kelvin: f32) -> [f32; 3] {
    let t = (kelvin - 800.0) / (3000.0 - 800.0); // normalize to [0, 1]
    let t2 = t * t;
    let t3 = t2 * t;

    // Polynomial approximation of Planckian locus for 800K-3000K
    // R: starts near 1.0 (hot), stays high
    // G: increases from ~0.1 to ~0.7
    // B: increases from ~0.0 to ~0.4
    let r = 1.0 - 0.3 * t + 0.2 * t2;
    let g = 0.1 + 0.6 * t - 0.15 * t2 + 0.1 * t3;
    let b = 0.0 + 0.4 * t - 0.2 * t2 + 0.15 * t3;

    [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]
}
