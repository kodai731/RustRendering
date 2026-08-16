/// Linear sRGB chromaticity of a Planckian radiator, normalized so the largest
/// channel is 1 (radiance magnitude is left to the caller); valid 1000-15000 K.
pub fn blackbody_rgb(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(1000.0, 15000.0) as f64;
    let (u, v) = planckian_uv(t);
    let denominator = 2.0 * u - 8.0 * v + 4.0;
    let x = 3.0 * u / denominator;
    let y = 2.0 * v / denominator;
    let xyz = [x / y, 1.0, (1.0 - x - y) / y];

    let linear = [
        3.2406 * xyz[0] - 1.5372 * xyz[1] - 0.4986 * xyz[2],
        -0.9689 * xyz[0] + 1.8758 * xyz[1] + 0.0415 * xyz[2],
        0.0557 * xyz[0] - 0.2040 * xyz[1] + 1.0570 * xyz[2],
    ];
    let peak = linear.iter().cloned().fold(0.0f64, f64::max).max(1e-6);
    let mut rgb = [0.0f32; 3];
    for (channel, value) in rgb.iter_mut().zip(linear) {
        *channel = (value / peak).clamp(0.0, 1.0) as f32;
    }
    rgb
}

// Krystek 1985 rational fit of the Planckian locus in CIE 1960 (u, v).
fn planckian_uv(t: f64) -> (f64, f64) {
    let u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t)
        / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t);
    let v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t)
        / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t);
    (u, v)
}
