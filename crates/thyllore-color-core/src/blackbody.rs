const KELVIN_MIN: f64 = 1000.0;
const KELVIN_MAX: f64 = 15000.0;

// sRGB D65 XYZ -> linear RGB matrix (IEC 61966-2-1, http://www.brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html).
const XYZ_TO_LINEAR_SRGB: [[f64; 3]; 3] = [
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
];

/// Linear sRGB chromaticity of a Planckian radiator, normalized so the largest
/// channel is 1 (radiance magnitude is left to the caller); valid 1000-15000 K.
pub fn blackbody_rgb(kelvin: f32) -> [f32; 3] {
    let t = (kelvin as f64).clamp(KELVIN_MIN, KELVIN_MAX);
    let (u, v) = planckian_uv(t);
    let xyz = cie_uv_to_xyz_unit_luminance(u, v);

    let linear = XYZ_TO_LINEAR_SRGB.map(|row| row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2]);
    let peak = linear.iter().cloned().fold(0.0f64, f64::max).max(1e-6);
    let mut rgb = [0.0f32; 3];
    for (channel, value) in rgb.iter_mut().zip(linear) {
        *channel = (value / peak).clamp(0.0, 1.0) as f32;
    }
    rgb
}

// CIE 1960 (u, v) -> CIE 1931 (x, y) -> XYZ with Y = 1 (https://en.wikipedia.org/wiki/CIE_1960_color_space).
fn cie_uv_to_xyz_unit_luminance(u: f64, v: f64) -> [f64; 3] {
    let denominator = 2.0 * u - 8.0 * v + 4.0;
    let x = 3.0 * u / denominator;
    let y = 2.0 * v / denominator;
    [x / y, 1.0, (1.0 - x - y) / y]
}

// Krystek 1985 rational fit of the Planckian locus in CIE 1960 (u, v), https://doi.org/10.1002/col.5080100109
fn planckian_uv(t: f64) -> (f64, f64) {
    let u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t)
        / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t);
    let v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t)
        / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t);
    (u, v)
}
