use cgmath::{InnerSpace, Vector3};

pub const WATER_WAVE_MODE_COUNT: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WaterWaveMode {
    pub m: i32,
    pub n: i32,
    pub amplitude: f32,
    pub omega: f32,
    pub phase: f32,
}

/// Deterministic LCG (Linear Congruential Generator) with fixed seed.
fn lcg_next(state: &mut u64) -> u64 {
    *state = (*state)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

/// Fixed coefficient tables F[k] and G[k] in [0.5, 2.0].
const F: [f32; WATER_WAVE_MODE_COUNT] = [1.0, 1.5, 0.5, 2.0, 1.2, 0.8, 1.8, 1.3];
const G: [f32; WATER_WAVE_MODE_COUNT] = [1.5, 0.5, 2.0, 1.0, 1.7, 1.1, 0.6, 1.9];

pub fn generate_water_wave_modes(
    wave_amplitude: f32,
    wave_frequency: f32,
    wave_speed: f32,
) -> [WaterWaveMode; WATER_WAVE_MODE_COUNT] {
    let mut state: u64 = 12345;

    let mut modes: [WaterWaveMode; WATER_WAVE_MODE_COUNT] =
        [WaterWaveMode::default(); WATER_WAVE_MODE_COUNT];

    // First pass: compute m, n, omega, phase and raw amplitudes
    for k in 0..WATER_WAVE_MODE_COUNT {
        let mut m = (wave_frequency * F[k]).round() as i32;
        let mut n = (wave_frequency * G[k]).round() as i32;

        // If (m, n) == (0, 0), set m = 1
        if m == 0 && n == 0 {
            m = 1;
        }

        let omega = wave_speed * ((m * m + n * n) as f32).sqrt();

        // Phase from LCG in [0, 2π)
        let phase =
            (lcg_next(&mut state) as f64 / (1u64 << 63) as f64 * 2.0 * std::f64::consts::PI) as f32;

        // Raw amplitude = wave_amplitude * 2^(-k/2)
        let raw_amplitude = wave_amplitude * (2.0_f32).powi(-(k as i32) / 2);

        modes[k] = WaterWaveMode {
            m,
            n,
            amplitude: raw_amplitude,
            omega,
            phase,
        };
    }

    // Normalize amplitudes so Σ amplitude ≈ wave_amplitude
    let sum: f32 = modes.iter().map(|m| m.amplitude).sum();
    if sum > 1e-6 {
        let scale = wave_amplitude / sum;
        for mode in &mut modes {
            mode.amplitude *= scale;
        }
    }

    modes
}

/// Compute water surface height and gradient at (u, v) at given time.
/// Returns (h, h_u, h_v).
/// Phase φ' = m(u + a*t) + n(v + b*t) - ω*t + φ.
/// h = Σ a * cos(φ'), h_u = -Σ a * m * sin(φ'), h_v = -Σ a * n * sin(φ').
pub fn water_height_and_gradient(
    u: f32,
    v: f32,
    time: f32,
    flow: (f32, f32),
    modes: &[WaterWaveMode],
) -> (f32, f32, f32) {
    let (a, b) = flow;

    let mut h = 0.0f32;
    let mut h_u = 0.0f32;
    let mut h_v = 0.0f32;

    for mode in modes {
        let phase_prime = mode.m as f32 * (u + a * time) + mode.n as f32 * (v + b * time)
            - mode.omega * time
            + mode.phase;

        let cos_val = phase_prime.cos();
        let sin_val = phase_prime.sin();

        h += mode.amplitude * cos_val;
        h_u -= mode.amplitude * mode.m as f32 * sin_val;
        h_v -= mode.amplitude * mode.n as f32 * sin_val;
    }

    (h, h_u, h_v)
}

/// Compute perturbed normal at (u, v) given height and gradient.
/// e_u = (-sin u, 0, cos u), e_v = (-sin v cos u, cos v, -sin v sin u), n = (cos v cos u, sin v, cos v sin u).
/// κ1 = 1/r, κ2 = cos v / (R + r cos v).
/// n' = normalize((1+hκ1)(1+hκ2)n - (1+hκ1)*h_u/(R + r cos v)*e_u - (1+hκ2)*h_v/r*e_v).
pub fn water_perturbed_normal(
    u: f32,
    v: f32,
    h: f32,
    h_u: f32,
    h_v: f32,
    major_radius: f32,
    minor_radius: f32,
) -> Vector3<f32> {
    let cos_u = u.cos();
    let sin_u = u.sin();
    let cos_v = v.cos();
    let sin_v = v.sin();

    // Tangent vectors and surface normal
    let e_u = Vector3::new(-sin_u, 0.0, cos_u);
    let e_v = Vector3::new(-sin_v * cos_u, cos_v, -sin_v * sin_u);
    let n = Vector3::new(cos_v * cos_u, sin_v, cos_v * sin_u);

    // Curvatures
    let kappa1 = 1.0 / minor_radius;
    let kappa2 = cos_v / (major_radius + minor_radius * cos_v);

    // Perturbed normal
    let mut n_prime = (1.0 + h * kappa1) * (1.0 + h * kappa2) * n
        - (1.0 + h * kappa1) * h_u / (major_radius + minor_radius * cos_v) * e_u
        - (1.0 + h * kappa2) * h_v / minor_radius * e_v;

    let mag = n_prime.magnitude();
    if mag > 1e-6 {
        n_prime /= mag;
    } else {
        n_prime = n;
    }

    n_prime
}
