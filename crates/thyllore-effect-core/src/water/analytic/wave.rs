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

const DETERMINISTIC_MODE_COUNT: usize = 4;

fn next_unit_f64(state: &mut u64) -> f64 {
    lcg_next(state) as f64 / (1u64 << 63) as f64
}

fn build_deterministic_mode(
    slot: usize,
    wave_amplitude: f32,
    wave_frequency: f32,
    wave_speed: f32,
    state: &mut u64,
) -> WaterWaveMode {
    let mut m = (wave_frequency * F[slot]).round() as i32;
    let n = (wave_frequency * G[slot]).round() as i32;

    if m == 0 && n == 0 {
        m = 1;
    }

    WaterWaveMode {
        m,
        n,
        amplitude: wave_amplitude * (2.0_f32).powi(-(slot as i32) / 2),
        omega: wave_speed * ((m * m + n * n) as f32).sqrt(),
        phase: (next_unit_f64(state) * 2.0 * std::f64::consts::PI) as f32,
    }
}

/// Deep-water dispersion sample: |k| log-uniform in [0.5, 3.0] * wave_frequency,
/// direction uniform, omega = wave_speed * sqrt(|k|), amplitude proportional to 1 / |k|.
fn sample_dispersive_mode(wave_frequency: f32, wave_speed: f32, state: &mut u64) -> WaterWaveMode {
    let log_min = (wave_frequency.max(1e-6) * 0.5).ln() as f64;
    let log_max = (wave_frequency.max(1e-6) * 3.0).ln() as f64;

    let wave_number = (log_min + next_unit_f64(state) * (log_max - log_min)).exp();
    let theta = next_unit_f64(state) * 2.0 * std::f64::consts::PI;

    let mut m = (wave_number * theta.cos()).round() as i32;
    let n = (wave_number * theta.sin()).round() as i32;

    if m == 0 && n == 0 {
        m = 1;
    }

    WaterWaveMode {
        m,
        n,
        amplitude: (1.0 / wave_number) as f32,
        omega: wave_speed * (wave_number as f32).sqrt(),
        phase: (next_unit_f64(state) * 2.0 * std::f64::consts::PI) as f32,
    }
}

fn rescale_amplitudes(modes: &mut [WaterWaveMode], target_sum: f32) {
    let sum: f32 = modes.iter().map(|mode| mode.amplitude).sum();
    if sum <= 1e-6 {
        return;
    }

    let scale = target_sum / sum;
    for mode in modes {
        mode.amplitude *= scale;
    }
}

pub fn generate_water_wave_modes(
    wave_amplitude: f32,
    wave_frequency: f32,
    wave_speed: f32,
    dispersion: f32,
    frame_index: u32,
) -> [WaterWaveMode; WATER_WAVE_MODE_COUNT] {
    let mut modes = [WaterWaveMode::default(); WATER_WAVE_MODE_COUNT];
    let mut deterministic_state: u64 = 12345;

    if dispersion <= 0.0 {
        for slot in 0..WATER_WAVE_MODE_COUNT {
            modes[slot] = build_deterministic_mode(
                slot,
                wave_amplitude,
                wave_frequency,
                wave_speed,
                &mut deterministic_state,
            );
        }

        rescale_amplitudes(&mut modes, wave_amplitude);
        return modes;
    }

    let dispersion = dispersion.min(1.0);

    for slot in 0..DETERMINISTIC_MODE_COUNT {
        modes[slot] = build_deterministic_mode(
            slot,
            wave_amplitude,
            wave_frequency,
            wave_speed,
            &mut deterministic_state,
        );
    }

    let mut dispersive_state: u64 =
        12345 ^ (frame_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for slot in DETERMINISTIC_MODE_COUNT..WATER_WAVE_MODE_COUNT {
        modes[slot] = sample_dispersive_mode(wave_frequency, wave_speed, &mut dispersive_state);
    }

    let (deterministic, dispersive) = modes.split_at_mut(DETERMINISTIC_MODE_COUNT);
    rescale_amplitudes(deterministic, wave_amplitude * (1.0 - dispersion));
    rescale_amplitudes(dispersive, wave_amplitude * dispersion);

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

    let e_u = Vector3::new(-sin_u, 0.0, cos_u);
    let e_v = Vector3::new(-sin_v * cos_u, cos_v, -sin_v * sin_u);
    let n = Vector3::new(cos_v * cos_u, sin_v, cos_v * sin_u);
    let kappa1 = 1.0 / minor_radius;
    let kappa2 = cos_v / (major_radius + minor_radius * cos_v);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispersion_zero_matches_old_output() {
        let modes = generate_water_wave_modes(0.02, 6.0, 1.0, 0.0, 0);

        let sum: f32 = modes.iter().map(|m| m.amplitude).sum();
        assert!(
            (sum - 0.02).abs() < 1e-5,
            "dispersion=0.0: total amplitude sum {} != 0.02",
            sum
        );

        let expected_m = [6, 9, 3, 12];
        let expected_n = [9, 3, 12, 6];
        for (i, (m, n)) in expected_m.iter().zip(expected_n.iter()).enumerate() {
            assert_eq!(modes[i].m, *m, "slot {}: m mismatch", i);
            assert_eq!(modes[i].n, *n, "slot {}: n mismatch", i);
        }

        let expected_m_4 = [7, 5, 11, 8];
        let expected_n_4 = [10, 7, 4, 11];
        for (i, (m, n)) in expected_m_4.iter().zip(expected_n_4.iter()).enumerate() {
            assert_eq!(modes[i + 4].m, *m, "slot {}: m mismatch", i + 4);
            assert_eq!(modes[i + 4].n, *n, "slot {}: n mismatch", i + 4);
        }
    }

    #[test]
    fn test_dispersion_half_different_across_frames() {
        let modes_frame0 = generate_water_wave_modes(0.02, 6.0, 1.0, 0.5, 0);
        let modes_frame1 = generate_water_wave_modes(0.02, 6.0, 1.0, 0.5, 1);

        for i in 0..4 {
            assert_eq!(
                modes_frame0[i].m, modes_frame1[i].m,
                "slot {}: m differs",
                i
            );
            assert_eq!(
                modes_frame0[i].n, modes_frame1[i].n,
                "slot {}: n differs",
                i
            );
            assert!(
                (modes_frame0[i].omega - modes_frame1[i].omega).abs() < 1e-6,
                "slot {}: omega differs",
                i
            );
        }

        let mut differs = false;
        for i in 4..8 {
            if modes_frame0[i].m != modes_frame1[i].m
                || modes_frame0[i].n != modes_frame1[i].n
                || (modes_frame0[i].omega - modes_frame1[i].omega).abs() > 1e-6
            {
                differs = true;
            }
        }
        assert!(
            differs,
            "slots 4..7 should differ between frame_index 0 and 1"
        );

        let sum0: f32 = modes_frame0.iter().map(|m| m.amplitude).sum();
        let sum1: f32 = modes_frame1.iter().map(|m| m.amplitude).sum();
        assert!(
            (sum0 - 0.02).abs() < 1e-5,
            "frame 0: total amplitude sum {} != 0.02",
            sum0
        );
        assert!(
            (sum1 - 0.02).abs() < 1e-5,
            "frame 1: total amplitude sum {} != 0.02",
            sum1
        );
    }

    #[test]
    fn test_dispersion_full_sum_check() {
        let modes = generate_water_wave_modes(0.05, 8.0, 2.0, 1.0, 42);
        let sum: f32 = modes.iter().map(|m| m.amplitude).sum();
        assert!(
            (sum - 0.05).abs() < 1e-5,
            "dispersion=1.0: total amplitude sum {} != 0.05",
            sum
        );
    }
}
