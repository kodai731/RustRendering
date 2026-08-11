//! CPU mirror of the fbm value-noise lattice (GLSL flame_noise.glsl
//! hashCell / valueNoise3 / fbm3). The product samples fbm only on the GPU
//! (boundary displacement and contour-wiggle fallback); this mirror exists to
//! pin the statistics the wave basis is calibrated against
//! (WAVE_NOISE_MEAN = 0.4375, WAVE_NOISE_STD = 0.106).

fn glsl_fract(x: f32) -> f32 {
    x - x.floor()
}

pub fn hash_cell(cell: [f32; 3]) -> f32 {
    let mut p = [
        glsl_fract(cell[0] * 0.318_309_9 + 0.1),
        glsl_fract(cell[1] * 0.318_309_9 + 0.2),
        glsl_fract(cell[2] * 0.318_309_9 + 0.3),
    ];
    for value in &mut p {
        *value *= 17.0;
    }
    glsl_fract(p[0] * p[1] * p[2] * (p[0] + p[1] + p[2]))
}

pub fn value_noise3(p: [f32; 3]) -> f32 {
    let cell = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [glsl_fract(p[0]), glsl_fract(p[1]), glsl_fract(p[2])];
    let u = [
        f[0] * f[0] * (3.0 - 2.0 * f[0]),
        f[1] * f[1] * (3.0 - 2.0 * f[1]),
        f[2] * f[2] * (3.0 - 2.0 * f[2]),
    ];

    let h = |dx: f32, dy: f32, dz: f32| hash_cell([cell[0] + dx, cell[1] + dy, cell[2] + dz]);
    let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;

    let nx00 = mix(h(0.0, 0.0, 0.0), h(1.0, 0.0, 0.0), u[0]);
    let nx10 = mix(h(0.0, 1.0, 0.0), h(1.0, 1.0, 0.0), u[0]);
    let nx01 = mix(h(0.0, 0.0, 1.0), h(1.0, 0.0, 1.0), u[0]);
    let nx11 = mix(h(0.0, 1.0, 1.0), h(1.0, 1.0, 1.0), u[0]);
    let nxy0 = mix(nx00, nx10, u[1]);
    let nxy1 = mix(nx01, nx11, u[1]);
    mix(nxy0, nxy1, u[2])
}

pub fn fbm3(p: [f32; 3]) -> f32 {
    let mut sum = 0.0f32;
    let mut amplitude = 0.5f32;
    let mut q = p;
    for _ in 0..3 {
        sum += amplitude * value_noise3(q);
        for axis in 0..3 {
            q[axis] = q[axis] * 2.02 + 13.7;
        }
        amplitude *= 0.5;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use thyllore_effect_core::flame_wave::{WAVE_NOISE_MEAN, WAVE_NOISE_STD};

    /// The calibration constants the wave basis reproduces are the measured
    /// lattice statistics of this fbm (mean = 0.5 * (0.5 + 0.25 + 0.125)).
    #[test]
    fn test_fbm3_statistics_match_wave_calibration() {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let count = 48usize;
        let total = (count * count * count) as f64;
        for ix in 0..count {
            for iy in 0..count {
                for iz in 0..count {
                    let p = [
                        ix as f32 * 0.37 + 0.11,
                        iy as f32 * 0.41 + 0.23,
                        iz as f32 * 0.43 + 0.07,
                    ];
                    let value = fbm3(p) as f64;
                    sum += value;
                    sum_sq += value * value;
                }
            }
        }
        let mean = sum / total;
        let std = (sum_sq / total - mean * mean).sqrt();
        assert!(
            (mean - WAVE_NOISE_MEAN as f64).abs() < 0.01,
            "fbm mean {mean} vs {WAVE_NOISE_MEAN}"
        );
        assert!(
            (std - WAVE_NOISE_STD as f64).abs() < 0.01,
            "fbm std {std} vs {WAVE_NOISE_STD}"
        );
    }
}
