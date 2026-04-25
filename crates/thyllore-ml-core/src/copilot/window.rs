use super::sampling::sample_curve_linear;

pub const PAE_WINDOW_SIZE: usize = 64;

pub fn sample_window(
    times: &[f32],
    values: &[f32],
    t_start: f32,
    t_end: f32,
    curve_mean: f32,
    curve_std: f32,
) -> [f32; PAE_WINDOW_SIZE] {
    let mut window = [0.0_f32; PAE_WINDOW_SIZE];

    if times.is_empty() || (t_end - t_start).abs() < 1e-8 {
        return window;
    }

    let std_safe = curve_std.max(1e-6);
    let denom = (PAE_WINDOW_SIZE - 1) as f32;
    let span = t_end - t_start;

    for i in 0..PAE_WINDOW_SIZE {
        let t = t_start + (i as f32 / denom) * span;
        let value = sample_curve_linear(times, values, t);
        window[i] = (value - curve_mean) / std_safe;
    }

    window
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_size_is_fixed() {
        let window = sample_window(&[0.0, 1.0], &[0.0, 10.0], 0.0, 1.0, 0.0, 1.0);
        assert_eq!(window.len(), PAE_WINDOW_SIZE);
    }

    #[test]
    fn empty_curve_returns_zeros() {
        let window = sample_window(&[], &[], 0.0, 1.0, 0.0, 1.0);
        for v in window.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn zero_range_returns_zeros() {
        let window = sample_window(&[0.0, 1.0], &[5.0, 10.0], 0.5, 0.5, 0.0, 1.0);
        for v in window.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn normalization_subtracts_mean_divides_std() {
        let window = sample_window(&[0.0, 1.0], &[100.0, 100.0], 0.0, 1.0, 100.0, 1.0);
        for v in window.iter() {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn endpoints_match_t_range() {
        let window = sample_window(&[0.0, 1.0], &[0.0, 10.0], 0.0, 1.0, 0.0, 1.0);
        assert!((window[0] - 0.0).abs() < 1e-5);
        assert!((window[PAE_WINDOW_SIZE - 1] - 10.0).abs() < 1e-5);
    }
}
