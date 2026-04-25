pub fn sample_curve_linear(times: &[f32], values: &[f32], t: f32) -> f32 {
    debug_assert_eq!(
        times.len(),
        values.len(),
        "times and values must have the same length"
    );

    if times.is_empty() {
        return 0.0;
    }
    if times.len() == 1 || t <= times[0] {
        return values[0];
    }
    let last_idx = times.len() - 1;
    if t >= times[last_idx] {
        return values[last_idx];
    }

    for i in 0..last_idx {
        let t0 = times[i];
        let t1 = times[i + 1];
        if t >= t0 && t <= t1 {
            let dt = t1 - t0;
            if dt < 1e-9 {
                return values[i];
            }
            let ratio = (t - t0) / dt;
            return values[i] + (values[i + 1] - values[i]) * ratio;
        }
    }

    values[last_idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slice_returns_zero() {
        assert_eq!(sample_curve_linear(&[], &[], 0.5), 0.0);
    }

    #[test]
    fn single_point_returns_constant() {
        assert_eq!(sample_curve_linear(&[1.0], &[42.0], 0.0), 42.0);
        assert_eq!(sample_curve_linear(&[1.0], &[42.0], 5.0), 42.0);
    }

    #[test]
    fn before_first_returns_first_value() {
        assert_eq!(sample_curve_linear(&[1.0, 2.0], &[10.0, 20.0], 0.0), 10.0);
    }

    #[test]
    fn after_last_returns_last_value() {
        assert_eq!(sample_curve_linear(&[1.0, 2.0], &[10.0, 20.0], 5.0), 20.0);
    }

    #[test]
    fn midpoint_interpolates_linearly() {
        let result = sample_curve_linear(&[0.0, 2.0], &[0.0, 10.0], 1.0);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn coincident_times_returns_lower_value() {
        let result = sample_curve_linear(&[1.0, 1.0, 2.0], &[10.0, 999.0, 20.0], 1.0);
        assert_eq!(result, 10.0);
    }
}
