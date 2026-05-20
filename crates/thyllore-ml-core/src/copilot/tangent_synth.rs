const ZERO_EPSILON: f32 = 1.0e-8;

#[inline]
fn is_handle_zero(time_offset: f32, value_offset: f32) -> bool {
    time_offset.abs() < ZERO_EPSILON && value_offset.abs() < ZERO_EPSILON
}

pub fn resolve_in_tangent(
    stored_dt: f32,
    stored_dv: f32,
    prev: Option<(f32, f32)>,
    curr_time: f32,
    curr_value: f32,
) -> (f32, f32) {
    if !is_handle_zero(stored_dt, stored_dv) {
        return (stored_dt, stored_dv);
    }
    let Some((prev_time, prev_value)) = prev else {
        return (0.0, 0.0);
    };
    let dt = curr_time - prev_time;
    if dt.abs() < ZERO_EPSILON {
        return (0.0, 0.0);
    }
    let dv = curr_value - prev_value;
    (-dt / 3.0, -dv / 3.0)
}

pub fn resolve_out_tangent(
    stored_dt: f32,
    stored_dv: f32,
    curr_time: f32,
    curr_value: f32,
    next: Option<(f32, f32)>,
) -> (f32, f32) {
    if !is_handle_zero(stored_dt, stored_dv) {
        return (stored_dt, stored_dv);
    }
    let Some((next_time, next_value)) = next else {
        return (0.0, 0.0);
    };
    let dt = next_time - curr_time;
    if dt.abs() < ZERO_EPSILON {
        return (0.0, 0.0);
    }
    let dv = next_value - curr_value;
    (dt / 3.0, dv / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_in_tangent_is_preserved_when_non_zero() {
        let (dt, dv) = resolve_in_tangent(-0.2, -0.5, Some((0.0, 0.0)), 1.0, 1.0);
        assert!((dt - (-0.2)).abs() < 1e-6);
        assert!((dv - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn stored_out_tangent_is_preserved_when_non_zero() {
        let (dt, dv) = resolve_out_tangent(0.2, 0.5, 1.0, 1.0, Some((2.0, 2.0)));
        assert!((dt - 0.2).abs() < 1e-6);
        assert!((dv - 0.5).abs() < 1e-6);
    }

    #[test]
    fn missing_prev_neighbor_yields_zero_in_tangent() {
        let (dt, dv) = resolve_in_tangent(0.0, 0.0, None, 1.0, 1.0);
        assert_eq!(dt, 0.0);
        assert_eq!(dv, 0.0);
    }

    #[test]
    fn missing_next_neighbor_yields_zero_out_tangent() {
        let (dt, dv) = resolve_out_tangent(0.0, 0.0, 1.0, 1.0, None);
        assert_eq!(dt, 0.0);
        assert_eq!(dv, 0.0);
    }

    #[test]
    fn zero_in_tangent_is_synthesized_from_previous_segment() {
        let (dt, dv) = resolve_in_tangent(0.0, 0.0, Some((1.0, 1.0)), 4.0, 7.0);
        let segment_dt = 4.0 - 1.0;
        let segment_dv = 7.0 - 1.0;
        assert!((dt - (-segment_dt / 3.0)).abs() < 1e-6);
        assert!((dv - (-segment_dv / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn zero_out_tangent_is_synthesized_from_next_segment() {
        let (dt, dv) = resolve_out_tangent(0.0, 0.0, 1.0, 1.0, Some((4.0, 7.0)));
        let segment_dt = 4.0 - 1.0;
        let segment_dv = 7.0 - 1.0;
        assert!((dt - segment_dt / 3.0).abs() < 1e-6);
        assert!((dv - segment_dv / 3.0).abs() < 1e-6);
    }

    #[test]
    fn rising_linear_segment_produces_positive_out_dv_and_negative_in_dv() {
        let prev = (0.0, -0.48);
        let curr = (1.0, 0.39);
        let next = (2.0, 1.26);

        let (in_dt, in_dv) = resolve_in_tangent(0.0, 0.0, Some(prev), curr.0, curr.1);
        let (out_dt, out_dv) = resolve_out_tangent(0.0, 0.0, curr.0, curr.1, Some(next));

        assert!(in_dt < 0.0);
        assert!(in_dv < 0.0);
        assert!(out_dt > 0.0);
        assert!(out_dv > 0.0);
    }

    #[test]
    fn zero_segment_duration_yields_zero_tangent() {
        let (in_dt, in_dv) = resolve_in_tangent(0.0, 0.0, Some((1.0, 0.5)), 1.0, 0.7);
        assert_eq!(in_dt, 0.0);
        assert_eq!(in_dv, 0.0);

        let (out_dt, out_dv) = resolve_out_tangent(0.0, 0.0, 1.0, 0.7, Some((1.0, 0.5)));
        assert_eq!(out_dt, 0.0);
        assert_eq!(out_dv, 0.0);
    }
}
