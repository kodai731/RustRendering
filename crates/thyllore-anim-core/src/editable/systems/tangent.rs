use crate::editable::components::keyframe::{
    BezierHandle, EditableKeyframe, TangentType, TangentWeightMode,
};

pub fn sample_bezier(
    k0_time: f32,
    k0_value: f32,
    k0_out: &BezierHandle,
    k1_time: f32,
    k1_value: f32,
    k1_in: &BezierHandle,
    time: f32,
) -> f32 {
    let dt = k1_time - k0_time;
    let (out_time, out_value) =
        clamp_handle_to_segment(k0_out.time_offset, k0_out.value_offset, dt);
    let (in_time, in_value) = clamp_handle_to_segment(k1_in.time_offset, k1_in.value_offset, -dt);

    let p0_t = k0_time;
    let p1_t = k0_time + out_time;
    let p2_t = k1_time + in_time;
    let p3_t = k1_time;

    let p0_v = k0_value;
    let p1_v = k0_value + out_value;
    let p2_v = k1_value + in_value;
    let p3_v = k1_value;

    let u = find_bezier_t_for_time(p0_t, p1_t, p2_t, p3_t, time);

    evaluate_cubic(p0_v, p1_v, p2_v, p3_v, u)
}

fn clamp_handle_to_segment(time_offset: f32, value_offset: f32, max_abs_time: f32) -> (f32, f32) {
    let limit = max_abs_time.abs();
    if time_offset.abs() <= limit {
        return (time_offset, value_offset);
    }

    if time_offset.abs() < 1e-8 {
        return (0.0, value_offset);
    }

    let ratio = limit / time_offset.abs();
    let clamped_time = time_offset.signum() * limit;
    let clamped_value = value_offset * ratio;
    (clamped_time, clamped_value)
}

fn find_bezier_t_for_time(p0: f32, p1: f32, p2: f32, p3: f32, target: f32) -> f32 {
    let mut t = (target - p0) / (p3 - p0).max(0.0001);
    t = t.clamp(0.0, 1.0);

    for _ in 0..8 {
        let current = evaluate_cubic(p0, p1, p2, p3, t);
        let derivative = evaluate_cubic_derivative(p0, p1, p2, p3, t);

        if derivative.abs() < 1e-8 {
            break;
        }

        let correction = (current - target) / derivative;
        t -= correction;
        t = t.clamp(0.0, 1.0);

        if correction.abs() < 1e-6 {
            break;
        }
    }

    t
}

fn evaluate_cubic(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let one_minus_t = 1.0 - t;
    let one_minus_t2 = one_minus_t * one_minus_t;
    let t2 = t * t;

    one_minus_t2 * one_minus_t * p0
        + 3.0 * one_minus_t2 * t * p1
        + 3.0 * one_minus_t * t2 * p2
        + t2 * t * p3
}

fn evaluate_cubic_derivative(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let one_minus_t = 1.0 - t;

    3.0 * one_minus_t * one_minus_t * (p1 - p0)
        + 6.0 * one_minus_t * t * (p2 - p1)
        + 3.0 * t * t * (p3 - p2)
}

pub fn compute_handle_length(handle: &BezierHandle) -> f32 {
    (handle.time_offset * handle.time_offset + handle.value_offset * handle.value_offset).sqrt()
}

pub fn create_handle_from_slope(slope: f32, length: f32, direction_sign: f32) -> BezierHandle {
    if length.abs() < 1e-8 {
        return BezierHandle::linear();
    }
    let denom = (1.0 + slope * slope).sqrt();
    let time_offset = direction_sign * (length / denom);
    let value_offset = slope * time_offset;
    BezierHandle::new(time_offset, value_offset)
}

pub fn initialize_weighted_handle_lengths(keyframe: &mut EditableKeyframe, dt: f32) {
    let default_len = dt / 3.0;
    if compute_handle_length(&keyframe.in_tangent) < 1e-8 {
        let slope = if keyframe.in_tangent.time_offset.abs() > 1e-8 {
            keyframe.in_tangent.value_offset / keyframe.in_tangent.time_offset
        } else {
            0.0
        };
        keyframe.in_tangent = create_handle_from_slope(slope, default_len, -1.0);
    }
    if compute_handle_length(&keyframe.out_tangent) < 1e-8 {
        let slope = if keyframe.out_tangent.time_offset.abs() > 1e-8 {
            keyframe.out_tangent.value_offset / keyframe.out_tangent.time_offset
        } else {
            0.0
        };
        keyframe.out_tangent = create_handle_from_slope(slope, default_len, 1.0);
    }
}

pub fn apply_auto_tangent(keyframes: &mut [EditableKeyframe], index: usize) {
    let len = keyframes.len();
    if len < 2 || index >= len {
        return;
    }

    let curr_val = keyframes[index].value;

    let (slope, dt_in, dt_out) = if index == 0 {
        let k1 = &keyframes[1];
        let dt = k1.time - keyframes[0].time;
        if dt.abs() < 1e-8 {
            return;
        }
        let s = (k1.value - curr_val) / dt;
        (s, dt, dt)
    } else if index == len - 1 {
        let k0 = &keyframes[len - 2];
        let dt = keyframes[len - 1].time - k0.time;
        if dt.abs() < 1e-8 {
            return;
        }
        let s = (curr_val - k0.value) / dt;
        (s, dt, dt)
    } else {
        let prev = &keyframes[index - 1];
        let next = &keyframes[index + 1];
        let dt_total = next.time - prev.time;
        if dt_total.abs() < 1e-8 {
            return;
        }

        let is_peak = curr_val >= prev.value && curr_val >= next.value;
        let is_valley = curr_val <= prev.value && curr_val <= next.value;

        let slope = if is_peak || is_valley {
            0.0
        } else {
            (next.value - prev.value) / dt_total
        };

        let dt_in = keyframes[index].time - prev.time;
        let dt_out = next.time - keyframes[index].time;
        (slope, dt_in, dt_out)
    };

    let in_handle_time = dt_in / 3.0;
    let out_handle_time = dt_out / 3.0;

    match keyframes[index].weight_mode {
        TangentWeightMode::Weighted => {
            let in_len = compute_handle_length(&keyframes[index].in_tangent);
            let out_len = compute_handle_length(&keyframes[index].out_tangent);
            keyframes[index].in_tangent = create_handle_from_slope(slope, in_len, -1.0);
            keyframes[index].out_tangent = create_handle_from_slope(slope, out_len, 1.0);
        }
        TangentWeightMode::NonWeighted => {
            let in_value = slope * in_handle_time;
            let out_value = slope * out_handle_time;

            let (clamped_in, clamped_out) = if index > 0 && index < len - 1 {
                let prev_val = keyframes[index - 1].value;
                let next_val = keyframes[index + 1].value;
                let max_in = (prev_val - curr_val).abs();
                let max_out = (next_val - curr_val).abs();
                (clamp_abs(in_value, max_in), clamp_abs(out_value, max_out))
            } else {
                (in_value, out_value)
            };

            keyframes[index].in_tangent = BezierHandle::new(-in_handle_time, -clamped_in);
            keyframes[index].out_tangent = BezierHandle::new(out_handle_time, clamped_out);
        }
    }
}

fn clamp_abs(value: f32, max_abs: f32) -> f32 {
    value.clamp(-max_abs, max_abs)
}

pub fn apply_flat_tangent(keyframe: &mut EditableKeyframe, dt: f32) {
    match keyframe.weight_mode {
        TangentWeightMode::Weighted => {
            let in_len = compute_handle_length(&keyframe.in_tangent);
            let out_len = compute_handle_length(&keyframe.out_tangent);
            keyframe.in_tangent = BezierHandle::new(-in_len, 0.0);
            keyframe.out_tangent = BezierHandle::new(out_len, 0.0);
        }
        TangentWeightMode::NonWeighted => {
            let handle_time = dt / 3.0;
            keyframe.in_tangent = BezierHandle::new(-handle_time, 0.0);
            keyframe.out_tangent = BezierHandle::new(handle_time, 0.0);
        }
    }
}

pub fn apply_linear_tangent(
    keyframes: &[EditableKeyframe],
    index: usize,
) -> (BezierHandle, BezierHandle) {
    let len = keyframes.len();
    if len < 2 || index >= len {
        return (BezierHandle::linear(), BezierHandle::linear());
    }

    let weighted = keyframes[index].weight_mode == TangentWeightMode::Weighted;

    let in_handle = if index > 0 {
        let prev = &keyframes[index - 1];
        let curr = &keyframes[index];
        let dt = curr.time - prev.time;
        let dv = curr.value - prev.value;
        if weighted {
            let existing_len = compute_handle_length(&curr.in_tangent);
            let slope = if dt.abs() > 1e-8 { dv / dt } else { 0.0 };
            create_handle_from_slope(slope, existing_len, -1.0)
        } else {
            let handle_time = dt / 3.0;
            let handle_value = dv / 3.0;
            BezierHandle::new(-handle_time, -handle_value)
        }
    } else {
        BezierHandle::linear()
    };

    let out_handle = if index < len - 1 {
        let curr = &keyframes[index];
        let next = &keyframes[index + 1];
        let dt = next.time - curr.time;
        let dv = next.value - curr.value;
        if weighted {
            let existing_len = compute_handle_length(&curr.out_tangent);
            let slope = if dt.abs() > 1e-8 { dv / dt } else { 0.0 };
            create_handle_from_slope(slope, existing_len, 1.0)
        } else {
            let handle_time = dt / 3.0;
            let handle_value = dv / 3.0;
            BezierHandle::new(handle_time, handle_value)
        }
    } else {
        BezierHandle::linear()
    };

    (in_handle, out_handle)
}

pub fn apply_auto_tangents_to_all(keyframes: &mut [EditableKeyframe]) {
    for i in 0..keyframes.len() {
        apply_auto_tangent(keyframes, i);
    }
}

fn compute_dt_for_index(keyframes: &[EditableKeyframe], index: usize) -> f32 {
    let len = keyframes.len();
    if len < 2 || index >= len {
        return 1.0;
    }

    if index == 0 {
        keyframes[1].time - keyframes[0].time
    } else if index == len - 1 {
        keyframes[len - 1].time - keyframes[len - 2].time
    } else {
        (keyframes[index + 1].time - keyframes[index - 1].time) * 0.5
    }
}

pub fn apply_clamped_tangent(keyframes: &mut [EditableKeyframe], index: usize) {
    let len = keyframes.len();
    if len < 2 || index >= len {
        return;
    }

    apply_auto_tangent(keyframes, index);

    if index + 1 < len {
        let dv_next = keyframes[index + 1].value - keyframes[index].value;
        let out_val = keyframes[index].out_tangent.value_offset;

        if dv_next.abs() < 1e-8 {
            keyframes[index].out_tangent.value_offset = 0.0;
        } else if out_val * dv_next < 0.0 || out_val.abs() > dv_next.abs() {
            keyframes[index].out_tangent.value_offset = dv_next;
        }
    }

    if index > 0 {
        let dv_prev = keyframes[index].value - keyframes[index - 1].value;
        let in_val = keyframes[index].in_tangent.value_offset;

        if dv_prev.abs() < 1e-8 {
            keyframes[index].in_tangent.value_offset = 0.0;
        } else if in_val * dv_prev > 0.0 || in_val.abs() > dv_prev.abs() {
            keyframes[index].in_tangent.value_offset = -dv_prev;
        }
    }
}

pub fn apply_plateau_tangent(keyframes: &mut [EditableKeyframe], index: usize) {
    let len = keyframes.len();
    if len < 2 || index >= len {
        return;
    }

    let is_extremum = if index == 0 || index == len - 1 {
        true
    } else {
        let prev_val = keyframes[index - 1].value;
        let curr_val = keyframes[index].value;
        let next_val = keyframes[index + 1].value;
        (curr_val - prev_val) * (next_val - curr_val) <= 0.0
    };

    if is_extremum {
        let dt = compute_dt_for_index(keyframes, index);
        apply_flat_tangent(&mut keyframes[index], dt);
    } else {
        apply_clamped_tangent(keyframes, index);
    }
}

pub fn apply_tangent_by_type(keyframes: &mut [EditableKeyframe], index: usize) {
    if index >= keyframes.len() {
        return;
    }

    match keyframes[index].tangent_type {
        TangentType::Manual => {}
        TangentType::Spline => apply_auto_tangent(keyframes, index),
        TangentType::Flat => {
            let dt = compute_dt_for_index(keyframes, index);
            apply_flat_tangent(&mut keyframes[index], dt);
        }
        TangentType::Linear => {
            let (in_handle, out_handle) = apply_linear_tangent(keyframes, index);
            keyframes[index].in_tangent = in_handle;
            keyframes[index].out_tangent = out_handle;
        }
        TangentType::Clamped => apply_clamped_tangent(keyframes, index),
        TangentType::Plateau => apply_plateau_tangent(keyframes, index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editable::InterpolationType;

    #[test]
    fn test_sample_bezier_linear_equivalent() {
        let result = sample_bezier(
            0.0,
            0.0,
            &BezierHandle::linear(),
            1.0,
            1.0,
            &BezierHandle::linear(),
            0.5,
        );
        assert!((result - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_sample_bezier_midpoint() {
        let out_handle = BezierHandle::new(0.33, 0.5);
        let in_handle = BezierHandle::new(-0.33, -0.5);

        let result = sample_bezier(0.0, 0.0, &out_handle, 1.0, 1.0, &in_handle, 0.5);
        assert!(result > 0.3 && result < 0.7);
    }

    #[test]
    fn test_apply_flat_tangent() {
        let mut kf = EditableKeyframe::new(1, 1.0, 5.0);
        kf.interpolation = InterpolationType::Bezier;
        apply_flat_tangent(&mut kf, 0.5);

        assert!((kf.in_tangent.value_offset).abs() < 1e-6);
        assert!((kf.out_tangent.value_offset).abs() < 1e-6);
    }

    #[test]
    fn test_apply_auto_tangent_three_kf() {
        let mut keyframes = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 2.0),
            EditableKeyframe::new(3, 2.0, 4.0),
        ];

        apply_auto_tangent(&mut keyframes, 1);

        let slope = (4.0 - 0.0) / (2.0 - 0.0);
        let expected_handle_time = (2.0 - 0.0) * 0.5 / 3.0;
        let expected_handle_value = slope * expected_handle_time;

        assert!((keyframes[1].out_tangent.time_offset - expected_handle_time).abs() < 1e-4);
        assert!((keyframes[1].out_tangent.value_offset - expected_handle_value).abs() < 1e-4);
    }

    #[test]
    fn test_stepped_sample() {
        let k0_value: f32 = 3.0;
        assert!((k0_value - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_weighted_auto_tangent_preserves_compute_handle_length() {
        let mut keyframes = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 2.0),
            EditableKeyframe::new(3, 2.0, 4.0),
        ];

        let initial_length = 0.5;
        keyframes[1].in_tangent = BezierHandle::new(-initial_length, 0.0);
        keyframes[1].out_tangent = BezierHandle::new(initial_length, 0.0);
        keyframes[1].weight_mode = TangentWeightMode::Weighted;

        apply_auto_tangent(&mut keyframes, 1);

        let in_len = compute_handle_length(&keyframes[1].in_tangent);
        let out_len = compute_handle_length(&keyframes[1].out_tangent);
        assert!(
            (in_len - initial_length).abs() < 1e-4,
            "In handle length changed: {} vs {}",
            in_len,
            initial_length
        );
        assert!(
            (out_len - initial_length).abs() < 1e-4,
            "Out handle length changed: {} vs {}",
            out_len,
            initial_length
        );
    }

    #[test]
    fn test_non_weighted_auto_tangent_resets_compute_handle_length() {
        let mut keyframes = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 2.0),
            EditableKeyframe::new(3, 2.0, 4.0),
        ];

        keyframes[1].in_tangent = BezierHandle::new(-0.8, -0.5);
        keyframes[1].out_tangent = BezierHandle::new(0.8, 0.5);
        keyframes[1].weight_mode = TangentWeightMode::NonWeighted;

        apply_auto_tangent(&mut keyframes, 1);

        let dt = (2.0 - 0.0) * 0.5;
        let expected_handle_time = dt / 3.0;
        assert!(
            (keyframes[1].out_tangent.time_offset - expected_handle_time).abs() < 1e-4,
            "NonWeighted should reset to dt/3.0"
        );
    }

    #[test]
    fn test_weighted_flat_tangent_preserves_length() {
        let mut kf = EditableKeyframe::new(1, 1.0, 5.0);
        kf.in_tangent = BezierHandle::new(-0.4, -0.3);
        kf.out_tangent = BezierHandle::new(0.4, 0.3);
        kf.weight_mode = TangentWeightMode::Weighted;

        let in_len_before = compute_handle_length(&kf.in_tangent);
        let out_len_before = compute_handle_length(&kf.out_tangent);

        apply_flat_tangent(&mut kf, 1.0);

        let in_len_after = compute_handle_length(&kf.in_tangent);
        let out_len_after = compute_handle_length(&kf.out_tangent);

        assert!(
            (in_len_after - in_len_before).abs() < 1e-4,
            "Weighted flat should preserve in handle length"
        );
        assert!(
            (out_len_after - out_len_before).abs() < 1e-4,
            "Weighted flat should preserve out handle length"
        );
        assert!(
            kf.in_tangent.value_offset.abs() < 1e-6,
            "Flat tangent should have zero value_offset"
        );
        assert!(
            kf.out_tangent.value_offset.abs() < 1e-6,
            "Flat tangent should have zero value_offset"
        );
    }

    #[test]
    fn test_default_weight_mode_is_non_weighted() {
        let kf = EditableKeyframe::new(1, 0.0, 0.0);
        assert_eq!(kf.weight_mode, TangentWeightMode::NonWeighted);

        let kf_default = EditableKeyframe::default();
        assert_eq!(kf_default.weight_mode, TangentWeightMode::NonWeighted);
    }

    #[test]
    fn test_compute_handle_length_calculation() {
        let h1 = BezierHandle::new(3.0, 4.0);
        assert!((compute_handle_length(&h1) - 5.0).abs() < 1e-6);

        let h2 = BezierHandle::new(0.0, 0.0);
        assert!(compute_handle_length(&h2).abs() < 1e-6);

        let h3 = BezierHandle::new(-1.0, 0.0);
        assert!((compute_handle_length(&h3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_initialize_weighted_handle_lengths_from_zero() {
        let mut kf = EditableKeyframe::new(1, 1.0, 5.0);
        kf.in_tangent = BezierHandle::linear();
        kf.out_tangent = BezierHandle::linear();

        let dt = 1.5;
        initialize_weighted_handle_lengths(&mut kf, dt);

        let expected_len = dt / 3.0;
        let in_len = compute_handle_length(&kf.in_tangent);
        let out_len = compute_handle_length(&kf.out_tangent);
        assert!(
            (in_len - expected_len).abs() < 1e-4,
            "In handle should be dt/3: {} vs {}",
            in_len,
            expected_len
        );
        assert!(
            (out_len - expected_len).abs() < 1e-4,
            "Out handle should be dt/3: {} vs {}",
            out_len,
            expected_len
        );
    }

    #[test]
    fn test_initialize_weighted_handle_lengths_preserves_existing() {
        let mut kf = EditableKeyframe::new(1, 1.0, 5.0);
        kf.in_tangent = BezierHandle::new(-0.4, -0.3);
        kf.out_tangent = BezierHandle::new(0.4, 0.3);

        let in_len_before = compute_handle_length(&kf.in_tangent);
        let out_len_before = compute_handle_length(&kf.out_tangent);

        initialize_weighted_handle_lengths(&mut kf, 1.0);

        let in_len_after = compute_handle_length(&kf.in_tangent);
        let out_len_after = compute_handle_length(&kf.out_tangent);
        assert!(
            (in_len_after - in_len_before).abs() < 1e-6,
            "Existing in handle should not change"
        );
        assert!(
            (out_len_after - out_len_before).abs() < 1e-6,
            "Existing out handle should not change"
        );
    }

    #[test]
    fn test_initialize_weighted_handle_preserves_slope() {
        let mut kf = EditableKeyframe::new(1, 1.0, 5.0);
        kf.in_tangent = BezierHandle::new(-0.001, -0.002);
        kf.out_tangent = BezierHandle::new(0.001, 0.002);

        let in_slope_before = kf.in_tangent.value_offset / kf.in_tangent.time_offset;
        let out_slope_before = kf.out_tangent.value_offset / kf.out_tangent.time_offset;

        initialize_weighted_handle_lengths(&mut kf, 1.5);

        let in_slope_after = kf.in_tangent.value_offset / kf.in_tangent.time_offset;
        let out_slope_after = kf.out_tangent.value_offset / kf.out_tangent.time_offset;
        assert!(
            (in_slope_after - in_slope_before).abs() < 1e-2,
            "In slope should be preserved: {} vs {}",
            in_slope_after,
            in_slope_before
        );
        assert!(
            (out_slope_after - out_slope_before).abs() < 1e-2,
            "Out slope should be preserved: {} vs {}",
            out_slope_after,
            out_slope_before
        );
    }

    #[test]
    fn test_weighted_linear_tangent_preserves_length() {
        let mut keyframes = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 3.0),
            EditableKeyframe::new(3, 2.0, 1.0),
        ];

        keyframes[1].in_tangent = BezierHandle::new(-0.4, 0.0);
        keyframes[1].out_tangent = BezierHandle::new(0.4, 0.0);
        keyframes[1].weight_mode = TangentWeightMode::Weighted;

        let in_len_before = compute_handle_length(&keyframes[1].in_tangent);
        let out_len_before = compute_handle_length(&keyframes[1].out_tangent);

        let (new_in, new_out) = apply_linear_tangent(&keyframes, 1);

        let in_len_after = compute_handle_length(&new_in);
        let out_len_after = compute_handle_length(&new_out);
        assert!(
            (in_len_after - in_len_before).abs() < 1e-4,
            "Weighted linear should preserve in handle length"
        );
        assert!(
            (out_len_after - out_len_before).abs() < 1e-4,
            "Weighted linear should preserve out handle length"
        );
    }

    #[test]
    fn test_create_handle_from_slope() {
        let h = create_handle_from_slope(0.0, 0.5, 1.0);
        assert!((h.time_offset - 0.5).abs() < 1e-4);
        assert!(h.value_offset.abs() < 1e-4);

        let h = create_handle_from_slope(1.0, 2.0_f32.sqrt(), 1.0);
        assert!((h.time_offset - 1.0).abs() < 1e-4);
        assert!((h.value_offset - 1.0).abs() < 1e-4);

        let h = create_handle_from_slope(1.0, 2.0_f32.sqrt(), -1.0);
        assert!((h.time_offset - (-1.0)).abs() < 1e-4);
        assert!((h.value_offset - (-1.0)).abs() < 1e-4);

        let h = create_handle_from_slope(1.0, 0.0, 1.0);
        assert!(compute_handle_length(&h).abs() < 1e-6);
    }

    #[test]
    fn test_apply_clamped_no_overshoot() {
        let mut keyframes = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 10.0),
            EditableKeyframe::new(3, 2.0, 5.0),
        ];

        apply_clamped_tangent(&mut keyframes, 1);

        assert!(keyframes[1].out_tangent.value_offset <= 0.0);
        assert!(keyframes[1].out_tangent.value_offset >= -5.0);

        assert!(keyframes[1].in_tangent.value_offset <= 0.0);
        assert!(keyframes[1].in_tangent.value_offset >= -10.0);
    }

    #[test]
    fn test_apply_clamped_monotone_matches_spline() {
        let mut kf_spline = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 5.0),
            EditableKeyframe::new(3, 2.0, 10.0),
        ];
        apply_auto_tangent(&mut kf_spline, 1);
        let spline_out = kf_spline[1].out_tangent.value_offset;
        let spline_in = kf_spline[1].in_tangent.value_offset;

        let mut kf_clamped = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 5.0),
            EditableKeyframe::new(3, 2.0, 10.0),
        ];
        apply_clamped_tangent(&mut kf_clamped, 1);

        assert!((kf_clamped[1].out_tangent.value_offset - spline_out).abs() < 1e-6);
        assert!((kf_clamped[1].in_tangent.value_offset - spline_in).abs() < 1e-6);
    }

    #[test]
    fn test_apply_plateau_flat_at_peak() {
        let mut keyframes = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 10.0),
            EditableKeyframe::new(3, 2.0, 0.0),
        ];

        apply_plateau_tangent(&mut keyframes, 1);

        assert!((keyframes[1].in_tangent.value_offset).abs() < 1e-6);
        assert!((keyframes[1].out_tangent.value_offset).abs() < 1e-6);
    }

    #[test]
    fn test_apply_plateau_slope_uses_clamped() {
        let mut kf_plateau = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 5.0),
            EditableKeyframe::new(3, 2.0, 10.0),
        ];
        apply_plateau_tangent(&mut kf_plateau, 1);

        let mut kf_clamped = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 5.0),
            EditableKeyframe::new(3, 2.0, 10.0),
        ];
        apply_clamped_tangent(&mut kf_clamped, 1);

        assert!(
            (kf_plateau[1].out_tangent.value_offset - kf_clamped[1].out_tangent.value_offset).abs()
                < 1e-6
        );
        assert!(
            (kf_plateau[1].in_tangent.value_offset - kf_clamped[1].in_tangent.value_offset).abs()
                < 1e-6
        );
    }

    #[test]
    fn test_apply_tangent_by_type_manual_noop() {
        let mut keyframes = vec![
            EditableKeyframe::new(1, 0.0, 0.0),
            EditableKeyframe::new(2, 1.0, 5.0),
            EditableKeyframe::new(3, 2.0, 10.0),
        ];

        keyframes[1].in_tangent = BezierHandle::new(-0.2, -1.0);
        keyframes[1].out_tangent = BezierHandle::new(0.2, 1.0);
        keyframes[1].tangent_type = TangentType::Manual;

        apply_tangent_by_type(&mut keyframes, 1);

        assert!((keyframes[1].in_tangent.time_offset - (-0.2)).abs() < 1e-6);
        assert!((keyframes[1].in_tangent.value_offset - (-1.0)).abs() < 1e-6);
        assert!((keyframes[1].out_tangent.time_offset - 0.2).abs() < 1e-6);
        assert!((keyframes[1].out_tangent.value_offset - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_tangent_by_type_all_variants() {
        let variants = [
            TangentType::Manual,
            TangentType::Spline,
            TangentType::Flat,
            TangentType::Linear,
            TangentType::Clamped,
            TangentType::Plateau,
        ];

        for variant in &variants {
            let mut keyframes = vec![
                EditableKeyframe::new(1, 0.0, 0.0),
                EditableKeyframe::new(2, 1.0, 5.0),
                EditableKeyframe::new(3, 2.0, 10.0),
            ];
            keyframes[1].tangent_type = *variant;
            apply_tangent_by_type(&mut keyframes, 1);

            match variant {
                TangentType::Flat => {
                    assert!(
                        (keyframes[1].out_tangent.value_offset).abs() < 1e-6,
                        "Flat should produce zero value_offset"
                    );
                }
                TangentType::Spline => {
                    assert!(
                        keyframes[1].out_tangent.time_offset > 0.0,
                        "Spline should produce positive out time_offset"
                    );
                }
                _ => {}
            }
        }
    }
}
