pub const FEATURES_PER_KEYFRAME: usize = 6;

pub struct ContextWindow {
    pub flat: Vec<f32>,
    pub curve_mean: f32,
    pub curve_std: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn flatten_context(
    times: &[f32],
    values: &[f32],
    in_tangent_dt: &[f32],
    in_tangent_dv: &[f32],
    out_tangent_dt: &[f32],
    out_tangent_dv: &[f32],
    max_keyframes: usize,
    clip_duration: f32,
) -> ContextWindow {
    debug_assert_eq!(times.len(), values.len());
    debug_assert_eq!(times.len(), in_tangent_dt.len());
    debug_assert_eq!(times.len(), in_tangent_dv.len());
    debug_assert_eq!(times.len(), out_tangent_dt.len());
    debug_assert_eq!(times.len(), out_tangent_dv.len());

    let total = times.len();
    let count = total.min(max_keyframes);
    let start = total.saturating_sub(max_keyframes);

    let curve_mean = if count > 0 {
        values[start..start + count].iter().sum::<f32>() / count as f32
    } else {
        0.0
    };

    let curve_std = if count > 0 {
        let variance = values[start..start + count]
            .iter()
            .map(|&v| (v - curve_mean).powi(2))
            .sum::<f32>()
            / count as f32;
        variance.sqrt().max(1e-6)
    } else {
        1e-6
    };

    let total_size = max_keyframes * FEATURES_PER_KEYFRAME;
    let mut flat = vec![0.0_f32; total_size];
    let duration = clip_duration.max(0.001);
    let padding_offset = (max_keyframes - count) * FEATURES_PER_KEYFRAME;

    for i in 0..count {
        let src = start + i;
        let dst = padding_offset + i * FEATURES_PER_KEYFRAME;
        flat[dst] = times[src] / duration;
        flat[dst + 1] = (values[src] - curve_mean) / curve_std;
        flat[dst + 2] = in_tangent_dt[src] / duration;
        flat[dst + 3] = in_tangent_dv[src] / curve_std;
        flat[dst + 4] = out_tangent_dt[src] / duration;
        flat[dst + 5] = out_tangent_dv[src] / curve_std;
    }

    ContextWindow {
        flat,
        curve_mean,
        curve_std,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyframe_setup(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let times: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * 0.5).collect();
        let values: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let zero = vec![0.0_f32; n];
        (
            times,
            values,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        )
    }

    #[test]
    fn output_size_is_fixed() {
        let (t, v, idt, idv, odt, odv) = keyframe_setup(5);
        let ctx = flatten_context(&t, &v, &idt, &idv, &odt, &odv, 8, 4.0);
        assert_eq!(ctx.flat.len(), 8 * 6);
    }

    #[test]
    fn padding_is_left_aligned_zeros() {
        let (t, v, idt, idv, odt, odv) = keyframe_setup(2);
        let ctx = flatten_context(&t, &v, &idt, &idv, &odt, &odv, 8, 4.0);
        let padding = (8 - 2) * 6;
        for i in 0..padding {
            assert_eq!(ctx.flat[i], 0.0);
        }
    }

    #[test]
    fn normalization_is_correct() {
        let times = vec![1.0_f32, 2.0];
        let values = vec![90.0_f32, 180.0];
        let zero = vec![0.0_f32; 2];
        let ctx = flatten_context(&times, &values, &zero, &zero, &zero, &zero, 8, 4.0);

        assert!((ctx.curve_mean - 135.0).abs() < 0.001);
        assert!((ctx.curve_std - 45.0).abs() < 0.001);

        let padding = (8 - 2) * 6;
        assert!((ctx.flat[padding] - 0.25).abs() < 0.001);
        assert!((ctx.flat[padding + 1] - (-1.0)).abs() < 0.001);
        assert!((ctx.flat[padding + 6 + 1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn constant_curve_clamps_std() {
        let times = vec![0.0_f32, 1.0, 2.0];
        let values = vec![42.0_f32, 42.0, 42.0];
        let zero = vec![0.0_f32; 3];
        let ctx = flatten_context(&times, &values, &zero, &zero, &zero, &zero, 8, 4.0);

        assert!((ctx.curve_mean - 42.0).abs() < 0.001);
        assert!((ctx.curve_std - 1e-6).abs() < 1e-7);
    }

    #[test]
    fn empty_input_returns_zero_filled() {
        let empty: &[f32] = &[];
        let ctx = flatten_context(empty, empty, empty, empty, empty, empty, 8, 4.0);
        assert_eq!(ctx.flat.len(), 48);
        for v in &ctx.flat {
            assert_eq!(*v, 0.0);
        }
    }
}
