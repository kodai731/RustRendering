use crate::editable::components::curve::PropertyCurve;
use crate::editable::components::keyframe::{EditableKeyframe, InterpolationType};

pub fn collect_bake_times(curves: &[&PropertyCurve]) -> Vec<f32> {
    let has_bezier = curves.iter().any(|c| c.has_bezier_keyframes());

    if !has_bezier {
        return collect_unique_times(curves);
    }

    let mut times: Vec<f32> = Vec::new();

    for curve in curves {
        for kf in &curve.keyframes {
            times.push(kf.time);
        }

        for i in 0..curve.keyframes.len().saturating_sub(1) {
            let k0 = &curve.keyframes[i];
            let k1 = &curve.keyframes[i + 1];

            if k0.interpolation == InterpolationType::Bezier {
                let subdivisions = compute_bezier_subdivisions(k0, k1);
                for s in 1..subdivisions {
                    let frac = s as f32 / subdivisions as f32;
                    let mid_time = k0.time + (k1.time - k0.time) * frac;
                    times.push(mid_time);
                }
            }
        }
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 0.0001);
    times
}

fn collect_unique_times(curves: &[&PropertyCurve]) -> Vec<f32> {
    let mut times: Vec<f32> = curves
        .iter()
        .flat_map(|c| c.keyframes.iter().map(|k| k.time))
        .collect();

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    times.dedup_by(|a, b| (*a - *b).abs() < 0.0001);
    times
}

fn estimate_bezier_curvature(k0: &EditableKeyframe, k1: &EditableKeyframe) -> f32 {
    let dt = k1.time - k0.time;
    if dt.abs() < 1e-8 {
        return 0.0;
    }

    let linear_slope = (k1.value - k0.value) / dt;
    let out_deviation =
        (k0.out_tangent.value_offset - linear_slope * k0.out_tangent.time_offset).abs();
    let in_deviation =
        (k1.in_tangent.value_offset - linear_slope * k1.in_tangent.time_offset).abs();

    let value_range = (k1.value - k0.value).abs().max(1.0);
    (out_deviation + in_deviation) / value_range
}

fn compute_bezier_subdivisions(k0: &EditableKeyframe, k1: &EditableKeyframe) -> usize {
    let duration = k1.time - k0.time;
    let fps_based = (duration * 30.0).ceil() as usize;

    let curvature = estimate_bezier_curvature(k0, k1);
    let curvature_multiplier = 1.0 + curvature * 2.0;

    let subdivisions = (fps_based as f32 * curvature_multiplier).ceil() as usize;
    subdivisions.clamp(2, 64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editable::{BezierHandle, EditableKeyframe, InterpolationType};

    fn make_bezier_kf(
        time: f32,
        value: f32,
        out_t: f32,
        out_v: f32,
        in_t: f32,
        in_v: f32,
    ) -> EditableKeyframe {
        let mut kf = EditableKeyframe::new(0, time, value);
        kf.interpolation = InterpolationType::Bezier;
        kf.out_tangent = BezierHandle::new(out_t, out_v);
        kf.in_tangent = BezierHandle::new(in_t, in_v);
        kf
    }

    #[test]
    fn test_short_segment_few_subdivisions() {
        let k0 = make_bezier_kf(0.0, 0.0, 0.01, 0.0, 0.0, 0.0);
        let k1 = make_bezier_kf(0.05, 1.0, 0.0, 0.0, -0.01, 0.0);
        let subs = compute_bezier_subdivisions(&k0, &k1);
        assert!(subs >= 2 && subs <= 10, "short segment: got {}", subs);
    }

    #[test]
    fn test_long_segment_many_subdivisions() {
        let k0 = make_bezier_kf(0.0, 0.0, 0.3, 0.0, 0.0, 0.0);
        let k1 = make_bezier_kf(3.0, 10.0, 0.0, 0.0, -0.3, 0.0);
        let subs = compute_bezier_subdivisions(&k0, &k1);
        assert!(subs >= 60, "long segment: got {}", subs);
    }

    #[test]
    fn test_linear_bezier_minimal_subdivisions() {
        let k0 = make_bezier_kf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let k1 = make_bezier_kf(1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let subs = compute_bezier_subdivisions(&k0, &k1);
        assert!(subs <= 31, "linear bezier: got {}", subs);
    }

    #[test]
    fn test_s_curve_high_subdivisions() {
        let k0 = make_bezier_kf(0.0, 0.0, 0.3, 5.0, 0.0, 0.0);
        let k1 = make_bezier_kf(1.0, 1.0, 0.0, 0.0, -0.3, -5.0);
        let subs = compute_bezier_subdivisions(&k0, &k1);
        assert!(subs > 30, "s-curve: got {}", subs);
    }

    #[test]
    fn test_curvature_zero_for_linear_handles() {
        let k0 = make_bezier_kf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let k1 = make_bezier_kf(1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let curvature = estimate_bezier_curvature(&k0, &k1);
        assert!(curvature.abs() < 0.01, "curvature: {}", curvature);
    }
}
