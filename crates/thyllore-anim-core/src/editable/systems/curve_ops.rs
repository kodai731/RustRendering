use crate::editable::components::clip::EditableAnimationClip;
use crate::editable::components::curve::{PropertyCurve, PropertyType};
use crate::editable::components::keyframe::{
    BezierHandle, EditableKeyframe, InterpolationType, KeyframeId,
};
use crate::editable::systems::tangent::{apply_auto_tangent, sample_bezier};
use crate::BoneId;

pub fn curve_add_keyframe(curve: &mut PropertyCurve, time: f32, value: f32) -> KeyframeId {
    let id = curve.allocate_keyframe_id();
    let keyframe = EditableKeyframe::new(id, time, value);
    curve.keyframes.push(keyframe);
    curve_sort_keyframes(curve);
    id
}

pub fn curve_add_keyframe_with_tangents(
    curve: &mut PropertyCurve,
    time: f32,
    value: f32,
    in_tangent: BezierHandle,
    out_tangent: BezierHandle,
    interpolation: InterpolationType,
) -> KeyframeId {
    let id = curve.allocate_keyframe_id();
    let mut keyframe = EditableKeyframe::with_tangents(id, time, value, in_tangent, out_tangent);
    keyframe.interpolation = interpolation;
    curve.keyframes.push(keyframe);
    curve_sort_keyframes(curve);
    id
}

pub fn curve_remove_keyframe(curve: &mut PropertyCurve, keyframe_id: KeyframeId) -> bool {
    if let Some(pos) = curve.keyframes.iter().position(|k| k.id == keyframe_id) {
        curve.keyframes.remove(pos);
        true
    } else {
        false
    }
}

pub fn curve_set_keyframe_time(curve: &mut PropertyCurve, keyframe_id: KeyframeId, time: f32) {
    if let Some(kf) = curve.get_keyframe_mut(keyframe_id) {
        kf.time = time;
    }
    curve_sort_keyframes(curve);
}

pub fn curve_sort_keyframes(curve: &mut PropertyCurve) {
    curve.keyframes.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

pub fn curve_resample_at_fps(curve: &mut PropertyCurve, duration: f32, fps: f32) -> usize {
    if curve.keyframes.is_empty() || fps <= 0.0 || duration <= 0.0 {
        return curve.keyframes.len();
    }

    let frame_interval = 1.0 / fps;
    let last_frame = (duration * fps).round().max(1.0) as usize;
    let samples: Vec<(f32, f32)> = (0..=last_frame)
        .map(|frame| {
            let time = (frame as f32 * frame_interval).min(duration);
            (time, curve_sample(curve, time).unwrap_or(0.0))
        })
        .collect();

    curve.keyframes.clear();
    for (time, value) in samples {
        let id = curve.allocate_keyframe_id();
        curve.keyframes.push(EditableKeyframe::new(id, time, value));
    }
    curve_sort_keyframes(curve);
    curve.keyframes.len()
}

pub fn curve_sample(curve: &PropertyCurve, time: f32) -> Option<f32> {
    keyframes_sample(&curve.keyframes, time)
}

/// Sample a list of keyframes at `time`. Returns `None` if keyframes are empty.
/// Clamps `time` to the range of first and last keyframes.
/// Uses binary search (partition_point) to find the interval, then interpolates
/// linearly (or bezier if either endpoint is Bezier). Stepped holds the previous value.
pub fn keyframes_sample(keyframes: &[EditableKeyframe], time: f32) -> Option<f32> {
    if keyframes.is_empty() {
        return None;
    }

    if keyframes.len() == 1 {
        return Some(keyframes[0].value);
    }

    if time <= keyframes[0].time {
        return Some(keyframes[0].value);
    }

    if let Some(last) = keyframes.last() {
        if time >= last.time {
            return Some(last.value);
        }
    }

    let idx = keyframes.partition_point(|kf| kf.time <= time);
    let i = if idx == 0 {
        0
    } else {
        (idx - 1).min(keyframes.len().saturating_sub(2))
    };

    let k0 = &keyframes[i];
    let k1 = &keyframes[i + 1];

    if k0.interpolation == InterpolationType::Stepped {
        return Some(k0.value);
    }

    let either_bezier = k0.interpolation == InterpolationType::Bezier
        || k1.interpolation == InterpolationType::Bezier;

    Some(if either_bezier {
        let dt = k1.time - k0.time;
        let dv = k1.value - k0.value;

        let out_h = if k0.interpolation == InterpolationType::Bezier {
            k0.out_tangent.clone()
        } else {
            BezierHandle::new(dt / 3.0, dv / 3.0)
        };
        let in_h = if k1.interpolation == InterpolationType::Bezier {
            k1.in_tangent.clone()
        } else {
            BezierHandle::new(-dt / 3.0, -dv / 3.0)
        };

        sample_bezier(k0.time, k0.value, &out_h, k1.time, k1.value, &in_h, time)
    } else {
        let t = (time - k0.time) / (k1.time - k0.time);
        k0.value + (k1.value - k0.value) * t
    })
}

pub fn segment_uses_bezier(k0: &EditableKeyframe, k1: &EditableKeyframe) -> bool {
    k0.interpolation == InterpolationType::Bezier || k1.interpolation == InterpolationType::Bezier
}

pub fn curve_recalculate_auto_tangents(curve: &mut PropertyCurve) {
    for i in 0..curve.keyframes.len() {
        apply_auto_tangent(&mut curve.keyframes, i);
    }
}

pub fn clip_add_keyframe(
    clip: &mut EditableAnimationClip,
    bone_id: BoneId,
    property_type: PropertyType,
    time: f32,
    value: f32,
) -> Option<KeyframeId> {
    clip.get_track_mut(bone_id)
        .map(|track| curve_add_keyframe(track.get_curve_mut(property_type), time, value))
}

pub fn curve_recalculate_auto_tangent_at(curve: &mut PropertyCurve, keyframe_id: KeyframeId) {
    if let Some(idx) = curve.keyframes.iter().position(|k| k.id == keyframe_id) {
        if idx > 0 {
            apply_auto_tangent(&mut curve.keyframes, idx - 1);
        }
        apply_auto_tangent(&mut curve.keyframes, idx);
        if idx + 1 < curve.keyframes.len() {
            apply_auto_tangent(&mut curve.keyframes, idx + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editable::compute_handle_length;
    use crate::editable::TangentWeightMode;

    fn make_curve_with_keyframes(times_values: &[(f32, f32)]) -> PropertyCurve {
        let mut curve = PropertyCurve::new(1, PropertyType::TranslationX);
        for &(time, value) in times_values {
            curve_add_keyframe(&mut curve, time, value);
        }
        curve
    }

    #[test]
    fn test_curve_sample_linear() {
        let mut curve = make_curve_with_keyframes(&[(0.0, 0.0), (1.0, 10.0)]);
        for kf in &mut curve.keyframes {
            kf.interpolation = InterpolationType::Linear;
        }

        assert!((curve_sample(&curve, 0.5).unwrap() - 5.0).abs() < 1e-4);
        assert!((curve_sample(&curve, 0.0).unwrap() - 0.0).abs() < 1e-4);
        assert!((curve_sample(&curve, 1.0).unwrap() - 10.0).abs() < 1e-4);
    }

    #[test]
    fn test_curve_sample_stepped() {
        let mut curve = make_curve_with_keyframes(&[(0.0, 0.0), (1.0, 10.0)]);
        curve.keyframes[0].interpolation = InterpolationType::Stepped;

        assert!((curve_sample(&curve, 0.5).unwrap() - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_curve_sample_single_keyframe() {
        let curve = make_curve_with_keyframes(&[(0.5, 7.0)]);
        assert!((curve_sample(&curve, 0.0).unwrap() - 7.0).abs() < 1e-4);
        assert!((curve_sample(&curve, 1.0).unwrap() - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_curve_sample_empty() {
        let curve = PropertyCurve::new(1, PropertyType::TranslationX);
        assert!(curve_sample(&curve, 0.5).is_none());
    }

    #[test]
    fn test_curve_add_and_remove_keyframe() {
        let mut curve = PropertyCurve::new(1, PropertyType::TranslationX);
        let id1 = curve_add_keyframe(&mut curve, 0.0, 1.0);
        let id2 = curve_add_keyframe(&mut curve, 1.0, 2.0);
        assert_eq!(curve.keyframe_count(), 2);

        assert!(curve_remove_keyframe(&mut curve, id1));
        assert_eq!(curve.keyframe_count(), 1);
        assert_eq!(curve.keyframes[0].id, id2);
    }

    #[test]
    fn test_curve_sort_after_add() {
        let mut curve = PropertyCurve::new(1, PropertyType::TranslationX);
        curve_add_keyframe(&mut curve, 2.0, 20.0);
        curve_add_keyframe(&mut curve, 0.0, 0.0);
        curve_add_keyframe(&mut curve, 1.0, 10.0);

        assert!((curve.keyframes[0].time - 0.0).abs() < 1e-6);
        assert!((curve.keyframes[1].time - 1.0).abs() < 1e-6);
        assert!((curve.keyframes[2].time - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_set_keyframe_weight_mode() {
        let mut curve = make_curve_with_keyframes(&[(0.0, 0.0), (1.0, 10.0)]);
        let kf_id = curve.keyframes[0].id;

        assert_eq!(
            curve.keyframes[0].weight_mode,
            TangentWeightMode::NonWeighted
        );

        curve.set_keyframe_weight_mode(kf_id, TangentWeightMode::Weighted);
        assert_eq!(curve.keyframes[0].weight_mode, TangentWeightMode::Weighted);

        curve.set_keyframe_weight_mode(kf_id, TangentWeightMode::NonWeighted);
        assert_eq!(
            curve.keyframes[0].weight_mode,
            TangentWeightMode::NonWeighted
        );
    }

    #[test]
    fn test_weighted_bezier_custom_handles_change_curve_shape() {
        let mut curve_a = make_curve_with_keyframes(&[(0.0, 0.0), (2.0, 10.0)]);
        let mut curve_b = make_curve_with_keyframes(&[(0.0, 0.0), (2.0, 10.0)]);

        for kf in &mut curve_a.keyframes {
            kf.interpolation = InterpolationType::Bezier;
        }
        for kf in &mut curve_b.keyframes {
            kf.interpolation = InterpolationType::Bezier;
        }

        curve_a.keyframes[0].out_tangent = BezierHandle::new(0.66, 0.0);
        curve_a.keyframes[1].in_tangent = BezierHandle::new(-0.66, 0.0);

        curve_b.keyframes[0].out_tangent = BezierHandle::new(0.66, 8.0);
        curve_b.keyframes[1].in_tangent = BezierHandle::new(-0.66, -8.0);

        let val_a = curve_sample(&curve_a, 0.5).unwrap();
        let val_b = curve_sample(&curve_b, 0.5).unwrap();

        assert!(
            (val_a - val_b).abs() > 0.5,
            "Different tangent handles should produce different curves: a={}, b={}",
            val_a,
            val_b
        );
    }

    #[test]
    fn test_auto_tangent_recalculate_at_specific_keyframe() {
        let mut curve = make_curve_with_keyframes(&[(0.0, 0.0), (1.0, 5.0), (2.0, 0.0)]);
        for kf in &mut curve.keyframes {
            kf.interpolation = InterpolationType::Bezier;
        }

        let mid_id = curve.keyframes[1].id;
        curve_recalculate_auto_tangent_at(&mut curve, mid_id);

        let mid = &curve.keyframes[1];
        assert!(
            mid.in_tangent.time_offset < 0.0,
            "In tangent should point left"
        );
        assert!(
            mid.out_tangent.time_offset > 0.0,
            "Out tangent should point right"
        );
    }

    #[test]
    fn test_recalculate_auto_tangent_weighted_preserves_length() {
        let mut curve = make_curve_with_keyframes(&[(0.0, 0.0), (1.0, 5.0), (2.0, 10.0)]);
        for kf in &mut curve.keyframes {
            kf.interpolation = InterpolationType::Bezier;
        }

        curve.keyframes[1].weight_mode = TangentWeightMode::Weighted;
        curve.keyframes[1].in_tangent = BezierHandle::new(-0.3, 0.0);
        curve.keyframes[1].out_tangent = BezierHandle::new(0.3, 0.0);

        let in_len_before = compute_handle_length(&curve.keyframes[1].in_tangent);
        let out_len_before = compute_handle_length(&curve.keyframes[1].out_tangent);

        let mid_id = curve.keyframes[1].id;
        curve_recalculate_auto_tangent_at(&mut curve, mid_id);

        let in_len_after = compute_handle_length(&curve.keyframes[1].in_tangent);
        let out_len_after = compute_handle_length(&curve.keyframes[1].out_tangent);

        assert!(
            (in_len_after - in_len_before).abs() < 1e-4,
            "Weighted auto tangent should preserve in handle length"
        );
        assert!(
            (out_len_after - out_len_before).abs() < 1e-4,
            "Weighted auto tangent should preserve out handle length"
        );
    }
}
