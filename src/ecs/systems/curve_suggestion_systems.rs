use crate::animation::editable::{
    curve_add_keyframe_with_tangents, curve_sample, BezierHandle, EditableAnimationClip,
    EditableKeyframe, InterpolationType, PropertyCurve, PropertyType,
};
use crate::animation::BoneId;
use crate::ecs::resource::{
    BoneNameTokenCache, BoneRestPositionCache, BoneTopologyCache, CurveSuggestionPendingDump,
    CurveSuggestionState, GhostCurveSuggestion, InferenceActorState,
};
use crate::ml::{InferenceActorId, InferenceRequestKind, InferenceResultKind};
use thyllore_ml_core::copilot::context::flatten_context;
use thyllore_ml_core::copilot::dump::{dump_curve_copilot_inference, CurveCopilotInferenceDump};
use thyllore_ml_core::copilot::property::{property_kind_to_id, PropertyKind};
use thyllore_ml_core::copilot::query::generate_query_times;
use thyllore_ml_core::copilot::window::sample_window;
use thyllore_model_core::Skeleton;

use super::curve_suggestion_bone_context::{build_bone_context_arrays, BoneContextRequest};
use super::inference_actor_systems::{inference_actor_submit, inference_actor_take_results};

const MAX_CONTEXT_KEYFRAMES: usize = 8;
const MIN_CURVE_STD: f32 = 0.01;
const ANCHOR_EPSILON: f32 = 1.0e-6;

struct FlatKeyframes {
    times: Vec<f32>,
    values: Vec<f32>,
    in_dt: Vec<f32>,
    in_dv: Vec<f32>,
    out_dt: Vec<f32>,
    out_dv: Vec<f32>,
}

fn flatten_keyframes_before_anchor(curve: &PropertyCurve, anchor_time: f32) -> FlatKeyframes {
    let mut sorted: Vec<&crate::animation::editable::EditableKeyframe> = curve
        .keyframes
        .iter()
        .filter(|kf| kf.time <= anchor_time + ANCHOR_EPSILON)
        .collect();
    sorted.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut flat = FlatKeyframes {
        times: Vec::with_capacity(sorted.len()),
        values: Vec::with_capacity(sorted.len()),
        in_dt: Vec::with_capacity(sorted.len()),
        in_dv: Vec::with_capacity(sorted.len()),
        out_dt: Vec::with_capacity(sorted.len()),
        out_dv: Vec::with_capacity(sorted.len()),
    };
    for kf in sorted {
        flat.times.push(kf.time);
        flat.values.push(kf.value);
        flat.in_dt.push(kf.in_tangent.time_offset);
        flat.in_dv.push(kf.in_tangent.value_offset);
        flat.out_dt.push(kf.out_tangent.time_offset);
        flat.out_dv.push(kf.out_tangent.value_offset);
    }
    flat
}

fn resolve_anchor_time(curve: &PropertyCurve, current_time: f32) -> Option<f32> {
    curve
        .keyframes
        .iter()
        .map(|kf| kf.time)
        .filter(|t| *t <= current_time + ANCHOR_EPSILON)
        .fold(None, |acc, t| Some(acc.map_or(t, |best: f32| best.max(t))))
}

fn property_type_to_kind(pt: PropertyType) -> PropertyKind {
    match pt {
        PropertyType::TranslationX => PropertyKind::TranslationX,
        PropertyType::TranslationY => PropertyKind::TranslationY,
        PropertyType::TranslationZ => PropertyKind::TranslationZ,
        PropertyType::RotationX => PropertyKind::RotationX,
        PropertyType::RotationY => PropertyKind::RotationY,
        PropertyType::RotationZ => PropertyKind::RotationZ,
        PropertyType::ScaleX => PropertyKind::ScaleX,
        PropertyType::ScaleY => PropertyKind::ScaleY,
        PropertyType::ScaleZ => PropertyKind::ScaleZ,
    }
}

pub struct CurveSuggestionInputs<'a> {
    pub clip: &'a EditableAnimationClip,
    pub skeleton: &'a Skeleton,
    pub topology_cache: &'a BoneTopologyCache,
    pub name_token_cache: &'a BoneNameTokenCache,
    pub rest_position_cache: &'a BoneRestPositionCache,
}

pub fn curve_suggestion_submit(
    suggestion_state: &mut CurveSuggestionState,
    inference_state: &mut InferenceActorState,
    actor_id: InferenceActorId,
    inputs: CurveSuggestionInputs<'_>,
    property_type: PropertyType,
    bone_id: BoneId,
    current_time: f32,
) {
    if !suggestion_state.enabled {
        return;
    }

    log!(
        "CurveCopilot triggered: bone_id={} property={:?} current_time={:.4}",
        bone_id,
        property_type,
        current_time,
    );

    let Some(max_steps) = inference_state.actor_max_steps(actor_id) else {
        log_warn!(
            "curve_suggestion_submit: actor {} has no max_steps metadata; skipping request",
            actor_id
        );
        return;
    };

    let Some(track) = inputs.clip.get_track(bone_id) else {
        return;
    };
    let curve = track.get_curve(property_type);
    let clip_duration = inputs.clip.duration;

    let Some(anchor_time) = resolve_anchor_time(curve, current_time) else {
        return;
    };

    let flat = flatten_keyframes_before_anchor(curve, anchor_time);
    let context = flatten_context(
        &flat.times,
        &flat.values,
        &flat.in_dt,
        &flat.in_dv,
        &flat.out_dt,
        &flat.out_dv,
        MAX_CONTEXT_KEYFRAMES,
        clip_duration,
    );

    if context.curve_std < MIN_CURVE_STD {
        return;
    }

    let property_type_id = property_kind_to_id(property_type_to_kind(property_type));
    let topology_features = inputs.topology_cache.get(bone_id).to_vec();
    let bone_name_tokens = inputs.name_token_cache.get(bone_id).to_vec();

    let all_times: Vec<f32> = curve.keyframes.iter().map(|kf| kf.time).collect();
    let query_times = generate_query_times(&all_times, anchor_time, clip_duration, max_steps);

    let context_window_kfs = flat.times.len().min(MAX_CONTEXT_KEYFRAMES);
    let context_start_index = flat.times.len().saturating_sub(context_window_kfs);
    let context_start_time = flat
        .times
        .get(context_start_index)
        .copied()
        .unwrap_or(anchor_time);
    let curve_window = sample_window(
        &flat.times,
        &flat.values,
        context_start_time,
        anchor_time,
        context.curve_mean,
        context.curve_std,
    );

    let bone_context = build_bone_context_arrays(BoneContextRequest {
        clip: inputs.clip,
        skeleton: inputs.skeleton,
        topology_cache: inputs.topology_cache,
        rest_position_cache: inputs.rest_position_cache,
        property_type,
        anchor_time,
        clip_duration,
    });

    let denorm_query_times: Vec<f32> = query_times
        .iter()
        .map(|t| t * clip_duration.max(0.001))
        .collect();

    let bone_name = inputs
        .skeleton
        .bones
        .get(bone_id as usize)
        .map(|b| b.name.as_str())
        .unwrap_or("<unknown>");
    let token_tail: Vec<i64> = bone_name_tokens
        .iter()
        .skip_while(|&&t| t == 0)
        .copied()
        .collect();
    let mask_true_count = bone_context.mask.iter().filter(|&&m| m).count();
    log!(
        "CurveCopilot input: bone='{}' tokens_nonpad={:?} mask_true={}/{} \
         clip_dur={:.3} curve_mean={:.4} curve_std={:.4} anchor={:.4} property={:?}",
        bone_name,
        token_tail,
        mask_true_count,
        bone_context.mask.len(),
        clip_duration,
        context.curve_mean,
        context.curve_std,
        anchor_time,
        property_type,
    );

    let real_curve_samples: Vec<Option<f32>> = denorm_query_times
        .iter()
        .map(|t| curve_sample(curve, *t))
        .collect();
    let nearest_keyframes: Vec<Option<(f32, f32)>> = denorm_query_times
        .iter()
        .map(|t| find_nearest_keyframe(curve, *t).map(|kf| (kf.time, kf.value)))
        .collect();

    let dump_snapshot = if suggestion_state.dump_inference {
        Some(CurveSuggestionPendingDump {
            context_keyframes: context.flat.clone(),
            property_type_id,
            topology_features: topology_features.clone(),
            bone_name_tokens: bone_name_tokens.clone(),
            query_times_normalized: query_times.clone(),
            curve_window: curve_window.to_vec(),
            bone_context_keyframes: bone_context.keyframes.clone(),
            bone_context_topology: bone_context.topology.clone(),
            bone_context_rest_positions: bone_context.rest_positions.clone(),
            bone_context_mask: bone_context.mask.clone(),
        })
    } else {
        None
    };

    let kind = InferenceRequestKind::CurveCopilotPredict {
        context: context.flat,
        property_type_id,
        topology_features,
        bone_name_tokens,
        query_times,
        curve_window: curve_window.to_vec(),
        bone_context_keyframes: bone_context.keyframes,
        bone_context_topology: bone_context.topology,
        bone_context_rest_positions: bone_context.rest_positions,
        bone_context_mask: bone_context.mask,
    };

    if let Some(request_id) = inference_actor_submit(inference_state, actor_id, kind) {
        suggestion_state.pending_request_id = Some(request_id);
        suggestion_state.pending_bone_id = Some(bone_id);
        suggestion_state.pending_property_type = Some(property_type);
        suggestion_state.pending_clip_duration = Some(clip_duration);
        suggestion_state.pending_curve_mean = Some(context.curve_mean);
        suggestion_state.pending_curve_std = Some(context.curve_std);
        suggestion_state.pending_query_times = Some(denorm_query_times);
        suggestion_state.pending_real_curve_samples = Some(real_curve_samples);
        suggestion_state.pending_nearest_keyframes = Some(nearest_keyframes);
        suggestion_state.pending_dump = dump_snapshot;
    }
}

fn find_nearest_keyframe(curve: &PropertyCurve, time: f32) -> Option<&EditableKeyframe> {
    curve.keyframes.iter().min_by(|a, b| {
        let da = (a.time - time).abs();
        let db = (b.time - time).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

pub fn curve_suggestion_poll_results(
    suggestion_state: &mut CurveSuggestionState,
    inference_state: &mut InferenceActorState,
) {
    if suggestion_state.pending_request_id.is_none() {
        return;
    }

    let results = inference_actor_take_results(inference_state);

    for result in results {
        let pending_match = suggestion_state
            .pending_request_id
            .map_or(false, |id| id == result.request_id);

        if !pending_match {
            continue;
        }

        if let InferenceResultKind::CurveCopilotPredict { steps } = result.kind {
            let bone_id = suggestion_state.pending_bone_id.unwrap_or(0);
            let property_type = suggestion_state
                .pending_property_type
                .unwrap_or(PropertyType::TranslationX);

            let clip_duration = suggestion_state.pending_clip_duration.unwrap_or(1.0);
            let curve_mean = suggestion_state.pending_curve_mean.unwrap_or(0.0);
            let curve_std = suggestion_state.pending_curve_std.unwrap_or(1.0);
            let query_times = suggestion_state
                .pending_query_times
                .clone()
                .unwrap_or_default();
            let real_samples = suggestion_state
                .pending_real_curve_samples
                .clone()
                .unwrap_or_default();
            let nearest_kfs = suggestion_state
                .pending_nearest_keyframes
                .clone()
                .unwrap_or_default();

            let collapse_threshold_time = clip_duration.max(0.001) * 0.01;
            let collapse_threshold_value = curve_std.max(1e-6) * 0.05;
            let mut sign_violation_count = 0usize;
            let mut collapsed_count = 0usize;
            let mut max_abs_tan_time = 0.0f32;

            for (i, step) in steps.iter().enumerate() {
                let predicted_time = query_times.get(i).copied().unwrap_or(0.0);

                let denorm_value = step.value * curve_std + curve_mean;
                let denorm_tan_in = (
                    step.tangent_in.0 * clip_duration,
                    step.tangent_in.1 * curve_std,
                );
                let denorm_tan_out = (
                    step.tangent_out.0 * clip_duration,
                    step.tangent_out.1 * curve_std,
                );

                suggestion_state.suggestions.push(GhostCurveSuggestion {
                    bone_id,
                    property_type,
                    predicted_time,
                    predicted_value: denorm_value,
                    tangent_in: denorm_tan_in,
                    tangent_out: denorm_tan_out,
                    confidence: step.confidence,
                    request_id: result.request_id,
                });

                let in_sign_violation = denorm_tan_in.0 > 0.0;
                let out_sign_violation = denorm_tan_out.0 < 0.0;
                let sign_flag = match (in_sign_violation, out_sign_violation) {
                    (true, true) => "BOTH",
                    (true, false) => "IN",
                    (false, true) => "OUT",
                    (false, false) => "ok",
                };
                if in_sign_violation || out_sign_violation {
                    sign_violation_count += 1;
                }

                let is_collapsed = denorm_tan_in.0.abs() < collapse_threshold_time
                    && denorm_tan_out.0.abs() < collapse_threshold_time
                    && denorm_tan_in.1.abs() < collapse_threshold_value
                    && denorm_tan_out.1.abs() < collapse_threshold_value;
                if is_collapsed {
                    collapsed_count += 1;
                }

                max_abs_tan_time = max_abs_tan_time
                    .max(denorm_tan_in.0.abs())
                    .max(denorm_tan_out.0.abs());

                let real_value_str = match real_samples.get(i).copied().flatten() {
                    Some(v) => format!("{:+.4}", v),
                    None => "n/a".to_string(),
                };
                let delta_str = match real_samples.get(i).copied().flatten() {
                    Some(v) => format!("{:+.4}", denorm_value - v),
                    None => "n/a".to_string(),
                };
                let nearest_kf_str = match nearest_kfs.get(i).copied().flatten() {
                    Some((kf_t, kf_v)) => format!("t={:.4} val={:+.4}", kf_t, kf_v),
                    None => "none".to_string(),
                };

                log!(
                    "CurveCopilot: step {}/{} conf={:.2} val={:.4} t={:.4} \
                     in_tan=({:+.5},{:+.5}) out_tan=({:+.5},{:+.5}) sign={}{} \
                     real_val={} delta={} nearest_kf=[{}]",
                    i + 1,
                    steps.len(),
                    step.confidence,
                    denorm_value,
                    predicted_time,
                    denorm_tan_in.0,
                    denorm_tan_in.1,
                    denorm_tan_out.0,
                    denorm_tan_out.1,
                    sign_flag,
                    if is_collapsed { " COLLAPSED" } else { "" },
                    real_value_str,
                    delta_str,
                    nearest_kf_str,
                );
            }

            log!(
                "CurveCopilot summary: steps={} sign_violations={}/{} ({:.0}%) \
                 collapsed={}/{} ({:.0}%) max_abs_tan_time={:.5} (thr={:.5})",
                steps.len(),
                sign_violation_count,
                steps.len(),
                100.0 * sign_violation_count as f32 / steps.len().max(1) as f32,
                collapsed_count,
                steps.len(),
                100.0 * collapsed_count as f32 / steps.len().max(1) as f32,
                max_abs_tan_time,
                collapse_threshold_time,
            );

            if let Some(snapshot) = suggestion_state.pending_dump.take() {
                write_inference_dump(
                    &snapshot,
                    &steps,
                    clip_duration,
                    curve_mean,
                    curve_std,
                    &query_times,
                    &real_samples,
                    &nearest_kfs,
                );
            }

            suggestion_state.pending_request_id = None;
            suggestion_state.pending_bone_id = None;
            suggestion_state.pending_property_type = None;
            suggestion_state.pending_clip_duration = None;
            suggestion_state.pending_curve_mean = None;
            suggestion_state.pending_curve_std = None;
            suggestion_state.pending_query_times = None;
            suggestion_state.pending_real_curve_samples = None;
            suggestion_state.pending_nearest_keyframes = None;
        }
    }
}

fn write_inference_dump(
    snapshot: &CurveSuggestionPendingDump,
    steps: &[thyllore_ml_core::copilot::CopilotStepPrediction],
    clip_duration: f32,
    curve_mean: f32,
    curve_std: f32,
    denorm_query_times: &[f32],
    real_samples: &[Option<f32>],
    nearest_kfs: &[Option<(f32, f32)>],
) {
    let onnx_features_per_step = 5_i64;
    let mut onnx_predictions = Vec::with_capacity(steps.len() * onnx_features_per_step as usize);
    let mut onnx_confidences = Vec::with_capacity(steps.len());
    for step in steps {
        onnx_predictions.push(step.value);
        onnx_predictions.push(step.tangent_in.0);
        onnx_predictions.push(step.tangent_in.1);
        onnx_predictions.push(step.tangent_out.0);
        onnx_predictions.push(step.tangent_out.1);
        onnx_confidences.push(step.confidence);
    }

    let real_curve_samples: Vec<f32> = real_samples
        .iter()
        .map(|opt| opt.unwrap_or(f32::NAN))
        .collect();
    let nearest_keyframe_times: Vec<f32> = nearest_kfs
        .iter()
        .map(|opt| opt.map(|(t, _)| t).unwrap_or(f32::NAN))
        .collect();
    let nearest_keyframe_values: Vec<f32> = nearest_kfs
        .iter()
        .map(|opt| opt.map(|(_, v)| v).unwrap_or(f32::NAN))
        .collect();

    let dump = CurveCopilotInferenceDump {
        context_keyframes: &snapshot.context_keyframes,
        property_type_id: snapshot.property_type_id,
        topology_features: &snapshot.topology_features,
        bone_name_tokens: &snapshot.bone_name_tokens,
        query_times: &snapshot.query_times_normalized,
        curve_window: &snapshot.curve_window,
        bone_context_keyframes: &snapshot.bone_context_keyframes,
        bone_context_topology: &snapshot.bone_context_topology,
        bone_context_rest_positions: &snapshot.bone_context_rest_positions,
        bone_context_mask: &snapshot.bone_context_mask,
        onnx_predictions: &onnx_predictions,
        onnx_confidences: &onnx_confidences,
        onnx_prediction_steps: steps.len() as i64,
        onnx_prediction_features_per_step: onnx_features_per_step,
        curve_mean,
        curve_std,
        clip_duration,
        denorm_query_times,
        real_curve_samples: &real_curve_samples,
        nearest_keyframe_times: &nearest_keyframe_times,
        nearest_keyframe_values: &nearest_keyframe_values,
    };

    match dump_curve_copilot_inference(&dump, std::path::Path::new("tmp")) {
        Ok(path) => log!("CurveCopilot dump: saved {}", path.display()),
        Err(e) => log_warn!("CurveCopilot dump failed: {}", e),
    }
}

pub fn curve_suggestion_apply(suggestion: &GhostCurveSuggestion, curve: &mut PropertyCurve) {
    let in_tangent = BezierHandle::new(suggestion.tangent_in.0, suggestion.tangent_in.1);
    let out_tangent = BezierHandle::new(suggestion.tangent_out.0, suggestion.tangent_out.1);

    curve_add_keyframe_with_tangents(
        curve,
        suggestion.predicted_time,
        suggestion.predicted_value,
        in_tangent,
        out_tangent,
        InterpolationType::Bezier,
    );
}

pub fn curve_suggestion_dismiss(suggestion_state: &mut CurveSuggestionState) {
    suggestion_state.suggestions.clear();
    suggestion_state.pending_request_id = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::editable::{curve_add_keyframe, CurveId};

    fn create_test_curve(keyframe_count: usize) -> PropertyCurve {
        let mut curve = PropertyCurve::new(1 as CurveId, PropertyType::TranslationX);
        for i in 0..keyframe_count {
            let time = (i + 1) as f32 * 0.5;
            curve_add_keyframe(&mut curve, time, (i as f32).sin());
        }
        curve
    }

    #[test]
    fn test_suggestion_dismiss() {
        let mut state = CurveSuggestionState::default();
        state.suggestions.push(GhostCurveSuggestion {
            bone_id: 0,
            property_type: PropertyType::TranslationX,
            predicted_time: 1.0,
            predicted_value: 2.0,
            tangent_in: (0.0, 0.0),
            tangent_out: (0.0, 0.0),
            confidence: 0.9,
            request_id: 42,
        });
        state.pending_request_id = Some(100);

        curve_suggestion_dismiss(&mut state);

        assert!(state.suggestions.is_empty());
        assert!(state.pending_request_id.is_none());
    }

    #[test]
    fn resolve_anchor_time_picks_latest_kf_before_current_time() {
        let mut curve = PropertyCurve::new(1 as CurveId, PropertyType::TranslationX);
        curve_add_keyframe(&mut curve, 0.5, 0.0);
        curve_add_keyframe(&mut curve, 1.0, 1.0);
        curve_add_keyframe(&mut curve, 2.0, 2.0);
        curve_add_keyframe(&mut curve, 3.0, 3.0);

        let anchor = resolve_anchor_time(&curve, 2.5).expect("anchor found");
        assert!((anchor - 2.0).abs() < 1e-6);
    }

    #[test]
    fn resolve_anchor_time_returns_none_when_all_kfs_are_in_future() {
        let mut curve = PropertyCurve::new(1 as CurveId, PropertyType::TranslationX);
        curve_add_keyframe(&mut curve, 1.0, 1.0);
        curve_add_keyframe(&mut curve, 2.0, 2.0);

        assert!(resolve_anchor_time(&curve, 0.5).is_none());
    }

    #[test]
    fn flatten_keyframes_before_anchor_excludes_future_kfs() {
        let mut curve = PropertyCurve::new(1 as CurveId, PropertyType::TranslationX);
        curve_add_keyframe(&mut curve, 0.5, 0.0);
        curve_add_keyframe(&mut curve, 1.0, 1.0);
        curve_add_keyframe(&mut curve, 2.0, 2.0);
        curve_add_keyframe(&mut curve, 3.0, 3.0);

        let flat = flatten_keyframes_before_anchor(&curve, 2.0);
        assert_eq!(flat.times, vec![0.5, 1.0, 2.0]);
    }

    #[test]
    fn test_suggestion_apply() {
        let suggestion = GhostCurveSuggestion {
            bone_id: 0,
            property_type: PropertyType::TranslationX,
            predicted_time: 1.5,
            predicted_value: 0.8,
            tangent_in: (-0.1, 0.0),
            tangent_out: (0.1, 0.0),
            confidence: 0.9,
            request_id: 1,
        };

        let mut curve = create_test_curve(3);
        let before_count = curve.keyframe_count();

        curve_suggestion_apply(&suggestion, &mut curve);

        assert_eq!(curve.keyframe_count(), before_count + 1);
    }
}
