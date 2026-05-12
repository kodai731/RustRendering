use crate::animation::editable::{
    curve_add_keyframe_with_tangents, BezierHandle, InterpolationType, PropertyCurve, PropertyType,
};
use crate::animation::BoneId;
use crate::ecs::resource::{
    BoneNameTokenCache, BoneTopologyCache, CurveSuggestionState, GhostCurveSuggestion,
    InferenceActorState,
};
use crate::ml::{InferenceActorId, InferenceRequestKind, InferenceResultKind};
use thyllore_ml_core::copilot::context::flatten_context;
use thyllore_ml_core::copilot::property::{property_kind_to_id, PropertyKind};
use thyllore_ml_core::copilot::query::generate_query_times;
use thyllore_ml_core::copilot::window::sample_window;

use super::inference_actor_systems::{inference_actor_submit, inference_actor_take_results};

const MAX_CONTEXT_KEYFRAMES: usize = 8;
const MIN_CURVE_STD: f32 = 0.01;

struct FlatKeyframes {
    times: Vec<f32>,
    values: Vec<f32>,
    in_dt: Vec<f32>,
    in_dv: Vec<f32>,
    out_dt: Vec<f32>,
    out_dv: Vec<f32>,
}

fn flatten_keyframes(curve: &PropertyCurve) -> FlatKeyframes {
    let n = curve.keyframes.len();
    let mut flat = FlatKeyframes {
        times: Vec::with_capacity(n),
        values: Vec::with_capacity(n),
        in_dt: Vec::with_capacity(n),
        in_dv: Vec::with_capacity(n),
        out_dt: Vec::with_capacity(n),
        out_dv: Vec::with_capacity(n),
    };
    for kf in &curve.keyframes {
        flat.times.push(kf.time);
        flat.values.push(kf.value);
        flat.in_dt.push(kf.in_tangent.time_offset);
        flat.in_dv.push(kf.in_tangent.value_offset);
        flat.out_dt.push(kf.out_tangent.time_offset);
        flat.out_dv.push(kf.out_tangent.value_offset);
    }
    flat
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

pub fn curve_suggestion_submit(
    suggestion_state: &mut CurveSuggestionState,
    inference_state: &mut InferenceActorState,
    actor_id: InferenceActorId,
    curve: &PropertyCurve,
    property_type: PropertyType,
    bone_id: BoneId,
    clip_duration: f32,
    current_time: f32,
    topology_cache: &BoneTopologyCache,
    name_token_cache: &BoneNameTokenCache,
) {
    if !suggestion_state.enabled {
        return;
    }

    let Some(max_steps) = inference_state.actor_max_steps(actor_id) else {
        log_warn!(
            "curve_suggestion_submit: actor {} has no max_steps metadata; skipping request",
            actor_id
        );
        return;
    };

    let flat = flatten_keyframes(curve);
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
    let topology_features = topology_cache.get(bone_id).to_vec();
    let bone_name_tokens = name_token_cache.get(bone_id).to_vec();

    let query_times = generate_query_times(&flat.times, current_time, clip_duration, max_steps);

    let context_start_time = curve
        .keyframes
        .iter()
        .rev()
        .take(MAX_CONTEXT_KEYFRAMES)
        .last()
        .map(|kf| kf.time)
        .unwrap_or(0.0);
    let last_query_time = query_times
        .last()
        .map(|t| t * clip_duration.max(0.001))
        .unwrap_or(clip_duration);
    let curve_window = sample_window(
        &flat.times,
        &flat.values,
        context_start_time,
        last_query_time,
        context.curve_mean,
        context.curve_std,
    );

    let denorm_query_times: Vec<f32> = query_times
        .iter()
        .map(|t| t * clip_duration.max(0.001))
        .collect();

    let kind = InferenceRequestKind::CurveCopilotPredict {
        context: context.flat,
        property_type_id,
        topology_features,
        bone_name_tokens,
        query_times,
        curve_window: curve_window.to_vec(),
    };

    if let Some(request_id) = inference_actor_submit(inference_state, actor_id, kind) {
        suggestion_state.pending_request_id = Some(request_id);
        suggestion_state.pending_bone_id = Some(bone_id);
        suggestion_state.pending_property_type = Some(property_type);
        suggestion_state.pending_clip_duration = Some(clip_duration);
        suggestion_state.pending_curve_mean = Some(context.curve_mean);
        suggestion_state.pending_curve_std = Some(context.curve_std);
        suggestion_state.pending_query_times = Some(denorm_query_times);
    }
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

                log!(
                    "CurveCopilot: step {}/{}, confidence={:.2}, denorm_value={:.4}, time={:.4}",
                    i + 1,
                    steps.len(),
                    step.confidence,
                    denorm_value,
                    predicted_time,
                );
            }

            suggestion_state.pending_request_id = None;
            suggestion_state.pending_bone_id = None;
            suggestion_state.pending_property_type = None;
            suggestion_state.pending_clip_duration = None;
            suggestion_state.pending_curve_mean = None;
            suggestion_state.pending_curve_std = None;
            suggestion_state.pending_query_times = None;
        }
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
