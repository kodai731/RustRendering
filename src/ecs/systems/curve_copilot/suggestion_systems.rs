use crate::animation::editable::{
    curve_add_keyframe_with_tangents, curve_sample, BezierHandle, EditableAnimationClip,
    EditableKeyframe, InterpolationType, PropertyCurve, PropertyType,
};
use crate::animation::BoneId;
use crate::ecs::resource::{
    CurveSuggestionPendingDump, CurveSuggestionState, GhostCurveSuggestion, InferenceActorState,
    PendingSuggestionRequest,
};
use crate::ml::{
    CurveCopilotMode, FeedbackSenderHandle, InferenceActorId, InferenceRequestKind,
    InferenceResultKind,
};
use thyllore_ml_core::copilot::v2::dump::{
    dump_v2_curve_copilot_inference, V2CurveCopilotInferenceDump,
};
use thyllore_ml_core::copilot::v2::forecast;

use super::mode_systems::{
    curve_copilot_capture_feedback_context, curve_copilot_degrade_context,
    curve_copilot_send_feedback,
};
use crate::ecs::systems::inference_actor_systems::{
    inference_actor_submit, inference_actor_take_results,
};

fn resolve_anchor_time(curve: &PropertyCurve, current_time: f32) -> Option<f32> {
    let times: Vec<f32> = curve.keyframes.iter().map(|kf| kf.time).collect();
    forecast::resolve_origin_time(&times, current_time)
}

fn find_nearest_keyframe(curve: &PropertyCurve, time: f32) -> Option<&EditableKeyframe> {
    curve.keyframes.iter().min_by(|a, b| {
        let da = (a.time - time).abs();
        let db = (b.time - time).abs();
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn sample_or_hold(curve: &PropertyCurve, time: f32) -> f32 {
    curve_sample(curve, time)
        .or_else(|| find_nearest_keyframe(curve, time).map(|kf| kf.value))
        .unwrap_or(0.0)
}

fn build_v2_curve_copilot_context(curve: &PropertyCurve, origin_time: f32, dt: f32) -> Vec<f32> {
    forecast::context_sample_offsets()
        .iter()
        .map(|&offset| sample_or_hold(curve, origin_time + offset as f32 * dt))
        .collect()
}

pub struct CurveSuggestionInputs<'a> {
    pub clip: &'a EditableAnimationClip,
}

pub fn curve_suggestion_submit(
    suggestion_state: &mut CurveSuggestionState,
    inference_state: &mut InferenceActorState,
    actor_id: InferenceActorId,
    inputs: CurveSuggestionInputs<'_>,
    property_type: PropertyType,
    bone_id: BoneId,
    current_time: f32,
    mode: CurveCopilotMode,
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

    let Some(track) = inputs.clip.get_track(bone_id) else {
        return;
    };
    let curve = track.get_curve(property_type);

    if resolve_anchor_time(curve, current_time).is_none() {
        return;
    }
    let origin_time = current_time;

    let dt = 1.0 / forecast::DEPLOY_FPS;
    let origin_value = sample_or_hold(curve, origin_time);
    let mut context = build_v2_curve_copilot_context(curve, origin_time, dt);
    curve_copilot_degrade_context(mode, &mut context);

    log!(
        "CurveCopilot input: bone_id={} property={:?} origin={:.4} fps={:.1} forecast",
        bone_id,
        property_type,
        origin_time,
        forecast::DEPLOY_FPS,
    );

    let dump_snapshot = if suggestion_state.dump_inference {
        Some(CurveSuggestionPendingDump {
            context: context.clone(),
            fps: forecast::DEPLOY_FPS,
            anchor_time: origin_time,
        })
    } else {
        None
    };

    let feedback_context = curve_copilot_capture_feedback_context(mode, &context);

    let kind = InferenceRequestKind::CurveCopilotPredict {
        context,
        fps: forecast::DEPLOY_FPS,
    };

    if let Some(request_id) = inference_actor_submit(inference_state, actor_id, kind) {
        suggestion_state.pending = Some(PendingSuggestionRequest {
            request_id,
            bone_id,
            property_type,
            anchor_time: origin_time,
            origin_value,
            dt,
            dump: dump_snapshot,
            feedback_context,
        });
    }
}

pub fn curve_suggestion_poll_results(
    suggestion_state: &mut CurveSuggestionState,
    inference_state: &mut InferenceActorState,
    feedback_sender: Option<&FeedbackSenderHandle>,
) {
    if suggestion_state.pending.is_none() {
        return;
    }

    let results = inference_actor_take_results(inference_state);

    for result in results {
        let pending_match = suggestion_state
            .pending
            .as_ref()
            .map_or(false, |pending| pending.request_id == result.request_id);

        if !pending_match {
            continue;
        }

        if let InferenceResultKind::CurveCopilotPredict { mean_curve } = result.kind {
            let Some(pending) = suggestion_state.pending.take() else {
                continue;
            };

            let suggestions = build_suggestions_from_curve(
                &mean_curve,
                pending.origin_value,
                pending.anchor_time,
                pending.dt,
                pending.bone_id,
                pending.property_type,
                result.request_id,
            );

            log!(
                "CurveCopilot output: {} dense values -> {} ghost suggestions \
                 (anchor={:.4} dt={:.4})",
                mean_curve.len(),
                suggestions.len(),
                pending.anchor_time,
                pending.dt,
            );

            suggestion_state.suggestions.extend(suggestions);

            if let Some(snapshot) = pending.dump {
                write_inference_dump(&snapshot, &mean_curve);
            }

            if let (Some(sender), Some(context)) = (feedback_sender, pending.feedback_context) {
                curve_copilot_send_feedback(
                    sender,
                    pending.property_type,
                    pending.origin_value,
                    &context,
                    &mean_curve,
                );
            }
        }
    }
}

fn build_suggestions_from_curve(
    mean_curve: &[f32],
    origin_value: f32,
    anchor_time: f32,
    dt: f32,
    bone_id: BoneId,
    property_type: PropertyType,
    request_id: crate::ml::InferenceRequestId,
) -> Vec<GhostCurveSuggestion> {
    let handle_dt = forecast::SUGGESTION_STRIDE as f32 * dt / 3.0;
    let continuity_offset = forecast::continuity_offset(mean_curve, origin_value);
    let velocities = forecast::compute_velocities(mean_curve, 1.0 / dt);
    let mut suggestions = Vec::new();

    for i in forecast::suggestion_frame_indices() {
        if i >= mean_curve.len() {
            continue;
        }
        let handle_dv = velocities[i] * handle_dt;
        suggestions.push(GhostCurveSuggestion {
            bone_id,
            property_type,
            predicted_time: anchor_time + (i as f32 + 1.0) * dt,
            predicted_value: mean_curve[i] + continuity_offset,
            tangent_in: (-handle_dt, -handle_dv),
            tangent_out: (handle_dt, handle_dv),
            confidence: 1.0,
            request_id,
        });
    }
    suggestions
}

fn write_inference_dump(snapshot: &CurveSuggestionPendingDump, mean_curve: &[f32]) {
    let dump = V2CurveCopilotInferenceDump {
        context: &snapshot.context,
        mean_curve,
        fps: snapshot.fps,
        anchor_time: snapshot.anchor_time,
    };

    match dump_v2_curve_copilot_inference(&dump, std::path::Path::new("tmp")) {
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
    suggestion_state.pending = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::editable::{curve_add_keyframe, CurveId};
    use thyllore_ml_core::copilot::v2::inference::{CONTEXT_LENGTH, MAX_HORIZON};

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
        state.pending = Some(PendingSuggestionRequest {
            request_id: 100,
            bone_id: 0,
            property_type: PropertyType::TranslationX,
            anchor_time: 0.0,
            origin_value: 0.0,
            dt: 1.0 / forecast::DEPLOY_FPS,
            dump: None,
            feedback_context: None,
        });

        curve_suggestion_dismiss(&mut state);

        assert!(state.suggestions.is_empty());
        assert!(state.pending.is_none());
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
    fn build_context_has_fixed_length() {
        let mut curve = PropertyCurve::new(1 as CurveId, PropertyType::TranslationX);
        curve_add_keyframe(&mut curve, 0.0, 0.0);
        let dt = 1.0 / forecast::DEPLOY_FPS;

        let context = build_v2_curve_copilot_context(&curve, 0.0, dt);
        assert_eq!(context.len(), CONTEXT_LENGTH);
    }

    #[test]
    fn first_suggestion_is_one_frame_after_origin_and_continuous() {
        let mean_curve: Vec<f32> = (0..MAX_HORIZON).map(|i| 100.0 + i as f32).collect();
        let origin_time = 0.5;
        let origin_value = 7.0;
        let dt = 1.0 / forecast::DEPLOY_FPS;

        let suggestions = build_suggestions_from_curve(
            &mean_curve,
            origin_value,
            origin_time,
            dt,
            0,
            PropertyType::TranslationX,
            1,
        );

        assert_eq!(suggestions.len(), forecast::SUGGESTION_FRAME_COUNT);
        assert!((suggestions[0].predicted_time - (origin_time + dt)).abs() < 1e-6);
        assert!((suggestions[0].predicted_value - origin_value).abs() < 1e-6);
        assert!((suggestions[1].predicted_value - (origin_value + 1.0)).abs() < 1e-6);
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
