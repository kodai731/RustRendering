use crate::animation::editable::{
    curve_add_keyframe_with_tangents, curve_sample, BezierHandle, EditableAnimationClip,
    EditableKeyframe, InterpolationType, PropertyCurve, PropertyType,
};
use crate::animation::BoneId;
use crate::ecs::resource::{
    CurveSuggestionPendingDump, CurveSuggestionState, GhostCurveSuggestion, InferenceActorState,
};
use crate::ml::{InferenceActorId, InferenceRequestKind, InferenceResultKind};
use thyllore_ml_core::copilot::dump::{dump_rawfuture_inference, RawFutureInferenceDump};
use thyllore_ml_core::copilot::rawfuture::{CONTEXT_LENGTH, MAX_HORIZON};

use super::inference_actor_systems::{inference_actor_submit, inference_actor_take_results};

const ANCHOR_EPSILON: f32 = 1.0e-6;
const DEPLOY_FPS: f32 = 60.0;
const SUGGESTION_STRIDE: usize = 1;
const SUGGESTION_FRAME_COUNT: usize = 8;

fn resolve_anchor_time(curve: &PropertyCurve, current_time: f32) -> Option<f32> {
    curve
        .keyframes
        .iter()
        .map(|kf| kf.time)
        .filter(|t| *t <= current_time + ANCHOR_EPSILON)
        .fold(None, |acc, t| Some(acc.map_or(t, |best: f32| best.max(t))))
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

struct RawFutureWindows {
    context: Vec<f32>,
    future: Vec<f32>,
}

fn build_rawfuture_windows(curve: &PropertyCurve, origin_time: f32, dt: f32) -> RawFutureWindows {
    let context: Vec<f32> = (0..CONTEXT_LENGTH)
        .map(|i| {
            sample_or_hold(
                curve,
                origin_time + (i as f32 - (CONTEXT_LENGTH as f32 - 1.0)) * dt,
            )
        })
        .collect();

    let future: Vec<f32> = (0..MAX_HORIZON)
        .map(|i| sample_or_hold(curve, origin_time + (i as f32 + 1.0) * dt))
        .collect();

    RawFutureWindows { context, future }
}

fn suggestion_frame_indices() -> impl Iterator<Item = usize> {
    let end = SUGGESTION_FRAME_COUNT.min(MAX_HORIZON);
    (SUGGESTION_STRIDE.saturating_sub(1)..end).step_by(SUGGESTION_STRIDE)
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

    let dt = 1.0 / DEPLOY_FPS;
    let origin_value = sample_or_hold(curve, origin_time);
    let windows = build_rawfuture_windows(curve, origin_time, dt);
    let reveal_mask = vec![false; MAX_HORIZON];

    log!(
        "CurveCopilot input: bone_id={} property={:?} origin={:.4} fps={:.1} forecast",
        bone_id,
        property_type,
        origin_time,
        DEPLOY_FPS,
    );

    let dump_snapshot = if suggestion_state.dump_inference {
        Some(CurveSuggestionPendingDump {
            context: windows.context.clone(),
            future: windows.future.clone(),
            reveal_mask: reveal_mask.clone(),
            fps: DEPLOY_FPS,
            anchor_time: origin_time,
        })
    } else {
        None
    };

    let kind = InferenceRequestKind::CurveCopilotPredict {
        context: windows.context,
        future: windows.future,
        reveal_mask,
        fps: DEPLOY_FPS,
    };

    if let Some(request_id) = inference_actor_submit(inference_state, actor_id, kind) {
        suggestion_state.pending_request_id = Some(request_id);
        suggestion_state.pending_bone_id = Some(bone_id);
        suggestion_state.pending_property_type = Some(property_type);
        suggestion_state.pending_anchor_time = Some(origin_time);
        suggestion_state.pending_origin_value = Some(origin_value);
        suggestion_state.pending_dt = Some(dt);
        suggestion_state.pending_dump = dump_snapshot;
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

        if let InferenceResultKind::CurveCopilotPredict { mean_curve } = result.kind {
            let bone_id = suggestion_state.pending_bone_id.unwrap_or(0);
            let property_type = suggestion_state
                .pending_property_type
                .unwrap_or(PropertyType::TranslationX);
            let anchor_time = suggestion_state.pending_anchor_time.unwrap_or(0.0);
            let dt = suggestion_state.pending_dt.unwrap_or(1.0 / DEPLOY_FPS);
            let origin_value = suggestion_state
                .pending_origin_value
                .unwrap_or_else(|| mean_curve.first().copied().unwrap_or(0.0));

            let suggestions = build_suggestions_from_curve(
                &mean_curve,
                origin_value,
                anchor_time,
                dt,
                bone_id,
                property_type,
                result.request_id,
            );

            log!(
                "CurveCopilot output: {} dense values -> {} ghost suggestions \
                 (anchor={:.4} dt={:.4})",
                mean_curve.len(),
                suggestions.len(),
                anchor_time,
                dt,
            );

            suggestion_state.suggestions.extend(suggestions);

            if let Some(snapshot) = suggestion_state.pending_dump.take() {
                write_inference_dump(&snapshot, &mean_curve);
            }

            suggestion_state.pending_request_id = None;
            suggestion_state.pending_bone_id = None;
            suggestion_state.pending_property_type = None;
            suggestion_state.pending_anchor_time = None;
            suggestion_state.pending_origin_value = None;
            suggestion_state.pending_dt = None;
        }
    }
}

fn predicted_velocity(mean_curve: &[f32], i: usize, dt: f32) -> f32 {
    let n = mean_curve.len();
    if n < 2 {
        return 0.0;
    }
    if i == 0 {
        (mean_curve[1] - mean_curve[0]) / dt
    } else if i == n - 1 {
        (mean_curve[n - 1] - mean_curve[n - 2]) / dt
    } else {
        (mean_curve[i + 1] - mean_curve[i - 1]) / (2.0 * dt)
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
    let handle_dt = SUGGESTION_STRIDE as f32 * dt / 3.0;
    let continuity_offset = mean_curve.first().map_or(0.0, |first| origin_value - first);
    let mut suggestions = Vec::new();

    for i in suggestion_frame_indices() {
        if i >= mean_curve.len() {
            continue;
        }
        let velocity = predicted_velocity(mean_curve, i, dt);
        let handle_dv = velocity * handle_dt;
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
    let dump = RawFutureInferenceDump {
        context: &snapshot.context,
        future: &snapshot.future,
        reveal_mask: &snapshot.reveal_mask,
        mean_curve,
        fps: snapshot.fps,
        anchor_time: snapshot.anchor_time,
    };

    match dump_rawfuture_inference(&dump, std::path::Path::new("tmp")) {
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
    fn build_windows_have_fixed_lengths() {
        let mut curve = PropertyCurve::new(1 as CurveId, PropertyType::TranslationX);
        curve_add_keyframe(&mut curve, 0.0, 0.0);
        let dt = 1.0 / DEPLOY_FPS;

        let windows = build_rawfuture_windows(&curve, 0.0, dt);
        assert_eq!(windows.context.len(), CONTEXT_LENGTH);
        assert_eq!(windows.future.len(), MAX_HORIZON);
    }

    #[test]
    fn first_suggestion_is_one_frame_after_origin_and_continuous() {
        let mean_curve: Vec<f32> = (0..MAX_HORIZON).map(|i| 100.0 + i as f32).collect();
        let origin_time = 0.5;
        let origin_value = 7.0;
        let dt = 1.0 / DEPLOY_FPS;

        let suggestions = build_suggestions_from_curve(
            &mean_curve,
            origin_value,
            origin_time,
            dt,
            0,
            PropertyType::TranslationX,
            1,
        );

        assert_eq!(suggestions.len(), SUGGESTION_FRAME_COUNT);
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
