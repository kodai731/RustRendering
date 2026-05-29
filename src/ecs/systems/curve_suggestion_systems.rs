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
const DEPLOY_FPS: f32 = 30.0;
const SUGGESTION_STRIDE: usize = 8;

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

/// Sample the curve at `time`, holding the nearest keyframe value when the time falls
/// outside the keyframed range (so the dense reconstruction never produces a gap).
fn sample_or_hold(curve: &PropertyCurve, time: f32) -> f32 {
    curve_sample(curve, time)
        .or_else(|| find_nearest_keyframe(curve, time).map(|kf| kf.value))
        .unwrap_or(0.0)
}

/// Dense raw windows fed to the rawfuture model, plus the anchors already known in the
/// future horizon (A/B auto: any existing keyframe after the anchor becomes a revealed
/// anchor, so the model interpolates; with none it forecasts).
struct RawFutureWindows {
    context: Vec<f32>,
    future: Vec<f32>,
    reveal_mask: Vec<bool>,
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

    let mut reveal_mask = vec![false; MAX_HORIZON];
    for kf in &curve.keyframes {
        if kf.time <= origin_time + ANCHOR_EPSILON {
            continue;
        }
        let frame = ((kf.time - origin_time) / dt - 1.0).round();
        if frame >= 0.0 && (frame as usize) < MAX_HORIZON {
            let i = frame as usize;
            if (origin_time + (i as f32 + 1.0) * dt - kf.time).abs() <= dt * 0.5 {
                reveal_mask[i] = true;
            }
        }
    }

    RawFutureWindows {
        context,
        future,
        reveal_mask,
    }
}

/// Indices of the future frames sampled into ghost suggestions (every `SUGGESTION_STRIDE`).
fn suggestion_frame_indices() -> impl Iterator<Item = usize> {
    (SUGGESTION_STRIDE.saturating_sub(1)..MAX_HORIZON).step_by(SUGGESTION_STRIDE)
}

/// True when at least one suggestion frame is unrevealed, i.e. there is a gap to fill.
fn has_strided_gap(reveal_mask: &[bool]) -> bool {
    suggestion_frame_indices().any(|i| !reveal_mask.get(i).copied().unwrap_or(false))
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

    // Forecast origin is the current edit time (playhead), not the last keyframe, so the
    // prediction always starts from "now" and continues the motion leading up to it,
    // rather than collapsing onto the animation's final settled pose. A keyframe must
    // exist at or before now so the context window carries real prior motion.
    if resolve_anchor_time(curve, current_time).is_none() {
        return;
    }
    let origin_time = current_time;

    let dt = 1.0 / DEPLOY_FPS;
    let mut windows = build_rawfuture_windows(curve, origin_time, dt);

    // A/B auto: keep revealed anchors only when there is a gap to fill at a suggestion
    // frame (sparse curve -> interpolation). A densely keyframed curve leaves no gap, so
    // fall back to forecast (reveal nothing) and predict the continuation from context.
    if !has_strided_gap(&windows.reveal_mask) {
        windows.reveal_mask = vec![false; MAX_HORIZON];
    }
    let revealed_count = windows.reveal_mask.iter().filter(|&&m| m).count();
    let regime = if revealed_count > 0 {
        "A/interp"
    } else {
        "B/forecast"
    };

    log!(
        "CurveCopilot input: bone_id={} property={:?} origin={:.4} fps={:.1} \
         regime={} revealed_anchors={}",
        bone_id,
        property_type,
        origin_time,
        DEPLOY_FPS,
        regime,
        revealed_count,
    );

    let dump_snapshot = if suggestion_state.dump_inference {
        Some(CurveSuggestionPendingDump {
            context: windows.context.clone(),
            future: windows.future.clone(),
            reveal_mask: windows.reveal_mask.clone(),
            fps: DEPLOY_FPS,
            anchor_time: origin_time,
        })
    } else {
        None
    };

    let kind = InferenceRequestKind::CurveCopilotPredict {
        context: windows.context,
        future: windows.future,
        reveal_mask: windows.reveal_mask.clone(),
        fps: DEPLOY_FPS,
    };

    if let Some(request_id) = inference_actor_submit(inference_state, actor_id, kind) {
        suggestion_state.pending_request_id = Some(request_id);
        suggestion_state.pending_bone_id = Some(bone_id);
        suggestion_state.pending_property_type = Some(property_type);
        suggestion_state.pending_anchor_time = Some(origin_time);
        suggestion_state.pending_dt = Some(dt);
        suggestion_state.pending_reveal_mask = Some(windows.reveal_mask);
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
            let reveal_mask = suggestion_state
                .pending_reveal_mask
                .clone()
                .unwrap_or_default();

            let suggestions = build_suggestions_from_curve(
                &mean_curve,
                &reveal_mask,
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
            suggestion_state.pending_dt = None;
            suggestion_state.pending_reveal_mask = None;
        }
    }
}

/// Velocity (value/second) of the dense predicted curve at frame `i`, matching the
/// central / one-sided finite differences used to build the model's tangent feature.
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

/// Sample the dense predicted curve at a fixed stride into ghost keyframe suggestions.
/// Frames that already hold a keyframe (revealed anchors) are skipped, so only the
/// model's filled-in / forecast frames are offered. Bezier handles span a third of the
/// stride, following the editor's 1/3 tangent convention.
fn build_suggestions_from_curve(
    mean_curve: &[f32],
    reveal_mask: &[bool],
    anchor_time: f32,
    dt: f32,
    bone_id: BoneId,
    property_type: PropertyType,
    request_id: crate::ml::InferenceRequestId,
) -> Vec<GhostCurveSuggestion> {
    let handle_dt = SUGGESTION_STRIDE as f32 * dt / 3.0;
    let mut suggestions = Vec::new();

    for i in suggestion_frame_indices() {
        if i >= mean_curve.len() || reveal_mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        let velocity = predicted_velocity(mean_curve, i, dt);
        let handle_dv = velocity * handle_dt;
        suggestions.push(GhostCurveSuggestion {
            bone_id,
            property_type,
            predicted_time: anchor_time + (i as f32 + 1.0) * dt,
            predicted_value: mean_curve[i],
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
    fn build_windows_marks_future_keyframe_as_revealed_anchor() {
        let mut curve = PropertyCurve::new(1 as CurveId, PropertyType::TranslationX);
        curve_add_keyframe(&mut curve, 0.0, 0.0);
        let dt = 1.0 / DEPLOY_FPS;
        let future_kf_time = (8.0 + 1.0) * dt;
        curve_add_keyframe(&mut curve, future_kf_time, 5.0);

        let windows = build_rawfuture_windows(&curve, 0.0, dt);
        assert_eq!(windows.context.len(), CONTEXT_LENGTH);
        assert_eq!(windows.future.len(), MAX_HORIZON);
        assert!(
            windows.reveal_mask[8],
            "future keyframe should reveal frame 8"
        );
        assert_eq!(windows.reveal_mask.iter().filter(|&&m| m).count(), 1);
    }

    #[test]
    fn dense_reveal_mask_has_no_strided_gap_triggering_forecast_fallback() {
        let all_revealed = vec![true; MAX_HORIZON];
        assert!(!has_strided_gap(&all_revealed));

        let mut sparse = vec![true; MAX_HORIZON];
        sparse[SUGGESTION_STRIDE - 1] = false;
        assert!(has_strided_gap(&sparse));
    }

    #[test]
    fn build_suggestions_skips_revealed_frames_and_uses_predicted_values() {
        let mean_curve: Vec<f32> = (0..MAX_HORIZON).map(|i| i as f32).collect();
        let mut reveal_mask = vec![false; MAX_HORIZON];
        reveal_mask[SUGGESTION_STRIDE - 1] = true;

        let suggestions = build_suggestions_from_curve(
            &mean_curve,
            &reveal_mask,
            0.0,
            1.0 / DEPLOY_FPS,
            0,
            PropertyType::TranslationX,
            1,
        );

        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .all(|s| (s.predicted_value - (s.predicted_value.round())).abs() < 1e-3));
        let first_index = SUGGESTION_STRIDE - 1;
        assert!(
            suggestions
                .iter()
                .all(|s| (s.predicted_value - first_index as f32).abs() > 1e-3),
            "revealed frame must be skipped"
        );
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
