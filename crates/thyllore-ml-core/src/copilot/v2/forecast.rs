use anyhow::Result;

use crate::copilot::v2::inference::{
    V2CurveCopilotRequest, V2CurveCopilotSession, CONTEXT_LENGTH, MAX_HORIZON,
};

const ANCHOR_EPSILON: f32 = 1.0e-6;

pub fn context_sample_offsets() -> [i64; CONTEXT_LENGTH] {
    let mut offsets = [0i64; CONTEXT_LENGTH];
    for (i, offset) in offsets.iter_mut().enumerate() {
        *offset = i as i64 - (CONTEXT_LENGTH as i64 - 1);
    }
    offsets
}

pub fn future_sample_offsets() -> [i64; MAX_HORIZON] {
    let mut offsets = [0i64; MAX_HORIZON];
    for (i, offset) in offsets.iter_mut().enumerate() {
        *offset = i as i64 + 1;
    }
    offsets
}

pub const DEPLOY_FPS: f32 = 60.0;
pub const SUGGESTION_STRIDE: usize = 1;
pub const SUGGESTION_FRAME_COUNT: usize = 16;

pub fn suggestion_frame_indices() -> impl Iterator<Item = usize> {
    let end = SUGGESTION_FRAME_COUNT.min(MAX_HORIZON);
    (SUGGESTION_STRIDE.saturating_sub(1)..end).step_by(SUGGESTION_STRIDE)
}

pub fn resolve_origin_time(keyframe_times: &[f32], playhead: f32) -> Option<f32> {
    keyframe_times
        .iter()
        .copied()
        .filter(|time| *time <= playhead + ANCHOR_EPSILON)
        .fold(None, |best, time| {
            Some(best.map_or(time, |b: f32| b.max(time)))
        })
}

pub fn compute_velocities(mean_curve: &[f32], fps: f32) -> Vec<f32> {
    let n = mean_curve.len();
    if n < 2 || fps <= 0.0 {
        return vec![0.0; n];
    }

    let dt = 1.0 / fps;
    let mut velocities = vec![0.0_f32; n];
    velocities[0] = (mean_curve[1] - mean_curve[0]) / dt;
    velocities[n - 1] = (mean_curve[n - 1] - mean_curve[n - 2]) / dt;
    for i in 1..n - 1 {
        velocities[i] = (mean_curve[i + 1] - mean_curve[i - 1]) / (2.0 * dt);
    }
    velocities
}

pub fn continuity_offset(mean_curve: &[f32], origin_value: f32) -> f32 {
    origin_value - mean_curve.first().copied().unwrap_or(0.0)
}

pub struct ForecastPreview {
    pub mean_curve: Vec<f32>,
    pub continuity_offset: f32,
    pub velocities: Vec<f32>,
    #[cfg(feature = "python")]
    pub ghost_points: Vec<(f32, f32)>,
}

#[cfg(feature = "python")]
fn assemble_ghost_points(
    mean_curve: &[f32],
    continuity_offset: f32,
    origin: f32,
    frame_step: f32,
) -> Vec<(f32, f32)> {
    let offsets = future_sample_offsets();
    let mut points = Vec::with_capacity(SUGGESTION_FRAME_COUNT + 1);

    let origin_value = mean_curve.first().copied().unwrap_or(0.0) + continuity_offset;
    points.push((origin, origin_value));
    for i in suggestion_frame_indices() {
        if i >= mean_curve.len() {
            continue;
        }
        points.push((
            origin + offsets[i] as f32 * frame_step,
            mean_curve[i] + continuity_offset,
        ));
    }
    points
}

pub fn assemble_forecast(
    mean_curve: Vec<f32>,
    origin: f32,
    origin_value: f32,
    fps: f32,
    frame_step: f32,
) -> ForecastPreview {
    #[cfg(not(feature = "python"))]
    let _ = (origin, frame_step);

    let continuity_offset = continuity_offset(&mean_curve, origin_value);
    let velocities = compute_velocities(&mean_curve, fps);

    ForecastPreview {
        #[cfg(feature = "python")]
        ghost_points: assemble_ghost_points(&mean_curve, continuity_offset, origin, frame_step),
        mean_curve,
        continuity_offset,
        velocities,
    }
}

pub fn build_forecast_preview(
    session: &mut V2CurveCopilotSession,
    context: &[f32],
    fps: f32,
    origin: f32,
    origin_value: f32,
    frame_step: f32,
) -> Result<ForecastPreview> {
    let mean_curve = session.predict_mean_curve(V2CurveCopilotRequest { context, fps })?;
    Ok(assemble_forecast(
        mean_curve,
        origin,
        origin_value,
        fps,
        frame_step,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_offsets_end_at_origin() {
        let offsets = context_sample_offsets();
        assert_eq!(offsets[CONTEXT_LENGTH - 1], 0);
        assert_eq!(offsets[0], -(CONTEXT_LENGTH as i64 - 1));
    }

    #[test]
    fn future_offsets_start_one_after_origin() {
        let offsets = future_sample_offsets();
        assert_eq!(offsets[0], 1);
        assert_eq!(offsets[MAX_HORIZON - 1], MAX_HORIZON as i64);
    }

    #[test]
    fn resolve_origin_picks_latest_before_playhead() {
        let times = [0.5_f32, 1.0, 2.0, 3.0];
        assert_eq!(resolve_origin_time(&times, 2.5), Some(2.0));
        assert_eq!(resolve_origin_time(&times, 0.0), None);
    }

    #[test]
    fn continuity_offset_anchors_first_sample() {
        let mean = [10.0_f32, 11.0, 12.0];
        let offset = continuity_offset(&mean, 4.0);
        assert!((mean[0] + offset - 4.0).abs() < 1e-6);
    }

    #[test]
    fn velocities_match_central_difference() {
        let mean = [0.0_f32, 1.0, 4.0, 9.0];
        let v = compute_velocities(&mean, 2.0);
        assert!((v[0] - 2.0).abs() < 1e-6);
        assert!((v[1] - 4.0).abs() < 1e-6);
        assert!((v[3] - 10.0).abs() < 1e-6);
    }
}
