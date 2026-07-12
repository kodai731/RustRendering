use anyhow::{ensure, Result};
use ort::session::Session as OrtSession;
use ort::value::Tensor;

pub const CONTEXT_LENGTH: usize = 64;
pub const MAX_HORIZON: usize = 64;
pub const STD_FLOOR: f32 = 0.05;
pub const CLIP: f32 = 8.0;

const INPUT_CONTEXT: &str = "context";
const INPUT_CONTEXT_TIMES: &str = "context_times";
const INPUT_FUTURE_TIMES: &str = "future_times";
const INPUT_CONTEXT_TANGENT: &str = "context_tangent";
const OUTPUT_MEAN_CURVE: &str = "mean_curve";

pub struct V2CurveCopilotRequest<'a> {
    pub context: &'a [f32],
    pub fps: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DeploySafeStats {
    pub mu: f32,
    pub sd: f32,
}

struct V2CurveCopilotTensors {
    context: Tensor<f32>,
    context_times: Tensor<f32>,
    future_times: Tensor<f32>,
    context_tangent: Tensor<f32>,
    stats: DeploySafeStats,
}

pub struct V2CurveCopilotSession {
    ort: OrtSession,
}

impl V2CurveCopilotSession {
    pub fn from_onnx_path(path: &str) -> Result<Self> {
        let ort = OrtSession::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(Self { ort })
    }

    pub fn predict_mean_curve(&mut self, request: V2CurveCopilotRequest<'_>) -> Result<Vec<f32>> {
        let tensors = build_v2_curve_copilot_tensors(&request)?;
        let stats = tensors.stats;

        let outputs = self.ort.run(ort::inputs![
            INPUT_CONTEXT => tensors.context,
            INPUT_CONTEXT_TIMES => tensors.context_times,
            INPUT_FUTURE_TIMES => tensors.future_times,
            INPUT_CONTEXT_TANGENT => tensors.context_tangent
        ])?;

        let (_shape, mean_norm) = outputs[OUTPUT_MEAN_CURVE].try_extract_tensor::<f32>()?;
        Ok(denormalize(mean_norm, stats))
    }

    pub fn run_raw(
        &mut self,
        context: &[f32],
        context_times: &[f32],
        future_times: &[f32],
        context_tangent: &[f32],
    ) -> Result<Vec<f32>> {
        ensure!(
            context.len() == CONTEXT_LENGTH
                && context_times.len() == CONTEXT_LENGTH
                && future_times.len() == MAX_HORIZON
                && context_tangent.len() == CONTEXT_LENGTH,
            "raw inputs must have context/context_times/context_tangent len {CONTEXT_LENGTH} \
             and future_times len {MAX_HORIZON}"
        );

        let row_ctx = vec![1i64, CONTEXT_LENGTH as i64];
        let row_fut = vec![1i64, MAX_HORIZON as i64];
        let outputs = self.ort.run(ort::inputs![
            INPUT_CONTEXT => Tensor::from_array((row_ctx.clone(), context.to_vec()))?,
            INPUT_CONTEXT_TIMES => Tensor::from_array((row_ctx.clone(), context_times.to_vec()))?,
            INPUT_FUTURE_TIMES => Tensor::from_array((row_fut, future_times.to_vec()))?,
            INPUT_CONTEXT_TANGENT => Tensor::from_array((row_ctx, context_tangent.to_vec()))?
        ])?;

        let (_shape, mean_curve) = outputs[OUTPUT_MEAN_CURVE].try_extract_tensor::<f32>()?;
        Ok(mean_curve.to_vec())
    }
}

pub fn deploy_safe_stats(context: &[f32]) -> DeploySafeStats {
    let count = context.len().max(1) as f32;
    let mu = context.iter().sum::<f32>() / count;
    let variance = context.iter().map(|&v| (v - mu).powi(2)).sum::<f32>() / count;
    let sd = variance.sqrt().max(STD_FLOOR);
    DeploySafeStats { mu, sd }
}

fn normalize_clip(values: &[f32], stats: DeploySafeStats) -> Vec<f32> {
    values
        .iter()
        .map(|&v| ((v - stats.mu) / stats.sd).clamp(-CLIP, CLIP))
        .collect()
}

fn denormalize(values: &[f32], stats: DeploySafeStats) -> Vec<f32> {
    values.iter().map(|&v| v * stats.sd + stats.mu).collect()
}

fn seconds_times(len: usize, offset: i64, dt: f32) -> Vec<f32> {
    (0..len).map(|i| (i as i64 + offset) as f32 * dt).collect()
}

fn finite_difference(values: &[f32], dt: f32) -> Vec<f32> {
    let n = values.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut gradient = vec![0.0_f32; n];
    gradient[0] = (values[1] - values[0]) / dt;
    gradient[n - 1] = (values[n - 1] - values[n - 2]) / dt;
    for i in 1..n - 1 {
        gradient[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt);
    }
    gradient
}

fn build_v2_curve_copilot_tensors(
    request: &V2CurveCopilotRequest<'_>,
) -> Result<V2CurveCopilotTensors> {
    ensure!(
        request.context.len() == CONTEXT_LENGTH,
        "context must have {CONTEXT_LENGTH} values, got {}",
        request.context.len()
    );
    ensure!(
        request.fps > 0.0,
        "fps must be positive, got {}",
        request.fps
    );

    let stats = deploy_safe_stats(request.context);
    let dt = 1.0 / request.fps;

    let context_norm = normalize_clip(request.context, stats);
    let context_times = seconds_times(CONTEXT_LENGTH, -(CONTEXT_LENGTH as i64) + 1, dt);
    let future_times = seconds_times(MAX_HORIZON, 1, dt);
    let context_tangent = finite_difference(&context_norm, dt);

    let row_ctx = vec![1i64, CONTEXT_LENGTH as i64];
    let row_fut = vec![1i64, MAX_HORIZON as i64];
    Ok(V2CurveCopilotTensors {
        context: Tensor::from_array((row_ctx.clone(), context_norm))?,
        context_times: Tensor::from_array((row_ctx.clone(), context_times))?,
        future_times: Tensor::from_array((row_fut, future_times))?,
        context_tangent: Tensor::from_array((row_ctx, context_tangent))?,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deploy_safe_stats_use_context_only() {
        let mut context = vec![0.0_f32; CONTEXT_LENGTH];
        context[0] = 4.0;

        let stats = deploy_safe_stats(&context);
        let expected_mu = 4.0 / CONTEXT_LENGTH as f32;
        assert!((stats.mu - expected_mu).abs() < 1e-6);
        assert!(stats.sd >= STD_FLOOR);
    }

    #[test]
    fn std_floor_clamps_flat_context() {
        let context = vec![42.0_f32; CONTEXT_LENGTH];

        let stats = deploy_safe_stats(&context);
        assert!((stats.mu - 42.0).abs() < 1e-4);
        assert!((stats.sd - STD_FLOOR).abs() < 1e-7);
    }

    #[test]
    fn finite_difference_matches_numpy_gradient() {
        let values = [0.0_f32, 1.0, 4.0, 9.0, 16.0];
        let gradient = finite_difference(&values, 0.5);
        assert_eq!(gradient, vec![2.0, 4.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn seconds_times_span_expected_ranges() {
        let dt = 0.25;
        let ctx = seconds_times(CONTEXT_LENGTH, -(CONTEXT_LENGTH as i64) + 1, dt);
        let fut = seconds_times(MAX_HORIZON, 1, dt);
        assert!((ctx[CONTEXT_LENGTH - 1] - 0.0).abs() < 1e-6);
        assert!((ctx[0] - (-(CONTEXT_LENGTH as f32 - 1.0) * dt)).abs() < 1e-6);
        assert!((fut[0] - dt).abs() < 1e-6);
        assert!((fut[MAX_HORIZON - 1] - (MAX_HORIZON as f32 * dt)).abs() < 1e-6);
    }

    #[test]
    fn normalize_then_denormalize_round_trips_within_clip() {
        let stats = DeploySafeStats { mu: 1.5, sd: 2.0 };
        let values = vec![1.5_f32, 3.5, -0.5];
        let restored = denormalize(&normalize_clip(&values, stats), stats);
        for (a, b) in values.iter().zip(&restored) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
