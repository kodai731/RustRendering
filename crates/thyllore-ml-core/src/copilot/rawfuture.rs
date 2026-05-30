use anyhow::{ensure, Result};
use ort::session::Session as OrtSession;
use ort::value::Tensor;

pub const CONTEXT_LENGTH: usize = 32;
pub const MAX_HORIZON: usize = 64;
pub const STD_FLOOR: f32 = 0.05;
pub const CLIP: f32 = 8.0;

const INPUT_CONTEXT: &str = "context";
const INPUT_FUTURE: &str = "future";
const INPUT_REVEAL_MASK: &str = "reveal_mask";
const INPUT_CONTEXT_TIMES: &str = "context_times";
const INPUT_FUTURE_TIMES: &str = "future_times";
const INPUT_CONTEXT_TANGENT: &str = "context_tangent";
const INPUT_FUTURE_TANGENT: &str = "future_tangent";
const OUTPUT_MEAN_CURVE: &str = "mean_curve";

/// Deploy inputs for `curve_copilot_20260531_rawfuture_v1_tangent.onnx` (RawFutureCombined, use_tangent=True).
///
/// `context` is the dense block of `CONTEXT_LENGTH` raw values immediately before the
/// edited region; `future` is the dense block of `MAX_HORIZON` raw values the model
/// predicts. Only the steps where `reveal_mask` is set are treated as known anchors —
/// the rest are reconstructed (e.g. lerp-through-anchors) so the finite-difference
/// tangent is defined densely. Values are raw (un-normalized): the deploy-safe z-score
/// is applied here and the output is denormalized back to this scale.
pub struct RawFutureRequest<'a> {
    pub context: &'a [f32],
    pub future: &'a [f32],
    pub reveal_mask: &'a [bool],
    pub fps: f32,
}

/// Per-row deploy-safe normalization statistics, retained to denormalize the output.
#[derive(Clone, Copy, Debug)]
pub struct DeploySafeStats {
    pub mu: f32,
    pub sd: f32,
}

struct RawFutureTensors {
    context: Tensor<f32>,
    future: Tensor<f32>,
    reveal_mask: Tensor<f32>,
    context_times: Tensor<f32>,
    future_times: Tensor<f32>,
    context_tangent: Tensor<f32>,
    future_tangent: Tensor<f32>,
    stats: DeploySafeStats,
}

pub struct RawFutureSession {
    ort: OrtSession,
}

impl RawFutureSession {
    pub fn from_onnx_path(path: &str) -> Result<Self> {
        let ort = OrtSession::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(Self { ort })
    }

    /// Run the MDN mixture-mean deploy path and return the denormalized dense curve
    /// (`MAX_HORIZON` values in the same units as `request.context`/`request.future`).
    pub fn predict_mean_curve(&mut self, request: RawFutureRequest<'_>) -> Result<Vec<f32>> {
        let tensors = build_rawfuture_tensors(&request)?;
        let stats = tensors.stats;

        let outputs = self.ort.run(ort::inputs![
            INPUT_CONTEXT => tensors.context,
            INPUT_FUTURE => tensors.future,
            INPUT_REVEAL_MASK => tensors.reveal_mask,
            INPUT_CONTEXT_TIMES => tensors.context_times,
            INPUT_FUTURE_TIMES => tensors.future_times,
            INPUT_CONTEXT_TANGENT => tensors.context_tangent,
            INPUT_FUTURE_TANGENT => tensors.future_tangent
        ])?;

        let (_shape, mean_norm) = outputs[OUTPUT_MEAN_CURVE].try_extract_tensor::<f32>()?;
        Ok(denormalize(mean_norm, stats))
    }
}

/// Deploy-safe statistics from KNOWN values only (context ⊕ revealed anchors).
///
/// Mirrors the training collate: the target scale is unknown at inference, so the row is
/// standardized by the mean/std of values the model can actually see. `std` is the
/// population standard deviation (matching numpy default), floored so a near-flat known
/// region cannot blow up the normalized magnitudes.
pub fn deploy_safe_stats(context: &[f32], future: &[f32], reveal_mask: &[bool]) -> DeploySafeStats {
    let mut known: Vec<f32> = context.to_vec();
    known.extend(
        future
            .iter()
            .zip(reveal_mask)
            .filter_map(|(&v, &revealed)| revealed.then_some(v)),
    );

    let count = known.len().max(1) as f32;
    let mu = known.iter().sum::<f32>() / count;
    let variance = known.iter().map(|&v| (v - mu).powi(2)).sum::<f32>() / count;
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

/// Seconds-based positions: context spans `[-(len-1)*dt ..= 0]`, future spans
/// `[dt ..= len*dt]`, matching the fps-agnostic time stamps the collate emits.
fn seconds_times(len: usize, offset: i64, dt: f32) -> Vec<f32> {
    (0..len).map(|i| (i as i64 + offset) as f32 * dt).collect()
}

/// Finite-difference derivative w.r.t. seconds, matching `numpy.gradient` with a uniform
/// coordinate array and the default `edge_order=1`: second-order central differences in
/// the interior, first-order one-sided differences at the two boundaries.
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

fn build_rawfuture_tensors(request: &RawFutureRequest<'_>) -> Result<RawFutureTensors> {
    ensure!(
        request.context.len() == CONTEXT_LENGTH,
        "context must have {CONTEXT_LENGTH} values, got {}",
        request.context.len()
    );
    ensure!(
        request.future.len() == MAX_HORIZON,
        "future must have {MAX_HORIZON} values, got {}",
        request.future.len()
    );
    ensure!(
        request.reveal_mask.len() == MAX_HORIZON,
        "reveal_mask must have {MAX_HORIZON} values, got {}",
        request.reveal_mask.len()
    );
    ensure!(
        request.fps > 0.0,
        "fps must be positive, got {}",
        request.fps
    );

    let stats = deploy_safe_stats(request.context, request.future, request.reveal_mask);
    let dt = 1.0 / request.fps;

    let context_norm = normalize_clip(request.context, stats);
    let future_norm = normalize_clip(request.future, stats);
    let context_times = seconds_times(CONTEXT_LENGTH, -(CONTEXT_LENGTH as i64) + 1, dt);
    let future_times = seconds_times(MAX_HORIZON, 1, dt);
    let context_tangent = finite_difference(&context_norm, dt);
    let future_tangent = finite_difference(&future_norm, dt);
    let reveal: Vec<f32> = request
        .reveal_mask
        .iter()
        .map(|&revealed| if revealed { 1.0 } else { 0.0 })
        .collect();

    let row_ctx = vec![1i64, CONTEXT_LENGTH as i64];
    let row_fut = vec![1i64, MAX_HORIZON as i64];
    Ok(RawFutureTensors {
        context: Tensor::from_array((row_ctx.clone(), context_norm))?,
        future: Tensor::from_array((row_fut.clone(), future_norm))?,
        reveal_mask: Tensor::from_array((row_fut.clone(), reveal))?,
        context_times: Tensor::from_array((row_ctx.clone(), context_times))?,
        future_times: Tensor::from_array((row_fut.clone(), future_times))?,
        context_tangent: Tensor::from_array((row_ctx, context_tangent))?,
        future_tangent: Tensor::from_array((row_fut, future_tangent))?,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deploy_safe_stats_use_known_values_only() {
        let context = vec![0.0_f32; CONTEXT_LENGTH];
        let mut future = vec![100.0_f32; MAX_HORIZON];
        future[0] = 4.0;
        let mut reveal = vec![false; MAX_HORIZON];
        reveal[0] = true;

        let stats = deploy_safe_stats(&context, &future, &reveal);
        let expected_mu = 4.0 / (CONTEXT_LENGTH as f32 + 1.0);
        assert!((stats.mu - expected_mu).abs() < 1e-6);
        assert!(stats.sd >= STD_FLOOR);
    }

    #[test]
    fn std_floor_clamps_flat_known_region() {
        let context = vec![42.0_f32; CONTEXT_LENGTH];
        let future = vec![42.0_f32; MAX_HORIZON];
        let reveal = vec![true; MAX_HORIZON];

        let stats = deploy_safe_stats(&context, &future, &reveal);
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
