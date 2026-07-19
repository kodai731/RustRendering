use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::error::{anyhow_to_pyerr, shape_mismatch};
use crate::copilot::v2::forecast;
use crate::copilot::v2::inference::{
    V2CurveCopilotRequest, V2CurveCopilotSession, CONTEXT_LENGTH, MAX_HORIZON,
};
use crate::degrade::{degrade_context_window, now_unix, DegradeGate, DEGRADED_CONTEXT_LENGTH};
use crate::mode::CurveCopilotMode;

fn degrade_gate() -> &'static DegradeGate {
    static GATE: OnceLock<DegradeGate> = OnceLock::new();
    GATE.get_or_init(DegradeGate::from_build_env)
}

fn parse_mode(mode: Option<&str>) -> PyResult<Option<CurveCopilotMode>> {
    match mode {
        None => Ok(None),
        Some(value) => CurveCopilotMode::parse(value).map(Some).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown curve copilot mode: {value} (expected full, degrade or private)"
            ))
        }),
    }
}

/// Private is the zero-network distribution: full context unconditionally,
/// no token gate. Degrade never consults the gate either. Full (and callers
/// that pass no mode) stay token-gated via [`DegradeGate`].
fn should_degrade_context(mode: Option<CurveCopilotMode>, full_token: Option<&str>) -> bool {
    match mode {
        Some(CurveCopilotMode::Private) => false,
        Some(CurveCopilotMode::Degrade) => true,
        Some(CurveCopilotMode::Full) | None => {
            degrade_gate().should_degrade(full_token, now_unix())
        }
    }
}

fn apply_context_degrade(
    context: &mut [f32],
    mode: Option<CurveCopilotMode>,
    full_token: Option<&str>,
) {
    if should_degrade_context(mode, full_token) {
        degrade_context_window(context);
    }
}

#[pyfunction]
pub fn capabilities() -> Vec<&'static str> {
    vec!["curve_forecast"]
}

#[pyfunction]
#[pyo3(signature = (full_token=None, mode=None))]
pub fn effective_context_length(full_token: Option<&str>, mode: Option<&str>) -> PyResult<usize> {
    let mode = parse_mode(mode)?;
    Ok(if should_degrade_context(mode, full_token) {
        DEGRADED_CONTEXT_LENGTH
    } else {
        CONTEXT_LENGTH
    })
}

#[pyfunction]
pub fn degraded_context_length() -> usize {
    DEGRADED_CONTEXT_LENGTH
}

#[pyfunction]
pub fn deploy_fps() -> f32 {
    forecast::DEPLOY_FPS
}

#[pyfunction]
pub fn resolve_curve_copilot_model_path() -> Option<String> {
    crate::model_path::resolve_v2_curve_copilot_model_path()
}

#[pyfunction]
pub fn forecast_sample_offsets() -> (Vec<i64>, Vec<i64>) {
    (
        forecast::context_sample_offsets().to_vec(),
        forecast::future_sample_offsets().to_vec(),
    )
}

#[pyfunction]
pub fn resolve_origin_frame(keyframe_times: Vec<f32>, playhead: f32) -> Option<f32> {
    forecast::resolve_origin_time(&keyframe_times, playhead)
}

#[pyclass(name = "PyV2CurveCopilotSession", module = "thyllore_ml_core")]
pub struct PyV2CurveCopilotSession {
    inner: V2CurveCopilotSession,
}

#[pymethods]
impl PyV2CurveCopilotSession {
    #[staticmethod]
    fn from_onnx_path(path: &str) -> PyResult<Self> {
        let inner = V2CurveCopilotSession::from_onnx_path(path).map_err(anyhow_to_pyerr)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn context_length() -> usize {
        CONTEXT_LENGTH
    }

    #[staticmethod]
    fn max_horizon() -> usize {
        MAX_HORIZON
    }

    #[pyo3(signature = (context, fps, full_token=None, mode=None))]
    fn predict_mean_curve<'py>(
        &mut self,
        py: Python<'py>,
        context: PyReadonlyArray1<'py, f32>,
        fps: f32,
        full_token: Option<&str>,
        mode: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mode = parse_mode(mode)?;
        let mut context = context.as_slice()?.to_vec();

        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }
        apply_context_degrade(&mut context, mode, full_token);

        let mean_curve = py
            .detach(|| {
                self.inner.predict_mean_curve(V2CurveCopilotRequest {
                    context: &context,
                    fps,
                })
            })
            .map_err(anyhow_to_pyerr)?;

        Ok(mean_curve.into_pyarray(py))
    }

    #[pyo3(signature = (context, fps, origin, origin_value, frame_step, full_token=None, mode=None))]
    fn build_forecast_preview(
        &mut self,
        py: Python<'_>,
        mut context: Vec<f32>,
        fps: f32,
        origin: f32,
        origin_value: f32,
        frame_step: f32,
        full_token: Option<&str>,
        mode: Option<&str>,
    ) -> PyResult<Vec<(f32, f32)>> {
        let mode = parse_mode(mode)?;
        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }
        apply_context_degrade(&mut context, mode, full_token);

        let preview = py
            .detach(|| {
                forecast::build_forecast_preview(
                    &mut self.inner,
                    &context,
                    fps,
                    origin,
                    origin_value,
                    frame_step,
                )
            })
            .map_err(anyhow_to_pyerr)?;

        Ok(preview.ghost_points)
    }
}
