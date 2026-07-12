use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::error::{anyhow_to_pyerr, shape_mismatch};
use crate::copilot::v2::forecast;
use crate::copilot::v2::inference::{
    V2CurveCopilotRequest, V2CurveCopilotSession, CONTEXT_LENGTH, MAX_HORIZON,
};
use crate::degrade::{degrade_context_window, now_unix, DegradeGate, DEGRADED_CONTEXT_LENGTH};

fn degrade_gate() -> &'static DegradeGate {
    static GATE: OnceLock<DegradeGate> = OnceLock::new();
    GATE.get_or_init(DegradeGate::from_build_env)
}

fn apply_degrade_gate(context: &mut [f32], unlock_token: Option<&str>) {
    if degrade_gate().should_degrade(unlock_token, now_unix()) {
        degrade_context_window(context);
    }
}

#[pyfunction]
pub fn capabilities() -> Vec<&'static str> {
    vec!["curve_forecast"]
}

#[pyfunction]
#[pyo3(signature = (unlock_token=None))]
pub fn effective_context_length(unlock_token: Option<&str>) -> usize {
    degrade_gate().effective_context_length(unlock_token, now_unix())
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

    #[pyo3(signature = (context, fps, unlock_token=None))]
    fn predict_mean_curve<'py>(
        &mut self,
        py: Python<'py>,
        context: PyReadonlyArray1<'py, f32>,
        fps: f32,
        unlock_token: Option<&str>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let mut context = context.as_slice()?.to_vec();

        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }
        apply_degrade_gate(&mut context, unlock_token);

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

    #[pyo3(signature = (context, fps, origin, origin_value, frame_step, unlock_token=None))]
    fn build_forecast_preview(
        &mut self,
        py: Python<'_>,
        mut context: Vec<f32>,
        fps: f32,
        origin: f32,
        origin_value: f32,
        frame_step: f32,
        unlock_token: Option<&str>,
    ) -> PyResult<Vec<(f32, f32)>> {
        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }
        apply_degrade_gate(&mut context, unlock_token);

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
