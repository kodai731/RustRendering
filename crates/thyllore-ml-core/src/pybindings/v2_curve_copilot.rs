use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::error::{anyhow_to_pyerr, shape_mismatch};
use crate::copilot::v2::forecast;
use crate::copilot::v2::inference::{
    V2CurveCopilotRequest, V2CurveCopilotSession, CONTEXT_LENGTH, MAX_HORIZON,
};

#[pyfunction]
pub fn capabilities() -> Vec<&'static str> {
    vec!["curve_forecast"]
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

    fn predict_mean_curve<'py>(
        &mut self,
        py: Python<'py>,
        context: PyReadonlyArray1<'py, f32>,
        fps: f32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let context = context.as_slice()?;

        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }

        let mean_curve = py
            .detach(|| {
                self.inner
                    .predict_mean_curve(V2CurveCopilotRequest { context, fps })
            })
            .map_err(anyhow_to_pyerr)?;

        Ok(mean_curve.into_pyarray(py))
    }

    fn build_forecast_preview(
        &mut self,
        py: Python<'_>,
        context: Vec<f32>,
        fps: f32,
        origin: f32,
        origin_value: f32,
        frame_step: f32,
    ) -> PyResult<Vec<(f32, f32)>> {
        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }

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
