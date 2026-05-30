use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::error::{anyhow_to_pyerr, shape_mismatch};
use crate::copilot::forecast;
use crate::copilot::rawfuture::{RawFutureRequest, RawFutureSession, CONTEXT_LENGTH, MAX_HORIZON};

/// Feature ids the wheel supports, for the addon's `poll` UI gating. The addon
/// checks `"curve_forecast" in tml.capabilities()`.
#[pyfunction]
pub fn capabilities() -> Vec<&'static str> {
    vec!["curve_forecast"]
}

#[pyfunction]
pub fn deploy_fps() -> f32 {
    forecast::DEPLOY_FPS
}

/// Resolve the shared rawfuture curve-copilot model path (same source of truth
/// the engine uses). Returns ``None`` when unset/missing so the addon can fall
/// back to its bundled copy.
#[pyfunction]
pub fn resolve_curve_copilot_model_path() -> Option<String> {
    crate::model_path::resolve_rawfuture_curve_copilot_model_path()
}

/// Frame offsets (relative to the origin) to sample for the context and future
/// windows. The addon samples the FCurve at `origin + offset` for each.
#[pyfunction]
pub fn forecast_sample_offsets() -> (Vec<i64>, Vec<i64>) {
    (
        forecast::context_sample_offsets().to_vec(),
        forecast::future_sample_offsets().to_vec(),
    )
}

/// Latest keyframe time at or before `playhead` — the forecast origin.
#[pyfunction]
pub fn resolve_origin_frame(keyframe_times: Vec<f32>, playhead: f32) -> Option<f32> {
    forecast::resolve_origin_time(&keyframe_times, playhead)
}

#[pyclass(name = "PyRawFutureSession", module = "thyllore_ml_core")]
pub struct PyRawFutureSession {
    inner: RawFutureSession,
}

#[pymethods]
impl PyRawFutureSession {
    #[staticmethod]
    fn from_onnx_path(path: &str) -> PyResult<Self> {
        let inner = RawFutureSession::from_onnx_path(path).map_err(anyhow_to_pyerr)?;
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
        future: PyReadonlyArray1<'py, f32>,
        reveal_mask: PyReadonlyArray1<'py, bool>,
        fps: f32,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let context = context.as_slice()?;
        let future = future.as_slice()?;
        let reveal_mask = reveal_mask.as_slice()?;

        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }
        if future.len() != MAX_HORIZON {
            return Err(shape_mismatch("future", MAX_HORIZON, future.len()));
        }
        if reveal_mask.len() != MAX_HORIZON {
            return Err(shape_mismatch(
                "reveal_mask",
                MAX_HORIZON,
                reveal_mask.len(),
            ));
        }

        let mean_curve = py
            .allow_threads(|| {
                self.inner.predict_mean_curve(RawFutureRequest {
                    context,
                    future,
                    reveal_mask,
                    fps,
                })
            })
            .map_err(anyhow_to_pyerr)?;

        Ok(mean_curve.into_pyarray(py))
    }

    /// Run the forecast and return the ghost preview polyline `[(frame, value)]`
    /// in FCurve space. Inputs are plain lists so the addon needs no numpy; all
    /// numeric work happens in the shared Rust core.
    #[allow(clippy::too_many_arguments)]
    fn build_forecast_preview(
        &mut self,
        py: Python<'_>,
        context: Vec<f32>,
        future: Vec<f32>,
        reveal_mask: Vec<bool>,
        fps: f32,
        origin: f32,
        origin_value: f32,
        frame_step: f32,
    ) -> PyResult<Vec<(f32, f32)>> {
        if context.len() != CONTEXT_LENGTH {
            return Err(shape_mismatch("context", CONTEXT_LENGTH, context.len()));
        }
        if future.len() != MAX_HORIZON {
            return Err(shape_mismatch("future", MAX_HORIZON, future.len()));
        }
        if reveal_mask.len() != MAX_HORIZON {
            return Err(shape_mismatch(
                "reveal_mask",
                MAX_HORIZON,
                reveal_mask.len(),
            ));
        }

        let preview = py
            .allow_threads(|| {
                forecast::build_forecast_preview(
                    &mut self.inner,
                    &context,
                    &future,
                    &reveal_mask,
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
