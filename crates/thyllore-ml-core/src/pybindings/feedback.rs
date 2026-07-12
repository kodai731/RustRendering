use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};

use super::error::anyhow_to_pyerr;
use crate::degrade::now_unix;
use crate::feedback;
use crate::feedback::{FeedbackEndpoint, FeedbackRecordInputs};

fn pythonize_to_pyerr(err: pythonize::PythonizeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
#[pyo3(signature = (
    model_hash, channel_kind, array_index, scene_fps, deploy_fps, frame_step,
    origin_value, context, prediction, ground_truth=None, signal="ignored", ts=None
))]
#[allow(clippy::too_many_arguments)]
pub fn build_feedback_record<'py>(
    py: Python<'py>,
    model_hash: &str,
    channel_kind: &str,
    array_index: u32,
    scene_fps: f64,
    deploy_fps: f64,
    frame_step: f64,
    origin_value: f64,
    context: Vec<f64>,
    prediction: Vec<f64>,
    ground_truth: Option<Vec<f64>>,
    signal: &str,
    ts: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    let record = feedback::build_feedback_record(FeedbackRecordInputs {
        model_hash,
        channel_kind,
        array_index,
        scene_fps,
        deploy_fps,
        frame_step,
        origin_value,
        context: &context,
        prediction: &prediction,
        ground_truth,
        signal,
        ts: ts.unwrap_or_else(now_unix),
    });
    pythonize(py, &record).map_err(pythonize_to_pyerr)
}

#[pyfunction]
pub fn send_feedback_batch<'py>(
    py: Python<'py>,
    feedback_url: &str,
    ingest_token: &str,
    anon_id: &str,
    records: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let records: Vec<serde_json::Value> = depythonize(&records).map_err(pythonize_to_pyerr)?;
    let endpoint = FeedbackEndpoint {
        url: feedback_url.to_string(),
        ingest_token: ingest_token.to_string(),
        anon_id: anon_id.to_string(),
    };

    let response = py
        .detach(|| feedback::send_feedback_batch(&endpoint, &records))
        .map_err(anyhow_to_pyerr)?;

    let payload = serde_json::json!({
        "full_token": response.full_token,
        "exp": response.exp,
    });
    pythonize(py, &payload).map_err(pythonize_to_pyerr)
}

#[pyfunction]
pub fn send_message(
    py: Python<'_>,
    feedback_url: &str,
    ingest_token: &str,
    anon_id: &str,
    text: &str,
    addon_version: &str,
) -> PyResult<()> {
    let endpoint = FeedbackEndpoint {
        url: feedback_url.to_string(),
        ingest_token: ingest_token.to_string(),
        anon_id: anon_id.to_string(),
    };

    py.detach(|| feedback::send_message(&endpoint, text, addon_version, now_unix()))
        .map_err(anyhow_to_pyerr)
}
