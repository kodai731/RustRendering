//! L3 boundary adapter — module-level entries that go through the
//! [`thyllore_ml_api::MlOps`] trait via [`crate::MlCoreImpl`].
//!
//! The wheel keeps a single shared [`MlCoreImpl`] instance for the lifetime
//! of the process; sessions are cached internally per onnx model path. This
//! module exposes only what the L4 Blender operator needs:
//!
//! - `__abi_marker__` (re-exported by `mod.rs`) for the addon's startup check.
//! - `capabilities()` for `Operator.poll()` to gate UI elements on what the
//!   wheel actually implements.
//! - `call_op_json(name, payload)` as the generic dispatch escape hatch for
//!   features added without bumping the ABI marker.

use std::sync::OnceLock;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pythonize::{depythonize, pythonize};

use thyllore_ml_api::{MlError, MlOps};

use crate::MlCoreImpl;

fn shared_impl() -> &'static MlCoreImpl {
    static IMPL: OnceLock<MlCoreImpl> = OnceLock::new();
    IMPL.get_or_init(MlCoreImpl::new)
}

#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyObject {
    let caps = shared_impl().capabilities();
    let list: Vec<&str> = caps.iter().map(|s| *s).collect();
    list.into_pyobject(py)
        .expect("Vec<&str> always converts to PyList")
        .unbind()
        .into_any()
}

#[pyfunction]
pub fn call_op_json<'py>(
    py: Python<'py>,
    op_name: &str,
    payload: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let payload_value: serde_json::Value = depythonize(&payload)
        .map_err(|e| PyValueError::new_err(format!("payload not JSON-compatible: {e}")))?;

    let payload_bytes = serde_json::to_vec(&payload_value)
        .map_err(|e| PyRuntimeError::new_err(format!("payload encode failed: {e}")))?;

    let response_bytes = py
        .allow_threads(|| shared_impl().call_op(op_name, &payload_bytes))
        .map_err(ml_error_to_pyerr)?;

    let response_value: serde_json::Value = serde_json::from_slice(&response_bytes)
        .map_err(|e| PyRuntimeError::new_err(format!("response decode failed: {e}")))?;

    pythonize(py, &response_value).map_err(|e| {
        PyRuntimeError::new_err(format!("response could not be returned to Python: {e}"))
    })
}

fn ml_error_to_pyerr(err: MlError) -> PyErr {
    match err {
        MlError::InvalidRequest(msg) => PyValueError::new_err(format!("invalid request: {msg}")),
        MlError::SchemaMismatch { op, message } => {
            PyValueError::new_err(format!("schema mismatch in '{op}': {message}"))
        }
        MlError::UnsupportedOp(name) => {
            PyRuntimeError::new_err(format!("operation '{name}' is not supported by this build"))
        }
        MlError::InferenceFailed(msg) => {
            PyRuntimeError::new_err(format!("inference failed: {msg}"))
        }
        MlError::Internal(msg) => PyRuntimeError::new_err(format!("internal error: {msg}")),
    }
}
