use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pub fn anyhow_to_pyerr(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{:#}", err))
}

pub fn shape_mismatch(field: &str, expected: usize, actual: usize) -> PyErr {
    PyValueError::new_err(format!(
        "{} shape mismatch: expected {} elements, got {}",
        field, expected, actual
    ))
}
