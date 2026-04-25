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

pub fn shape_mismatch_2d(field: &str, expected: (usize, usize), actual: (usize, usize)) -> PyErr {
    PyValueError::new_err(format!(
        "{} shape mismatch: expected {:?}, got {:?}",
        field, expected, actual
    ))
}

pub trait IntoPyResult<T> {
    fn into_py(self) -> PyResult<T>;
}

impl<T> IntoPyResult<T> for anyhow::Result<T> {
    fn into_py(self) -> PyResult<T> {
        self.map_err(anyhow_to_pyerr)
    }
}
