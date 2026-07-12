//! PyO3 wrappers for the debug file logger. Present only when the wheel is
//! built with the `debug-log` feature; production builds omit these functions
//! so the addon's `_debuglog.py` finds nothing to call and stays silent.

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

/// Open (create/append) the addon debug log at `path`.
#[pyfunction]
pub fn debug_log_init(path: &str) -> PyResult<()> {
    crate::debug_log::init(path).map_err(|e| PyOSError::new_err(e.to_string()))
}

/// Append one timestamped line to the addon debug log.
#[pyfunction]
pub fn debug_log(message: &str) {
    crate::debug_log::log_line(message);
}
