use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::license;

#[pyfunction]
pub fn refresh_license<'py>(
    py: Python<'py>,
    license_endpoint: &str,
    license_key: &str,
    device_id: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let result = py.detach(|| license::refresh_license(license_endpoint, license_key, device_id));

    let dict = PyDict::new(py);
    match result {
        Ok(session) => {
            dict.set_item("ok", true)?;
            dict.set_item("full_token", session.full_token)?;
            dict.set_item("exp", session.exp)?;
        }
        Err(refusal) => {
            dict.set_item("ok", false)?;
            dict.set_item("reason", refusal.reason)?;
            dict.set_item("discard_cached_token", refusal.discard_cached_token)?;
        }
    }
    Ok(dict)
}
