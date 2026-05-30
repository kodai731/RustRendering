use pyo3::prelude::*;

#[cfg(feature = "debug-log")]
mod debuglog;
mod error;
mod rawfuture;

pub use rawfuture::{
    capabilities, forecast_sample_offsets, resolve_curve_copilot_model_path, resolve_origin_frame,
    PyRawFutureSession,
};

#[pymodule]
fn thyllore_ml_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__abi_marker__", thyllore_ml_api::ABI_MARKER)?;

    m.add_class::<PyRawFutureSession>()?;

    m.add_function(wrap_pyfunction!(rawfuture::capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(
        rawfuture::resolve_curve_copilot_model_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rawfuture::forecast_sample_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(rawfuture::resolve_origin_frame, m)?)?;

    #[cfg(feature = "debug-log")]
    {
        m.add_function(wrap_pyfunction!(debuglog::debug_log_init, m)?)?;
        m.add_function(wrap_pyfunction!(debuglog::debug_log, m)?)?;
    }

    Ok(())
}
