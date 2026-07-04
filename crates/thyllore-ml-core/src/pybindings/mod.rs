use pyo3::prelude::*;

#[cfg(feature = "debug-log")]
mod debuglog;
mod error;
mod v2_curve_copilot;

pub use v2_curve_copilot::PyV2CurveCopilotSession;

#[pymodule]
fn thyllore_ml_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__abi_marker__", thyllore_ml_api::ABI_MARKER)?;

    m.add_class::<PyV2CurveCopilotSession>()?;

    m.add_function(wrap_pyfunction!(v2_curve_copilot::capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(v2_curve_copilot::deploy_fps, m)?)?;
    m.add_function(wrap_pyfunction!(
        v2_curve_copilot::resolve_curve_copilot_model_path,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        v2_curve_copilot::forecast_sample_offsets,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(v2_curve_copilot::resolve_origin_frame, m)?)?;

    #[cfg(feature = "debug-log")]
    {
        m.add_function(wrap_pyfunction!(debuglog::debug_log_init, m)?)?;
        m.add_function(wrap_pyfunction!(debuglog::debug_log, m)?)?;
    }

    Ok(())
}
