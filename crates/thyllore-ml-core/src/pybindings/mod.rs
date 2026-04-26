use pyo3::prelude::*;

mod error;
mod functions;
mod mlops;
mod session;
mod skeleton;

pub use session::PySession;
pub use skeleton::PySkeleton;

#[pymodule]
fn thyllore_ml_core(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__abi_marker__", thyllore_ml_api::ABI_MARKER)?;

    m.add_class::<PySkeleton>()?;
    m.add_class::<PySession>()?;

    m.add_function(wrap_pyfunction!(functions::compute_topology, m)?)?;
    m.add_function(wrap_pyfunction!(functions::tokenize_bone_names, m)?)?;
    m.add_function(wrap_pyfunction!(functions::tokenize_bone_name_string, m)?)?;
    m.add_function(wrap_pyfunction!(functions::flatten_context, m)?)?;
    m.add_function(wrap_pyfunction!(functions::sample_window, m)?)?;
    m.add_function(wrap_pyfunction!(functions::generate_query_times, m)?)?;

    m.add_function(wrap_pyfunction!(mlops::capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(mlops::call_op_json, m)?)?;

    Ok(())
}
