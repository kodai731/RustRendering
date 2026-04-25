use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use super::error::{anyhow_to_pyerr, IntoPyResult};
use crate::copilot::session::Session;

#[pyclass(name = "PySession", module = "thyllore_ml_core")]
pub struct PySession {
    inner: Session,
}

#[pymethods]
impl PySession {
    #[staticmethod]
    fn from_onnx_path(path: &str) -> PyResult<Self> {
        let inner = Session::from_onnx_path(path).map_err(anyhow_to_pyerr)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (context, property_type_id, topology_features, bone_name_tokens, query_times, curve_window))]
    fn run_curve_copilot<'py>(
        &mut self,
        py: Python<'py>,
        context: PyReadonlyArray1<'py, f32>,
        property_type_id: u32,
        topology_features: PyReadonlyArray1<'py, f32>,
        bone_name_tokens: PyReadonlyArray1<'py, i64>,
        query_times: PyReadonlyArray1<'py, f32>,
        curve_window: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)> {
        let ctx = context.as_slice()?.to_vec();
        let topo = topology_features.as_slice()?.to_vec();
        let tokens = bone_name_tokens.as_slice()?.to_vec();
        let qt = query_times.as_slice()?.to_vec();
        let cw = curve_window.as_slice()?.to_vec();

        let steps = py
            .allow_threads(|| {
                self.inner
                    .run_curve_copilot(&ctx, property_type_id, &topo, &tokens, &qt, &cw)
            })
            .into_py()?;

        let n = steps.len();
        let mut pred_flat = Vec::with_capacity(n * 5);
        let mut conf_flat = Vec::with_capacity(n);
        for s in &steps {
            pred_flat.push(s.value);
            pred_flat.push(s.tangent_in.0);
            pred_flat.push(s.tangent_in.1);
            pred_flat.push(s.tangent_out.0);
            pred_flat.push(s.tangent_out.1);
            conf_flat.push(s.confidence);
        }

        let pred_arr = pred_flat.into_pyarray(py).reshape([n, 5])?;
        let conf_arr = conf_flat.into_pyarray(py);
        Ok((pred_arr, conf_arr))
    }

    fn run_curve_predict<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let input_vec = input.as_slice()?.to_vec();
        let output = py
            .allow_threads(|| self.inner.run_curve_predict(&input_vec))
            .into_py()?;
        Ok(output.into_pyarray(py))
    }
}
