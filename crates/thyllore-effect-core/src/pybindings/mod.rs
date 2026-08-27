use crate::flame::{
    apply_flame_preset, build_flame_ubo, overwrite_persisted_fields, parameter_owner,
    refresh_flame_coefficients, FlameBaked, FlameEffect, FlameTemporalAccum, FlameUBO,
    FLAME_PRESET_NAMES, FLAME_UI_PARAMS,
};
use cgmath::{Quaternion, Vector3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyfunction]
fn flame_preset_names() -> Vec<&'static str> {
    FLAME_PRESET_NAMES.to_vec()
}

#[pyfunction]
fn flame_ui_params(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let default_dict: Bound<'_, PyDict> =
        pythonize::pythonize(py, &FlameEffect::default())?.cast_into::<PyDict>()?;
    let list = PyList::empty(py);
    for param in FLAME_UI_PARAMS {
        let owner = parameter_owner(param.name);

        let dict = PyDict::new(py);
        dict.set_item("name", param.name)?;
        dict.set_item("label", param.display_label())?;
        dict.set_item("min", param.min)?;
        dict.set_item("max", param.max)?;
        dict.set_item("format", param.format)?;
        dict.set_item("tooltip", param.tooltip)?;

        let Some(default_value) = default_dict.get_item(param.name)? else {
            continue;
        };
        dict.set_item("default", default_value)?;

        match owner {
            Some(crate::flame::ParameterOwner::Frame) => {
                dict.set_item("owner", "frame")?;
            }
            Some(crate::flame::ParameterOwner::Shape) => {
                dict.set_item("owner", "shape")?;
            }
            Some(crate::flame::ParameterOwner::Style) => {
                dict.set_item("owner", "style")?;
            }
            None => {
                dict.set_item("owner", "unknown")?;
            }
        }

        list.append(dict)?;
    }
    Ok(list)
}

#[pyfunction]
fn flame_preset_params<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let mut effect = FlameEffect::default();
    if !apply_flame_preset(&mut effect, name) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown preset: {}",
            name
        )));
    }

    let dict: Bound<'py, PyDict> = pythonize::pythonize(py, &effect)?.cast_into::<PyDict>()?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (params, time, position, rotation, light_position=None, frame_index=0))]
fn pack_flame_ubo(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    time: f32,
    position: [f32; 3],
    rotation: [f32; 4],
    light_position: Option<[f32; 3]>,
    frame_index: u64,
) -> PyResult<Vec<u8>> {
    let merged: Bound<'_, PyDict> =
        pythonize::pythonize(py, &FlameEffect::default())?.cast_into::<PyDict>()?;
    for key in params.keys() {
        if !merged.contains(&key)? {
            let key_str: &str = key.extract()?;
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "unknown parameter: {}",
                key_str
            )));
        }
    }

    merged.update(params.as_mapping())?;

    let source: FlameEffect = pythonize::depythonize(&merged).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "failed to deserialize parameters: {}",
            e
        ))
    })?;

    let mut effect = FlameEffect::default();
    overwrite_persisted_fields(&mut effect, &source);

    effect.time = time;
    effect.position = Vector3::new(position[0], position[1], position[2]);
    effect.rotation = Quaternion::new(rotation[0], rotation[1], rotation[2], rotation[3]);

    if let Some(lp) = light_position {
        effect.light_position_world = Vector3::new(lp[0], lp[1], lp[2]);
    }

    let baked = FlameBaked::default();
    refresh_flame_coefficients(&mut effect, &baked);

    let temporal = FlameTemporalAccum {
        frame_index,
        ..Default::default()
    };
    let ubo = build_flame_ubo(&effect, &baked, &temporal);

    let bytes = unsafe {
        std::slice::from_raw_parts(
            &ubo as *const FlameUBO as *const u8,
            std::mem::size_of::<FlameUBO>(),
        )
    };
    Ok(bytes.to_vec())
}

#[pyfunction]
fn flame_ubo_size() -> usize {
    std::mem::size_of::<FlameUBO>()
}

#[pymodule]
fn thyllore_effect_core(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flame_preset_names, m)?)?;
    m.add_function(wrap_pyfunction!(flame_ui_params, m)?)?;
    m.add_function(wrap_pyfunction!(flame_preset_params, m)?)?;
    m.add_function(wrap_pyfunction!(pack_flame_ubo, m)?)?;
    m.add_function(wrap_pyfunction!(flame_ubo_size, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
