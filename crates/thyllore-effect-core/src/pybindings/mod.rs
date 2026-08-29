use crate::flame::{
    apply_flame_preset, build_flame_model_matrix, build_flame_ubo, flame_bend_offset,
    flame_local_bounds, flame_local_bounds_corners, flame_proxy_pad, flame_support_scale,
    overwrite_persisted_fields, parameter_owner, refresh_flame_coefficients, FlameBaked,
    FlameEffect, FlameTemporalAccum, FlameUBO, FLAME_PRESET_NAMES, FLAME_UI_PARAMS,
};
use cgmath::{Quaternion, Vector3, Vector4};
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

fn build_effect_from_params(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    time: f32,
    position: [f32; 3],
    rotation: [f32; 4],
    light_position: Option<[f32; 3]>,
) -> PyResult<(FlameEffect, FlameBaked)> {
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
    Ok((effect, baked))
}

fn build_ubo_from_params(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    time: f32,
    position: [f32; 3],
    rotation: [f32; 4],
    light_position: Option<[f32; 3]>,
    frame_index: u64,
) -> PyResult<FlameUBO> {
    let (effect, baked) =
        build_effect_from_params(py, params, time, position, rotation, light_position)?;

    let temporal = FlameTemporalAccum {
        frame_index,
        ..Default::default()
    };
    Ok(build_flame_ubo(&effect, &baked, &temporal))
}

/// World-space corners of the shell proxy box, the same box the engine scissors its
/// flame pass to (`compute_flame_scissor`) and picks against.
#[pyfunction]
fn flame_bounds_corners(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    position: [f32; 3],
    rotation: [f32; 4],
) -> PyResult<Vec<[f32; 3]>> {
    let (effect, baked) = build_effect_from_params(py, params, 0.0, position, rotation, None)?;

    let bounds = flame_local_bounds(
        flame_bend_offset(&effect),
        flame_support_scale(&effect),
        effect.support_margin,
        flame_proxy_pad(&effect, &baked),
    );
    let model = build_flame_model_matrix(&effect);

    Ok(flame_local_bounds_corners(&bounds)
        .iter()
        .map(|corner| {
            let world = model * Vector4::new(corner.x, corner.y, corner.z, 1.0);
            [world.x, world.y, world.z]
        })
        .collect())
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
    let ubo = build_ubo_from_params(
        py,
        params,
        time,
        position,
        rotation,
        light_position,
        frame_index,
    )?;

    let bytes = unsafe {
        std::slice::from_raw_parts(
            &ubo as *const FlameUBO as *const u8,
            std::mem::size_of::<FlameUBO>(),
        )
    };
    Ok(bytes.to_vec())
}

/// Uniform members that only select a code path in `flameResolveFragment.frag`.
/// The Blender addon bakes them into the GLSL as constants so the dead paths are
/// not compiled; the values come from the same UBO the shader would read.
#[pyfunction]
fn flame_shader_specialization<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let ubo = build_ubo_from_params(py, params, 0.0, [0.0; 3], [1.0, 0.0, 0.0, 0.0], None, 0)?;

    let dict = PyDict::new(py);
    dict.set_item("flame.emitterParams.kind", ubo.emitter_params.kind)?;
    dict.set_item("flame.contourParams.rteBands", ubo.contour_params.rte_bands)?;
    dict.set_item("flame.trailMeta.sampleCount", ubo.trail_meta.sample_count)?;
    Ok(dict)
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
    m.add_function(wrap_pyfunction!(flame_shader_specialization, m)?)?;
    m.add_function(wrap_pyfunction!(flame_bounds_corners, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
