use crate::flame::{
    apply_flame_preset, build_flame_model_matrix, build_flame_ubo, effective_sigma_t,
    flame_bend_offset, flame_local_bounds, flame_local_bounds_corners, flame_proxy_pad,
    flame_support_scale, overwrite_persisted_fields, parameter_owner, refresh_flame_coefficients,
    FlameBaked, FlameEffect, FlameTemporalAccum, FlameUBO, FLAME_PRESET_NAMES, FLAME_UI_PARAMS,
    MIN_FLAME_EXTENT,
};
use crate::water::{
    apply_water_preset, build_water_model_matrix, build_water_ubo, inverse_view_proj_f64,
    overwrite_water_persisted_fields, WaterTorusEffect, WaterUBO, ABSORPTION_REFERENCE_DISTANCE,
    WATER_PRESET_NAMES, WATER_UI_PARAMS,
};
use cgmath::{Matrix4, Quaternion, Vector3, Vector4};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use thyllore_math_core::torus_local_bounds_corners;
use thyllore_scene_core::{UiKind, UiParam};

fn ui_kind_name(kind: UiKind) -> &'static str {
    match kind {
        UiKind::Scalar => "scalar",
        UiKind::Color => "color",
        UiKind::Absorption => "absorption",
    }
}

fn fill_ui_param_dict(dict: &Bound<'_, PyDict>, param: &UiParam) -> PyResult<()> {
    dict.set_item("name", param.name)?;
    dict.set_item("label", param.display_label())?;
    dict.set_item("kind", ui_kind_name(param.kind))?;
    dict.set_item("min", param.min)?;
    dict.set_item("max", param.max)?;
    dict.set_item("format", param.format)?;
    dict.set_item("tooltip", param.tooltip)?;
    dict.set_item("persisted", param.persisted)?;
    match param.kind {
        UiKind::Scalar | UiKind::Color => {}
        UiKind::Absorption => {
            dict.set_item("reference_distance", ABSORPTION_REFERENCE_DISTANCE)?;
        }
    }
    Ok(())
}

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
        fill_ui_param_dict(&dict, param)?;

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

/// Optical depth the effect really renders with: `optical_depth` when set,
/// otherwise `sigma_t * radius` (the `0 = use sigma_t directly` convention).
#[pyfunction]
fn flame_effective_optical_depth(py: Python<'_>, params: &Bound<'_, PyDict>) -> PyResult<f32> {
    let (effect, _) =
        build_effect_from_params(py, params, 0.0, [0.0; 3], [1.0, 0.0, 0.0, 0.0], None)?;
    Ok(effective_sigma_t(&effect) * effect.radius.max(MIN_FLAME_EXTENT))
}
#[pyfunction]
fn flame_ubo_size() -> usize {
    std::mem::size_of::<FlameUBO>()
}

#[pyfunction]
fn water_preset_names() -> Vec<&'static str> {
    WATER_PRESET_NAMES.to_vec()
}

#[pyfunction]
fn water_ui_params(py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
    let default_dict: Bound<'_, PyDict> =
        pythonize::pythonize(py, &WaterTorusEffect::default())?.cast_into::<PyDict>()?;
    let list = PyList::empty(py);
    for param in WATER_UI_PARAMS {
        let dict = PyDict::new(py);
        fill_ui_param_dict(&dict, param)?;

        let Some(default_value) = default_dict.get_item(param.name)? else {
            continue;
        };
        dict.set_item("default", default_value)?;
        dict.set_item("owner", "frame")?;

        list.append(dict)?;
    }
    Ok(list)
}

#[pyfunction]
fn water_preset_params<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let mut effect = WaterTorusEffect::default();
    if !apply_water_preset(&mut effect, name) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unknown preset: {}",
            name
        )));
    }

    let dict: Bound<'py, PyDict> = pythonize::pythonize(py, &effect)?.cast_into::<PyDict>()?;
    Ok(dict)
}

fn build_water_effect_from_params(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    time: f32,
    position: [f32; 3],
    rotation: [f32; 4],
) -> PyResult<WaterTorusEffect> {
    let merged: Bound<'_, PyDict> =
        pythonize::pythonize(py, &WaterTorusEffect::default())?.cast_into::<PyDict>()?;
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

    let source: WaterTorusEffect = pythonize::depythonize(&merged).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "failed to deserialize parameters: {}",
            e
        ))
    })?;

    let mut effect = WaterTorusEffect::default();
    overwrite_water_persisted_fields(&mut effect, &source);

    effect.time = time;
    effect.position = Vector3::new(position[0], position[1], position[2]);
    effect.rotation = Quaternion::new(rotation[0], rotation[1], rotation[2], rotation[3]);

    Ok(effect)
}

#[pyfunction]
#[pyo3(signature = (params, time, position, rotation, view, proj, frame_index=0))]
fn pack_water_ubo(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    time: f32,
    position: [f32; 3],
    rotation: [f32; 4],
    view: [f32; 16],
    proj: [f32; 16],
    frame_index: u32,
) -> PyResult<Vec<u8>> {
    let effect = build_water_effect_from_params(py, params, time, position, rotation)?;
    let mut ubo = build_water_ubo(&effect, frame_index);

    // Compute inv_view_proj from view and proj matrices (column-major, same as pack_frame_ubo)
    let view_mat: Matrix4<f32> = Matrix4::new(
        view[0], view[1], view[2], view[3], view[4], view[5], view[6], view[7], view[8], view[9],
        view[10], view[11], view[12], view[13], view[14], view[15],
    );
    let proj_mat: Matrix4<f32> = Matrix4::new(
        proj[0], proj[1], proj[2], proj[3], proj[4], proj[5], proj[6], proj[7], proj[8], proj[9],
        proj[10], proj[11], proj[12], proj[13], proj[14], proj[15],
    );
    ubo.inv_view_proj = inverse_view_proj_f64(proj_mat, view_mat);

    let bytes = unsafe {
        std::slice::from_raw_parts(
            &ubo as *const WaterUBO as *const u8,
            std::mem::size_of::<WaterUBO>(),
        )
    };
    Ok(bytes.to_vec())
}

#[pyfunction]
fn water_bounds_corners(
    py: Python<'_>,
    params: &Bound<'_, PyDict>,
    position: [f32; 3],
    rotation: [f32; 4],
) -> PyResult<Vec<[f32; 3]>> {
    let effect = build_water_effect_from_params(py, params, 0.0, position, rotation)?;

    let model = build_water_model_matrix(&effect);

    Ok(
        torus_local_bounds_corners(effect.major_radius, effect.minor_radius)
            .iter()
            .map(|corner| {
                let world = model * Vector4::new(corner.x, corner.y, corner.z, 1.0);
                [world.x, world.y, world.z]
            })
            .collect(),
    )
}

#[pyfunction]
fn water_ubo_size() -> usize {
    std::mem::size_of::<WaterUBO>()
}

#[pymodule]
fn thyllore_effect_core(_py: Python<'_>, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(flame_preset_names, m)?)?;
    m.add_function(wrap_pyfunction!(flame_ui_params, m)?)?;
    m.add_function(wrap_pyfunction!(flame_effective_optical_depth, m)?)?;
    m.add_function(wrap_pyfunction!(flame_preset_params, m)?)?;
    m.add_function(wrap_pyfunction!(pack_flame_ubo, m)?)?;
    m.add_function(wrap_pyfunction!(flame_ubo_size, m)?)?;
    m.add_function(wrap_pyfunction!(flame_shader_specialization, m)?)?;
    m.add_function(wrap_pyfunction!(flame_bounds_corners, m)?)?;

    m.add_function(wrap_pyfunction!(water_preset_names, m)?)?;
    m.add_function(wrap_pyfunction!(water_ui_params, m)?)?;
    m.add_function(wrap_pyfunction!(water_preset_params, m)?)?;
    m.add_function(wrap_pyfunction!(pack_water_ubo, m)?)?;
    m.add_function(wrap_pyfunction!(water_bounds_corners, m)?)?;
    m.add_function(wrap_pyfunction!(water_ubo_size, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests;
