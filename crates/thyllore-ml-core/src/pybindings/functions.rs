use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use super::error::{shape_mismatch, shape_mismatch_2d};
use super::skeleton::PySkeleton;

#[pyfunction]
pub fn compute_topology<'py>(
    py: Python<'py>,
    skeleton: &PySkeleton,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let features = crate::topology::compute_bone_topology(skeleton.as_skeleton());

    let bone_count = skeleton.as_skeleton().bones.len();
    let mut flat = Vec::with_capacity(bone_count * 6);
    for bone in &skeleton.as_skeleton().bones {
        let f = features.get(&bone.id).cloned().unwrap_or_default();
        flat.extend_from_slice(&f.to_vec());
    }

    let arr = flat.into_pyarray(py).reshape([bone_count, 6])?;
    Ok(arr)
}

#[pyfunction]
pub fn tokenize_bone_names<'py>(
    py: Python<'py>,
    skeleton: &PySkeleton,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let tokens_map = crate::tokenizer::compute_bone_name_tokens(skeleton.as_skeleton());

    let bone_count = skeleton.as_skeleton().bones.len();
    let mut flat = Vec::with_capacity(bone_count * 32);
    for bone in &skeleton.as_skeleton().bones {
        let tokens = tokens_map.get(&bone.id).cloned().unwrap_or([0_i64; 32]);
        flat.extend_from_slice(&tokens);
    }

    let arr = flat.into_pyarray(py).reshape([bone_count, 32])?;
    Ok(arr)
}

#[pyfunction]
pub fn tokenize_bone_name_string<'py>(py: Python<'py>, name: &str) -> Bound<'py, PyArray1<i64>> {
    let tokens = crate::tokenizer::tokenize_bone_name(name);
    tokens.to_vec().into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (keyframes, max_keyframes, clip_duration))]
pub fn flatten_context<'py>(
    py: Python<'py>,
    keyframes: PyReadonlyArray2<'py, f32>,
    max_keyframes: usize,
    clip_duration: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, f32, f32)> {
    let shape = keyframes.shape();
    if shape.len() != 2 || shape[1] != 6 {
        return Err(shape_mismatch_2d(
            "keyframes",
            (shape[0], 6),
            (shape[0], shape.get(1).copied().unwrap_or(0)),
        ));
    }

    let n = shape[0];
    let slice = keyframes.as_slice()?;

    let mut times = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut in_dt = Vec::with_capacity(n);
    let mut in_dv = Vec::with_capacity(n);
    let mut out_dt = Vec::with_capacity(n);
    let mut out_dv = Vec::with_capacity(n);

    for i in 0..n {
        let row = &slice[i * 6..i * 6 + 6];
        times.push(row[0]);
        values.push(row[1]);
        in_dt.push(row[2]);
        in_dv.push(row[3]);
        out_dt.push(row[4]);
        out_dv.push(row[5]);
    }

    let ctx = crate::copilot::context::flatten_context(
        &times,
        &values,
        &in_dt,
        &in_dv,
        &out_dt,
        &out_dv,
        max_keyframes,
        clip_duration,
    );

    Ok((ctx.flat.into_pyarray(py), ctx.curve_mean, ctx.curve_std))
}

#[pyfunction]
#[pyo3(signature = (times, values, t_start, t_end, curve_mean, curve_std))]
pub fn sample_window<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f32>,
    values: PyReadonlyArray1<'py, f32>,
    t_start: f32,
    t_end: f32,
    curve_mean: f32,
    curve_std: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let times_len = times.shape()[0];
    let values_len = values.shape()[0];
    if times_len != values_len {
        return Err(shape_mismatch(
            "values (must equal times length)",
            times_len,
            values_len,
        ));
    }
    let window = crate::copilot::window::sample_window(
        times.as_slice()?,
        values.as_slice()?,
        t_start,
        t_end,
        curve_mean,
        curve_std,
    );
    Ok(window.to_vec().into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (times, current_time, clip_duration, max_steps))]
pub fn generate_query_times<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f32>,
    current_time: f32,
    clip_duration: f32,
    max_steps: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let result = crate::copilot::query::generate_query_times(
        times.as_slice()?,
        current_time,
        clip_duration,
        max_steps,
    );
    Ok(result.into_pyarray(py))
}
