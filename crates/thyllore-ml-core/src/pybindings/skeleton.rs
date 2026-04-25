use cgmath::Matrix4;
use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use thyllore_model_core::{BoneId, Skeleton};

use super::error::{shape_mismatch, shape_mismatch_2d};

#[pyclass(name = "PySkeleton", module = "thyllore_ml_core")]
pub struct PySkeleton {
    pub(crate) inner: Skeleton,
}

#[pymethods]
impl PySkeleton {
    #[staticmethod]
    fn from_flat<'py>(
        bone_names: Vec<String>,
        parent_indices: Vec<i32>,
        local_matrices: PyReadonlyArray3<'py, f32>,
    ) -> PyResult<Self> {
        let n = bone_names.len();
        if parent_indices.len() != n {
            return Err(shape_mismatch("parent_indices", n, parent_indices.len()));
        }

        let shape = local_matrices.shape();
        if shape.len() != 3 || shape[0] != n || shape[1] != 4 || shape[2] != 4 {
            return Err(shape_mismatch_2d(
                "local_matrices (expected (N, 4, 4))",
                (n, 16),
                (shape[0], shape.iter().skip(1).product()),
            ));
        }

        let matrices_slice = local_matrices.as_slice()?;
        let mut skeleton = Skeleton::new("from_python");

        for i in 0..n {
            let parent = if parent_indices[i] < 0 {
                None
            } else {
                Some(parent_indices[i] as BoneId)
            };
            skeleton.add_bone(&bone_names[i], parent);

            let m = matrix_from_flat_row_major(&matrices_slice[i * 16..i * 16 + 16]);
            if let Some(bone) = skeleton.bones.get_mut(i) {
                bone.local_transform = m;
            }
        }

        Ok(Self { inner: skeleton })
    }

    #[getter]
    fn bone_count(&self) -> usize {
        self.inner.bones.len()
    }

    fn bone_name(&self, index: usize) -> PyResult<String> {
        self.inner
            .bones
            .get(index)
            .map(|b| b.name.clone())
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "bone index {} out of range (bone_count = {})",
                    index,
                    self.inner.bones.len()
                ))
            })
    }
}

fn matrix_from_flat_row_major(flat: &[f32]) -> Matrix4<f32> {
    debug_assert_eq!(flat.len(), 16);
    Matrix4::new(
        flat[0], flat[4], flat[8], flat[12], flat[1], flat[5], flat[9], flat[13], flat[2], flat[6],
        flat[10], flat[14], flat[3], flat[7], flat[11], flat[15],
    )
}

impl PySkeleton {
    pub(crate) fn as_skeleton(&self) -> &Skeleton {
        &self.inner
    }
}
