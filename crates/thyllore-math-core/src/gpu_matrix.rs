use cgmath::{Matrix4, SquareMatrix};

use crate::matrix::{array_from_mat4, mat4_from_array, Mat4};

/// Column-major 4x4 matrix in the layout GLSL `mat4` expects (std140 / std430).
/// Uploaded to UBO / SSBO memory as-is; convert through `Mat4` for arithmetic.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuMat4 {
    pub columns: [[f32; 4]; 4],
}

impl GpuMat4 {
    pub const IDENTITY: Self = Self {
        columns: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub const ZERO: Self = Self {
        columns: [[0.0; 4]; 4],
    };

    pub fn from_mat4(matrix: Mat4) -> Self {
        Self {
            columns: array_from_mat4(matrix),
        }
    }

    pub fn to_mat4(self) -> Mat4 {
        mat4_from_array(self.columns)
    }

    pub fn normal_matrix_of(model: Mat4) -> Self {
        let inverse = model.invert().unwrap_or_else(Matrix4::identity);
        Self::from_mat4(cgmath::Matrix::transpose(&inverse))
    }
}

impl From<Mat4> for GpuMat4 {
    fn from(matrix: Mat4) -> Self {
        Self::from_mat4(matrix)
    }
}

impl From<GpuMat4> for Mat4 {
    fn from(matrix: GpuMat4) -> Self {
        matrix.to_mat4()
    }
}

impl Default for GpuMat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Row-major 3x4 affine transform: three rows of `[x, y, z, translation]`.
/// This is the layout of `VkTransformMatrixKHR` used by acceleration structure instances.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AffineRows3x4 {
    pub rows: [[f32; 4]; 3],
}

impl AffineRows3x4 {
    pub const IDENTITY: Self = Self {
        rows: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    };

    pub fn from_mat4(matrix: Mat4) -> Self {
        let m = matrix;
        Self {
            rows: [
                [m[0][0], m[1][0], m[2][0], m[3][0]],
                [m[0][1], m[1][1], m[2][1], m[3][1]],
                [m[0][2], m[1][2], m[2][2], m[3][2]],
            ],
        }
    }

    pub fn to_mat4(self) -> Mat4 {
        let r = self.rows;
        Mat4::new(
            r[0][0], r[1][0], r[2][0], 0.0, r[0][1], r[1][1], r[2][1], 0.0, r[0][2], r[1][2],
            r[2][2], 0.0, r[0][3], r[1][3], r[2][3], 1.0,
        )
    }
}

impl From<Mat4> for AffineRows3x4 {
    fn from(matrix: Mat4) -> Self {
        Self::from_mat4(matrix)
    }
}

impl Default for AffineRows3x4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::approx_equal_mat4;
    use cgmath::{Rad, Vector3};

    fn sample_transform() -> Mat4 {
        Mat4::from_translation(Vector3::new(1.0, 2.0, 3.0))
            * Mat4::from_angle_y(Rad(0.7))
            * Mat4::from_nonuniform_scale(2.0, 3.0, 4.0)
    }

    #[test]
    fn gpu_mat4_identity_matches_cgmath_identity() {
        assert_eq!(GpuMat4::IDENTITY, GpuMat4::from_mat4(Mat4::identity()));
        assert_eq!(std::mem::size_of::<GpuMat4>(), 64);
    }

    #[test]
    fn gpu_mat4_round_trips_column_major() {
        let original = sample_transform();
        let gpu = GpuMat4::from_mat4(original);
        assert_eq!(gpu.columns[3][0], 1.0);
        assert_eq!(gpu.columns[3][1], 2.0);
        assert_eq!(gpu.columns[3][2], 3.0);
        assert!(approx_equal_mat4(&original, &gpu.to_mat4()));
    }

    #[test]
    fn affine_rows_keep_translation_in_last_column() {
        let original = sample_transform();
        let rows = AffineRows3x4::from_mat4(original);
        assert_eq!(rows.rows[0][3], 1.0);
        assert_eq!(rows.rows[1][3], 2.0);
        assert_eq!(rows.rows[2][3], 3.0);
        assert_eq!(std::mem::size_of::<AffineRows3x4>(), 48);
        assert!(approx_equal_mat4(&original, &rows.to_mat4()));
    }

    #[test]
    fn affine_rows_identity_matches_cgmath_identity() {
        assert_eq!(
            AffineRows3x4::IDENTITY,
            AffineRows3x4::from_mat4(Mat4::identity())
        );
    }

    #[test]
    fn normal_matrix_is_inverse_transpose() {
        let model = sample_transform();
        let expected = cgmath::Matrix::transpose(&model.invert().unwrap());
        assert!(approx_equal_mat4(
            &expected,
            &GpuMat4::normal_matrix_of(model).to_mat4()
        ));
    }
}
