use crate::vector::{approx_equal_vec4, ToArray4};
use cgmath::{InnerSpace, Matrix3, Matrix4, Quaternion, Vector3};

pub type Mat3 = cgmath::Matrix3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

pub fn array_from_mat4(m: Mat4) -> [[f32; 4]; 4] {
    [
        m.x.to_array(),
        m.y.to_array(),
        m.z.to_array(),
        m.w.to_array(),
    ]
}

pub fn mat4_from_array(a: [[f32; 4]; 4]) -> Mat4 {
    Mat4::from_cols(a[0].into(), a[1].into(), a[2].into(), a[3].into())
}

pub fn mat4_from_array_transpose(a: [[f32; 4]; 4]) -> Mat4 {
    Mat4::new(
        a[0][0], a[1][0], a[2][0], a[3][0], a[0][1], a[1][1], a[2][1], a[3][1], a[0][2], a[1][2],
        a[2][2], a[3][2], a[0][3], a[1][3], a[2][3], a[3][3],
    )
}

pub fn decompose(m: &Matrix4<f32>) -> (Vector3<f32>, Quaternion<f32>, Vector3<f32>) {
    let translation = Vector3::new(m.w.x, m.w.y, m.w.z);

    let scale_x = Vector3::new(m.x.x, m.x.y, m.x.z).magnitude();
    let scale_y = Vector3::new(m.y.x, m.y.y, m.y.z).magnitude();
    let scale_z = Vector3::new(m.z.x, m.z.y, m.z.z).magnitude();
    let scale = Vector3::new(scale_x, scale_y, scale_z);

    let rot_x = Vector3::new(m.x.x, m.x.y, m.x.z) / scale_x;
    let rot_y = Vector3::new(m.y.x, m.y.y, m.y.z) / scale_y;
    let rot_z = Vector3::new(m.z.x, m.z.y, m.z.z) / scale_z;

    let rot_matrix = Matrix3::from_cols(rot_x, rot_y, rot_z);

    let rotation = Quaternion::from(rot_matrix);

    (translation, rotation, scale)
}

pub fn swap(q: &Quaternion<f32>) -> Quaternion<f32> {
    Quaternion::new(q.s, q.v[0], q.v[1], q.v[2])
}

pub unsafe fn rodrigues(
    rotate: &mut cgmath::Matrix3<f32>,
    c: f32,
    s: f32,
    n: &cgmath::Vector3<f32>,
) -> anyhow::Result<()> {
    let ac = 1.0 - c;
    let xyac = n.x * n.y * ac;
    let yzac = n.y * n.z * ac;
    let zxac = n.x * n.z * ac;
    let xs = n.x * s;
    let ys = n.y * s;
    let zs = n.z * s;
    *rotate = cgmath::Matrix3::new(
        c + n.x * n.x * ac,
        xyac + zs,
        zxac - ys,
        xyac - zs,
        c + n.y * n.y * ac,
        yzac + xs,
        zxac + ys,
        yzac - xs,
        c + n.z * n.z * ac,
    );
    Ok(())
}

pub fn approx_equal_mat4(a: &Mat4, b: &Mat4) -> bool {
    approx_equal_vec4(&a.x, &b.x)
        && approx_equal_vec4(&a.y, &b.y)
        && approx_equal_vec4(&a.z, &b.z)
        && approx_equal_vec4(&a.w, &b.w)
}

#[test]
fn test_mat4_array_conversion() {
    let original = Mat4::from_cols(
        cgmath::Vector4::new(1.0, 0.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 1.0, 0.0, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 1.0, 0.0),
        cgmath::Vector4::new(0.0, 0.0, 0.0, 1.0),
    );

    let arr = array_from_mat4(original);
    let converted = mat4_from_array(arr);

    assert!(approx_equal_mat4(&original, &converted));
}

#[test]
fn test_swap_quaternion() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let swapped = swap(&q);
    assert_eq!(swapped.s, 1.0);
    assert_eq!(swapped.v.x, 2.0);
    assert_eq!(swapped.v.y, 3.0);
    assert_eq!(swapped.v.z, 4.0);
}

#[test]
fn test_decompose_identity() {
    let identity = Matrix4::from_scale(1.0);
    let (translation, rotation, scale) = decompose(&identity);

    assert!((translation.x).abs() < 1e-5);
    assert!((translation.y).abs() < 1e-5);
    assert!((translation.z).abs() < 1e-5);

    assert!((scale.x - 1.0).abs() < 1e-5);
    assert!((scale.y - 1.0).abs() < 1e-5);
    assert!((scale.z - 1.0).abs() < 1e-5);
}

#[test]
fn test_rodrigues_rotation() {
    unsafe {
        let mut rotate = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let angle = std::f32::consts::PI / 2.0;
        let c = angle.cos();
        let s = angle.sin();
        let n = Vector3::new(0.0, 0.0, 1.0);

        let result = rodrigues(&mut rotate, c, s, &n);
        assert!(result.is_ok());
    }
}
