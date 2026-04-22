use cgmath::{Matrix4, SquareMatrix, Vector4};

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct FrameUBO {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
    pub camera_pos: Vector4<f32>,
    pub light_pos: Vector4<f32>,
    pub light_color: Vector4<f32>,
}

impl Default for FrameUBO {
    fn default() -> Self {
        Self {
            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            camera_pos: Vector4::new(0.0, 0.0, 0.0, 1.0),
            light_pos: Vector4::new(0.0, 0.0, 0.0, 1.0),
            light_color: Vector4::new(1.0, 1.0, 1.0, 1.0),
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct ObjectUBO {
    pub model: Matrix4<f32>,
}

impl Default for ObjectUBO {
    fn default() -> Self {
        Self {
            model: Matrix4::identity(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct MaterialUBO {
    pub base_color: Vector4<f32>,
    pub metallic: f32,
    pub roughness: f32,
    pub _padding: [f32; 2],
}

impl Default for MaterialUBO {
    fn default() -> Self {
        Self {
            base_color: Vector4::new(1.0, 1.0, 1.0, 1.0),
            metallic: 0.0,
            roughness: 0.5,
            _padding: [0.0; 2],
        }
    }
}
