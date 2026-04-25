use cgmath::Vector3;

#[repr(C)]
#[derive(Clone, Debug, Copy, Default)]
pub struct ColorVertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum GizmoAxis {
    #[default]
    None,
    X,
    Y,
    Z,
    Center,
}

#[derive(Clone, Debug)]
pub struct GizmoPosition {
    pub position: Vector3<f32>,
}

impl Default for GizmoPosition {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GizmoSelectable {
    pub is_selected: bool,
    pub selected_axis: GizmoAxis,
}

#[derive(Clone, Debug)]
pub struct GizmoDraggable {
    pub drag_axis: GizmoAxis,
    pub initial_position: Vector3<f32>,
}

impl Default for GizmoDraggable {
    fn default() -> Self {
        Self {
            drag_axis: GizmoAxis::None,
            initial_position: Vector3::new(0.0, 0.0, 0.0),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum BoneDisplayStyle {
    #[default]
    Stick,
    Octahedral,
    Box,
    Sphere,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TransformGizmoHandle {
    #[default]
    None,
    AxisX,
    AxisY,
    AxisZ,
    PlaneXY,
    PlaneXZ,
    PlaneYZ,
    Center,
    RingX,
    RingY,
    RingZ,
    Trackball,
}
