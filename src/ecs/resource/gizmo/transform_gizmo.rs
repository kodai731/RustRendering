use cgmath::{Quaternion, Vector3};

use crate::animation::BoneId;
use crate::ecs::component::{GizmoDraggable, GizmoPosition, GizmoSelectable, LineMesh, RenderInfo};
use crate::ecs::Entity;

pub use thyllore_render_core::TransformGizmoHandle;

#[derive(Clone, Debug)]
pub struct TransformGizmoData {
    pub visible: bool,
    pub position: GizmoPosition,
    pub selectable: GizmoSelectable,
    pub draggable: GizmoDraggable,
    pub active_handle: TransformGizmoHandle,
    pub drag_active: bool,
    pub line_mesh: LineMesh,
    pub solid_mesh: LineMesh,
    pub line_render_info: RenderInfo,
    pub solid_render_info: RenderInfo,
    pub drag_start_position: Vector3<f32>,
    pub drag_start_rotation: Quaternion<f32>,
    pub drag_start_scale: Vector3<f32>,
    pub drag_plane_normal: Vector3<f32>,
    pub drag_initial_hit: Vector3<f32>,
    pub drag_initial_angle: f32,
    pub target_bone_id: Option<BoneId>,
    pub target_entity: Option<Entity>,
}

impl Default for TransformGizmoData {
    fn default() -> Self {
        Self {
            visible: false,
            position: GizmoPosition::default(),
            selectable: GizmoSelectable::default(),
            draggable: GizmoDraggable::default(),
            active_handle: TransformGizmoHandle::None,
            drag_active: false,
            line_mesh: LineMesh::default(),
            solid_mesh: LineMesh::default(),
            line_render_info: RenderInfo::default(),
            solid_render_info: RenderInfo::default(),
            drag_start_position: Vector3::new(0.0, 0.0, 0.0),
            drag_start_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            drag_start_scale: Vector3::new(1.0, 1.0, 1.0),
            drag_plane_normal: Vector3::new(0.0, 1.0, 0.0),
            drag_initial_hit: Vector3::new(0.0, 0.0, 0.0),
            drag_initial_angle: 0.0,
            target_bone_id: None,
            target_entity: None,
        }
    }
}
