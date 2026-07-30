use anyhow::Result;
use cgmath::{InnerSpace, Matrix4, Vector3, Vector4};

use crate::ecs::component::mesh::presets::{POSITION, TEX_COORD_0};
use crate::ecs::component::mesh::{MeshData, PrimitiveTopology};
use crate::ecs::component::{CameraState, RenderInfo};
use crate::ecs::resource::billboard::{
    BillboardData, BillboardMesh, BillboardRenderState, BillboardTransform, BillboardVertex,
};
use crate::ecs::world::{BillboardBehavior, Transform, World};
use crate::render::BillboardBackend;

pub fn create_billboard_mesh_data(size: f32) -> MeshData {
    let positions = vec![
        [-size, -size, 0.0],
        [size, -size, 0.0],
        [size, size, 0.0],
        [-size, size, 0.0],
    ];

    let tex_coords = vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    let indices = vec![0, 1, 2, 0, 2, 3];

    MeshData::new(PrimitiveTopology::TriangleList)
        .with_inserted_attribute(POSITION, positions)
        .with_inserted_attribute(TEX_COORD_0, tex_coords)
        .with_indices(indices)
}

#[deprecated(note = "Use create_billboard_mesh_data() instead")]
pub fn create_billboard() -> BillboardData {
    let billboard_size = 0.5;
    let vertices = vec![
        BillboardVertex {
            pos: [-billboard_size, -billboard_size, 0.0],
            tex_coord: [0.0, 1.0],
        },
        BillboardVertex {
            pos: [billboard_size, -billboard_size, 0.0],
            tex_coord: [1.0, 1.0],
        },
        BillboardVertex {
            pos: [billboard_size, billboard_size, 0.0],
            tex_coord: [1.0, 0.0],
        },
        BillboardVertex {
            pos: [-billboard_size, billboard_size, 0.0],
            tex_coord: [0.0, 0.0],
        },
    ];

    let indices = vec![0, 1, 2, 0, 2, 3];

    BillboardData {
        mesh: BillboardMesh {
            vertices,
            indices,
            vertex_buffer_handles: [Default::default(); 2],
            index_buffer_handles: [Default::default(); 2],
            last_written_slot: 0,
        },
        transform: None,
        render_info: RenderInfo::default(),
        render_state: BillboardRenderState::default(),
    }
}

pub fn create_billboard_transform(position: Vector3<f32>) -> BillboardTransform {
    BillboardTransform {
        position,
        model_matrix: Matrix4::from_translation(position),
    }
}

pub fn billboard_system(world: &mut World, camera: &CameraState) {
    let billboards = world.query_billboards();

    for entity in billboards {
        let Some(behavior) = world.get_component::<BillboardBehavior>(entity) else {
            continue;
        };

        if !behavior.always_face_camera {
            continue;
        }

        let Some(transform) = world.get_component_mut::<Transform>(entity) else {
            continue;
        };

        let position = transform.translation;
        let to_camera = camera.position - position;

        if to_camera.magnitude() > 0.001 {
            let forward = to_camera.normalize();
            let up = Vector3::new(0.0, 1.0, 0.0);
            let right = up.cross(forward).normalize();
            let adjusted_up = forward.cross(right);

            transform.rotation =
                cgmath::Quaternion::from(cgmath::Matrix3::from_cols(right, adjusted_up, forward));
        }
    }
}

pub unsafe fn billboard_create_buffers(
    billboard: &mut BillboardData,
    backend: &mut dyn BillboardBackend,
) -> Result<()> {
    backend.create_billboard_buffers(billboard)
}

pub fn billboard_transform_update_look_at(
    transform: &mut BillboardTransform,
    camera_position: Vector3<f32>,
    world_up: Vector3<f32>,
) {
    let forward = (camera_position - transform.position).normalize();
    let right = world_up.cross(forward).normalize();
    let up = forward.cross(right);

    let rotation = Matrix4::from_cols(
        right.extend(0.0),
        up.extend(0.0),
        forward.extend(0.0),
        Vector4::new(0.0, 0.0, 0.0, 1.0),
    );

    let translation = Matrix4::from_translation(transform.position);
    transform.model_matrix = translation * rotation;
}

pub fn billboard_transform_set_position(
    transform: &mut BillboardTransform,
    position: Vector3<f32>,
) {
    transform.position = position;
}
