use anyhow::Result;
use cgmath::{InnerSpace, Matrix4, SquareMatrix, Vector3, Vector4};

use crate::animation::Skeleton;
use crate::ecs::component::{ColorVertex, LineMesh};
use crate::ecs::resource::gizmo::{BoneGizmoData, BoneSelectionState};
use crate::ecs::resource::MeshAssets;
use crate::math::{ray_to_triangle_barycentric, ray_to_triangle_intersection};
use crate::render::RenderBackend;

const BONE_LINE_COLOR: [f32; 3] = [0.0, 0.8, 0.0];
const JOINT_MARKER_COLOR: [f32; 3] = [1.0, 1.0, 0.0];
const JOINT_CROSS_SIZE: f32 = 0.02;

const BONE_SOLID_COLOR: [f32; 3] = [0.2, 0.45, 0.7];
const BONE_WIRE_COLOR: [f32; 3] = [0.05, 0.15, 0.35];
const BONE_SELECTED_SOLID_COLOR: [f32; 3] = [0.4, 0.7, 1.0];
const BONE_SELECTED_WIRE_COLOR: [f32; 3] = [0.1, 0.3, 0.55];
const BONE_ACTIVE_SOLID_COLOR: [f32; 3] = [1.0, 0.6, 0.2];
const BONE_ACTIVE_WIRE_COLOR: [f32; 3] = [0.5, 0.3, 0.1];
const OCTA_WIDTH_RATIO: f32 = 0.1;
const OCTA_DEPTH_RATIO: f32 = 0.1;
const BOX_WIDTH_RATIO: f32 = 0.08;
const SPHERE_RADIUS_RATIO: f32 = 0.06;
const SPHERE_RING_COUNT: usize = 6;
const SPHERE_SEGMENT_COUNT: usize = 8;

pub fn compute_bone_local_offsets(
    skeleton: &Skeleton,
    _rest_global_transforms: &[Matrix4<f32>],
) -> Vec<[f32; 3]> {
    vec![[0.0, 0.0, 0.0]; skeleton.bones.len()]
}

pub fn build_bone_line_mesh(
    skeleton: &Skeleton,
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    mesh_scale: f32,
    skeleton_for_display: Option<&Skeleton>,
    mesh: &mut LineMesh,
) {
    mesh.vertices.clear();
    mesh.indices.clear();

    if global_transforms.is_empty() {
        return;
    }

    let display_transforms = compute_display_transforms_with_skeleton(
        global_transforms,
        bone_local_offsets,
        mesh_scale,
        skeleton_for_display,
    );

    for bone in &skeleton.bones {
        let bone_idx = bone.id as usize;
        if bone_idx >= display_transforms.len() {
            continue;
        }

        let pos = display_transforms[bone_idx];

        if let Some(parent_id) = bone.parent_id {
            let parent_idx = parent_id as usize;
            if parent_idx >= display_transforms.len() {
                continue;
            }

            let parent_pos = display_transforms[parent_idx];

            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(ColorVertex {
                pos: parent_pos,
                color: BONE_LINE_COLOR,
            });
            mesh.vertices.push(ColorVertex {
                pos,
                color: BONE_LINE_COLOR,
            });
            mesh.indices.push(base);
            mesh.indices.push(base + 1);
        }

        append_joint_cross_marker(mesh, pos);
    }
}

pub fn build_octahedral_bone_meshes(
    skeleton: &Skeleton,
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    mesh_scale: f32,
    solid_mesh: &mut LineMesh,
    wire_mesh: &mut LineMesh,
) {
    let default_selection = BoneSelectionState::default();
    build_octahedral_bone_meshes_with_selection(
        skeleton,
        global_transforms,
        bone_local_offsets,
        &default_selection,
        1.0,
        mesh_scale,
        None,
        solid_mesh,
        wire_mesh,
    );
}

pub fn build_octahedral_bone_meshes_with_selection(
    skeleton: &Skeleton,
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    selection: &BoneSelectionState,
    visual_scale: f32,
    mesh_scale: f32,
    skeleton_for_display: Option<&Skeleton>,
    solid_mesh: &mut LineMesh,
    wire_mesh: &mut LineMesh,
) {
    solid_mesh.vertices.clear();
    solid_mesh.indices.clear();
    wire_mesh.vertices.clear();
    wire_mesh.indices.clear();

    if global_transforms.is_empty() {
        return;
    }

    let display_transforms = compute_display_transforms_with_skeleton(
        global_transforms,
        bone_local_offsets,
        mesh_scale,
        skeleton_for_display,
    );

    for bone in &skeleton.bones {
        let bone_idx = bone.id as usize;
        if bone_idx >= display_transforms.len() {
            continue;
        }

        let Some(parent_id) = bone.parent_id else {
            continue;
        };
        let parent_idx = parent_id as usize;
        if parent_idx >= display_transforms.len() {
            continue;
        }

        let head = display_transforms[parent_idx];
        let tail = display_transforms[bone_idx];

        let Some((bone_length, bone_dir, right, forward)) = compute_bone_axes(head, tail) else {
            continue;
        };

        let w = bone_length * OCTA_WIDTH_RATIO * visual_scale;
        let d = bone_length * OCTA_DEPTH_RATIO * visual_scale;

        let mid = [
            head[0] + bone_dir[0] * d,
            head[1] + bone_dir[1] * d,
            head[2] + bone_dir[2] * d,
        ];

        let verts: [[f32; 3]; 6] = [
            head,
            [
                mid[0] + right[0] * w,
                mid[1] + right[1] * w,
                mid[2] + right[2] * w,
            ],
            [
                mid[0] + forward[0] * w,
                mid[1] + forward[1] * w,
                mid[2] + forward[2] * w,
            ],
            [
                mid[0] - right[0] * w,
                mid[1] - right[1] * w,
                mid[2] - right[2] * w,
            ],
            [
                mid[0] - forward[0] * w,
                mid[1] - forward[1] * w,
                mid[2] - forward[2] * w,
            ],
            tail,
        ];

        let (solid_color, wire_color) = resolve_bone_colors(bone_idx, selection);
        append_octahedral_solid_colored(solid_mesh, &verts, solid_color);
        append_octahedral_wire_colored(wire_mesh, &verts, wire_color);
    }
}

fn resolve_bone_colors(bone_index: usize, selection: &BoneSelectionState) -> ([f32; 3], [f32; 3]) {
    if selection.active_bone_index == Some(bone_index) {
        return (BONE_ACTIVE_SOLID_COLOR, BONE_ACTIVE_WIRE_COLOR);
    }

    if selection.selected_bone_indices.contains(&bone_index) {
        return (BONE_SELECTED_SOLID_COLOR, BONE_SELECTED_WIRE_COLOR);
    }

    (BONE_SOLID_COLOR, BONE_WIRE_COLOR)
}

fn append_octahedral_solid_colored(mesh: &mut LineMesh, verts: &[[f32; 3]; 6], color: [f32; 3]) {
    let base = mesh.vertices.len() as u32;

    for v in verts {
        mesh.vertices.push(ColorVertex { pos: *v, color });
    }

    let tris: [[u32; 3]; 8] = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [5, 2, 1],
        [5, 3, 2],
        [5, 4, 3],
        [5, 1, 4],
    ];

    for tri in &tris {
        mesh.indices.push(base + tri[0]);
        mesh.indices.push(base + tri[1]);
        mesh.indices.push(base + tri[2]);
    }
}

fn append_octahedral_wire_colored(mesh: &mut LineMesh, verts: &[[f32; 3]; 6], color: [f32; 3]) {
    let base = mesh.vertices.len() as u32;

    for v in verts {
        mesh.vertices.push(ColorVertex { pos: *v, color });
    }

    let edges: [[u32; 2]; 12] = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [5, 1],
        [5, 2],
        [5, 3],
        [5, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
    ];

    for edge in &edges {
        mesh.indices.push(base + edge[0]);
        mesh.indices.push(base + edge[1]);
    }
}

fn compute_bone_axes(
    head: [f32; 3],
    tail: [f32; 3],
) -> Option<(f32, [f32; 3], [f32; 3], [f32; 3])> {
    let dx = tail[0] - head[0];
    let dy = tail[1] - head[1];
    let dz = tail[2] - head[2];
    let bone_length = (dx * dx + dy * dy + dz * dz).sqrt();

    if bone_length < 1e-6 {
        return None;
    }

    let inv_len = 1.0 / bone_length;
    let bone_dir = [dx * inv_len, dy * inv_len, dz * inv_len];

    let up_candidate = if bone_dir[1].abs() < 0.99 {
        [0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };

    let right = cross(bone_dir, up_candidate);
    let right = normalize(right);
    let forward = cross(bone_dir, right);
    let forward = normalize(forward);

    Some((bone_length, bone_dir, right, forward))
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 0.0];
    }
    let inv = 1.0 / len;
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

pub(crate) fn compute_display_transforms(
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    mesh_scale: f32,
) -> Vec<[f32; 3]> {
    compute_display_transforms_with_skeleton(
        global_transforms,
        bone_local_offsets,
        mesh_scale,
        None,
    )
}

pub(crate) fn compute_display_transforms_with_skeleton(
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    mesh_scale: f32,
    skeleton: Option<&Skeleton>,
) -> Vec<[f32; 3]> {
    (0..global_transforms.len())
        .map(|idx| {
            // `global_transforms` are joint world transforms (parent * local).
            // The joint display position is that transform applied to the bone's
            // local display offset. `inverse_bind_pose` is for skinning only and
            // must NOT enter here: `global * inverse_bind` collapses to identity
            // at rest, placing every bone at the origin.
            let _ = skeleton;
            let offset = bone_local_offsets.get(idx).copied().unwrap_or([0.0; 3]);
            let world_pos =
                global_transforms[idx] * Vector4::new(offset[0], offset[1], offset[2], 1.0);
            [
                world_pos.x * mesh_scale,
                world_pos.y * mesh_scale,
                world_pos.z * mesh_scale,
            ]
        })
        .collect()
}

fn append_joint_cross_marker(mesh: &mut LineMesh, pos: [f32; 3]) {
    let s = JOINT_CROSS_SIZE;
    let offsets = [
        ([-s, 0.0, 0.0], [s, 0.0, 0.0]),
        ([0.0, -s, 0.0], [0.0, s, 0.0]),
        ([0.0, 0.0, -s], [0.0, 0.0, s]),
    ];

    for (neg, posi) in &offsets {
        let base = mesh.vertices.len() as u32;

        mesh.vertices.push(ColorVertex {
            pos: [pos[0] + neg[0], pos[1] + neg[1], pos[2] + neg[2]],
            color: JOINT_MARKER_COLOR,
        });
        mesh.vertices.push(ColorVertex {
            pos: [pos[0] + posi[0], pos[1] + posi[1], pos[2] + posi[2]],
            color: JOINT_MARKER_COLOR,
        });

        mesh.indices.push(base);
        mesh.indices.push(base + 1);
    }
}

const OCTAHEDRAL_TRIANGLES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 1],
    [5, 2, 1],
    [5, 3, 2],
    [5, 4, 3],
    [5, 1, 4],
];

pub fn compute_octahedral_vertices_per_bone(
    skeleton: &Skeleton,
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    mesh_scale: f32,
    skeleton_for_display: Option<&Skeleton>,
) -> Vec<(usize, [[f32; 3]; 6])> {
    if global_transforms.is_empty() {
        return Vec::new();
    }

    let display_transforms = compute_display_transforms_with_skeleton(
        global_transforms,
        bone_local_offsets,
        mesh_scale,
        skeleton_for_display,
    );

    let mut result = Vec::new();

    for bone in &skeleton.bones {
        let bone_idx = bone.id as usize;
        if bone_idx >= display_transforms.len() {
            continue;
        }

        let Some(parent_id) = bone.parent_id else {
            continue;
        };
        let parent_idx = parent_id as usize;
        if parent_idx >= display_transforms.len() {
            continue;
        }

        let head = display_transforms[parent_idx];
        let tail = display_transforms[bone_idx];

        let Some((bone_length, bone_dir, right, forward)) = compute_bone_axes(head, tail) else {
            continue;
        };

        let w = bone_length * OCTA_WIDTH_RATIO;
        let d = bone_length * OCTA_DEPTH_RATIO;

        let mid = [
            head[0] + bone_dir[0] * d,
            head[1] + bone_dir[1] * d,
            head[2] + bone_dir[2] * d,
        ];

        let verts: [[f32; 3]; 6] = [
            head,
            [
                mid[0] + right[0] * w,
                mid[1] + right[1] * w,
                mid[2] + right[2] * w,
            ],
            [
                mid[0] + forward[0] * w,
                mid[1] + forward[1] * w,
                mid[2] + forward[2] * w,
            ],
            [
                mid[0] - right[0] * w,
                mid[1] - right[1] * w,
                mid[2] - right[2] * w,
            ],
            [
                mid[0] - forward[0] * w,
                mid[1] - forward[1] * w,
                mid[2] - forward[2] * w,
            ],
            tail,
        ];

        result.push((bone_idx, verts));
    }

    result
}

pub fn select_bone_by_ray(
    ray_origin: Vector3<f32>,
    ray_direction: Vector3<f32>,
    skeleton: &Skeleton,
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    mesh_scale: f32,
    skeleton_for_display: Option<&Skeleton>,
) -> Option<(usize, f32)> {
    let bone_verts = compute_octahedral_vertices_per_bone(
        skeleton,
        global_transforms,
        bone_local_offsets,
        mesh_scale,
        skeleton_for_display,
    );

    let mut closest: Option<(usize, f32)> = None;

    for (bone_idx, verts) in &bone_verts {
        for tri_indices in &OCTAHEDRAL_TRIANGLES {
            let v0 = verts[tri_indices[0]];
            let v1 = verts[tri_indices[1]];
            let v2 = verts[tri_indices[2]];

            if let Some(t) = ray_to_triangle_intersection(
                ray_origin,
                ray_direction,
                Vector3::new(v0[0], v0[1], v0[2]),
                Vector3::new(v1[0], v1[1], v1[2]),
                Vector3::new(v2[0], v2[1], v2[2]),
            ) {
                let is_closer = closest.map_or(true, |(_, prev_t)| t < prev_t);
                if is_closer {
                    closest = Some((*bone_idx, t));
                }
            }
        }
    }

    closest
}

pub unsafe fn update_bone_gizmo_buffers(
    bone_gizmo: &mut BoneGizmoData,
    backend: &mut dyn RenderBackend,
) -> Result<()> {
    backend.update_or_create_line_buffers(&mut bone_gizmo.stick_mesh)
}

pub fn select_bone_by_mesh_ray(
    ray_origin_world: Vector3<f32>,
    ray_direction_world: Vector3<f32>,
    mesh_assets: &MeshAssets,
    entity_world_transform: Matrix4<f32>,
) -> Option<(usize, f32)> {
    let inv_world = entity_world_transform.invert()?;
    let (ray_origin, ray_direction) =
        transform_ray_to_local(&inv_world, ray_origin_world, ray_direction_world);

    let hit = find_closest_triangle_hit(mesh_assets, ray_origin, ray_direction)?;

    let mesh = &mesh_assets.meshes[hit.mesh_index];
    let skin = mesh.skin_data.as_ref()?;
    let bone_index = pick_dominant_bone_from_triangle(skin, &hit)?;

    Some((bone_index, hit.t_world))
}

struct MeshTriangleHit {
    mesh_index: usize,
    vertex_indices: [usize; 3],
    barycentric: [f32; 3],
    t_world: f32,
}

fn transform_ray_to_local(
    inv_world: &Matrix4<f32>,
    ray_origin_world: Vector3<f32>,
    ray_direction_world: Vector3<f32>,
) -> (Vector3<f32>, Vector3<f32>) {
    let o = inv_world
        * Vector4::new(
            ray_origin_world.x,
            ray_origin_world.y,
            ray_origin_world.z,
            1.0,
        );
    let d = inv_world
        * Vector4::new(
            ray_direction_world.x,
            ray_direction_world.y,
            ray_direction_world.z,
            0.0,
        );

    (Vector3::new(o.x, o.y, o.z), Vector3::new(d.x, d.y, d.z))
}

fn find_closest_triangle_hit(
    mesh_assets: &MeshAssets,
    ray_origin_local: Vector3<f32>,
    ray_direction_local: Vector3<f32>,
) -> Option<MeshTriangleHit> {
    let direction_length = ray_direction_local.dot(ray_direction_local).sqrt();
    if direction_length < f32::EPSILON {
        return None;
    }

    let mut best: Option<MeshTriangleHit> = None;

    for (mesh_index, mesh) in mesh_assets.meshes.iter().enumerate() {
        if mesh.skin_data.is_none() {
            continue;
        }
        let vertices = &mesh.vertex_data.vertices;
        let indices = &mesh.vertex_data.indices;
        if vertices.is_empty() || indices.len() < 3 {
            continue;
        }

        for tri in indices.chunks_exact(3) {
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
                continue;
            }

            let p0 = vertex_position(&vertices[i0]);
            let p1 = vertex_position(&vertices[i1]);
            let p2 = vertex_position(&vertices[i2]);

            let Some((t_local, u, v)) =
                ray_to_triangle_barycentric(ray_origin_local, ray_direction_local, p0, p1, p2)
            else {
                continue;
            };

            let t_world = t_local * direction_length;
            let is_closer = best.as_ref().map_or(true, |h| t_world < h.t_world);
            if is_closer {
                best = Some(MeshTriangleHit {
                    mesh_index,
                    vertex_indices: [i0, i1, i2],
                    barycentric: [1.0 - u - v, u, v],
                    t_world,
                });
            }
        }
    }

    best
}

fn vertex_position(vertex: &crate::vulkanr::data::Vertex) -> Vector3<f32> {
    Vector3::new(vertex.pos.x, vertex.pos.y, vertex.pos.z)
}

fn pick_dominant_bone_from_triangle(
    skin: &crate::animation::SkinData,
    hit: &MeshTriangleHit,
) -> Option<usize> {
    let mut accumulated: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();

    for (slot, &vertex_index) in hit.vertex_indices.iter().enumerate() {
        if vertex_index >= skin.bone_indices.len() || vertex_index >= skin.bone_weights.len() {
            continue;
        }
        let bary_weight = hit.barycentric[slot];
        let indices = &skin.bone_indices[vertex_index];
        let weights = &skin.bone_weights[vertex_index];

        let influences = [
            (indices.x, weights.x),
            (indices.y, weights.y),
            (indices.z, weights.z),
            (indices.w, weights.w),
        ];

        for (bone_id, weight) in influences {
            if weight <= 0.0 {
                continue;
            }
            *accumulated.entry(bone_id).or_insert(0.0) += bary_weight * weight;
        }
    }

    accumulated
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(bone_id, _)| bone_id as usize)
}

pub fn build_box_bone_meshes_with_selection(
    skeleton: &Skeleton,
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    selection: &BoneSelectionState,
    visual_scale: f32,
    mesh_scale: f32,
    skeleton_for_display: Option<&Skeleton>,
    solid_mesh: &mut LineMesh,
    wire_mesh: &mut LineMesh,
) {
    solid_mesh.vertices.clear();
    solid_mesh.indices.clear();
    wire_mesh.vertices.clear();
    wire_mesh.indices.clear();

    if global_transforms.is_empty() {
        return;
    }

    let display_transforms = compute_display_transforms_with_skeleton(
        global_transforms,
        bone_local_offsets,
        mesh_scale,
        skeleton_for_display,
    );

    for bone in &skeleton.bones {
        let bone_idx = bone.id as usize;
        if bone_idx >= display_transforms.len() {
            continue;
        }

        let Some(parent_id) = bone.parent_id else {
            continue;
        };
        let parent_idx = parent_id as usize;
        if parent_idx >= display_transforms.len() {
            continue;
        }

        let head = display_transforms[parent_idx];
        let tail = display_transforms[bone_idx];

        let Some((bone_length, _bone_dir, right, forward)) = compute_bone_axes(head, tail) else {
            continue;
        };

        let w = bone_length * BOX_WIDTH_RATIO * visual_scale;
        let box_verts = compute_box_vertices(head, tail, right, forward, w);

        let (solid_color, wire_color) = resolve_bone_colors(bone_idx, selection);
        append_box_solid_colored(solid_mesh, &box_verts, solid_color);
        append_box_wire_colored(wire_mesh, &box_verts, wire_color);
    }
}

fn compute_box_vertices(
    head: [f32; 3],
    tail: [f32; 3],
    right: [f32; 3],
    forward: [f32; 3],
    w: f32,
) -> [[f32; 3]; 8] {
    let mut verts = [[0.0f32; 3]; 8];

    let signs: [[f32; 2]; 4] = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]];

    for (i, [sr, sf]) in signs.iter().enumerate() {
        for k in 0..3 {
            verts[i][k] = head[k] + right[k] * w * sr + forward[k] * w * sf;
            verts[i + 4][k] = tail[k] + right[k] * w * sr + forward[k] * w * sf;
        }
    }

    verts
}

fn append_box_solid_colored(mesh: &mut LineMesh, verts: &[[f32; 3]; 8], color: [f32; 3]) {
    let base = mesh.vertices.len() as u32;

    for v in verts {
        mesh.vertices.push(ColorVertex { pos: *v, color });
    }

    let tris: [[u32; 3]; 12] = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [2, 6, 7],
        [2, 7, 3],
        [1, 5, 6],
        [1, 6, 2],
        [0, 3, 7],
        [0, 7, 4],
    ];

    for tri in &tris {
        mesh.indices.push(base + tri[0]);
        mesh.indices.push(base + tri[1]);
        mesh.indices.push(base + tri[2]);
    }
}

fn append_box_wire_colored(mesh: &mut LineMesh, verts: &[[f32; 3]; 8], color: [f32; 3]) {
    let base = mesh.vertices.len() as u32;

    for v in verts {
        mesh.vertices.push(ColorVertex { pos: *v, color });
    }

    let edges: [[u32; 2]; 12] = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];

    for edge in &edges {
        mesh.indices.push(base + edge[0]);
        mesh.indices.push(base + edge[1]);
    }
}

pub fn build_sphere_bone_meshes_with_selection(
    skeleton: &Skeleton,
    global_transforms: &[Matrix4<f32>],
    bone_local_offsets: &[[f32; 3]],
    selection: &BoneSelectionState,
    visual_scale: f32,
    mesh_scale: f32,
    skeleton_for_display: Option<&Skeleton>,
    solid_mesh: &mut LineMesh,
    wire_mesh: &mut LineMesh,
) {
    solid_mesh.vertices.clear();
    solid_mesh.indices.clear();
    wire_mesh.vertices.clear();
    wire_mesh.indices.clear();

    if global_transforms.is_empty() {
        return;
    }

    let display_transforms = compute_display_transforms_with_skeleton(
        global_transforms,
        bone_local_offsets,
        mesh_scale,
        skeleton_for_display,
    );

    for bone in &skeleton.bones {
        let bone_idx = bone.id as usize;
        if bone_idx >= display_transforms.len() {
            continue;
        }

        let Some(parent_id) = bone.parent_id else {
            continue;
        };
        let parent_idx = parent_id as usize;
        if parent_idx >= display_transforms.len() {
            continue;
        }

        let head = display_transforms[parent_idx];
        let tail = display_transforms[bone_idx];

        let Some((bone_length, bone_dir, right, forward)) = compute_bone_axes(head, tail) else {
            continue;
        };

        let center = [
            (head[0] + tail[0]) * 0.5,
            (head[1] + tail[1]) * 0.5,
            (head[2] + tail[2]) * 0.5,
        ];
        let radius = bone_length * SPHERE_RADIUS_RATIO * visual_scale;

        let (solid_color, wire_color) = resolve_bone_colors(bone_idx, selection);
        append_sphere_solid_colored(
            solid_mesh,
            center,
            radius,
            bone_dir,
            right,
            forward,
            solid_color,
        );
        append_sphere_wire_colored(
            wire_mesh, center, radius, bone_dir, right, forward, wire_color,
        );
    }
}

fn sphere_point(
    center: [f32; 3],
    radius: f32,
    bone_dir: [f32; 3],
    right: [f32; 3],
    forward: [f32; 3],
    theta: f32,
    phi: f32,
) -> [f32; 3] {
    let st = theta.sin();
    let ct = theta.cos();
    let sp = phi.sin();
    let cp = phi.cos();

    let x = st * cp;
    let y = ct;
    let z = st * sp;

    [
        center[0] + radius * (right[0] * x + bone_dir[0] * y + forward[0] * z),
        center[1] + radius * (right[1] * x + bone_dir[1] * y + forward[1] * z),
        center[2] + radius * (right[2] * x + bone_dir[2] * y + forward[2] * z),
    ]
}

fn append_sphere_solid_colored(
    mesh: &mut LineMesh,
    center: [f32; 3],
    radius: f32,
    bone_dir: [f32; 3],
    right: [f32; 3],
    forward: [f32; 3],
    color: [f32; 3],
) {
    let rings = SPHERE_RING_COUNT;
    let segments = SPHERE_SEGMENT_COUNT;
    let base = mesh.vertices.len() as u32;

    for ring in 0..=rings {
        let theta = std::f32::consts::PI * ring as f32 / rings as f32;
        for seg in 0..=segments {
            let phi = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
            let pos = sphere_point(center, radius, bone_dir, right, forward, theta, phi);
            mesh.vertices.push(ColorVertex { pos, color });
        }
    }

    let cols = (segments + 1) as u32;
    for ring in 0..rings as u32 {
        for seg in 0..segments as u32 {
            let tl = base + ring * cols + seg;
            let tr = tl + 1;
            let bl = tl + cols;
            let br = bl + 1;

            mesh.indices.push(tl);
            mesh.indices.push(bl);
            mesh.indices.push(tr);

            mesh.indices.push(tr);
            mesh.indices.push(bl);
            mesh.indices.push(br);
        }
    }
}

fn append_sphere_wire_colored(
    mesh: &mut LineMesh,
    center: [f32; 3],
    radius: f32,
    bone_dir: [f32; 3],
    right: [f32; 3],
    forward: [f32; 3],
    color: [f32; 3],
) {
    let rings = SPHERE_RING_COUNT;
    let segments = SPHERE_SEGMENT_COUNT;

    for ring in 1..rings {
        let theta = std::f32::consts::PI * ring as f32 / rings as f32;
        let mut prev = sphere_point(center, radius, bone_dir, right, forward, theta, 0.0);

        for seg in 1..=segments {
            let phi = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
            let curr = sphere_point(center, radius, bone_dir, right, forward, theta, phi);

            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(ColorVertex { pos: prev, color });
            mesh.vertices.push(ColorVertex { pos: curr, color });
            mesh.indices.push(base);
            mesh.indices.push(base + 1);

            prev = curr;
        }
    }

    for seg in 0..segments {
        let phi = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
        let mut prev = sphere_point(center, radius, bone_dir, right, forward, 0.0, phi);

        for ring in 1..=rings {
            let theta = std::f32::consts::PI * ring as f32 / rings as f32;
            let curr = sphere_point(center, radius, bone_dir, right, forward, theta, phi);

            let base = mesh.vertices.len() as u32;
            mesh.vertices.push(ColorVertex { pos: prev, color });
            mesh.vertices.push(ColorVertex { pos: curr, color });
            mesh.indices.push(base);
            mesh.indices.push(base + 1);

            prev = curr;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::SkinData;
    use crate::ecs::{apply_skinning, compute_pose_global_transforms, create_pose_from_rest};
    use crate::loader::{LoadedNode, ModelLoadResult};
    use crate::vulkanr::data::{Vertex, VertexData};
    use crate::vulkanr::resource::graphics_resource::MeshBuffer;

    #[test]
    fn display_position_equals_joint_world_position_with_bind_pose() {
        // A bone whose world rest position is (1, 2, 3). Skeletal models (USD,
        // glTF, FBX) load inverse_bind_pose = bind_world^-1, so the gizmo must
        // still report the joint's world position — not collapse to the origin.
        let mut skeleton = Skeleton::new("rig");
        let bone_id = skeleton.add_bone("joint", None);
        let bind_world = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));
        if let Some(bone) = skeleton.get_bone_mut(bone_id) {
            bone.local_transform = bind_world;
            bone.inverse_bind_pose = bind_world.invert().unwrap();
        }

        let globals = vec![bind_world]; // rest pose global == bind
        let offsets = vec![[0.0f32; 3]];
        let display =
            compute_display_transforms_with_skeleton(&globals, &offsets, 1.0, Some(&skeleton));

        assert_eq!(
            display[0],
            [1.0, 2.0, 3.0],
            "bone display must be the joint world position, not the origin"
        );
    }

    fn make_mesh_buffer_single_triangle(
        bone_indices: [[u32; 4]; 3],
        bone_weights: [[f32; 4]; 3],
        positions: [Vector3<f32>; 3],
    ) -> MeshBuffer {
        let mut mesh = MeshBuffer::default();
        let mut vertex_data = VertexData::default();
        for p in &positions {
            let mut v = Vertex::default();
            v.pos.x = p.x;
            v.pos.y = p.y;
            v.pos.z = p.z;
            vertex_data.vertices.push(v);
        }
        vertex_data.indices = vec![0, 1, 2];
        mesh.vertex_data = vertex_data;

        let mut skin = SkinData::default();
        skin.bone_indices = bone_indices
            .iter()
            .map(|bi| cgmath::Vector4::new(bi[0], bi[1], bi[2], bi[3]))
            .collect();
        skin.bone_weights = bone_weights
            .iter()
            .map(|bw| cgmath::Vector4::new(bw[0], bw[1], bw[2], bw[3]))
            .collect();
        skin.base_positions = positions.to_vec();
        mesh.skin_data = Some(skin);

        mesh
    }

    #[test]
    fn select_bone_by_mesh_ray_picks_dominant_bone_on_centroid_hit() {
        let mesh = make_mesh_buffer_single_triangle(
            [[5, 0, 0, 0], [5, 0, 0, 0], [5, 0, 0, 0]],
            [[1.0, 0.0, 0.0, 0.0]; 3],
            [
                Vector3::new(-1.0, -1.0, 0.0),
                Vector3::new(1.0, -1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
        );
        let mesh_assets = MeshAssets { meshes: vec![mesh] };

        let ray_origin = Vector3::new(0.0, 0.0, 5.0);
        let ray_direction = Vector3::new(0.0, 0.0, -1.0);

        let hit =
            select_bone_by_mesh_ray(ray_origin, ray_direction, &mesh_assets, Matrix4::identity());

        let (bone_index, _t) = hit.expect("expected a mesh ray hit");
        assert_eq!(
            bone_index, 5,
            "all vertices weighted to bone 5, expected selection"
        );
    }

    #[test]
    fn select_bone_by_mesh_ray_uses_barycentric_weighting() {
        let mesh = make_mesh_buffer_single_triangle(
            [[1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]],
            [[1.0, 0.0, 0.0, 0.0]; 3],
            [
                Vector3::new(-1.0, -1.0, 0.0),
                Vector3::new(1.0, -1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
        );
        let mesh_assets = MeshAssets { meshes: vec![mesh] };

        let ray_origin = Vector3::new(0.9, -0.9, 5.0);
        let ray_direction = Vector3::new(0.0, 0.0, -1.0);

        let hit =
            select_bone_by_mesh_ray(ray_origin, ray_direction, &mesh_assets, Matrix4::identity());

        let (bone_index, _t) = hit.expect("expected a mesh ray hit near vertex 1");
        assert_eq!(
            bone_index, 2,
            "hit near vertex 1 (bone 2), expected dominant bone 2"
        );
    }

    #[test]
    fn select_bone_by_mesh_ray_returns_none_when_ray_misses_mesh() {
        let mesh = make_mesh_buffer_single_triangle(
            [[1, 0, 0, 0]; 3],
            [[1.0, 0.0, 0.0, 0.0]; 3],
            [
                Vector3::new(-1.0, -1.0, 0.0),
                Vector3::new(1.0, -1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
        );
        let mesh_assets = MeshAssets { meshes: vec![mesh] };

        let ray_origin = Vector3::new(10.0, 10.0, 5.0);
        let ray_direction = Vector3::new(0.0, 0.0, -1.0);

        let hit =
            select_bone_by_mesh_ray(ray_origin, ray_direction, &mesh_assets, Matrix4::identity());

        assert!(hit.is_none(), "ray outside triangle should not hit");
    }

    #[test]
    fn select_bone_by_mesh_ray_applies_entity_world_transform() {
        let mesh = make_mesh_buffer_single_triangle(
            [[7, 0, 0, 0]; 3],
            [[1.0, 0.0, 0.0, 0.0]; 3],
            [
                Vector3::new(-1.0, -1.0, 0.0),
                Vector3::new(1.0, -1.0, 0.0),
                Vector3::new(0.0, 1.0, 0.0),
            ],
        );
        let mesh_assets = MeshAssets { meshes: vec![mesh] };

        let entity_transform = Matrix4::from_translation(Vector3::new(100.0, 0.0, 0.0));

        let ray_origin = Vector3::new(100.0, 0.0, 5.0);
        let ray_direction = Vector3::new(0.0, 0.0, -1.0);

        let hit =
            select_bone_by_mesh_ray(ray_origin, ray_direction, &mesh_assets, entity_transform);

        let (bone_index, _t) = hit.expect("expected mesh hit after applying entity transform");
        assert_eq!(bone_index, 7);
    }

    fn load_stickman() -> Option<ModelLoadResult> {
        let path = "assets/models/stickman/stickman.glb";
        if !std::path::Path::new(path).exists() {
            return None;
        }
        let gltf = unsafe { crate::loader::gltf::load_gltf_file(path) }.ok()?;
        Some(ModelLoadResult::from_gltf(gltf))
    }

    fn load_phoenix() -> Option<ModelLoadResult> {
        let path = "tests/testmodels/glTF/skinning/glb/phoenixBird.glb";
        if !std::path::Path::new(path).exists() {
            return None;
        }
        let gltf = unsafe { crate::loader::gltf::load_gltf_file(path) }.ok()?;
        Some(ModelLoadResult::from_gltf(gltf))
    }

    fn compute_node_global_transforms(nodes: &[LoadedNode]) -> Vec<Matrix4<f32>> {
        let count = nodes.len();
        let mut globals = vec![Matrix4::identity(); count];
        let mut computed = vec![false; count];

        fn compute(
            nodes: &[LoadedNode],
            idx: usize,
            computed: &mut [bool],
            globals: &mut [Matrix4<f32>],
        ) -> Matrix4<f32> {
            if computed[idx] {
                return globals[idx];
            }
            let local = nodes[idx].local_transform;
            let global = if let Some(parent_idx) = nodes[idx].parent_index {
                if let Some(pi) = nodes.iter().position(|n| n.index == parent_idx) {
                    compute(nodes, pi, computed, globals) * local
                } else {
                    local
                }
            } else {
                local
            };
            globals[idx] = global;
            computed[idx] = true;
            global
        }

        for i in 0..count {
            compute(nodes, i, &mut computed, &mut globals);
        }
        globals
    }

    fn find_bone_for_node(
        nodes: &[LoadedNode],
        node_array_idx: usize,
        skeleton: &Skeleton,
    ) -> Option<usize> {
        let node = &nodes[node_array_idx];
        if let Some(bone) = skeleton.bones.iter().find(|b| b.name == node.name) {
            return Some(bone.id as usize);
        }
        if let Some(parent_idx) = node.parent_index {
            if let Some(pi) = nodes.iter().position(|n| n.index == parent_idx) {
                return find_bone_for_node(nodes, pi, skeleton);
            }
        }
        None
    }

    fn compute_node_mesh_center_per_bone(
        result: &ModelLoadResult,
        skeleton: &Skeleton,
    ) -> Vec<Option<Vector3<f32>>> {
        let bone_count = skeleton.bones.len();
        let mut sums: Vec<Vector3<f32>> = vec![Vector3::new(0.0, 0.0, 0.0); bone_count];
        let mut counts = vec![0u32; bone_count];
        let node_globals = compute_node_global_transforms(&result.nodes);
        let scale = result.node_animation_scale;

        for mesh in &result.meshes {
            let Some(node_idx) = mesh.node_index else {
                continue;
            };
            let Some(nai) = result.nodes.iter().position(|n| n.index == node_idx) else {
                continue;
            };
            let Some(bone_idx) = find_bone_for_node(&result.nodes, nai, skeleton) else {
                continue;
            };
            let transform = node_globals[nai];

            for v in &mesh.local_vertices {
                let pos = transform * Vector4::new(v.pos.x, v.pos.y, v.pos.z, 1.0);
                sums[bone_idx] += Vector3::new(pos.x * scale, pos.y * scale, pos.z * scale);
                counts[bone_idx] += 1;
            }
        }

        sums.iter()
            .zip(counts.iter())
            .map(|(s, &c)| if c > 0 { Some(*s / c as f32) } else { None })
            .collect()
    }

    fn compute_skinned_mesh_center_per_bone(
        skin_data: &SkinData,
        global_transforms: &[Matrix4<f32>],
        skeleton: &Skeleton,
    ) -> Vec<Option<Vector3<f32>>> {
        let vc = skin_data.base_positions.len();
        let mut positions = vec![Vector3::new(0.0f32, 0.0, 0.0); vc];
        let mut normals = vec![Vector3::new(0.0f32, 1.0, 0.0); vc];
        apply_skinning(
            skin_data,
            global_transforms,
            skeleton,
            &mut positions,
            &mut normals,
        );

        let bc = skeleton.bones.len();
        let mut sums = vec![Vector3::new(0.0f32, 0.0, 0.0); bc];
        let mut counts = vec![0u32; bc];

        for i in 0..vc {
            let idx = &skin_data.bone_indices[i];
            let wts = &skin_data.bone_weights[i];
            let dominant = [
                (idx.x as usize, wts.x),
                (idx.y as usize, wts.y),
                (idx.z as usize, wts.z),
                (idx.w as usize, wts.w),
            ]
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|&(i, _)| i)
            .unwrap_or(0);
            if dominant < bc {
                sums[dominant] += positions[i];
                counts[dominant] += 1;
            }
        }

        sums.iter()
            .zip(counts.iter())
            .map(|(s, &c)| if c > 0 { Some(*s / c as f32) } else { None })
            .collect()
    }

    fn build_node_globals_for_skeleton(
        nodes: &[LoadedNode],
        skeleton: &Skeleton,
    ) -> Vec<Matrix4<f32>> {
        let node_globals = compute_node_global_transforms(nodes);
        let mut transforms = vec![Matrix4::identity(); skeleton.bones.len()];
        for bone in &skeleton.bones {
            if let Some(nai) = nodes.iter().position(|n| n.name == bone.name) {
                transforms[bone.id as usize] = node_globals[nai];
            }
        }
        transforms
    }

    #[test]
    fn test_node_animation_bone_gizmo_matches_mesh() {
        let Some(result) = load_stickman() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let scale = result.node_animation_scale;
        let globals = build_node_globals_for_skeleton(&result.nodes, skeleton);

        let offsets = compute_bone_local_offsets(skeleton, &globals);
        let display = compute_display_transforms(&globals, &offsets, scale);

        let mesh_centers = compute_node_mesh_center_per_bone(&result, skeleton);

        let tolerance = 0.3;
        let mut tested = 0;
        let mut max_dist = 0.0f32;

        for (bi, center) in mesh_centers.iter().enumerate() {
            let Some(center) = center else { continue };
            let gizmo = Vector3::new(display[bi][0], display[bi][1], display[bi][2]);
            let dist = (gizmo - center).magnitude();
            max_dist = max_dist.max(dist);
            tested += 1;

            assert!(
                dist < tolerance,
                "bone[{}] '{}': gizmo=({:.4},{:.4},{:.4}) mesh=({:.4},{:.4},{:.4}) dist={:.4}",
                bi,
                skeleton.bones[bi].name,
                gizmo.x,
                gizmo.y,
                gizmo.z,
                center.x,
                center.y,
                center.z,
                dist
            );
        }

        assert!(tested > 0, "should test at least one bone");
        eprintln!("node gizmo: tested={}, max_dist={:.4}", tested, max_dist);
    }

    #[test]
    fn test_skinned_animation_bone_gizmo_matches_mesh() {
        let Some(result) = load_phoenix() else {
            eprintln!("Skipping: phoenix model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let pose = create_pose_from_rest(skeleton);
        let globals = compute_pose_global_transforms(skeleton, &pose);

        let skin_mesh = result.meshes.iter().find(|m| m.skin_data.is_some());
        let Some(mesh) = skin_mesh else {
            panic!("phoenix should have skinned mesh")
        };
        let skin_data = mesh.skin_data.as_ref().unwrap();

        let offsets = compute_bone_local_offsets(skeleton, &globals);
        let display =
            compute_display_transforms_with_skeleton(&globals, &offsets, 1.0, Some(skeleton));

        let mesh_centers = compute_skinned_mesh_center_per_bone(skin_data, &globals, skeleton);

        let tolerance = 0.8;
        let mut tested = 0;
        let mut max_dist = 0.0f32;
        let mut failures = Vec::new();

        for (bi, center) in mesh_centers.iter().enumerate() {
            let Some(center) = center else { continue };
            let gizmo = Vector3::new(display[bi][0], display[bi][1], display[bi][2]);
            let dist = (gizmo - center).magnitude();
            max_dist = max_dist.max(dist);
            tested += 1;

            if dist >= tolerance {
                failures.push(format!(
                    "bone[{}] '{}': gizmo=({:.3},{:.3},{:.3}) mesh=({:.3},{:.3},{:.3}) dist={:.3}",
                    bi,
                    skeleton.bones[bi].name,
                    gizmo.x,
                    gizmo.y,
                    gizmo.z,
                    center.x,
                    center.y,
                    center.z,
                    dist
                ));
            }
        }

        assert!(tested > 0, "should test at least one bone");
        eprintln!("skinned gizmo: tested={}, max_dist={:.3}", tested, max_dist);
        assert!(
            failures.is_empty(),
            "bones too far from mesh:\n{}",
            failures.join("\n")
        );
    }

    fn to_node_data(
        nodes: &[LoadedNode],
    ) -> Vec<crate::vulkanr::resource::graphics_resource::NodeData> {
        nodes
            .iter()
            .map(|n| crate::vulkanr::resource::graphics_resource::NodeData {
                index: n.index,
                name: n.name.clone(),
                parent_index: n.parent_index,
                local_transform: n.local_transform,
                global_transform: Matrix4::identity(),
            })
            .collect()
    }

    fn simulate_runtime_node_gizmo_transforms(
        result: &ModelLoadResult,
        skeleton: &Skeleton,
    ) -> Vec<Matrix4<f32>> {
        use crate::vulkanr::resource::graphics_resource::NodeData;

        let pose = create_pose_from_rest(skeleton);
        let mut nodes = to_node_data(&result.nodes);
        crate::ecs::systems::animation::apply::compute_node_global_transforms(
            &mut nodes, skeleton, &pose,
        );
        crate::ecs::systems::animation::apply::build_node_based_bone_transforms(&nodes, skeleton)
    }

    #[test]
    fn skinned_bone_gizmo_must_not_collapse_to_origin() {
        let Some(result) = load_phoenix() else {
            eprintln!("Skipping: phoenix model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let pose = create_pose_from_rest(skeleton);
        let globals = compute_pose_global_transforms(skeleton, &pose);

        let offsets = compute_bone_local_offsets(skeleton, &globals);
        let display = compute_display_transforms(&globals, &offsets, 1.0);

        let mut collapsed = 0u32;
        let mut tested = 0u32;

        for (idx, bone) in skeleton.bones.iter().enumerate() {
            if bone.parent_id.is_none() {
                continue;
            }
            tested += 1;

            let pos = display[idx];
            let dist_from_origin = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();

            if dist_from_origin < 0.01 {
                collapsed += 1;
            }
        }

        assert!(tested > 0);
        assert!(
            collapsed == 0,
            "{}/{} bones collapsed to origin. Bone gizmo display is broken.",
            collapsed,
            tested,
        );
    }

    #[test]
    fn runtime_skinned_bone_gizmo_display_matches_mesh() {
        let Some(result) = load_phoenix() else {
            eprintln!("Skipping: phoenix model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let pose = create_pose_from_rest(skeleton);
        let globals = compute_pose_global_transforms(skeleton, &pose);

        let skin_mesh = result.meshes.iter().find(|m| m.skin_data.is_some());
        let Some(mesh) = skin_mesh else {
            panic!("phoenix should have skinned mesh");
        };
        let skin_data = mesh.skin_data.as_ref().unwrap();

        let offsets = compute_bone_local_offsets(skeleton, &globals);
        let display = compute_display_transforms(&globals, &offsets, 1.0);
        let mesh_centers = compute_skinned_mesh_center_per_bone(skin_data, &globals, skeleton);

        let tolerance = 0.5;
        let mut tested = 0u32;
        let mut failures = Vec::new();

        for (bi, center) in mesh_centers.iter().enumerate() {
            let Some(center) = center else { continue };
            tested += 1;

            let gizmo = Vector3::new(display[bi][0], display[bi][1], display[bi][2]);
            let dist = (gizmo - center).magnitude();

            if dist >= tolerance {
                failures.push(format!(
                    "  bone[{}] '{}': gizmo=({:.3},{:.3},{:.3}) mesh=({:.3},{:.3},{:.3}) dist={:.3}",
                    bi, skeleton.bones[bi].name,
                    gizmo.x, gizmo.y, gizmo.z,
                    center.x, center.y, center.z, dist,
                ));
            }
        }

        assert!(tested > 0, "should test at least one bone");
        assert!(
            failures.is_empty(),
            "Bone gizmo display (with current skeleton_for_display) too far from mesh:\n{}",
            failures.join("\n"),
        );
    }

    #[test]
    fn runtime_node_gizmo_transforms_are_not_identity() {
        let Some(result) = load_stickman() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let transforms = simulate_runtime_node_gizmo_transforms(&result, skeleton);

        let mut identity_count = 0u32;
        let mut total = 0u32;

        for (idx, bone) in skeleton.bones.iter().enumerate() {
            if bone.parent_id.is_none() {
                continue;
            }
            total += 1;

            if transforms[idx] == Matrix4::identity() {
                identity_count += 1;
                eprintln!(
                    "  bone[{}] '{}' has identity transform (node_index={:?})",
                    idx, bone.name, bone.node_index,
                );
            }
        }

        assert!(total > 0);
        assert!(
            identity_count == 0,
            "{}/{} non-root bones have identity transforms. \
             build_node_based_bone_transforms failed to match these bones to nodes.",
            identity_count,
            total,
        );
    }

    #[test]
    fn runtime_node_bone_gizmo_matches_mesh() {
        let Some(result) = load_stickman() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let scale = result.node_animation_scale;
        let transforms = simulate_runtime_node_gizmo_transforms(&result, skeleton);

        let offsets = compute_bone_local_offsets(skeleton, &transforms);
        let display = compute_display_transforms(&transforms, &offsets, scale);

        let mesh_centers = compute_node_mesh_center_per_bone(&result, skeleton);

        let tolerance = 0.3;
        let mut tested = 0u32;
        let mut failures = Vec::new();

        for (bi, center) in mesh_centers.iter().enumerate() {
            let Some(center) = center else { continue };
            tested += 1;

            let gizmo = Vector3::new(display[bi][0], display[bi][1], display[bi][2]);
            let dist = (gizmo - center).magnitude();

            if dist >= tolerance {
                failures.push(format!(
                    "  bone[{}] '{}': gizmo=({:.3},{:.3},{:.3}) mesh=({:.3},{:.3},{:.3}) dist={:.3}",
                    bi, skeleton.bones[bi].name,
                    gizmo.x, gizmo.y, gizmo.z,
                    center.x, center.y, center.z, dist,
                ));
            }
        }

        assert!(tested > 0, "should test at least one bone");
        assert!(
            failures.is_empty(),
            "Node bone gizmo (runtime path) too far from mesh:\n{}",
            failures.join("\n"),
        );
    }

    #[test]
    fn runtime_node_transform_gizmo_not_at_origin() {
        let Some(result) = load_stickman() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let scale = result.node_animation_scale;
        let transforms = simulate_runtime_node_gizmo_transforms(&result, skeleton);
        let offsets = compute_bone_local_offsets(skeleton, &transforms);

        let mut gizmo = crate::ecs::resource::gizmo::TransformGizmoData::default();
        let mut at_origin_count = 0u32;
        let mut total = 0u32;

        for (idx, bone) in skeleton.bones.iter().enumerate() {
            if bone.parent_id.is_none() {
                continue;
            }
            total += 1;

            crate::ecs::systems::transform_gizmo_systems::transform_gizmo_sync_to_bone(
                &mut gizmo,
                Some(idx),
                &transforms,
                &offsets,
                scale,
            );

            let pos = gizmo.position.position;
            let dist = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();

            if dist < 0.001 {
                at_origin_count += 1;
                eprintln!("  bone[{}] '{}': transform gizmo at origin", idx, bone.name,);
            }
        }

        assert!(total > 0);
        assert!(
            at_origin_count == 0,
            "{}/{} bones have transform gizmo stuck at origin.",
            at_origin_count,
            total,
        );
    }

    #[test]
    fn runtime_skinned_transform_gizmo_matches_bone_display() {
        let Some(result) = load_phoenix() else {
            eprintln!("Skipping: phoenix model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");
        let pose = create_pose_from_rest(skeleton);
        let globals = compute_pose_global_transforms(skeleton, &pose);
        let offsets = compute_bone_local_offsets(skeleton, &globals);

        let display = compute_display_transforms(&globals, &offsets, 1.0);

        let mut gizmo = crate::ecs::resource::gizmo::TransformGizmoData::default();
        let mut failures = Vec::new();

        for (idx, bone) in skeleton.bones.iter().enumerate() {
            if bone.parent_id.is_none() {
                continue;
            }

            crate::ecs::systems::transform_gizmo_systems::transform_gizmo_sync_to_bone(
                &mut gizmo,
                Some(idx),
                &globals,
                &offsets,
                1.0,
            );

            let gizmo_pos = gizmo.position.position;
            let bone_display = display[idx];

            let dist = ((gizmo_pos.x - bone_display[0]).powi(2)
                + (gizmo_pos.y - bone_display[1]).powi(2)
                + (gizmo_pos.z - bone_display[2]).powi(2))
            .sqrt();

            if dist > 0.01 {
                failures.push(format!(
                    "  bone[{}] '{}': tg=({:.4},{:.4},{:.4}) bone=({:.4},{:.4},{:.4}) dist={:.4}",
                    idx,
                    bone.name,
                    gizmo_pos.x,
                    gizmo_pos.y,
                    gizmo_pos.z,
                    bone_display[0],
                    bone_display[1],
                    bone_display[2],
                    dist,
                ));
            }
        }

        assert!(
            failures.is_empty(),
            "Transform gizmo and bone gizmo display positions differ:\n{}",
            failures.join("\n"),
        );
    }

    #[test]
    fn node_index_fallback_finds_bone_transform() {
        use crate::vulkanr::resource::graphics_resource::NodeData;

        let mut skeleton = Skeleton::new("test");
        let bone_id = skeleton.add_bone("mismatched_name", None);
        if let Some(bone) = skeleton.get_bone_mut(bone_id) {
            bone.node_index = Some(42);
        }

        let node_transform = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));
        let nodes = vec![NodeData {
            index: 42,
            name: "different_name".to_string(),
            parent_index: None,
            local_transform: node_transform,
            global_transform: node_transform,
        }];

        let transforms = crate::ecs::systems::animation::apply::build_node_based_bone_transforms(
            &nodes, &skeleton,
        );

        assert_eq!(
            transforms[0], node_transform,
            "node_index fallback should match the node by index when name does not match"
        );
    }

    #[test]
    fn node_index_fallback_prefers_name_match() {
        use crate::vulkanr::resource::graphics_resource::NodeData;

        let mut skeleton = Skeleton::new("test");
        let bone_id = skeleton.add_bone("correct_name", None);
        if let Some(bone) = skeleton.get_bone_mut(bone_id) {
            bone.node_index = Some(99);
        }

        let name_transform = Matrix4::from_translation(Vector3::new(1.0, 0.0, 0.0));
        let index_transform = Matrix4::from_translation(Vector3::new(9.0, 9.0, 9.0));

        let nodes = vec![
            NodeData {
                index: 0,
                name: "correct_name".to_string(),
                parent_index: None,
                local_transform: name_transform,
                global_transform: name_transform,
            },
            NodeData {
                index: 99,
                name: "other_name".to_string(),
                parent_index: None,
                local_transform: index_transform,
                global_transform: index_transform,
            },
        ];

        let transforms = crate::ecs::systems::animation::apply::build_node_based_bone_transforms(
            &nodes, &skeleton,
        );

        assert_eq!(
            transforms[0], name_transform,
            "name match should take priority over node_index fallback"
        );
    }

    #[test]
    fn debug_phoenix_bone_usage_per_mesh() {
        let Some(result) = load_phoenix() else {
            eprintln!("Skipping: phoenix model not found");
            return;
        };
        let skeleton = result.skeletons.first().expect("no skeleton");

        let skinned_meshes: Vec<_> = result
            .meshes
            .iter()
            .enumerate()
            .filter(|(_, m)| m.skin_data.is_some())
            .collect();

        eprintln!(
            "Phoenix: {} skinned meshes, {} bones",
            skinned_meshes.len(),
            skeleton.bones.len()
        );

        let mut per_mesh_bones: Vec<std::collections::HashSet<u32>> = Vec::new();

        for (mi, mesh) in &skinned_meshes {
            let sd = mesh.skin_data.as_ref().unwrap();
            let mut used_bones = std::collections::HashSet::new();

            for (vi, indices) in sd.bone_indices.iter().enumerate() {
                let weights = &sd.bone_weights[vi];
                if weights.x > 0.0 {
                    used_bones.insert(indices.x);
                }
                if weights.y > 0.0 {
                    used_bones.insert(indices.y);
                }
                if weights.z > 0.0 {
                    used_bones.insert(indices.z);
                }
                if weights.w > 0.0 {
                    used_bones.insert(indices.w);
                }
            }

            eprintln!(
                "  mesh[{}]: {} verts, {} bones used",
                mi,
                sd.base_positions.len(),
                used_bones.len(),
            );
            per_mesh_bones.push(used_bones);
        }

        if per_mesh_bones.len() == 2 {
            let shared: std::collections::HashSet<_> =
                per_mesh_bones[0].intersection(&per_mesh_bones[1]).collect();
            let only_0: std::collections::HashSet<_> =
                per_mesh_bones[0].difference(&per_mesh_bones[1]).collect();
            let only_1: std::collections::HashSet<_> =
                per_mesh_bones[1].difference(&per_mesh_bones[0]).collect();

            eprintln!("  shared bones: {}", shared.len());
            eprintln!("  only mesh[0]: {}", only_0.len());
            eprintln!("  only mesh[1]: {}", only_1.len());

            for &b in &shared {
                if let Some(bone) = skeleton.bones.get(*b as usize) {
                    eprintln!("    shared: bone[{}] '{}'", b, bone.name);
                }
            }
        }
    }
}
