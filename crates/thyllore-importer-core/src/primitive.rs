use thyllore_anim_core::{AnimationSystem, MorphAnimationSystem};
use thyllore_math_core::{Vec2, Vec3, Vec4};
use thyllore_model_core::mesh::{Vertex, VertexData};

use crate::model_result::{LoadedMesh, ModelLoadResult};

/// Build a box model with face normals (24 vertices, 12 triangles).
pub fn build_box_model(size_x: f32, size_y: f32, size_z: f32, color: [f32; 4]) -> ModelLoadResult {
    let hx = size_x * 0.5;
    let hy = size_y * 0.5;
    let hz = size_z * 0.5;

    // Each face has 4 vertices (2 triangles), 6 faces = 24 vertices, 36 indices.
    let mut vertices: Vec<Vertex> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    let vertex_color = Vec4::new(color[0], color[1], color[2], color[3]);

    // Helper to add a quad (2 triangles) with a given normal direction.
    let mut add_face = |nx: f32,
                        ny: f32,
                        nz: f32,
                        ax: f32,
                        ay: f32,
                        az: f32,
                        bx: f32,
                        by: f32,
                        bz: f32,
                        cx: f32,
                        cy: f32,
                        cz: f32,
                        dx: f32,
                        dy: f32,
                        dz: f32| {
        let base = vertices.len() as u32;
        let normal = Vec3::new(nx, ny, nz);
        vertices.push(Vertex::new_with_normal(
            Vec3::new(ax, ay, az),
            vertex_color,
            Vec2::new(0.0, 0.0),
            normal,
        ));
        vertices.push(Vertex::new_with_normal(
            Vec3::new(bx, by, bz),
            vertex_color,
            Vec2::new(0.0, 0.0),
            normal,
        ));
        vertices.push(Vertex::new_with_normal(
            Vec3::new(cx, cy, cz),
            vertex_color,
            Vec2::new(0.0, 0.0),
            normal,
        ));
        vertices.push(Vertex::new_with_normal(
            Vec3::new(dx, dy, dz),
            vertex_color,
            Vec2::new(0.0, 0.0),
            normal,
        ));
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    };

    // +X face (right)
    add_face(
        1.0, 0.0, 0.0, hx, -hy, hz, hx, hy, hz, hx, hy, -hz, hx, -hy, -hz,
    );
    // -X face (left)
    add_face(
        -1.0, 0.0, 0.0, -hx, -hy, -hz, -hx, hy, -hz, -hx, hy, hz, -hx, -hy, hz,
    );
    // +Y face (top)
    add_face(
        0.0, 1.0, 0.0, -hx, hy, hz, hx, hy, hz, hx, hy, -hz, -hx, hy, -hz,
    );
    // -Y face (bottom)
    add_face(
        0.0, -1.0, 0.0, -hx, -hy, -hz, hx, -hy, -hz, hx, -hy, hz, -hx, -hy, hz,
    );
    // +Z face (front)
    add_face(
        0.0, 0.0, 1.0, -hx, -hy, hz, hx, -hy, hz, hx, hy, hz, -hx, hy, hz,
    );
    // -Z face (back)
    add_face(
        0.0, 0.0, -1.0, hx, -hy, -hz, -hx, -hy, -hz, -hx, hy, -hz, hx, hy, -hz,
    );

    let vertex_data = VertexData { vertices, indices };

    ModelLoadResult {
        meshes: vec![LoadedMesh {
            vertex_data,
            skin_data: None,
            skeleton_id: None,
            node_index: None,
            local_vertices: Vec::new(),
            texture: None,
            base_color_factor: color,
        }],
        nodes: Vec::new(),
        skeletons: Vec::new(),
        animation_system: AnimationSystem::default(),
        clips: Vec::new(),
        morph_animation: MorphAnimationSystem::default(),
        has_skinned_meshes: false,
        node_animation_scale: 1.0,
        constraints: Vec::new(),
        spring_bone_setup: None,
    }
}

/// Build a cube model with face normals (24 vertices, 12 triangles).
pub fn build_cube_model(size: f32) -> ModelLoadResult {
    build_box_model(size, size, size, [0.8, 0.15, 0.15, 1.0])
}

/// Build a UV sphere model with smooth normals per face.
pub fn build_uv_sphere_model(radius: f32, segments: u32, rings: u32) -> ModelLoadResult {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let color = Vec4::new(0.9, 0.9, 0.9, 1.0);

    // Generate vertices for each (ring, segment) position.
    // rings go from 0..=rings (top to bottom), segments from 0..segments.
    let mut positions: Vec<(Vec3, Vec3)> = Vec::new();

    for ring in 0..=rings {
        let phi = std::f32::consts::PI * ring as f32 / rings as f32;
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();
        for seg in 0..segments {
            let theta = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
            let x = radius * sin_phi * theta.cos();
            let y = radius * cos_phi;
            let z = radius * sin_phi * theta.sin();
            let pos = Vec3::new(x, y, z);
            // Normal is just the normalized position (unit sphere scaled by radius).
            let normal = Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin());
            positions.push((pos, normal));
        }
    }

    // Build faces: each quad is two triangles.
    for ring in 0..rings {
        for seg in 0..segments {
            let a = ring * segments + seg;
            let b = ring * segments + (seg + 1) % segments;
            let c = (ring + 1) * segments + (seg + 1) % segments;
            let d = (ring + 1) * segments + seg;

            // Use per-face normals: each vertex gets the normal of its position.
            let (pos_a, norm_a) = positions[a as usize];
            let (pos_b, norm_b) = positions[b as usize];
            let (pos_c, norm_c) = positions[c as usize];
            let (pos_d, norm_d) = positions[d as usize];

            let base = vertices.len() as u32;
            vertices.push(Vertex::new_with_normal(
                pos_a,
                color,
                Vec2::new(0.0, 0.0),
                norm_a,
            ));
            vertices.push(Vertex::new_with_normal(
                pos_b,
                color,
                Vec2::new(0.0, 0.0),
                norm_b,
            ));
            vertices.push(Vertex::new_with_normal(
                pos_c,
                color,
                Vec2::new(0.0, 0.0),
                norm_c,
            ));
            vertices.push(Vertex::new_with_normal(
                pos_d,
                color,
                Vec2::new(0.0, 0.0),
                norm_d,
            ));

            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    let vertex_data = VertexData { vertices, indices };

    ModelLoadResult {
        meshes: vec![LoadedMesh {
            vertex_data,
            skin_data: None,
            skeleton_id: None,
            node_index: None,
            local_vertices: Vec::new(),
            texture: None,
            base_color_factor: [0.9, 0.9, 0.9, 1.0],
        }],
        nodes: Vec::new(),
        skeletons: Vec::new(),
        animation_system: AnimationSystem::default(),
        clips: Vec::new(),
        morph_animation: MorphAnimationSystem::default(),
        has_skinned_meshes: false,
        node_animation_scale: 1.0,
        constraints: Vec::new(),
        spring_bone_setup: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_normals_normalized(meshes: &[LoadedMesh]) {
        for mesh in meshes {
            for v in &mesh.vertex_data.vertices {
                let len_sq = v.normal[0] * v.normal[0]
                    + v.normal[1] * v.normal[1]
                    + v.normal[2] * v.normal[2];
                assert!(
                    (len_sq - 1.0).abs() < 1e-5,
                    "normal {:?} is not normalized (length^2 = {})",
                    v.normal,
                    len_sq
                );
            }
        }
    }

    fn assert_indices_in_range(meshes: &[LoadedMesh]) {
        for mesh in meshes {
            let count = mesh.vertex_data.vertices.len() as u32;
            for &idx in &mesh.vertex_data.indices {
                assert!(
                    idx < count,
                    "index {} is out of range (vertex count = {})",
                    idx,
                    count
                );
            }
        }
    }

    #[test]
    fn test_cube_vertex_count() {
        let result = build_cube_model(1.0);
        assert_eq!(result.meshes.len(), 1);
        assert_eq!(result.meshes[0].vertex_data.vertices.len(), 24);
    }

    #[test]
    fn test_cube_normals_normalized() {
        let result = build_cube_model(1.0);
        assert_normals_normalized(&result.meshes);
    }

    #[test]
    fn test_cube_indices_in_range() {
        let result = build_cube_model(1.0);
        assert_indices_in_range(&result.meshes);
    }

    #[test]
    fn test_sphere_vertex_count() {
        let segments = 8;
        let rings = 6;
        let result = build_uv_sphere_model(1.0, segments, rings);
        assert_eq!(result.meshes.len(), 1);
        // Each face quad has 4 vertices, and there are rings * segments quads.
        let expected = (rings * segments * 4) as usize;
        assert_eq!(result.meshes[0].vertex_data.vertices.len(), expected);
    }

    #[test]
    fn test_sphere_normals_normalized() {
        let result = build_uv_sphere_model(1.0, 16, 12);
        assert_normals_normalized(&result.meshes);
    }

    #[test]
    fn test_sphere_indices_in_range() {
        let result = build_uv_sphere_model(1.0, 16, 12);
        assert_indices_in_range(&result.meshes);
    }
}
