use cgmath::{Vector3, Vector4};

use crate::animation::{BoneId, SkinData};
use crate::ecs::resource::gizmo::BoneSelectionState;
use crate::ecs::resource::{HeatmapApplication, WeightHeatmapState};
use crate::math::Vec4;
use crate::vulkanr::data::Vertex;
use crate::vulkanr::resource::graphics_resource::GraphicsResources;
use crate::vulkanr::resource::mesh_buffer::MeshBuffer;

pub fn update_weight_heatmap(
    graphics: &mut GraphicsResources,
    heatmap: &mut WeightHeatmapState,
    selection: &BoneSelectionState,
) -> Vec<usize> {
    let desired = if heatmap.enabled {
        HeatmapApplication::Applied(selection.active_bone_index.map(|i| i as BoneId))
    } else {
        HeatmapApplication::NotApplied
    };

    if &desired == heatmap.last_applied() {
        return Vec::new();
    }

    log!(
        "WeightHeatmap: state transition {:?} -> {:?} (meshes={})",
        heatmap.last_applied(),
        desired,
        graphics.meshes.len()
    );

    let mut changed_meshes = Vec::new();
    let mut skipped_no_base_colors = 0;
    let mut skipped_length_mismatch = 0;
    for (mesh_index, mesh) in graphics.meshes.iter_mut().enumerate() {
        match update_mesh_colors(mesh, &desired) {
            MeshUpdateOutcome::Updated => changed_meshes.push(mesh_index),
            MeshUpdateOutcome::SkippedNoBaseColors => skipped_no_base_colors += 1,
            MeshUpdateOutcome::SkippedLengthMismatch => skipped_length_mismatch += 1,
        }
    }

    log!(
        "WeightHeatmap: updated={}, skipped_no_base_colors={}, skipped_length_mismatch={}",
        changed_meshes.len(),
        skipped_no_base_colors,
        skipped_length_mismatch
    );

    heatmap.set_last_applied(desired);
    changed_meshes
}

enum MeshUpdateOutcome {
    Updated,
    SkippedNoBaseColors,
    SkippedLengthMismatch,
}

fn update_mesh_colors(mesh: &mut MeshBuffer, desired: &HeatmapApplication) -> MeshUpdateOutcome {
    let MeshBuffer {
        vertex_data,
        base_colors,
        skin_data,
        ..
    } = mesh;
    let Some(base_colors) = base_colors.as_ref() else {
        return MeshUpdateOutcome::SkippedNoBaseColors;
    };
    if base_colors.len() != vertex_data.vertices.len() {
        return MeshUpdateOutcome::SkippedLengthMismatch;
    }

    let target_bone = match desired {
        HeatmapApplication::Applied(bone_opt) => *bone_opt,
        HeatmapApplication::NotApplied => None,
    };

    match (desired, target_bone, skin_data.as_ref()) {
        (HeatmapApplication::Applied(_), Some(bone_id), Some(skin)) => {
            apply_heatmap_to_vertices(&mut vertex_data.vertices, skin, base_colors, bone_id);
        }
        _ => {
            restore_base_colors(&mut vertex_data.vertices, base_colors);
        }
    }
    MeshUpdateOutcome::Updated
}

fn apply_heatmap_to_vertices(
    vertices: &mut [Vertex],
    skin: &SkinData,
    base_colors: &[Vector4<f32>],
    bone_id: BoneId,
) {
    debug_assert_eq!(vertices.len(), skin.bone_indices.len());
    debug_assert_eq!(vertices.len(), skin.bone_weights.len());
    debug_assert_eq!(vertices.len(), base_colors.len());

    for i in 0..vertices.len() {
        let weight = extract_weight_for_bone(&skin.bone_indices[i], &skin.bone_weights[i], bone_id);
        let rgb = weight_to_heatmap_rgb(weight);
        let base_alpha = base_colors[i].w;
        vertices[i].color = Vec4::new(rgb.x, rgb.y, rgb.z, base_alpha);
    }
}

fn restore_base_colors(vertices: &mut [Vertex], base_colors: &[Vector4<f32>]) {
    debug_assert_eq!(vertices.len(), base_colors.len());
    for (i, base) in base_colors.iter().enumerate() {
        vertices[i].color = Vec4::new(base.x, base.y, base.z, base.w);
    }
}

fn extract_weight_for_bone(indices: &Vector4<u32>, weights: &Vector4<f32>, bone_id: BoneId) -> f32 {
    let slots = [
        (indices.x, weights.x),
        (indices.y, weights.y),
        (indices.z, weights.z),
        (indices.w, weights.w),
    ];
    slots
        .iter()
        .filter(|(bone, _)| *bone == bone_id)
        .map(|(_, weight)| *weight)
        .sum()
}

fn weight_to_heatmap_rgb(w: f32) -> Vector3<f32> {
    let w = w.clamp(0.0, 1.0);
    let stops = [
        (0.00_f32, Vector3::new(0.0, 0.0, 1.0)),
        (0.25_f32, Vector3::new(0.0, 1.0, 1.0)),
        (0.50_f32, Vector3::new(0.0, 1.0, 0.0)),
        (0.75_f32, Vector3::new(1.0, 1.0, 0.0)),
        (1.00_f32, Vector3::new(1.0, 0.0, 0.0)),
    ];
    for window in stops.windows(2) {
        let (t0, c0) = window[0];
        let (t1, c1) = window[1];
        if w <= t1 {
            let k = (w - t0) / (t1 - t0);
            return c0 * (1.0 - k) + c1 * k;
        }
    }
    stops[stops.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_vec3(a: Vector3<f32>, b: Vector3<f32>) -> bool {
        (a.x - b.x).abs() < 1e-5 && (a.y - b.y).abs() < 1e-5 && (a.z - b.z).abs() < 1e-5
    }

    #[test]
    fn weight_to_heatmap_rgb_returns_blue_at_zero() {
        let color = weight_to_heatmap_rgb(0.0);
        assert!(approx_eq_vec3(color, Vector3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn weight_to_heatmap_rgb_returns_red_at_one() {
        let color = weight_to_heatmap_rgb(1.0);
        assert!(approx_eq_vec3(color, Vector3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn weight_to_heatmap_rgb_returns_green_at_half() {
        let color = weight_to_heatmap_rgb(0.5);
        assert!(approx_eq_vec3(color, Vector3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn weight_to_heatmap_rgb_clamps_below_zero() {
        let color = weight_to_heatmap_rgb(-0.5);
        assert!(approx_eq_vec3(color, Vector3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn weight_to_heatmap_rgb_clamps_above_one() {
        let color = weight_to_heatmap_rgb(2.0);
        assert!(approx_eq_vec3(color, Vector3::new(1.0, 0.0, 0.0)));
    }

    #[test]
    fn extract_weight_for_bone_returns_matching_slot() {
        let indices = Vector4::new(0, 3, 5, 7);
        let weights = Vector4::new(0.1, 0.4, 0.3, 0.2);

        assert!((extract_weight_for_bone(&indices, &weights, 3) - 0.4).abs() < 1e-6);
        assert!((extract_weight_for_bone(&indices, &weights, 5) - 0.3).abs() < 1e-6);
    }

    #[test]
    fn extract_weight_for_bone_returns_zero_when_not_present() {
        let indices = Vector4::new(0, 1, 2, 3);
        let weights = Vector4::new(0.25, 0.25, 0.25, 0.25);

        assert!(extract_weight_for_bone(&indices, &weights, 99).abs() < 1e-6);
    }

    #[test]
    fn extract_weight_for_bone_sums_duplicate_slots() {
        let indices = Vector4::new(5, 5, 3, 1);
        let weights = Vector4::new(0.2, 0.3, 0.4, 0.1);

        assert!((extract_weight_for_bone(&indices, &weights, 5) - 0.5).abs() < 1e-6);
    }

    fn make_skin_single_bone(vertex_count: usize, bone_id: BoneId) -> SkinData {
        let mut skin = SkinData::default();
        skin.bone_indices = vec![Vector4::new(bone_id, 0, 0, 0); vertex_count];
        skin.bone_weights = vec![Vector4::new(1.0, 0.0, 0.0, 0.0); vertex_count];
        skin.base_positions = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
        skin.base_normals = vec![Vector3::new(0.0, 1.0, 0.0); vertex_count];
        skin
    }

    #[test]
    fn apply_heatmap_to_vertices_writes_red_for_full_weight() {
        let mut vertices = vec![Vertex::default(); 3];
        let skin = make_skin_single_bone(3, 5);
        let base_colors = vec![Vector4::new(0.9, 0.8, 0.7, 0.5); 3];

        apply_heatmap_to_vertices(&mut vertices, &skin, &base_colors, 5);

        for vertex in &vertices {
            assert!((vertex.color.x - 1.0).abs() < 1e-5);
            assert!(vertex.color.y.abs() < 1e-5);
            assert!(vertex.color.z.abs() < 1e-5);
            assert!((vertex.color.w - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn apply_heatmap_to_vertices_writes_blue_for_non_matching_bone() {
        let mut vertices = vec![Vertex::default(); 3];
        let skin = make_skin_single_bone(3, 5);
        let base_colors = vec![Vector4::new(0.9, 0.8, 0.7, 1.0); 3];

        apply_heatmap_to_vertices(&mut vertices, &skin, &base_colors, 99);

        for vertex in &vertices {
            assert!(vertex.color.x.abs() < 1e-5);
            assert!(vertex.color.y.abs() < 1e-5);
            assert!((vertex.color.z - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn restore_base_colors_copies_exact_values() {
        let mut vertices = vec![Vertex::default(); 2];
        vertices[0].color = Vec4::new(1.0, 0.0, 0.0, 1.0);
        vertices[1].color = Vec4::new(0.0, 1.0, 0.0, 1.0);

        let base_colors = vec![
            Vector4::new(0.1, 0.2, 0.3, 0.4),
            Vector4::new(0.5, 0.6, 0.7, 0.8),
        ];

        restore_base_colors(&mut vertices, &base_colors);

        assert!((vertices[0].color.x - 0.1).abs() < 1e-5);
        assert!((vertices[0].color.w - 0.4).abs() < 1e-5);
        assert!((vertices[1].color.z - 0.7).abs() < 1e-5);
        assert!((vertices[1].color.w - 0.8).abs() < 1e-5);
    }

    fn build_graphics_with_single_skinned_mesh(bone_id: BoneId) -> GraphicsResources {
        let mut graphics = GraphicsResources::default();
        let mut mesh = MeshBuffer::default();

        let vertex_count = 3;
        mesh.vertex_data.vertices = vec![Vertex::default(); vertex_count];
        mesh.skin_data = Some(make_skin_single_bone(vertex_count, bone_id));
        mesh.base_colors = Some(vec![Vector4::new(0.2, 0.2, 0.2, 1.0); vertex_count]);

        graphics.meshes.push(mesh);
        graphics
    }

    #[test]
    fn update_weight_heatmap_is_idempotent_when_state_unchanged() {
        let mut graphics = build_graphics_with_single_skinned_mesh(5);
        let mut heatmap = WeightHeatmapState::default();
        let mut selection = BoneSelectionState::default();
        heatmap.enabled = true;
        selection.active_bone_index = Some(5);

        let first = update_weight_heatmap(&mut graphics, &mut heatmap, &selection);
        let second = update_weight_heatmap(&mut graphics, &mut heatmap, &selection);

        assert_eq!(first.len(), 1);
        assert!(second.is_empty());
    }

    #[test]
    fn update_weight_heatmap_applies_red_when_enabled_with_matching_bone() {
        let mut graphics = build_graphics_with_single_skinned_mesh(5);
        let mut heatmap = WeightHeatmapState::default();
        let mut selection = BoneSelectionState::default();
        heatmap.enabled = true;
        selection.active_bone_index = Some(5);

        update_weight_heatmap(&mut graphics, &mut heatmap, &selection);

        for vertex in &graphics.meshes[0].vertex_data.vertices {
            assert!((vertex.color.x - 1.0).abs() < 1e-5);
        }
        assert_eq!(
            heatmap.last_applied(),
            &HeatmapApplication::Applied(Some(5))
        );
    }

    #[test]
    fn update_weight_heatmap_restores_base_colors_when_disabled() {
        let mut graphics = build_graphics_with_single_skinned_mesh(5);
        let mut heatmap = WeightHeatmapState::default();
        let mut selection = BoneSelectionState::default();

        heatmap.enabled = true;
        selection.active_bone_index = Some(5);
        update_weight_heatmap(&mut graphics, &mut heatmap, &selection);

        heatmap.enabled = false;
        let changed = update_weight_heatmap(&mut graphics, &mut heatmap, &selection);

        assert_eq!(changed.len(), 1);
        for vertex in &graphics.meshes[0].vertex_data.vertices {
            assert!((vertex.color.x - 0.2).abs() < 1e-5);
            assert!((vertex.color.w - 1.0).abs() < 1e-5);
        }
        assert_eq!(heatmap.last_applied(), &HeatmapApplication::NotApplied);
    }

    #[test]
    fn update_weight_heatmap_enabled_with_no_selection_restores_base_colors() {
        let mut graphics = build_graphics_with_single_skinned_mesh(5);
        let mut heatmap = WeightHeatmapState::default();
        let selection = BoneSelectionState::default();
        heatmap.enabled = true;

        update_weight_heatmap(&mut graphics, &mut heatmap, &selection);

        for vertex in &graphics.meshes[0].vertex_data.vertices {
            assert!((vertex.color.x - 0.2).abs() < 1e-5);
        }
        assert_eq!(heatmap.last_applied(), &HeatmapApplication::Applied(None));
    }
}
