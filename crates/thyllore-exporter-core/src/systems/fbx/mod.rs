pub mod animation;
pub(crate) mod build;
pub(crate) mod connections;
pub(crate) mod curves;
pub(crate) mod geometry;
pub(crate) mod mesh_material;
pub(crate) mod skin;
pub(crate) mod writer;

use std::path::Path;

use fbxcel::low::FbxVersion;
use fbxcel::writer::v7400::binary::Writer;

use thyllore_anim_core::editable::EditableAnimationClip;
use thyllore_anim_core::Skeleton;
use thyllore_importer_core::fbx::fbx::FbxModel;

use crate::systems::fbx::build::build_full_export_data;
use crate::systems::fbx::writer::*;

pub fn export_full_fbx(
    fbx_model: &FbxModel,
    clip: Option<&EditableAnimationClip>,
    skeleton: &Skeleton,
    path: &Path,
) -> anyhow::Result<()> {
    let export_data = build_full_export_data(fbx_model, clip, skeleton, path)?;

    let file = std::fs::File::create(path)?;
    let writer = Writer::new(file, FbxVersion::V7_4)
        .map_err(|e| anyhow::anyhow!("FBX writer init failed: {}", e))?;

    write_full_fbx_binary(writer, &export_data)
        .map_err(|e| anyhow::anyhow!("FBX write failed: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::systems::fbx::geometry::{
        convert_positions_to_fbx, convert_uvs_to_fbx, encode_triangle_polygon_indices,
    };
    use crate::systems::fbx::skin::matrix4_to_flat_f64_scaled;
    use cgmath::Matrix4;
    use std::path::PathBuf;
    use thyllore_importer_core::fbx::fbx::FbxData;

    #[test]
    fn test_encode_triangle_polygon_indices() {
        let indices = vec![0, 1, 2, 3, 4, 5];
        let encoded = encode_triangle_polygon_indices(&indices);
        assert_eq!(encoded, vec![0, 1, -3, 3, 4, -6]);
    }

    #[test]
    fn test_encode_triangle_polygon_indices_single() {
        let indices = vec![0, 1, 2];
        let encoded = encode_triangle_polygon_indices(&indices);
        assert_eq!(encoded, vec![0, 1, -3]);
    }

    #[test]
    fn test_convert_uvs_to_fbx_flip() {
        let mut fbx_data = FbxData::new();
        fbx_data.tex_coords = vec![[0.5, 0.3]];
        let uv_values = convert_uvs_to_fbx(&fbx_data);
        assert!((uv_values[0] - 0.5).abs() < 1e-6);
        assert!((uv_values[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_matrix4_to_flat_f64_scaled() {
        use cgmath::SquareMatrix;
        let identity = Matrix4::<f32>::identity();
        let flat = matrix4_to_flat_f64_scaled(&identity, 2.0);
        assert!((flat[0] - 1.0).abs() < 1e-8);
        assert!((flat[5] - 1.0).abs() < 1e-8);
        assert!((flat[10] - 1.0).abs() < 1e-8);
        assert!((flat[15] - 1.0).abs() < 1e-8);
        assert!((flat[12] - 0.0).abs() < 1e-8);
    }

    #[test]
    fn test_convert_positions_to_fbx_no_scale() {
        let mut fbx_data = FbxData::new();
        fbx_data.positions = vec![cgmath::Vector3::new(0.01, 0.02, 0.03)];
        fbx_data.local_positions = vec![];

        let positions = convert_positions_to_fbx(&fbx_data, 1.0);
        assert!((positions[0] - 0.01).abs() < 1e-6);
        assert!((positions[1] - 0.02).abs() < 1e-6);
        assert!((positions[2] - 0.03).abs() < 1e-6);
    }

    #[test]
    fn test_fbx_roundtrip_stickman() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping roundtrip test: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");

        let result =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");
        let (load_result, _) = result;

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("roundtrip_test.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        let orig_axes = &original_scene.settings.axes;
        let exp_axes = &exported_scene.settings.axes;
        assert_eq!(
            orig_axes.up as i32, exp_axes.up as i32,
            "UpAxis mismatch: original={:?}, exported={:?}",
            orig_axes.up, exp_axes.up
        );
        assert_eq!(
            orig_axes.front as i32, exp_axes.front as i32,
            "FrontAxis mismatch: original={:?}, exported={:?}",
            orig_axes.front, exp_axes.front
        );
        assert_eq!(
            orig_axes.right as i32, exp_axes.right as i32,
            "CoordAxis mismatch: original={:?}, exported={:?}",
            orig_axes.right, exp_axes.right
        );

        let orig_non_root_nodes: Vec<_> =
            original_scene.nodes.iter().filter(|n| !n.is_root).collect();
        let exp_non_root_nodes: Vec<_> =
            exported_scene.nodes.iter().filter(|n| !n.is_root).collect();

        let orig_names: std::collections::HashSet<String> = orig_non_root_nodes
            .iter()
            .map(|n| n.element.name.to_string())
            .collect();
        let exp_names: std::collections::HashSet<String> = exp_non_root_nodes
            .iter()
            .map(|n| n.element.name.to_string())
            .collect();

        let missing_in_export: Vec<_> = orig_names.difference(&exp_names).collect();
        let extra_in_export: Vec<_> = exp_names.difference(&orig_names).collect();
        assert!(
            missing_in_export.is_empty(),
            "Nodes missing in exported FBX: {:?}",
            missing_in_export
        );
        if !extra_in_export.is_empty() {
            eprintln!("Extra nodes in export (acceptable): {:?}", extra_in_export);
        }

        for orig_node in &orig_non_root_nodes {
            let name = orig_node.element.name.to_string();
            let exp_node = exp_non_root_nodes
                .iter()
                .find(|n| n.element.name.to_string() == name);

            let Some(exp_node) = exp_node else {
                continue;
            };

            let orig_t = &orig_node.local_transform;
            let exp_t = &exp_node.local_transform;

            let position_tolerance = 0.1;
            let orig_pos = [
                orig_t.translation.x,
                orig_t.translation.y,
                orig_t.translation.z,
            ];
            let exp_pos = [
                exp_t.translation.x,
                exp_t.translation.y,
                exp_t.translation.z,
            ];

            for axis in 0..3 {
                let diff = (orig_pos[axis] - exp_pos[axis]).abs();
                assert!(
                    diff < position_tolerance,
                    "Node '{}' position[{}] mismatch: original={}, exported={}, diff={}",
                    name,
                    axis,
                    orig_pos[axis],
                    exp_pos[axis],
                    diff
                );
            }
        }

        assert!(
            !exported_scene.anim_stacks.is_empty(),
            "Exported FBX has no animation stacks"
        );

        std::fs::remove_file(&export_path).ok();
    }

    #[test]
    fn test_fbx_roundtrip_stickman_with_animation() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);
        assert!(editable.duration > 0.0);
        assert!(!editable.tracks.is_empty());

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("roundtrip_anim_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX with animation");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        assert!(!exported_scene.anim_stacks.is_empty());
        assert!(
            (exported_scene.settings.frames_per_second - original_scene.settings.frames_per_second)
                .abs()
                < 1.0
        );

        let anim_stack = &exported_scene.anim_stacks[0];
        let time_span = anim_stack.time_end - anim_stack.time_begin;
        assert!(
            time_span > 0.1,
            "Animation time span too short: {:.4}s",
            time_span
        );

        let baked = ufbx::bake_anim(
            &exported_scene,
            &exported_scene.anim_stacks[0].anim,
            ufbx::BakeOpts::default(),
        )
        .expect("Failed to bake exported animation");

        let orig_baked = ufbx::bake_anim(
            &original_scene,
            &original_scene.anim_stacks[0].anim,
            ufbx::BakeOpts::default(),
        )
        .expect("Failed to bake original animation");

        let animated_count = baked
            .nodes
            .iter()
            .filter(|n| n.rotation_keys.len() > 1)
            .count();
        assert!(animated_count > 0, "Exported FBX has no animated nodes");

        let mesh_node_names: Vec<String> = fbx_model
            .fbx_data
            .iter()
            .filter_map(|d| d.mesh_node_name.clone())
            .collect();

        for mesh_name in &mesh_node_names {
            let orig_parent = original_scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == *mesh_name)
                .and_then(|n| n.parent.as_ref())
                .map(|p| p.element.name.to_string())
                .unwrap_or_default();
            let exp_parent = exported_scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == *mesh_name)
                .and_then(|n| n.parent.as_ref())
                .map(|p| p.element.name.to_string())
                .unwrap_or_default();
            assert_eq!(
                orig_parent, exp_parent,
                "Parent mismatch for mesh '{}'",
                mesh_name
            );
        }

        for orig_bn in &orig_baked.nodes {
            let orig_idx = orig_bn.typed_id as usize;
            if orig_idx >= original_scene.nodes.len() || orig_bn.rotation_keys.len() <= 2 {
                continue;
            }
            let name = original_scene.nodes[orig_idx].element.name.to_string();

            let exp_bn = baked.nodes.iter().find(|bn| {
                let idx = bn.typed_id as usize;
                idx < exported_scene.nodes.len()
                    && exported_scene.nodes[idx].element.name.to_string() == name
            });
            let exp_bn = match exp_bn {
                Some(b) => b,
                None => continue,
            };

            assert_eq!(
                orig_bn.rotation_keys.len(),
                exp_bn.rotation_keys.len(),
                "Rotation key count mismatch for bone '{}'",
                name
            );

            let sample_indices = [0, 100, 500, 1000];
            for &idx in &sample_indices {
                if idx >= orig_bn.rotation_keys.len() {
                    break;
                }
                let o = &orig_bn.rotation_keys[idx];
                let e = &exp_bn.rotation_keys[idx];
                let max_diff = (o.value.w - e.value.w)
                    .abs()
                    .max((o.value.x - e.value.x).abs())
                    .max((o.value.y - e.value.y).abs())
                    .max((o.value.z - e.value.z).abs());
                assert!(
                    max_diff < 0.01,
                    "Rotation value mismatch for bone '{}' at key {}: diff={}",
                    name,
                    idx,
                    max_diff
                );
            }
        }
    }

    #[test]
    fn test_exported_bone_node_types_match_original() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_compat_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        let mesh_node_names: std::collections::HashSet<String> = fbx_model
            .fbx_data
            .iter()
            .filter_map(|d| d.mesh_node_name.clone())
            .collect();

        let bone_only_names: std::collections::HashSet<String> = skeleton
            .bones
            .iter()
            .filter(|b| !mesh_node_names.contains(&b.name))
            .map(|b| b.name.clone())
            .collect();

        for exp_node in exported_scene.nodes.iter() {
            let name = exp_node.element.name.to_string();
            if !bone_only_names.contains(&name) {
                continue;
            }

            let orig_node = original_scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == name);

            if let Some(orig_node) = orig_node {
                assert_eq!(
                    exp_node.attrib_type as i32, orig_node.attrib_type as i32,
                    "Node '{}' attrib_type mismatch: exported={:?}, original={:?}",
                    name, exp_node.attrib_type, orig_node.attrib_type,
                );
            }

            assert_eq!(
                exp_node.attrib_type,
                ufbx::ElementType::Unknown,
                "Node '{}' should have no NodeAttribute (attrib_type=Unknown) for Blender object-level animation, got {:?}",
                name,
                exp_node.attrib_type,
            );
        }

        let exported_bone_count = exported_scene
            .nodes
            .iter()
            .filter(|n| n.attrib_type == ufbx::ElementType::Bone)
            .count();
        assert_eq!(
            exported_bone_count, 0,
            "Exported FBX should have 0 Bone-type nodes for object-level animation, found {}",
            exported_bone_count,
        );

        std::fs::remove_file(&export_path).ok();
    }

    #[test]
    fn test_compare_anim_structure() {
        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("anim_structure_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX with animation");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        print_scene_anim_structure("ORIGINAL", &original_scene);
        eprintln!("\n{}\n", "=".repeat(80));
        print_scene_anim_structure("EXPORTED", &exported_scene);

        eprintln!("\n{}\n", "=".repeat(80));
        print_anim_prop_connections("ORIGINAL", &original_scene);
        eprintln!("\n{}\n", "=".repeat(80));
        print_anim_prop_connections("EXPORTED", &exported_scene);

        std::fs::remove_file(&export_path).ok();
    }

    fn canonicalize_no_prefix(path: &std::path::Path) -> PathBuf {
        match path.canonicalize() {
            Ok(p) => {
                let s = p.to_string_lossy();
                if let Some(stripped) = s.strip_prefix(r"\\?\") {
                    PathBuf::from(stripped)
                } else {
                    p
                }
            }
            Err(_) => path.to_path_buf(),
        }
    }

    fn read_blender_path() -> Option<String> {
        let paths_file = std::path::Path::new(".claude/local/paths.md");
        let content = std::fs::read_to_string(paths_file).ok()?;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("- BlenderPath = ") {
                let path = rest.trim().to_string();
                if std::path::Path::new(&path).exists() {
                    return Some(path);
                }
            }
        }
        None
    }

    #[test]
    fn test_blender_animation_import() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured in .claude/local/paths.md");
                return;
            }
        };

        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let script_path = "scripts/blender_fbx_diagnostic.py";
        if !std::path::Path::new(script_path).exists() {
            eprintln!("Skipping: {} not found", script_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();
        let anim_clip = load_result.clips.first().expect("No animation clip found");

        let bone_names: std::collections::HashMap<u32, String> = skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect();
        let editable = thyllore_anim_core::editable::clip_from_animation(1, anim_clip, &bone_names);

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_anim_test.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &export_path)
            .expect("Failed to export FBX");

        let abs_export = canonicalize_no_prefix(&export_path);
        let abs_script = canonicalize_no_prefix(std::path::Path::new(script_path));

        let abs_output = canonicalize_no_prefix(std::path::Path::new("assets/exports"))
            .join("blender_diagnostic.json");

        let output = std::process::Command::new(&blender_path)
            .args([
                "--background",
                "--python",
                abs_script.to_str().unwrap(),
                "--",
                abs_export.to_str().unwrap(),
                abs_output.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run Blender");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Blender stdout:\n{}", stdout);
        eprintln!("Blender stderr:\n{}", stderr);

        assert!(
            output.status.success(),
            "Blender exited with error: {:?}",
            output.status,
        );

        assert!(
            abs_output.exists(),
            "Blender diagnostic JSON not created at {:?}",
            abs_output,
        );

        let json_content =
            std::fs::read_to_string(&abs_output).expect("Failed to read diagnostic JSON");
        let diagnostic: serde_json::Value =
            serde_json::from_str(&json_content).expect("Failed to parse diagnostic JSON");

        let summary = &diagnostic["summary"];

        let total_actions = summary["total_actions"].as_u64().unwrap_or(0);
        eprintln!("Blender imported actions: {}", total_actions);
        assert!(
            total_actions > 0,
            "Blender should import at least 1 action, got {}",
            total_actions,
        );

        let total_fcurves = summary["total_fcurves"].as_u64().unwrap_or(0);
        eprintln!("Blender imported FCurves: {}", total_fcurves);
        assert!(
            total_fcurves > 0,
            "Blender should import FCurves, got {}",
            total_fcurves,
        );

        let moved = summary["moved"].as_array().map(|a| a.len()).unwrap_or(0);
        eprintln!("Objects that moved during playback: {}", moved);
        assert!(
            moved > 0,
            "At least some objects should move during animation playback, got {}",
            moved,
        );

        std::fs::remove_file(&export_path).ok();
        std::fs::remove_file(&abs_output).ok();
    }

    fn print_scene_anim_structure(label: &str, scene: &ufbx::Scene) {
        eprintln!("--- {} ---", label);
        eprintln!("  anim_stacks: {}", scene.anim_stacks.len());
        eprintln!("  anim_layers: {}", scene.anim_layers.len());
        eprintln!("  anim_values: {}", scene.anim_values.len());
        eprintln!("  anim_curves: {}", scene.anim_curves.len());
        eprintln!("  total nodes: {}", scene.nodes.len());

        for (i, stack) in scene.anim_stacks.iter().enumerate() {
            eprintln!(
                "  AnimStack[{}]: name='{}', time_begin={:.4}, time_end={:.4}, layers={}",
                i,
                stack.element.name,
                stack.time_begin,
                stack.time_end,
                stack.layers.len(),
            );
        }

        let bake_opts = ufbx::BakeOpts::default();
        let baked = ufbx::bake_anim(scene, &scene.anim_stacks[0].anim, bake_opts)
            .expect("Failed to bake animation");

        let mut bone_only_animated = 0u32;
        let mut mesh_node_animated = 0u32;

        eprintln!("  Baked nodes total: {}", baked.nodes.len());
        for bn in &baked.nodes {
            let has_translation_anim = bn.translation_keys.len() > 1;
            let has_rotation_anim = bn.rotation_keys.len() > 1;
            let has_scale_anim = bn.scale_keys.len() > 1;
            if !has_translation_anim && !has_rotation_anim && !has_scale_anim {
                continue;
            }

            let node_idx = bn.typed_id as usize;
            if node_idx >= scene.nodes.len() {
                continue;
            }
            let node = &scene.nodes[node_idx];
            let name = node.element.name.to_string();
            let has_mesh = node.mesh.is_some();
            let attrib_type = node.attrib_type;

            if has_mesh {
                mesh_node_animated += 1;
            } else {
                bone_only_animated += 1;
            }

            eprintln!(
                "    ANIMATED: '{}' attrib={:?} has_mesh={} t_keys={} r_keys={} s_keys={} const_t={} const_r={} const_s={}",
                name,
                attrib_type,
                has_mesh,
                bn.translation_keys.len(),
                bn.rotation_keys.len(),
                bn.scale_keys.len(),
                bn.constant_translation,
                bn.constant_rotation,
                bn.constant_scale,
            );
        }

        eprintln!(
            "  SUMMARY: bone-only animated={}, mesh-node animated={}",
            bone_only_animated, mesh_node_animated
        );
    }

    #[test]
    fn test_fbx_roundtrip_skinned_fly() {
        let original_path = "tests/testmodels/fbx/skinning/source/fly.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("roundtrip_skinned_fly.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let original_scene = ufbx::load_file(original_path, ufbx::LoadOpts::default())
            .expect("Failed to load original with ufbx");
        let exported_scene =
            ufbx::load_file(export_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to load exported with ufbx");

        assert!(
            !exported_scene.meshes.is_empty(),
            "Exported FBX should have at least one mesh",
        );

        assert!(
            !exported_scene.materials.is_empty(),
            "Exported FBX should have at least one material",
        );

        assert!(
            !exported_scene.skin_clusters.is_empty(),
            "Exported FBX should have skin clusters",
        );

        let orig_bone_names: std::collections::HashSet<String> = original_scene
            .skin_clusters
            .iter()
            .filter_map(|c| c.bone_node.as_ref().map(|n| n.element.name.to_string()))
            .collect();
        let exp_bone_names: std::collections::HashSet<String> = exported_scene
            .skin_clusters
            .iter()
            .filter_map(|c| c.bone_node.as_ref().map(|n| n.element.name.to_string()))
            .collect();
        let missing_bones: Vec<_> = orig_bone_names.difference(&exp_bone_names).collect();
        assert!(
            missing_bones.is_empty(),
            "Missing bone references in exported clusters: {:?}",
            missing_bones,
        );

        for exp_cluster in exported_scene.skin_clusters.iter() {
            assert!(
                exp_cluster.bone_node.is_some(),
                "Exported cluster should have a bone_node reference",
            );
            assert!(
                exp_cluster.num_weights > 0,
                "Exported cluster should have vertex weights",
            );
        }

        for exp_mesh in exported_scene.meshes.iter() {
            assert!(
                !exp_mesh.materials.is_empty(),
                "Exported mesh should have at least one material reference",
            );
            assert!(
                !exp_mesh.skin_deformers.is_empty(),
                "Exported skinned mesh should have a skin deformer",
            );
        }

        assert!(
            (original_scene.settings.unit_meters - exported_scene.settings.unit_meters).abs()
                < 1e-6,
            "UnitScaleFactor mismatch: original unit_meters={}, exported unit_meters={}",
            original_scene.settings.unit_meters,
            exported_scene.settings.unit_meters,
        );

        let orig_g2b_map: std::collections::HashMap<String, &ufbx::Matrix> = original_scene
            .skin_clusters
            .iter()
            .filter_map(|c| {
                c.bone_node
                    .as_ref()
                    .map(|n| (n.element.name.to_string(), &c.geometry_to_bone))
            })
            .collect();

        for exp_cluster in exported_scene.skin_clusters.iter() {
            let bone_name = exp_cluster
                .bone_node
                .as_ref()
                .map(|n| n.element.name.to_string())
                .unwrap_or_default();

            if let Some(&orig_g2b) = orig_g2b_map.get(&bone_name) {
                let diff = (orig_g2b.m03 - exp_cluster.geometry_to_bone.m03).abs()
                    + (orig_g2b.m13 - exp_cluster.geometry_to_bone.m13).abs()
                    + (orig_g2b.m23 - exp_cluster.geometry_to_bone.m23).abs();
                assert!(
                    diff < 0.1,
                    "geometry_to_bone translation mismatch for bone '{}': \
                     orig=[{:.4}, {:.4}, {:.4}], exp=[{:.4}, {:.4}, {:.4}]",
                    bone_name,
                    orig_g2b.m03,
                    orig_g2b.m13,
                    orig_g2b.m23,
                    exp_cluster.geometry_to_bone.m03,
                    exp_cluster.geometry_to_bone.m13,
                    exp_cluster.geometry_to_bone.m23,
                );
            }
        }

        let orig_mat_map: std::collections::HashMap<String, &ufbx::Material> = original_scene
            .materials
            .iter()
            .map(|m| (m.element.name.to_string(), m.as_ref()))
            .collect();

        for exp_mat in exported_scene.materials.iter() {
            let name = exp_mat.element.name.to_string();
            if let Some(orig_mat) = orig_mat_map.get(&name) {
                let orig_dc = &orig_mat.fbx.diffuse_color.value_vec4;
                let exp_dc = &exp_mat.fbx.diffuse_color.value_vec4;
                let color_diff = (orig_dc.x - exp_dc.x).abs()
                    + (orig_dc.y - exp_dc.y).abs()
                    + (orig_dc.z - exp_dc.z).abs();
                assert!(
                    color_diff < 0.01,
                    "DiffuseColor mismatch for '{}': orig=[{:.3},{:.3},{:.3}] exp=[{:.3},{:.3},{:.3}]",
                    name, orig_dc.x, orig_dc.y, orig_dc.z, exp_dc.x, exp_dc.y, exp_dc.z,
                );

                assert!(
                    exp_mat.fbx.diffuse_factor.has_value,
                    "DiffuseFactor must be explicitly set for '{}'",
                    name,
                );
            }
        }

        for exp_mat in exported_scene.materials.iter() {
            let name = exp_mat.element.name.to_string();
            if let Some(orig_mat) = orig_mat_map.get(&name) {
                let orig_has_tex = orig_mat.fbx.diffuse_color.texture.is_some();
                let exp_has_tex = exp_mat.fbx.diffuse_color.texture.is_some();
                assert_eq!(
                    orig_has_tex, exp_has_tex,
                    "Texture presence mismatch for '{}': original={}, exported={}",
                    name, orig_has_tex, exp_has_tex,
                );

                if let (Some(orig_tex), Some(exp_tex)) = (
                    orig_mat.fbx.diffuse_color.texture.as_ref(),
                    exp_mat.fbx.diffuse_color.texture.as_ref(),
                ) {
                    let orig_stem = Path::new(&orig_tex.filename.to_string())
                        .file_stem()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    let exp_basename = Path::new(&exp_tex.filename.to_string())
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    assert!(
                        exp_basename.starts_with(&orig_stem),
                        "Texture filename mismatch for '{}': original stem='{}', exported='{}'",
                        name,
                        orig_stem,
                        exp_basename,
                    );
                }
            }
        }

        std::fs::remove_file(&export_path).ok();
    }

    fn print_anim_prop_connections(label: &str, scene: &ufbx::Scene) {
        eprintln!("--- {} AnimProp connections ---", label);

        if scene.anim_layers.is_empty() {
            eprintln!("  No anim layers");
            return;
        }

        let layer = &scene.anim_layers[0];
        eprintln!(
            "  AnimLayer '{}' has {} anim_props",
            layer.element.name,
            layer.anim_props.len()
        );

        let mut node_prop_map: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();

        for ap in &layer.anim_props {
            let target_name = ap.element.name.to_string();
            let prop_name = ap.prop_name.to_string();
            node_prop_map
                .entry(target_name)
                .or_default()
                .push(prop_name);
        }

        for (target_name, props) in &node_prop_map {
            let node = scene
                .nodes
                .iter()
                .find(|n| n.element.name.to_string() == *target_name);

            let (has_mesh, attrib_type) = match node {
                Some(n) => (n.mesh.is_some(), format!("{:?}", n.attrib_type)),
                None => (false, "NOT_A_NODE".to_string()),
            };

            eprintln!(
                "    target='{}' attrib={} has_mesh={} props={:?}",
                target_name, attrib_type, has_mesh, props
            );
        }
    }

    #[test]
    fn test_blender_skinned_fly_import() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured");
                return;
            }
        };

        let original_path = "tests/testmodels/fbx/skinning/source/fly.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let script_path = "scripts/blender_fbx_diagnostic.py";
        if !std::path::Path::new(script_path).exists() {
            eprintln!("Skipping: {} not found", script_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load original FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_skinned_fly.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let abs_export = canonicalize_no_prefix(&export_path);
        let abs_script = canonicalize_no_prefix(std::path::Path::new(script_path));

        let abs_output = canonicalize_no_prefix(std::path::Path::new("assets/exports"))
            .join("blender_skinned_diagnostic.json");

        let output = std::process::Command::new(&blender_path)
            .args([
                "--background",
                "--python",
                abs_script.to_str().unwrap(),
                "--",
                abs_export.to_str().unwrap(),
                abs_output.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run Blender");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("Blender stdout:\n{}", stdout);
        if !stderr.is_empty() {
            eprintln!("Blender stderr:\n{}", stderr);
        }

        assert!(
            output.status.success(),
            "Blender exited with error: {:?}",
            output.status,
        );

        assert!(
            abs_output.exists(),
            "Blender diagnostic JSON not created at {:?}",
            abs_output,
        );

        let json_content =
            std::fs::read_to_string(&abs_output).expect("Failed to read diagnostic JSON");
        let diagnostic: serde_json::Value =
            serde_json::from_str(&json_content).expect("Failed to parse diagnostic JSON");

        let summary = &diagnostic["summary"];

        let total_materials = summary["total_materials"].as_u64().unwrap_or(0);
        eprintln!("Blender imported materials: {}", total_materials);
        assert!(
            total_materials > 0,
            "Blender should import at least 1 material, got {}",
            total_materials,
        );

        let missing_textures = summary["textures_missing"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0);
        eprintln!("Missing textures: {}", missing_textures);
        assert_eq!(
            missing_textures, 0,
            "All textures should be found, but {} are missing: {:?}",
            missing_textures, summary["textures_missing"],
        );

        if let Some(mesh_bounds) = diagnostic["mesh_bounds"].as_array() {
            for mb in mesh_bounds {
                let name = mb["name"].as_str().unwrap_or("");
                let bbox_min = &mb["bbox_min"];
                let bbox_max = &mb["bbox_max"];
                eprintln!("Mesh '{}': min={}, max={}", name, bbox_min, bbox_max);

                let max_coord = bbox_max
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    max_coord < 100.0,
                    "Mesh '{}' bbox is too large (max_coord={}), likely wrong scale",
                    name,
                    max_coord,
                );
            }
        }

        std::fs::remove_file(&export_path).ok();
        std::fs::remove_file(&abs_output).ok();
    }

    fn run_blender_import(blender_path: &str, fbx_path: &Path) -> (String, String, bool) {
        let script = r#"
import bpy, sys
argv = sys.argv
idx = argv.index("--") if "--" in argv else len(argv)
fbx_path = argv[idx + 1]
for obj in bpy.data.objects:
    obj.select_set(True)
bpy.ops.object.delete()
bpy.ops.import_scene.fbx(filepath=fbx_path)
print("IMPORT_DONE")
"#;

        let temp_script = std::env::temp_dir().join("blender_import_check.py");
        std::fs::write(&temp_script, script).expect("Failed to write temp script");

        let abs_fbx = canonicalize_no_prefix(fbx_path);

        let output = std::process::Command::new(blender_path)
            .args([
                "--background",
                "--python",
                temp_script.to_str().unwrap(),
                "--",
                abs_fbx.to_str().unwrap(),
            ])
            .output()
            .expect("Failed to run Blender");

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        std::fs::remove_file(&temp_script).ok();
        (stdout, stderr, output.status.success())
    }

    fn collect_fbx_import_warnings(stdout: &str) -> Vec<String> {
        stdout
            .lines()
            .filter(|line| {
                let lower = line.to_lowercase();
                lower.starts_with("warning") && lower.contains("layer")
            })
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn test_blender_no_import_warnings_stickman() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured");
                return;
            }
        };

        let original_path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");
        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_warn_test_stickman.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let (stdout, _stderr, success) = run_blender_import(&blender_path, &export_path);
        assert!(success, "Blender exited with error");
        assert!(
            stdout.contains("IMPORT_DONE"),
            "Blender import did not complete",
        );

        let warnings = collect_fbx_import_warnings(&stdout);
        eprintln!("FBX import warnings: {:?}", warnings);
        assert!(
            warnings.is_empty(),
            "Blender FBX import produced warnings:\n{}",
            warnings.join("\n"),
        );

        std::fs::remove_file(&export_path).ok();
    }

    #[test]
    fn test_blender_no_import_warnings_skinned() {
        let blender_path = match read_blender_path() {
            Some(p) => p,
            None => {
                eprintln!("Skipping: BlenderPath not configured");
                return;
            }
        };

        let original_path = "tests/testmodels/fbx/skinning/source/fly.fbx";
        if !std::path::Path::new(original_path).exists() {
            eprintln!("Skipping: {} not found", original_path);
            return;
        }

        let fbx_model = thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(original_path)
            .expect("Failed to load FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(original_path)
                .expect("Failed to load graphics resources");
        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton found")
            .clone();

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let export_path = export_dir.join("blender_warn_test_skinned.fbx");

        export_full_fbx(&fbx_model, None, &skeleton, &export_path).expect("Failed to export FBX");

        let (stdout, _stderr, success) = run_blender_import(&blender_path, &export_path);
        assert!(success, "Blender exited with error");
        assert!(
            stdout.contains("IMPORT_DONE"),
            "Blender import did not complete",
        );

        let warnings = collect_fbx_import_warnings(&stdout);
        eprintln!("FBX import warnings: {:?}", warnings);
        assert!(
            warnings.is_empty(),
            "Blender FBX import produced warnings:\n{}",
            warnings.join("\n"),
        );

        std::fs::remove_file(&export_path).ok();
    }

    fn load_stickman_for_roundtrip() -> Option<(
        FbxModel,
        thyllore_anim_core::Skeleton,
        thyllore_anim_core::AnimationClip,
    )> {
        let path = "assets/models/stickman/stickman_bin.fbx";
        if !std::path::Path::new(path).exists() {
            return None;
        }

        let fbx_model =
            thyllore_importer_core::fbx::fbx::load_fbx_with_ufbx(path).expect("Failed to load FBX");
        let (load_result, _) =
            thyllore_importer_core::fbx::loader::load_fbx_to_graphics_resources(path)
                .expect("Failed to load graphics");

        let skeleton = load_result
            .animation_system
            .get_skeleton(0)
            .expect("No skeleton")
            .clone();
        let clip = load_result.clips.first().expect("No clip").clone();

        Some((fbx_model, skeleton, clip))
    }

    fn build_bone_name_map(
        skeleton: &thyllore_anim_core::Skeleton,
    ) -> std::collections::HashMap<u32, String> {
        skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| (i as u32, b.name.clone()))
            .collect()
    }

    fn find_rotation_x_curve_for_bone<'a>(
        scene: &'a ufbx::Scene,
        bone_name: &str,
    ) -> Option<&'a ufbx::AnimCurve> {
        if scene.anim_layers.is_empty() {
            return None;
        }

        let layer = &scene.anim_layers[0];
        for ap in &layer.anim_props {
            let target_name = ap.element.name.to_string();
            let prop_name = ap.prop_name.to_string();

            if target_name == bone_name && prop_name == "Lcl Rotation" {
                return ap.anim_value.curves[0].as_ref().map(|r| &**r);
            }
        }

        None
    }

    #[test]
    fn test_weighted_tangent_preserved_on_fbx_roundtrip() {
        let Some((fbx_model, skeleton, clip)) = load_stickman_for_roundtrip() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };

        let bone_names = build_bone_name_map(&skeleton);
        let mut editable = thyllore_anim_core::editable::clip_from_animation(1, &clip, &bone_names);

        let target_bone_name = skeleton.bones[2].name.clone();

        use thyllore_anim_core::editable::{BezierHandle, InterpolationType, TangentWeightMode};

        fn set_weighted_tangents(
            editable: &mut EditableAnimationClip,
            bone_id: u32,
            in_handle: &BezierHandle,
            out_handle: &BezierHandle,
        ) {
            let track = editable
                .tracks
                .get_mut(&bone_id)
                .expect("Bone track not found");
            for kf in &mut track.rotation_x.keyframes {
                kf.interpolation = InterpolationType::Bezier;
                kf.weight_mode = TangentWeightMode::Weighted;
                kf.out_tangent = out_handle.clone();
                kf.in_tangent = in_handle.clone();
            }
        }

        set_weighted_tangents(
            &mut editable,
            2,
            &BezierHandle::new(-0.15, -5.0),
            &BezierHandle::new(0.15, 5.0),
        );

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let weighted_path = export_dir.join("weighted_tangent_roundtrip.fbx");
        let flat_path = export_dir.join("flat_tangent_roundtrip.fbx");

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &weighted_path)
            .expect("Failed to export weighted");

        set_weighted_tangents(
            &mut editable,
            2,
            &BezierHandle::new(-0.15, 0.0),
            &BezierHandle::new(0.15, 0.0),
        );

        export_full_fbx(&fbx_model, Some(&editable), &skeleton, &flat_path)
            .expect("Failed to export flat");

        let weighted_scene =
            ufbx::load_file(weighted_path.to_str().unwrap(), ufbx::LoadOpts::default())
                .expect("Failed to reload weighted");
        let flat_scene = ufbx::load_file(flat_path.to_str().unwrap(), ufbx::LoadOpts::default())
            .expect("Failed to reload flat");

        let weighted_curve = find_rotation_x_curve_for_bone(&weighted_scene, &target_bone_name)
            .expect("Weighted curve not found");
        let flat_curve = find_rotation_x_curve_for_bone(&flat_scene, &target_bone_name)
            .expect("Flat curve not found");

        assert!(
            weighted_curve.keyframes.len() >= 2,
            "Weighted curve should have keyframes"
        );
        assert_eq!(
            weighted_curve.keyframes.len(),
            flat_curve.keyframes.len(),
            "Both exports should have the same number of keyframes"
        );

        let has_cubic = weighted_curve
            .keyframes
            .iter()
            .any(|kf| kf.interpolation == ufbx::Interpolation::Cubic);
        assert!(
            has_cubic,
            "Re-imported curve should have cubic interpolation"
        );

        let mut max_shape_diff: f64 = 0.0;
        let duration = editable.duration as f64;
        for i in 1..10 {
            let t = i as f64 * duration / 10.0;
            let weighted_val = ufbx::evaluate_curve(weighted_curve, t, 0.0);
            let flat_val = ufbx::evaluate_curve(flat_curve, t, 0.0);
            let diff = (weighted_val - flat_val).abs();
            if diff > max_shape_diff {
                max_shape_diff = diff;
            }
        }

        assert!(
            max_shape_diff > 0.5,
            "Weighted tangent handles should produce a different curve shape than flat handles, max_diff={:.4}",
            max_shape_diff
        );

        let has_nonzero_tangent = weighted_curve.keyframes.iter().any(|kf| {
            kf.right.dx.abs() > 1e-6
                || kf.right.dy.abs() > 1e-6
                || kf.left.dx.abs() > 1e-6
                || kf.left.dy.abs() > 1e-6
        });
        assert!(
            has_nonzero_tangent,
            "Weighted tangent should have at least one non-zero tangent"
        );

        std::fs::remove_file(&weighted_path).ok();
        std::fs::remove_file(&flat_path).ok();
    }

    #[test]
    fn test_non_weighted_tangent_unchanged_on_fbx_roundtrip() {
        let Some((fbx_model, skeleton, clip)) = load_stickman_for_roundtrip() else {
            eprintln!("Skipping: stickman model not found");
            return;
        };

        let bone_names = build_bone_name_map(&skeleton);
        let mut editable_a =
            thyllore_anim_core::editable::clip_from_animation(1, &clip, &bone_names);
        let mut editable_b =
            thyllore_anim_core::editable::clip_from_animation(2, &clip, &bone_names);

        let target_bone_name = skeleton.bones[2].name.clone();

        use thyllore_anim_core::editable::{
            curve_recalculate_auto_tangents, InterpolationType, TangentWeightMode,
        };

        for editable in [&mut editable_a, &mut editable_b] {
            let track = editable.tracks.get_mut(&2).expect("Bone track not found");
            for kf in &mut track.rotation_x.keyframes {
                kf.interpolation = InterpolationType::Bezier;
                kf.weight_mode = TangentWeightMode::NonWeighted;
            }
            curve_recalculate_auto_tangents(&mut track.rotation_x);
        }

        let export_dir = std::path::Path::new("assets/exports");
        std::fs::create_dir_all(export_dir).ok();
        let path_a = export_dir.join("non_weighted_roundtrip_a.fbx");
        let path_b = export_dir.join("non_weighted_roundtrip_b.fbx");

        export_full_fbx(&fbx_model, Some(&editable_a), &skeleton, &path_a)
            .expect("Failed to export A");
        export_full_fbx(&fbx_model, Some(&editable_b), &skeleton, &path_b)
            .expect("Failed to export B");

        let scene_a = ufbx::load_file(path_a.to_str().unwrap(), ufbx::LoadOpts::default())
            .expect("Failed to reload A");
        let scene_b = ufbx::load_file(path_b.to_str().unwrap(), ufbx::LoadOpts::default())
            .expect("Failed to reload B");

        let curve_a =
            find_rotation_x_curve_for_bone(&scene_a, &target_bone_name).expect("Curve A not found");
        let curve_b =
            find_rotation_x_curve_for_bone(&scene_b, &target_bone_name).expect("Curve B not found");

        assert!(
            curve_a.keyframes.len() >= 2,
            "Re-imported curve should have keyframes"
        );
        assert_eq!(
            curve_a.keyframes.len(),
            curve_b.keyframes.len(),
            "Both exports should have the same number of keyframes"
        );

        let duration = editable_a.duration as f64;
        for i in 0..=10 {
            let t = i as f64 * duration / 10.0;
            let val_a = ufbx::evaluate_curve(curve_a, t, 0.0);
            let val_b = ufbx::evaluate_curve(curve_b, t, 0.0);
            let diff = (val_a - val_b).abs();
            assert!(
                diff < 1e-4,
                "Non-weighted tangent exports should be identical at t={:.2}: a={:.4}, b={:.4}, diff={:.6}",
                t, val_a, val_b, diff
            );
        }

        for (kf_a, kf_b) in curve_a.keyframes.iter().zip(curve_b.keyframes.iter()) {
            assert!(
                (kf_a.right.dx - kf_b.right.dx).abs() < 1e-4
                    && (kf_a.right.dy - kf_b.right.dy).abs() < 1e-4,
                "Non-weighted tangent data should be identical: a=({}, {}), b=({}, {})",
                kf_a.right.dx,
                kf_a.right.dy,
                kf_b.right.dx,
                kf_b.right.dy
            );
        }

        std::fs::remove_file(&path_a).ok();
        std::fs::remove_file(&path_b).ok();
    }
}
