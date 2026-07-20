use std::path::{Path, PathBuf};
use std::process::Command;

use thyllore_animation::animation::editable::{
    clip_add_keyframe, EditableAnimationClip, PropertyType,
};
use thyllore_animation::animation::{AnimationClip, Skeleton};

#[test]
fn test_gltf_export_preserves_structure() {
    let source_path = Path::new("assets/models/stickman/stickman.glb");
    if !source_path.exists() {
        println!("Skipping test: stickman.glb not found");
        return;
    }

    let raw_bytes = std::fs::read(source_path).unwrap();
    let glb = gltf::binary::Glb::from_slice(&raw_bytes).unwrap();
    let original_root: gltf::json::Root = gltf::json::Root::from_slice(&glb.json).unwrap();

    let original_node_count = original_root.nodes.len();
    let original_mesh_count = original_root.meshes.len();
    let original_skin_count = original_root.skins.len();
    let original_material_count = original_root.materials.len();

    println!(
        "Original: {} nodes, {} meshes, {} skins, {} materials",
        original_node_count, original_mesh_count, original_skin_count, original_material_count
    );

    let skeleton = build_skeleton_from_gltf(&original_root);
    let clip = build_test_clip_translation_only(&skeleton);

    let output_path = std::env::temp_dir().join("test_gltf_export.glb");

    thyllore_animation::exporter::gltf::export_gltf_animation(
        source_path,
        &clip,
        &skeleton,
        &output_path,
    )
    .unwrap();

    let exported_bytes = std::fs::read(&output_path).unwrap();
    let exported_glb = gltf::binary::Glb::from_slice(&exported_bytes).unwrap();
    let exported_root: gltf::json::Root = gltf::json::Root::from_slice(&exported_glb.json).unwrap();

    assert_eq!(
        exported_root.nodes.len(),
        original_node_count,
        "node count mismatch"
    );
    assert_eq!(
        exported_root.meshes.len(),
        original_mesh_count,
        "mesh count mismatch"
    );
    assert_eq!(
        exported_root.skins.len(),
        original_skin_count,
        "skin count mismatch"
    );
    assert_eq!(
        exported_root.materials.len(),
        original_material_count,
        "material count mismatch"
    );

    assert!(
        !exported_root.animations.is_empty(),
        "exported file should have animations"
    );

    let anim = &exported_root.animations[0];
    assert!(!anim.channels.is_empty(), "animation should have channels");
    assert!(!anim.samplers.is_empty(), "animation should have samplers");

    println!(
        "Exported: {} animations, {} channels",
        exported_root.animations.len(),
        anim.channels.len()
    );

    std::fs::remove_file(&output_path).ok();
}

#[test]
fn test_gltf_blender_roundtrip_node_animation() {
    let source_path = Path::new("assets/models/stickman/stickman.glb");
    if !source_path.exists() {
        eprintln!("Skipping: stickman.glb not found");
        return;
    }

    let raw_bytes = std::fs::read(source_path).unwrap();
    let glb = gltf::binary::Glb::from_slice(&raw_bytes).unwrap();
    let original_root: gltf::json::Root = gltf::json::Root::from_slice(&glb.json).unwrap();
    let skeleton = build_skeleton_from_gltf(&original_root);

    let clip = build_roundtrip_test_clip(&skeleton);
    let baked_clip = thyllore_animation::animation::editable::clip_to_animation(&clip);

    let exported_path = std::env::temp_dir().join("gltf_roundtrip_node_exported.glb");
    let blender_output_path = std::env::temp_dir().join("gltf_roundtrip_node_blender.glb");

    run_roundtrip_and_verify(
        source_path,
        &clip,
        &baked_clip,
        &skeleton,
        &exported_path,
        &blender_output_path,
        0.1,
        0.02,
    );

    std::fs::remove_file(&exported_path).ok();
    std::fs::remove_file(&blender_output_path).ok();
}

#[test]
fn test_gltf_blender_roundtrip_skeletal_animation() {
    let source_path = Path::new("assets/models/phoenix-bird/glb/phoenixBird.glb");
    if !source_path.exists() {
        eprintln!("Skipping: phoenixBird.glb not found");
        return;
    }

    let raw_bytes = std::fs::read(source_path).unwrap();
    let glb = gltf::binary::Glb::from_slice(&raw_bytes).unwrap();
    let original_root: gltf::json::Root = gltf::json::Root::from_slice(&glb.json).unwrap();
    let skeleton = build_skeleton_from_gltf(&original_root);

    assert!(
        !original_root.skins.is_empty(),
        "phoenixBird should have skin data for skeletal animation"
    );

    let clip = build_skeletal_roundtrip_test_clip(&skeleton);
    let baked_clip = thyllore_animation::animation::editable::clip_to_animation(&clip);

    let exported_path = std::env::temp_dir().join("gltf_roundtrip_skeletal_exported.glb");
    let blender_output_path = std::env::temp_dir().join("gltf_roundtrip_skeletal_blender.glb");

    run_roundtrip_and_verify(
        source_path,
        &clip,
        &baked_clip,
        &skeleton,
        &exported_path,
        &blender_output_path,
        0.1,
        0.02,
    );

    std::fs::remove_file(&exported_path).ok();
    std::fs::remove_file(&blender_output_path).ok();
}

fn run_roundtrip_and_verify(
    source_path: &Path,
    clip: &EditableAnimationClip,
    baked_clip: &AnimationClip,
    skeleton: &Skeleton,
    exported_path: &Path,
    blender_output_path: &Path,
    translation_tolerance: f32,
    rotation_tolerance: f32,
) {
    let blender_path = match read_blender_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: BlenderPath not configured in .claude/local/paths.md");
            return;
        }
    };

    let script_path = Path::new("scripts/blender_gltf_roundtrip.py");
    if !script_path.exists() {
        eprintln!("Skipping: blender_gltf_roundtrip.py not found");
        return;
    }

    thyllore_animation::exporter::gltf::export_gltf_animation(
        source_path,
        clip,
        skeleton,
        exported_path,
    )
    .expect("Failed to export glTF animation");

    let script_abs = canonicalize_no_prefix(script_path);
    let exported_abs = canonicalize_no_prefix(exported_path);
    let blender_output_abs = canonicalize_no_prefix(blender_output_path);

    let output = Command::new(&blender_path)
        .args([
            "--background",
            "--python",
            &script_abs.to_string_lossy(),
            "--",
            &exported_abs.to_string_lossy(),
            &blender_output_abs.to_string_lossy(),
        ])
        .output()
        .expect("Failed to run Blender");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("Blender stdout:\n{}", stdout);
    if !output.status.success() {
        panic!("Blender roundtrip failed:\nstderr: {}", stderr);
    }

    assert!(
        blender_output_path.exists(),
        "Blender did not produce output file"
    );

    let reloaded = unsafe {
        thyllore_animation::loader::gltf::load_gltf_file(&blender_output_path.to_string_lossy())
    }
    .expect("Failed to load exported glTF");

    assert!(
        !reloaded.clips.is_empty(),
        "Reloaded file should have animation clips"
    );

    let reloaded_clip = &reloaded.clips[0];
    eprintln!(
        "Original clip: duration={:.3}, channels={}",
        baked_clip.duration,
        baked_clip.channels.len()
    );
    eprintln!(
        "Reloaded clip: duration={:.3}, channels={}",
        reloaded_clip.duration,
        reloaded_clip.channels.len()
    );

    let reloaded_skeleton = reloaded
        .animation_system
        .skeletons
        .first()
        .expect("No skeleton in reloaded file");

    verify_animation_channels(
        baked_clip,
        reloaded_clip,
        skeleton,
        reloaded_skeleton,
        translation_tolerance,
        rotation_tolerance,
    );
}

fn verify_animation_channels(
    original_clip: &AnimationClip,
    reloaded_clip: &AnimationClip,
    original_skeleton: &Skeleton,
    reloaded_skeleton: &Skeleton,
    translation_tolerance: f32,
    rotation_tolerance: f32,
) {
    let bone_name_to_original_id: std::collections::HashMap<&str, u32> = original_skeleton
        .bones
        .iter()
        .map(|b| (b.name.as_str(), b.id))
        .collect();

    let mut matched_bones = 0;
    let mut max_translation_error = 0.0f32;
    let mut max_rotation_error = 0.0f32;

    for reloaded_bone in &reloaded_skeleton.bones {
        let Some(&original_bone_id) = bone_name_to_original_id.get(reloaded_bone.name.as_str())
        else {
            continue;
        };

        let Some(original_channel) = original_clip.channels.get(&original_bone_id) else {
            continue;
        };

        let Some(reloaded_channel) = reloaded_clip.channels.get(&reloaded_bone.id) else {
            continue;
        };

        matched_bones += 1;

        for orig_kf in &original_channel.translation {
            if let Some(reloaded_kf) =
                find_nearest_keyframe_vec3(&reloaded_channel.translation, orig_kf.time)
            {
                let dx = (orig_kf.value.x - reloaded_kf.value.x).abs();
                let dy = (orig_kf.value.y - reloaded_kf.value.y).abs();
                let dz = (orig_kf.value.z - reloaded_kf.value.z).abs();
                let error = dx.max(dy).max(dz);
                max_translation_error = max_translation_error.max(error);

                if error > 0.05 {
                    eprintln!(
                        "  Translation mismatch bone='{}' t={:.3}: \
                         orig=({:.4},{:.4},{:.4}) reload=({:.4},{:.4},{:.4}) err={:.4}",
                        reloaded_bone.name,
                        orig_kf.time,
                        orig_kf.value.x,
                        orig_kf.value.y,
                        orig_kf.value.z,
                        reloaded_kf.value.x,
                        reloaded_kf.value.y,
                        reloaded_kf.value.z,
                        error,
                    );
                }
            }
        }

        for orig_kf in &original_channel.rotation {
            if let Some(reloaded_kf) =
                find_nearest_keyframe_quat(&reloaded_channel.rotation, orig_kf.time)
            {
                let dot = (orig_kf.value.s * reloaded_kf.value.s
                    + orig_kf.value.v.x * reloaded_kf.value.v.x
                    + orig_kf.value.v.y * reloaded_kf.value.v.y
                    + orig_kf.value.v.z * reloaded_kf.value.v.z)
                    .abs();
                let error = 1.0 - dot.min(1.0);
                max_rotation_error = max_rotation_error.max(error);

                if error > 0.01 {
                    eprintln!(
                        "  Rotation mismatch bone='{}' t={:.3}: dot={:.6} error={:.6}",
                        reloaded_bone.name, orig_kf.time, dot, error,
                    );
                }
            }
        }
    }

    eprintln!("Matched bones: {}", matched_bones);
    eprintln!("Max translation error: {:.6}", max_translation_error);
    eprintln!("Max rotation error: {:.6}", max_rotation_error);

    assert!(
        matched_bones > 0,
        "No bone channels matched between original and reloaded"
    );

    assert!(
        max_translation_error < translation_tolerance,
        "Translation error too large: {:.6} (tolerance: {:.6})",
        max_translation_error,
        translation_tolerance
    );

    assert!(
        max_rotation_error < rotation_tolerance,
        "Rotation error too large: {:.6} (tolerance: {:.6})",
        max_rotation_error,
        rotation_tolerance
    );
}

fn read_blender_path() -> Option<String> {
    let paths_file = Path::new(".claude/local/paths.md");
    let content = std::fs::read_to_string(paths_file).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("- BlenderPath = ") {
            let path = rest.trim().to_string();
            if Path::new(&path).exists() {
                return Some(path);
            }
        }
    }
    None
}

fn canonicalize_no_prefix(path: &Path) -> PathBuf {
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

fn build_skeleton_from_gltf(root: &gltf::json::Root) -> Skeleton {
    let mut skeleton = Skeleton::new("test_skeleton");

    if root.skins.is_empty() {
        for (i, node) in root.nodes.iter().enumerate() {
            let fallback = format!("node_{}", i);
            let name = node.name.as_deref().unwrap_or(&fallback);
            skeleton.add_bone(name, None);
        }
        return skeleton;
    }

    let skin = &root.skins[0];
    let mut joint_to_bone: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();

    for (bone_idx, joint_index) in skin.joints.iter().enumerate() {
        let node_idx = joint_index.value();
        let node = &root.nodes[node_idx];
        let fallback = format!("joint_{}", bone_idx);
        let name = node.name.as_deref().unwrap_or(&fallback);

        let parent_bone_id = find_parent_node(root, node_idx)
            .and_then(|parent_node_idx| joint_to_bone.get(&parent_node_idx).copied());

        let bone_id = skeleton.add_bone(name, parent_bone_id);
        joint_to_bone.insert(node_idx, bone_id);
    }

    skeleton
}

fn find_parent_node(root: &gltf::json::Root, target: usize) -> Option<usize> {
    for (i, node) in root.nodes.iter().enumerate() {
        if let Some(ref children) = node.children {
            for child in children {
                if child.value() == target {
                    return Some(i);
                }
            }
        }
    }
    None
}

fn build_test_clip_translation_only(skeleton: &Skeleton) -> EditableAnimationClip {
    let mut clip = EditableAnimationClip::new(0, "test_animation".to_string());
    clip.duration = 1.0;

    for bone in &skeleton.bones {
        clip.add_track(bone.id, bone.name.clone());
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationX, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationX, 1.0, 1.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationY, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationY, 1.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationZ, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationZ, 1.0, 0.0);
    }

    clip
}

fn build_roundtrip_test_clip(skeleton: &Skeleton) -> EditableAnimationClip {
    let mut clip = EditableAnimationClip::new(0, "roundtrip_test".to_string());
    clip.duration = 1.0;

    for bone in &skeleton.bones {
        clip.add_track(bone.id, bone.name.clone());

        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationX, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationX, 0.5, 0.5);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationX, 1.0, 1.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationY, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationY, 0.5, 0.25);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationY, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationZ, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationZ, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationX, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationX, 1.0, 15.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationY, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationY, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationZ, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationZ, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleX, 0.0, 1.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleX, 1.0, 1.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleY, 0.0, 1.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleY, 1.0, 1.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleZ, 0.0, 1.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleZ, 1.0, 1.0);
    }

    clip
}

fn build_skeletal_roundtrip_test_clip(skeleton: &Skeleton) -> EditableAnimationClip {
    let mut clip = EditableAnimationClip::new(0, "skeletal_roundtrip_test".to_string());
    clip.duration = 1.0;

    for bone in &skeleton.bones {
        clip.add_track(bone.id, bone.name.clone());

        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationX, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationX, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationY, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationY, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationZ, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::TranslationZ, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationX, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationX, 0.5, 10.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationX, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationY, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationY, 0.5, 5.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationY, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationZ, 0.0, 0.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::RotationZ, 1.0, 0.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleX, 0.0, 1.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleX, 1.0, 1.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleY, 0.0, 1.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleY, 1.0, 1.0);

        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleZ, 0.0, 1.0);
        clip_add_keyframe(&mut clip, bone.id, PropertyType::ScaleZ, 1.0, 1.0);
    }

    clip
}

fn find_nearest_keyframe_vec3<'a>(
    keyframes: &'a [thyllore_animation::animation::Keyframe<cgmath::Vector3<f32>>],
    target_time: f32,
) -> Option<&'a thyllore_animation::animation::Keyframe<cgmath::Vector3<f32>>> {
    keyframes
        .iter()
        .min_by(|a, b| {
            let da = (a.time - target_time).abs();
            let db = (b.time - target_time).abs();
            da.partial_cmp(&db).unwrap()
        })
        .filter(|kf| (kf.time - target_time).abs() < 0.1)
}

fn find_nearest_keyframe_quat<'a>(
    keyframes: &'a [thyllore_animation::animation::Keyframe<cgmath::Quaternion<f32>>],
    target_time: f32,
) -> Option<&'a thyllore_animation::animation::Keyframe<cgmath::Quaternion<f32>>> {
    keyframes
        .iter()
        .min_by(|a, b| {
            let da = (a.time - target_time).abs();
            let db = (b.time - target_time).abs();
            da.partial_cmp(&db).unwrap()
        })
        .filter(|kf| (kf.time - target_time).abs() < 0.1)
}
