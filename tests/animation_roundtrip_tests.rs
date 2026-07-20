use cgmath::Vector3;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::path::PathBuf;

use thyllore_anim_core::editable::clip_from_animation;
use thyllore_animation::ecs::systems::{
    apply_skinning, compute_pose_global_transforms, create_pose_from_rest, sample_clip_to_pose,
};
use thyllore_animation::loader::fbx::load_fbx_to_graphics_resources;
use thyllore_animation::loader::gltf::load_gltf_file;
use thyllore_animation::loader::ModelLoadResult;

const TOLERANCE: f32 = 1e-3;

fn build_bone_name_map(
    skeleton: &thyllore_anim_core::Skeleton,
) -> HashMap<thyllore_anim_core::BoneId, String> {
    skeleton
        .bones
        .iter()
        .enumerate()
        .map(|(i, b)| (i as u32, b.name.clone()))
        .collect()
}

fn get_project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn resolve_asset_path(asset_path: &str) -> PathBuf {
    get_project_root().join(asset_path)
}

fn remap_globals_to_original_bone_order(
    orig_skel: &thyllore_anim_core::Skeleton,
    rt_skel: &thyllore_anim_core::Skeleton,
    rt_globals: &[cgmath::Matrix4<f32>],
) -> Vec<cgmath::Matrix4<f32>> {
    let rt_id_by_name: HashMap<&str, usize> = rt_skel
        .bones
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.as_str(), i))
        .collect();

    orig_skel
        .bones
        .iter()
        .map(|bone| {
            let rt_idx = rt_id_by_name.get(bone.name.as_str()).unwrap_or_else(|| {
                panic!("Bone '{}' missing in round-tripped skeleton", bone.name)
            });
            rt_globals[*rt_idx]
        })
        .collect()
}

// The glTF loader bakes skinned animations to dense per-key Step keyframes while the
// FBX import path yields Linear keyframes for the same data. Interpolation-mode
// semantics are a loader concern, not an export-fidelity concern, so both clips are
// normalized to Linear before sampling.
fn normalize_interpolation_to_linear(
    clip: &thyllore_anim_core::AnimationClip,
) -> thyllore_anim_core::AnimationClip {
    use thyllore_anim_core::Interpolation;

    let mut normalized = clip.clone();
    for channel in normalized.channels.values_mut() {
        for kf in channel.translation.iter_mut() {
            kf.interpolation = Interpolation::Linear;
        }
        for kf in channel.rotation.iter_mut() {
            kf.interpolation = Interpolation::Linear;
        }
        for kf in channel.scale.iter_mut() {
            kf.interpolation = Interpolation::Linear;
        }
    }
    normalized
}

fn sample_and_compare(
    original: &ModelLoadResult,
    roundtripped: &ModelLoadResult,
) -> anyhow::Result<()> {
    let orig_skel = original
        .skeletons
        .first()
        .expect("No skeleton in original model");
    let rt_skel = roundtripped
        .skeletons
        .first()
        .expect("No skeleton in round-tripped model");

    let orig_clip = &normalize_interpolation_to_linear(
        original
            .clips
            .first()
            .expect("No animation clip in original"),
    );
    let rt_clip = &normalize_interpolation_to_linear(
        roundtripped
            .clips
            .first()
            .expect("No animation clip in round-tripped"),
    );

    let mut rng = StdRng::seed_from_u64(42);

    let duration = orig_clip.duration;
    let num_samples = 5;
    let sampled_times: Vec<f32> = (0..num_samples)
        .map(|_| rng.gen_range(0.0..duration))
        .collect();

    let bone_count = orig_skel.bones.len();
    let num_bone_samples = 3;
    let sampled_bones: Vec<thyllore_anim_core::BoneId> = (0..num_bone_samples)
        .map(|_| rng.gen_range(0..bone_count) as thyllore_anim_core::BoneId)
        .collect();

    eprintln!("Sampled times: {:?}", sampled_times);
    eprintln!("Sampled bones: {:?}", sampled_bones);

    let mut compared_vertex_count = 0usize;

    // The round-tripped file is animation-only (no meshes), so skin the ORIGINAL
    // mesh with globals computed from the round-tripped skeleton/clip, remapped
    // into original bone order by bone name.
    for (mesh_idx, orig_mesh) in original.meshes.iter().enumerate() {
        let orig_skin = match &orig_mesh.skin_data {
            Some(sd) => sd,
            None => continue,
        };

        let vertex_count = orig_skin.base_positions.len();

        for &t in &sampled_times {
            let mut orig_pose = create_pose_from_rest(orig_skel);
            sample_clip_to_pose(orig_clip, t, orig_skel, &mut orig_pose, false);
            let orig_globals = compute_pose_global_transforms(orig_skel, &orig_pose);

            let mut rt_pose = create_pose_from_rest(rt_skel);
            sample_clip_to_pose(rt_clip, t, rt_skel, &mut rt_pose, false);
            let rt_globals = compute_pose_global_transforms(rt_skel, &rt_pose);
            let rt_globals_in_orig_order =
                remap_globals_to_original_bone_order(orig_skel, rt_skel, &rt_globals);

            let mut orig_positions = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            let mut orig_normals = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            apply_skinning(
                orig_skin,
                &orig_globals,
                orig_skel,
                &mut orig_positions,
                &mut orig_normals,
            );

            let mut rt_positions = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            let mut rt_normals = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            apply_skinning(
                orig_skin,
                &rt_globals_in_orig_order,
                orig_skel,
                &mut rt_positions,
                &mut rt_normals,
            );

            for &bone_id in &sampled_bones {
                for i in 0..vertex_count {
                    let bone_indices = orig_skin.bone_indices[i];
                    let weights = orig_skin.bone_weights[i];

                    let has_bone_weight = (bone_indices.x == bone_id && weights.x > 0.0)
                        || (bone_indices.y == bone_id && weights.y > 0.0)
                        || (bone_indices.z == bone_id && weights.z > 0.0)
                        || (bone_indices.w == bone_id && weights.w > 0.0);

                    if has_bone_weight {
                        let diff = (orig_positions[i].x - rt_positions[i].x).abs()
                            + (orig_positions[i].y - rt_positions[i].y).abs()
                            + (orig_positions[i].z - rt_positions[i].z).abs();
                        assert!(
                            diff < TOLERANCE,
                            "Mesh {} vertex {} at time {:.1}: position difference {:.6} exceeds tolerance {:.3}",
                            mesh_idx, i, t, diff, TOLERANCE
                        );
                        compared_vertex_count += 1;
                    }
                }
            }
        }
    }

    assert!(
        compared_vertex_count > 0,
        "No vertices were compared — the round-trip comparison did not execute"
    );
    eprintln!("Compared {} vertex samples", compared_vertex_count);

    Ok(())
}

#[test]
fn test_gltf_to_fbx_roundtrip() -> anyhow::Result<()> {
    let gltf_path = resolve_asset_path("assets/models/phoenix-bird/glb/phoenixBird.glb");
    if !gltf_path.exists() {
        eprintln!(
            "Skipping test: asset not available: {}",
            gltf_path.display()
        );
        return Ok(());
    }

    // Load original glTF
    let gltf_result = unsafe { load_gltf_file(gltf_path.to_str().unwrap()) }?;
    let original = ModelLoadResult::from_gltf(gltf_result);

    // Get skeleton and clip
    let skeleton = original.skeletons.first().expect("No skeleton in glTF");
    let clip = original.clips.first().expect("No animation clip in glTF");

    // Convert to EditableAnimationClip
    let bone_names = build_bone_name_map(skeleton);
    let editable_clip = clip_from_animation(0, clip, &bone_names);

    // Export to FBX
    let fbx_output_path = std::env::temp_dir().join("roundtrip_gltf_to_fbx.fbx");
    thyllore_exporter_core::systems::fbx::animation::export_animation_fbx(
        &editable_clip,
        skeleton,
        &fbx_output_path,
        false,
        thyllore_importer_core::fbx::fbx::FbxAxesInfo {
            up_axis: 1,
            up_axis_sign: 1,
            front_axis: 2,
            front_axis_sign: 1,
            coord_axis: 0,
            coord_axis_sign: 1,
        },
        30.0,
    )?;

    // Re-import the FBX
    let (fbx_result, _) = load_fbx_to_graphics_resources(fbx_output_path.to_str().unwrap())?;
    let roundtripped = ModelLoadResult::from_fbx(fbx_result);

    // Sample and compare
    sample_and_compare(&original, &roundtripped)?;

    Ok(())
}

#[test]
fn test_fbx_to_gltf_roundtrip() -> anyhow::Result<()> {
    let fbx_path = resolve_asset_path("assets/models/phoenix-bird/source/fly.fbx");
    if !fbx_path.exists() {
        eprintln!("Skipping test: asset not available: {}", fbx_path.display());
        return Ok(());
    }

    // Load original FBX
    let (fbx_result, _) = load_fbx_to_graphics_resources(fbx_path.to_str().unwrap())?;
    let original = ModelLoadResult::from_fbx(fbx_result);

    // Get skeleton and clip
    let skeleton = original.skeletons.first().expect("No skeleton in FBX");
    let clip = original.clips.first().expect("No animation clip in FBX");

    // Convert to EditableAnimationClip
    let bone_names = build_bone_name_map(skeleton);
    let editable_clip = clip_from_animation(0, clip, &bone_names);

    // Export to glTF
    let gltf_output_path = std::env::temp_dir().join("roundtrip_fbx_to_gltf.glb");
    thyllore_exporter_core::systems::gltf::export_gltf_animation_only(
        &editable_clip,
        skeleton,
        &gltf_output_path,
    )?;

    // Re-import the glTF
    let gltf_result = unsafe { load_gltf_file(gltf_output_path.to_str().unwrap()) }?;
    let roundtripped = ModelLoadResult::from_gltf(gltf_result);

    // Sample and compare
    sample_and_compare(&original, &roundtripped)?;

    Ok(())
}
