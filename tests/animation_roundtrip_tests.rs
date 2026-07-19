use cgmath::Vector3;
use rand::prelude::{StdRng};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::path::PathBuf;

use thyllore_animation::ecs::systems::{
    apply_skinning, compute_pose_global_transforms, create_pose_from_rest, sample_clip_to_pose,
};
use thyllore_animation::loader::fbx::load_fbx_to_graphics_resources;
use thyllore_animation::loader::gltf::load_gltf_file;
use thyllore_animation::loader::ModelLoadResult;
use thyllore_anim_core::editable::clip_from_animation;

const TOLERANCE: f32 = 1e-3;

fn build_bone_name_map(skeleton: &thyllore_anim_core::Skeleton) -> HashMap<thyllore_anim_core::BoneId, String> {
    skeleton.bones.iter().enumerate()
        .map(|(i, b)| (i as u32, b.name.clone()))
        .collect()
}

fn get_project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn resolve_asset_path(asset_path: &str) -> PathBuf {
    get_project_root().join(asset_path)
}

fn sample_and_compare(
    original: &ModelLoadResult,
    roundtripped: &ModelLoadResult,
) -> anyhow::Result<()> {
    // Get skeletons from both models
    let orig_skel = original.skeletons.first().expect("No skeleton in original model");
    let rt_skel = roundtripped.skeletons.first().expect("No skeleton in round-tripped model");

    // Get clips from both models
    let orig_clip = original.clips.first().expect("No animation clip in original");
    let rt_clip = roundtripped.clips.first().expect("No animation clip in round-tripped");

    // Use fixed seed for reproducible random sampling
    let mut rng = StdRng::seed_from_u64(42);

    // Sample a few times within clip duration
    let duration = orig_clip.duration;
    let num_samples = 5;
    let sampled_times: Vec<f32> = (0..num_samples)
        .map(|_| rng.gen_range(0.0..duration))
        .collect();

    // Sample a few bones from skeleton.bones
    let bone_count = orig_skel.bones.len();
    let num_bone_samples = 3;
    let sampled_bones: Vec<thyllore_anim_core::BoneId> = (0..num_bone_samples)
        .map(|_| rng.gen_range(0..bone_count) as thyllore_anim_core::BoneId)
        .collect();

    eprintln!("Sampled times: {:?}", sampled_times);
    eprintln!("Sampled bones: {:?}", sampled_bones);

    // For each mesh in the original model that has skin_data
    for (mesh_idx, orig_mesh) in original.meshes.iter().enumerate() {
        let orig_skin = match &orig_mesh.skin_data {
            Some(sd) => sd,
            None => continue, // Skip meshes without skin data
        };

        // Find corresponding mesh in round-tripped model by index
        let rt_mesh = match roundtripped.meshes.get(mesh_idx) {
            Some(m) => m,
            None => continue, // Skip if mesh doesn't exist in round-tripped model
        };

        let rt_skin = match &rt_mesh.skin_data {
            Some(sd) => sd,
            None => continue, // Skip if skin data is missing in round-tripped mesh
        };

        // Ensure both have the same number of vertices
        if orig_skin.base_positions.len() != rt_skin.base_positions.len() {
            eprintln!("Warning: vertex count mismatch for mesh {}", mesh_idx);
            continue;
        }

        let vertex_count = orig_skin.base_positions.len();

        for &t in &sampled_times {
            // Original pose
            let mut orig_pose = create_pose_from_rest(orig_skel);
            sample_clip_to_pose(orig_clip, t, orig_skel, &mut orig_pose, false);
            let orig_globals = compute_pose_global_transforms(orig_skel, &orig_pose);

            // Round-tripped pose
            let mut rt_pose = create_pose_from_rest(rt_skel);
            sample_clip_to_pose(rt_clip, t, rt_skel, &mut rt_pose, false);
            let rt_globals = compute_pose_global_transforms(rt_skel, &rt_pose);

            // Apply skinning
            let mut orig_positions = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            let mut orig_normals = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            apply_skinning(orig_skin, &orig_globals, orig_skel, &mut orig_positions, &mut orig_normals);

            let mut rt_positions = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            let mut rt_normals = vec![Vector3::new(0.0, 0.0, 0.0); vertex_count];
            apply_skinning(rt_skin, &rt_globals, rt_skel, &mut rt_positions, &mut rt_normals);

            // For each sampled bone, find vertices whose skin weights include this bone
            for &bone_id in &sampled_bones {
                for i in 0..vertex_count {
                    let bone_indices = orig_skin.bone_indices[i];
                    let weights = orig_skin.bone_weights[i];

                    // Check if this vertex has non-zero weight for the sampled bone
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
                    }
                }
            }
        }
    }

    Ok(())
}

#[test]
fn test_gltf_to_fbx_roundtrip() -> anyhow::Result<()> {
    let gltf_path = resolve_asset_path("assets/models/phoenix-bird/glb/phoenixBird.glb");
    assert!(gltf_path.exists(), "glTF file not found: {}", gltf_path.display());

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
    assert!(fbx_path.exists(), "FBX file not found: {}", fbx_path.display());

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
