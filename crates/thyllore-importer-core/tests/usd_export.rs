//! Round-trip test for the USD exporter: build an internal model, export it to
//! a `.usda` file, re-import it, and verify the geometry / skeleton / baked
//! animation / morph survive the round trip.

use cgmath::{Matrix4, Vector3, Vector4};

use thyllore_anim_core::MorphAnimation;
use thyllore_anim_core::{AnimationClip, Keyframe, Skeleton, TransformChannel};
use thyllore_importer_core::usd::{
    load_usd_file, save_usd_file, UsdExportBlendShape, UsdExportMesh, UsdExportScene,
};
use thyllore_math_core::{Vec2, Vec3, Vec4};
use thyllore_model_core::mesh::Vertex;
use thyllore_model_core::SkinData;

fn quad_vertices() -> Vec<Vertex> {
    let make = |x: f32, y: f32, u: f32, v: f32| Vertex {
        pos: Vec3::new(x, y, 0.0),
        color: Vec4::new(1.0, 1.0, 1.0, 1.0),
        tex_coord: Vec2::new(u, v),
        normal: Vec3::new(0.0, 0.0, 1.0),
    };
    vec![
        make(0.0, 0.0, 0.0, 0.0),
        make(1.0, 0.0, 1.0, 0.0),
        make(1.0, 1.0, 1.0, 1.0),
        make(0.0, 1.0, 0.0, 1.0),
    ]
}

fn skin_bound_to_bone(bone: u32, vertex_count: usize) -> SkinData {
    SkinData {
        skeleton_id: 0,
        bone_indices: vec![Vector4::new(bone, 0, 0, 0); vertex_count],
        bone_weights: vec![Vector4::new(1.0, 0.0, 0.0, 0.0); vertex_count],
        base_positions: vec![Vector3::new(0.0, 0.0, 0.0); vertex_count],
        base_normals: vec![Vector3::new(0.0, 0.0, 1.0); vertex_count],
    }
}

fn two_bone_skeleton() -> Skeleton {
    let mut skeleton = Skeleton::new("rig");
    let root = skeleton.add_bone("root", None);
    let child = skeleton.add_bone("child", Some(root));

    if let Some(bone) = skeleton.get_bone_mut(child) {
        bone.local_transform = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));
    }
    skeleton
}

fn translation_clip(bone: u32) -> AnimationClip {
    let mut channel = TransformChannel::default();
    channel
        .translation
        .push(Keyframe::new(0.0, Vector3::new(0.0, 1.0, 0.0)));
    channel
        .translation
        .push(Keyframe::new(1.0, Vector3::new(0.0, 2.0, 0.0)));

    let mut clip = AnimationClip::new("Take");
    clip.duration = 1.0;
    clip.add_channel(bone, channel);
    clip
}

#[test]
fn exports_and_reimports_skinned_animated_mesh() {
    let skeleton = two_bone_skeleton();
    let clip = translation_clip(1);
    let vertices = quad_vertices();
    let indices = vec![0u32, 1, 2, 0, 2, 3];
    let skin = skin_bound_to_bone(1, vertices.len());

    let blend_shape = UsdExportBlendShape {
        name: "smile".to_string(),
        position_offsets: vec![[0.0, 0.0, 0.1]; vertices.len()],
        normal_offsets: Vec::new(),
    };
    let mesh = UsdExportMesh {
        name: "quad",
        vertices: &vertices,
        indices: &indices,
        skin: Some(&skin),
        blend_shapes: std::slice::from_ref(&blend_shape),
    };

    let blend_order = vec!["smile".to_string()];
    let weights = vec![
        MorphAnimation {
            key_frame: 0.0,
            weights: vec![0.0],
        },
        MorphAnimation {
            key_frame: 1.0,
            weights: vec![1.0],
        },
    ];

    let scene = UsdExportScene {
        skeleton: Some(&skeleton),
        clip: Some(&clip),
        blend_shape_order: &blend_order,
        blend_shape_weights: &weights,
        meshes: std::slice::from_ref(&mesh),
        start_time_code: 0.0,
        end_time_code: 24.0,
        time_codes_per_second: 24.0,
    };

    let path = std::env::temp_dir().join("thyllore_usd_export_roundtrip.usda");
    let path_str = path.to_str().expect("temp path is valid UTF-8");
    save_usd_file(path_str, &scene).expect("USD export should succeed");

    let result = load_usd_file(path_str).expect("re-import should succeed");

    assert_eq!(result.animation_system.skeletons.len(), 1, "one skeleton");
    assert_eq!(
        result.animation_system.skeletons[0].bones.len(),
        2,
        "two bones round-trip"
    );

    assert_eq!(result.meshes.len(), 1, "one mesh");
    let mesh_out = &result.meshes[0];
    assert_eq!(
        mesh_out.vertex_data.vertices.len(),
        4,
        "per-vertex attributes must not split the quad"
    );
    assert_eq!(mesh_out.vertex_data.indices, indices, "topology preserved");
    assert!(result.has_skinned_meshes, "skin binding round-trips");

    assert_eq!(result.clips.len(), 1, "one animation clip");
    let baked_keyframes: usize = result.clips[0]
        .channels
        .values()
        .map(|c| c.translation.len())
        .sum();
    assert!(baked_keyframes > 1, "animation is baked into samples");

    assert!(!result.morph_animation.is_empty(), "morph round-trips");
    let morphed = result
        .morph_animation
        .targets
        .iter()
        .filter(|t| !t.is_empty())
        .count();
    assert_eq!(morphed, 1, "one morphed mesh");
}
