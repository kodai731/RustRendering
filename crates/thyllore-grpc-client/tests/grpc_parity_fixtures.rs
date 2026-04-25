//! Phase 3 — Generates parity fixtures for Python <-> Rust bit-identical proto verification.
//!
//! Run:
//!     cargo test -p thyllore-grpc-client --features text-to-motion \
//!         --test grpc_parity_fixtures generate_parity_fixtures -- --include-ignored
//!
//! Output:
//!     blender_addon/tests/fixtures/{rigging,motion,mesh}_request.{bin,json}

#![cfg(feature = "text-to-motion")]

use std::fs;
use std::path::PathBuf;

use prost::Message;
use thyllore_grpc_client::proto;

fn fixtures_dir() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .join("..")
        .join("..")
        .join("blender_addon")
        .join("tests")
        .join("fixtures")
}

fn write_pair(name: &str, bytes: &[u8], json: &serde_json::Value) {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).expect("create fixtures dir");

    let bin_path = dir.join(format!("{name}.bin"));
    let json_path = dir.join(format!("{name}.json"));

    fs::write(&bin_path, bytes).unwrap_or_else(|err| panic!("write {bin_path:?}: {err}"));
    let json_text = serde_json::to_string_pretty(json).expect("json serialize");
    fs::write(&json_path, json_text).unwrap_or_else(|err| panic!("write {json_path:?}: {err}"));
}

fn rigging_request_fixture() -> (Vec<u8>, serde_json::Value) {
    let request = proto::RiggingRequest {
        glb_data: vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03],
        params: Some(proto::RiggingParams {
            num_sample_points: 65536,
        }),
        model_type: proto::RiggingModelType::RiggingUnirig as i32,
    };
    let bytes = request.encode_to_vec();
    let json = serde_json::json!({
        "glb_data_hex": hex::encode(&request.glb_data),
        "num_sample_points": request.params.as_ref().unwrap().num_sample_points,
        "model_type": "RIGGING_UNIRIG",
    });
    (bytes, json)
}

fn motion_request_fixture() -> (Vec<u8>, serde_json::Value) {
    let request = proto::MotionRequest {
        prompt: "walking forward".to_string(),
        duration_seconds: 3.0,
        target_fps: 30,
        skeleton_type: proto::SkeletonType::Smpl22 as i32,
        bone_mappings: vec![
            proto::BoneMapping {
                source_joint_index: 0,
                target_bone_name: "hips".to_string(),
            },
            proto::BoneMapping {
                source_joint_index: 3,
                target_bone_name: "spine".to_string(),
            },
        ],
        glb_skeleton: Some(proto::GlbSkeletonSpec {
            glb_data: vec![0x01, 0x02, 0x03],
            skeleton_cache_id: "cache_xyz".to_string(),
        }),
        internal_use_only: false,
    };
    let bytes = request.encode_to_vec();
    let json = serde_json::json!({
        "prompt": request.prompt,
        "duration_seconds": request.duration_seconds,
        "target_fps": request.target_fps,
        "skeleton_type": "SMPL_22",
        "bone_mappings": [
            { "source_joint_index": 0, "target_bone_name": "hips" },
            { "source_joint_index": 3, "target_bone_name": "spine" },
        ],
        "glb_skeleton_glb_data_hex": hex::encode(&request.glb_skeleton.as_ref().unwrap().glb_data),
        "glb_skeleton_cache_id": request.glb_skeleton.as_ref().unwrap().skeleton_cache_id,
        "internal_use_only": request.internal_use_only,
    });
    (bytes, json)
}

fn mesh_request_fixture() -> (Vec<u8>, serde_json::Value) {
    let request = proto::MeshRequest {
        prompt: "a cute robot".to_string(),
        params: Some(proto::MeshGenerationParams {
            target_faces: 30000,
            seed: 42,
            image_size: 768,
            image_inference_steps: 30,
        }),
        input_image_png: vec![0xAA, 0xBB, 0xCC],
        input_mode: proto::MeshInputMode::TextToMesh as i32,
        model_type: proto::MeshModelType::Trellis as i32,
        t2i_model_type: proto::TextToImageModelType::T2iServerDefault as i32,
    };
    let bytes = request.encode_to_vec();
    let params = request.params.as_ref().unwrap();
    let json = serde_json::json!({
        "prompt": request.prompt,
        "target_faces": params.target_faces,
        "seed": params.seed,
        "image_size": params.image_size,
        "image_inference_steps": params.image_inference_steps,
        "input_image_png_hex": hex::encode(&request.input_image_png),
        "input_mode": "TEXT_TO_MESH",
        "model_type": "TRELLIS",
        "t2i_model_type": "T2I_SERVER_DEFAULT",
    });
    (bytes, json)
}

#[test]
#[ignore]
fn generate_parity_fixtures() {
    let (bytes, json) = rigging_request_fixture();
    write_pair("rigging_request", &bytes, &json);

    let (bytes, json) = motion_request_fixture();
    write_pair("motion_request", &bytes, &json);

    let (bytes, json) = mesh_request_fixture();
    write_pair("mesh_request", &bytes, &json);
}
