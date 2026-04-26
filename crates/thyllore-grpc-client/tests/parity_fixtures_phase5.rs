//! Phase 5 — Tier A proto wire-bytes fixture generator.
//!
//! Run from WSL2 (recommended):
//!     export THYLLORE_PHASE5_FIXTURE_OUTPUT=/home/kodai/Projects/SharedData/fixtures/ml_parity
//!     cargo test -p thyllore-grpc-client --features auto-rig,text-to-motion \
//!         --test parity_fixtures_phase5 generate_phase5_proto_fixtures \
//!         -- --ignored --nocapture
//!
//! Outputs (one *.bin per request/response pair):
//!     proto/rigging_request.bin     proto/rigging_response.bin
//!     proto/motion_request.bin      proto/motion_response.bin
//!     proto/mesh_request.bin        proto/mesh_response.bin
//!
//! These are produced via `prost::Message::encode_to_vec()` and are byte-for-byte
//! comparable against Python `protobuf.SerializeToString()`. Phase 5 mock server
//! reads `*_response.bin` and returns it verbatim; both Rust and Blender clients
//! re-encode `*_request.bin`-equivalent objects and the orchestrator confirms
//! sha256 of the wire bytes match.
//!
//! Map fields are intentionally absent from the proto definition: their
//! iteration order is non-deterministic in `protobuf`/`prost`, which would
//! break wire-bytes parity. See proto_invariants.rs for the assertion.

#![cfg(feature = "text-to-motion")]

use std::env;
use std::fs;
use std::path::PathBuf;

use prost::Message;
use thyllore_grpc_client::proto;

fn fixture_root() -> PathBuf {
    PathBuf::from(
        env::var("THYLLORE_PHASE5_FIXTURE_OUTPUT").expect(
            "set THYLLORE_PHASE5_FIXTURE_OUTPUT to the fixtures/ml_parity root \
             (e.g. /home/kodai/Projects/SharedData/fixtures/ml_parity)",
        ),
    )
}

#[test]
#[ignore]
fn generate_phase5_proto_fixtures() {
    let proto_dir = fixture_root().join("proto");
    fs::create_dir_all(&proto_dir).expect("create proto fixture dir");

    write_pair(
        &proto_dir,
        "rigging_request",
        rigging_request_fixture().encode_to_vec(),
    );
    write_pair(
        &proto_dir,
        "rigging_response",
        rigging_response_fixture().encode_to_vec(),
    );
    write_pair(
        &proto_dir,
        "motion_request",
        motion_request_fixture().encode_to_vec(),
    );
    write_pair(
        &proto_dir,
        "motion_response",
        motion_response_fixture().encode_to_vec(),
    );
    write_pair(
        &proto_dir,
        "mesh_request",
        mesh_request_fixture().encode_to_vec(),
    );
    write_pair(
        &proto_dir,
        "mesh_response",
        mesh_response_fixture().encode_to_vec(),
    );
}

fn write_pair(dir: &std::path::Path, name: &str, bytes: Vec<u8>) {
    let path = dir.join(format!("{name}.bin"));
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
    eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
}

fn rigging_request_fixture() -> proto::RiggingRequest {
    proto::RiggingRequest {
        glb_data: vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03],
        params: Some(proto::RiggingParams {
            num_sample_points: 65536,
        }),
        model_type: proto::RiggingModelType::RiggingUnirig as i32,
    }
}

fn rigging_response_fixture() -> proto::RiggingResponse {
    proto::RiggingResponse {
        rigged_glb_data: vec![0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80],
        metadata: Some(proto::RiggingMetadata {
            joint_count: 12,
            bone_count: 11,
            generation_time_ms: 42.5,
        }),
        skeleton_joints: vec![
            proto::SkeletonJoint {
                name: "root".to_string(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
                tail_x: 0.0,
                tail_y: 0.1,
                tail_z: 0.0,
                parent_index: -1,
            },
            proto::SkeletonJoint {
                name: "spine".to_string(),
                x: 0.0,
                y: 0.1,
                z: 0.0,
                tail_x: 0.0,
                tail_y: 0.3,
                tail_z: 0.0,
                parent_index: 0,
            },
            proto::SkeletonJoint {
                name: "head".to_string(),
                x: 0.0,
                y: 0.3,
                z: 0.0,
                tail_x: 0.0,
                tail_y: 0.5,
                tail_z: 0.0,
                parent_index: 1,
            },
        ],
    }
}

fn motion_request_fixture() -> proto::MotionRequest {
    proto::MotionRequest {
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
    }
}

fn motion_response_fixture() -> proto::MotionResponse {
    proto::MotionResponse {
        curves: vec![
            proto::AnimationCurve {
                bone_name: "hips".to_string(),
                property_type: proto::PropertyType::TranslationY as i32,
                keyframes: vec![
                    proto::CurveKeyframe {
                        time: 0.0,
                        value: 0.0,
                        tangent_in_dt: -0.1,
                        tangent_in_dv: 0.0,
                        tangent_out_dt: 0.1,
                        tangent_out_dv: 0.05,
                        interpolation: proto::InterpolationType::Bezier as i32,
                    },
                    proto::CurveKeyframe {
                        time: 1.0,
                        value: 0.05,
                        tangent_in_dt: -0.1,
                        tangent_in_dv: -0.02,
                        tangent_out_dt: 0.1,
                        tangent_out_dv: 0.02,
                        interpolation: proto::InterpolationType::Bezier as i32,
                    },
                    proto::CurveKeyframe {
                        time: 2.0,
                        value: 0.0,
                        tangent_in_dt: -0.1,
                        tangent_in_dv: 0.0,
                        tangent_out_dt: 0.1,
                        tangent_out_dv: 0.0,
                        interpolation: proto::InterpolationType::Bezier as i32,
                    },
                ],
            },
            proto::AnimationCurve {
                bone_name: "spine".to_string(),
                property_type: proto::PropertyType::RotationZ as i32,
                keyframes: vec![proto::CurveKeyframe {
                    time: 0.0,
                    value: 0.0,
                    tangent_in_dt: 0.0,
                    tangent_in_dv: 0.0,
                    tangent_out_dt: 0.0,
                    tangent_out_dv: 0.0,
                    interpolation: proto::InterpolationType::Linear as i32,
                }],
            },
        ],
        generation_time_ms: 250.0,
        model_used: "light_t2m_v1".to_string(),
    }
}

fn mesh_request_fixture() -> proto::MeshRequest {
    proto::MeshRequest {
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
    }
}

fn mesh_response_fixture() -> proto::MeshResponse {
    proto::MeshResponse {
        glb_data: vec![0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88],
        metadata: Some(proto::MeshMetadata {
            vertex_count: 1024,
            face_count: 2048,
            generation_time_ms: 5500.0,
            intermediate_image_png: vec![],
        }),
    }
}
