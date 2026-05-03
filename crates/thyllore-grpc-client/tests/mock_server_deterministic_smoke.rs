#![cfg(feature = "text-to-motion")]

mod common;

use std::path::{Path, PathBuf};

use prost::Message;
use sha2::{Digest, Sha256};

use thyllore_grpc_client::proto;

use crate::common::mock_server::{MockServerConfig, MockServerHandle};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn resolve_fixture_root() -> PathBuf {
    if let Ok(p) = std::env::var("THYLLORE_PARITY_FIXTURE_OUTPUT") {
        return PathBuf::from(p);
    }
    workspace_root().join("fixtures").join("ml_parity")
}

fn build_client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("client runtime")
}

#[test]
fn echo_mode_returns_synthesized_response_and_records_request() {
    let server = MockServerHandle::start(MockServerConfig::EchoDefault);
    let address = server.address.clone();

    let request = proto::RiggingRequest {
        glb_data: vec![1, 2, 3, 4, 5],
        params: Some(proto::RiggingParams {
            num_sample_points: 1024,
        }),
        model_type: proto::RiggingModelType::RiggingUnirig as i32,
    };
    let expected_wire_bytes = request.encode_to_vec();
    let expected_sha = format!("{:x}", Sha256::digest(&expected_wire_bytes));

    let runtime = build_client_runtime();
    let response = runtime.block_on(async {
        let mut client =
            proto::auto_rigging_service_client::AutoRiggingServiceClient::connect(address.clone())
                .await
                .expect("connect to mock");
        client
            .generate_rig(request)
            .await
            .expect("generate_rig succeeds")
            .into_inner()
    });

    assert_eq!(response.rigged_glb_data, vec![1, 2, 3, 4, 5]);

    let buffer = server.recv_buffer.lock().expect("recv_buffer");
    assert_eq!(buffer.count("AutoRiggingService", "GenerateRig"), 1);
    let recorded_sha = buffer
        .sha256_of("AutoRiggingService", "GenerateRig", 0)
        .expect("recorded sha");
    assert_eq!(recorded_sha, expected_sha);
}

#[test]
fn deterministic_mode_returns_fixture_response() {
    let root = resolve_fixture_root();
    if !root.join("proto/rigging_response.bin").exists() {
        eprintln!(
            "skip: rigging_response.bin missing at {}; run scripts/generate_parity_fixtures.sh",
            root.display()
        );
        return;
    }

    let expected_response_bytes =
        std::fs::read(root.join("proto/rigging_response.bin")).expect("read rigging_response.bin");
    let expected_response: proto::RiggingResponse =
        proto::RiggingResponse::decode(&*expected_response_bytes).expect("fixture decodes");

    let server = MockServerHandle::start(MockServerConfig::DeterministicFromFixtures(root));
    let address = server.address.clone();

    let runtime = build_client_runtime();
    let response = runtime.block_on(async {
        let mut client =
            proto::auto_rigging_service_client::AutoRiggingServiceClient::connect(address.clone())
                .await
                .expect("connect to mock");
        client
            .generate_rig(proto::RiggingRequest {
                glb_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
                params: Some(proto::RiggingParams {
                    num_sample_points: 65536,
                }),
                model_type: proto::RiggingModelType::RiggingUnirig as i32,
            })
            .await
            .expect("generate_rig")
            .into_inner()
    });

    let response_bytes = response.encode_to_vec();
    assert_eq!(
        response_bytes, expected_response_bytes,
        "deterministic response did not round-trip to the fixture wire bytes"
    );
    assert_eq!(response.rigged_glb_data, expected_response.rigged_glb_data);
    assert_eq!(
        response.skeleton_joints.len(),
        expected_response.skeleton_joints.len()
    );
}

#[test]
fn three_services_register_and_record_independently() {
    let server = MockServerHandle::start(MockServerConfig::EchoDefault);
    let address = server.address.clone();

    let runtime = build_client_runtime();
    runtime.block_on(async {
        let mut auto_rig_client =
            proto::auto_rigging_service_client::AutoRiggingServiceClient::connect(address.clone())
                .await
                .expect("auto rig connect");
        let _ = auto_rig_client
            .generate_rig(proto::RiggingRequest {
                glb_data: vec![],
                params: None,
                model_type: 0,
            })
            .await
            .expect("auto rig call");

        let mut motion_client =
            proto::text_to_motion_service_client::TextToMotionServiceClient::connect(
                address.clone(),
            )
            .await
            .expect("motion connect");
        let _ = motion_client
            .generate_motion(proto::MotionRequest {
                prompt: "smoke".into(),
                duration_seconds: 1.0,
                target_fps: 30,
                skeleton_type: 0,
                bone_mappings: vec![],
                glb_skeleton: None,
                internal_use_only: false,
            })
            .await
            .expect("motion call");

        let mut mesh_client =
            proto::mesh_generation_service_client::MeshGenerationServiceClient::connect(
                address.clone(),
            )
            .await
            .expect("mesh connect");
        let _ = mesh_client
            .generate_mesh(proto::MeshRequest {
                prompt: "smoke".into(),
                params: None,
                input_image_png: vec![],
                input_mode: 0,
                model_type: 0,
                t2i_model_type: 0,
            })
            .await
            .expect("mesh call");
    });

    let buffer = server.recv_buffer.lock().expect("recv_buffer");
    assert_eq!(buffer.count("AutoRiggingService", "GenerateRig"), 1);
    assert_eq!(buffer.count("TextToMotionService", "GenerateMotion"), 1);
    assert_eq!(buffer.count("MeshGenerationService", "GenerateMesh"), 1);
}
