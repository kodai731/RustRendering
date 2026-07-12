#![cfg(feature = "text-to-motion")]

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use prost::Message;
use sha2::{Digest, Sha256};

use thyllore_grpc_client::proto;

const RIGGING_REQUEST_FIXTURE: &str = "proto/rigging_request.bin";
const MOTION_REQUEST_FIXTURE: &str = "proto/motion_request.bin";
const MESH_REQUEST_FIXTURE: &str = "proto/mesh_request.bin";

const RIGGING_RESULT_NAME: &str = "rigging_response_rust.json";
const MOTION_RESULT_NAME: &str = "motion_response_rust.json";
const MESH_RESULT_NAME: &str = "mesh_response_rust.json";

#[test]
#[ignore]
fn run_for_orchestrator() {
    let context = OrchestratorContext::from_env();
    fs::create_dir_all(&context.result_dir).expect("create result dir");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("client runtime");

    runtime.block_on(async move {
        run_auto_rig(&context).await;
        run_text_to_motion(&context).await;
        run_mesh(&context).await;
    });
}

struct OrchestratorContext {
    server_url: String,
    fixture_root: PathBuf,
    result_dir: PathBuf,
}

impl OrchestratorContext {
    fn from_env() -> Self {
        let server_url = env::var("THYLLORE_PARITY_SERVER_URL")
            .expect("THYLLORE_PARITY_SERVER_URL must be set by orchestrator");
        let fixture_root = PathBuf::from(
            env::var("THYLLORE_PARITY_FIXTURE_ROOT")
                .expect("THYLLORE_PARITY_FIXTURE_ROOT must be set"),
        );
        let result_dir = PathBuf::from(
            env::var("THYLLORE_PARITY_RESULT_DIR").expect("THYLLORE_PARITY_RESULT_DIR must be set"),
        );
        Self {
            server_url,
            fixture_root,
            result_dir,
        }
    }
}

async fn run_auto_rig(context: &OrchestratorContext) {
    let request: proto::RiggingRequest =
        decode_fixture(&context.fixture_root, RIGGING_REQUEST_FIXTURE);

    let mut client = proto::auto_rigging_service_client::AutoRiggingServiceClient::connect(
        context.server_url.clone(),
    )
    .await
    .expect("connect AutoRigging");
    let response = client
        .generate_rig(request)
        .await
        .expect("generate_rig")
        .into_inner();

    write_canonical_json(
        &context.result_dir.join(RIGGING_RESULT_NAME),
        &auto_rig_response_to_canonical(&response),
    );
}

async fn run_text_to_motion(context: &OrchestratorContext) {
    let request: proto::MotionRequest =
        decode_fixture(&context.fixture_root, MOTION_REQUEST_FIXTURE);

    let mut client = proto::text_to_motion_service_client::TextToMotionServiceClient::connect(
        context.server_url.clone(),
    )
    .await
    .expect("connect TextToMotion");
    let response = client
        .generate_motion(request)
        .await
        .expect("generate_motion")
        .into_inner();

    write_canonical_json(
        &context.result_dir.join(MOTION_RESULT_NAME),
        &motion_response_to_canonical(&response),
    );
}

async fn run_mesh(context: &OrchestratorContext) {
    let request: proto::MeshRequest = decode_fixture(&context.fixture_root, MESH_REQUEST_FIXTURE);

    let mut client = proto::mesh_generation_service_client::MeshGenerationServiceClient::connect(
        context.server_url.clone(),
    )
    .await
    .expect("connect MeshGeneration");
    let response = client
        .generate_mesh(request)
        .await
        .expect("generate_mesh")
        .into_inner();

    write_canonical_json(
        &context.result_dir.join(MESH_RESULT_NAME),
        &mesh_response_to_canonical(&response),
    );
}

fn decode_fixture<M: Message + Default>(fixture_root: &Path, relative: &str) -> M {
    let path = fixture_root.join(relative);
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read fixture {}: {e}", path.display()));
    M::decode(bytes.as_slice()).unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
}

fn write_canonical_json(path: &Path, value: &serde_json::Value) {
    let text = serde_json::to_string_pretty(value)
        .unwrap_or_else(|e| panic!("serialize {}: {e}", path.display()));
    fs::write(path, text + "\n").unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn auto_rig_response_to_canonical(response: &proto::RiggingResponse) -> serde_json::Value {
    let mut root: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    root.insert(
        "rigged_glb_sha256".to_string(),
        serde_json::Value::String(sha256_hex(&response.rigged_glb_data)),
    );
    root.insert(
        "rigged_glb_size".to_string(),
        serde_json::Value::Number((response.rigged_glb_data.len() as u64).into()),
    );
    root.insert(
        "metadata".to_string(),
        rigging_metadata_to_canonical(response.metadata.as_ref()),
    );

    let joints: Vec<serde_json::Value> = response
        .skeleton_joints
        .iter()
        .map(skeleton_joint_to_canonical)
        .collect();
    root.insert(
        "skeleton_joints".to_string(),
        serde_json::Value::Array(joints),
    );

    serde_json::Value::Object(root.into_iter().collect())
}

fn skeleton_joint_to_canonical(joint: &proto::SkeletonJoint) -> serde_json::Value {
    let mut entry: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    entry.insert(
        "name".to_string(),
        serde_json::Value::String(joint.name.clone()),
    );
    entry.insert(
        "parent_index".to_string(),
        serde_json::Value::Number(joint.parent_index.into()),
    );
    entry.insert(
        "head_x_bits".to_string(),
        serde_json::Value::Number(joint.x.to_bits().into()),
    );
    entry.insert(
        "head_y_bits".to_string(),
        serde_json::Value::Number(joint.y.to_bits().into()),
    );
    entry.insert(
        "head_z_bits".to_string(),
        serde_json::Value::Number(joint.z.to_bits().into()),
    );
    entry.insert(
        "tail_x_bits".to_string(),
        serde_json::Value::Number(joint.tail_x.to_bits().into()),
    );
    entry.insert(
        "tail_y_bits".to_string(),
        serde_json::Value::Number(joint.tail_y.to_bits().into()),
    );
    entry.insert(
        "tail_z_bits".to_string(),
        serde_json::Value::Number(joint.tail_z.to_bits().into()),
    );
    serde_json::Value::Object(entry.into_iter().collect())
}

fn rigging_metadata_to_canonical(metadata: Option<&proto::RiggingMetadata>) -> serde_json::Value {
    let Some(metadata) = metadata else {
        return serde_json::Value::Null;
    };
    let mut entry: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    entry.insert(
        "joint_count".to_string(),
        serde_json::Value::Number(metadata.joint_count.into()),
    );
    entry.insert(
        "bone_count".to_string(),
        serde_json::Value::Number(metadata.bone_count.into()),
    );
    entry.insert(
        "generation_time_ms_bits".to_string(),
        serde_json::Value::Number(metadata.generation_time_ms.to_bits().into()),
    );
    serde_json::Value::Object(entry.into_iter().collect())
}

fn motion_response_to_canonical(response: &proto::MotionResponse) -> serde_json::Value {
    let mut root: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    root.insert(
        "model_used".to_string(),
        serde_json::Value::String(response.model_used.clone()),
    );
    root.insert(
        "generation_time_ms_bits".to_string(),
        serde_json::Value::Number(response.generation_time_ms.to_bits().into()),
    );

    let curves: Vec<serde_json::Value> = response
        .curves
        .iter()
        .map(animation_curve_to_canonical)
        .collect();
    root.insert("curves".to_string(), serde_json::Value::Array(curves));

    serde_json::Value::Object(root.into_iter().collect())
}

fn animation_curve_to_canonical(curve: &proto::AnimationCurve) -> serde_json::Value {
    let mut entry: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    entry.insert(
        "bone_name".to_string(),
        serde_json::Value::String(curve.bone_name.clone()),
    );
    entry.insert(
        "property_type".to_string(),
        serde_json::Value::Number(curve.property_type.into()),
    );

    let keyframes: Vec<serde_json::Value> =
        curve.keyframes.iter().map(keyframe_to_canonical).collect();
    entry.insert("keyframes".to_string(), serde_json::Value::Array(keyframes));

    serde_json::Value::Object(entry.into_iter().collect())
}

fn keyframe_to_canonical(keyframe: &proto::CurveKeyframe) -> serde_json::Value {
    let mut entry: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    entry.insert(
        "time_bits".to_string(),
        serde_json::Value::Number(keyframe.time.to_bits().into()),
    );
    entry.insert(
        "value_bits".to_string(),
        serde_json::Value::Number(keyframe.value.to_bits().into()),
    );
    entry.insert(
        "tangent_in_dt_bits".to_string(),
        serde_json::Value::Number(keyframe.tangent_in_dt.to_bits().into()),
    );
    entry.insert(
        "tangent_in_dv_bits".to_string(),
        serde_json::Value::Number(keyframe.tangent_in_dv.to_bits().into()),
    );
    entry.insert(
        "tangent_out_dt_bits".to_string(),
        serde_json::Value::Number(keyframe.tangent_out_dt.to_bits().into()),
    );
    entry.insert(
        "tangent_out_dv_bits".to_string(),
        serde_json::Value::Number(keyframe.tangent_out_dv.to_bits().into()),
    );
    entry.insert(
        "interpolation".to_string(),
        serde_json::Value::Number(keyframe.interpolation.into()),
    );
    serde_json::Value::Object(entry.into_iter().collect())
}

fn mesh_response_to_canonical(response: &proto::MeshResponse) -> serde_json::Value {
    let mut root: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    root.insert(
        "glb_sha256".to_string(),
        serde_json::Value::String(sha256_hex(&response.glb_data)),
    );
    root.insert(
        "glb_size".to_string(),
        serde_json::Value::Number((response.glb_data.len() as u64).into()),
    );
    root.insert(
        "metadata".to_string(),
        mesh_metadata_to_canonical(response.metadata.as_ref()),
    );
    serde_json::Value::Object(root.into_iter().collect())
}

fn mesh_metadata_to_canonical(metadata: Option<&proto::MeshMetadata>) -> serde_json::Value {
    let Some(metadata) = metadata else {
        return serde_json::Value::Null;
    };
    let mut entry: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    entry.insert(
        "vertex_count".to_string(),
        serde_json::Value::Number(metadata.vertex_count.into()),
    );
    entry.insert(
        "face_count".to_string(),
        serde_json::Value::Number(metadata.face_count.into()),
    );
    entry.insert(
        "generation_time_ms_bits".to_string(),
        serde_json::Value::Number(metadata.generation_time_ms.to_bits().into()),
    );
    entry.insert(
        "intermediate_image_png_sha256".to_string(),
        serde_json::Value::String(sha256_hex(&metadata.intermediate_image_png)),
    );
    entry.insert(
        "intermediate_image_png_size".to_string(),
        serde_json::Value::Number((metadata.intermediate_image_png.len() as u64).into()),
    );
    serde_json::Value::Object(entry.into_iter().collect())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}
