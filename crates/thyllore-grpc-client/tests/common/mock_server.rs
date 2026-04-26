#![cfg(feature = "text-to-motion")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use prost::Message;
use sha2::{Digest, Sha256};

use thyllore_grpc_client::proto;

const SERVICE_AUTO_RIG: &str = "AutoRiggingService";
const SERVICE_MOTION: &str = "TextToMotionService";
const SERVICE_MESH: &str = "MeshGenerationService";

const METHOD_GENERATE_RIG: &str = "GenerateRig";
const METHOD_GENERATE_MOTION: &str = "GenerateMotion";
const METHOD_GENERATE_MESH: &str = "GenerateMesh";

const RIGGING_RESPONSE_FIXTURE: &str = "rigging_response.bin";
const MOTION_RESPONSE_FIXTURE: &str = "motion_response.bin";
const MESH_RESPONSE_FIXTURE: &str = "mesh_response.bin";

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum MockServerConfig {
    EchoDefault,
    DeterministicFromFixtures(PathBuf),
}

#[derive(Default, Debug)]
pub struct RecvBuffer {
    re_encoded_messages_by_method: BTreeMap<(String, String), Vec<Vec<u8>>>,
}

impl RecvBuffer {
    pub fn record(&mut self, service: &str, method: &str, re_encoded_bytes: Vec<u8>) {
        self.re_encoded_messages_by_method
            .entry((service.to_string(), method.to_string()))
            .or_default()
            .push(re_encoded_bytes);
    }

    pub fn count(&self, service: &str, method: &str) -> usize {
        self.re_encoded_messages_by_method
            .get(&(service.to_string(), method.to_string()))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn bytes_at(&self, service: &str, method: &str, index: usize) -> Option<&[u8]> {
        self.re_encoded_messages_by_method
            .get(&(service.to_string(), method.to_string()))
            .and_then(|v| v.get(index))
            .map(Vec::as_slice)
    }

    pub fn sha256_of(&self, service: &str, method: &str, index: usize) -> Option<String> {
        let bytes = self.bytes_at(service, method, index)?;
        Some(format!("{:x}", Sha256::digest(bytes)))
    }
}

pub struct MockServerHandle {
    pub address: String,
    #[allow(dead_code)]
    pub port: u16,
    pub recv_buffer: Arc<Mutex<RecvBuffer>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join: Option<thread::JoinHandle<()>>,
}

impl MockServerHandle {
    pub fn start(config: MockServerConfig) -> Self {
        let recv_buffer = Arc::new(Mutex::new(RecvBuffer::default()));
        let recv_buffer_for_thread = recv_buffer.clone();
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let (port_tx, port_rx) = std::sync::mpsc::channel::<u16>();

        let join = thread::Builder::new()
            .name("phase5-mock-grpc-server".into())
            .spawn(move || {
                run_server_thread(config, recv_buffer_for_thread, shutdown_rx, port_tx);
            })
            .expect("spawn mock server thread");

        let port = port_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("receive port from server thread");

        Self {
            address: format!("http://127.0.0.1:{port}"),
            port,
            recv_buffer,
            shutdown_tx: Some(shutdown_tx),
            join: Some(join),
        }
    }

    #[allow(dead_code)]
    pub fn shutdown(mut self) {
        self.shutdown_inner();
    }

    fn shutdown_inner(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for MockServerHandle {
    fn drop(&mut self) {
        self.shutdown_inner();
    }
}

fn run_server_thread(
    config: MockServerConfig,
    recv_buffer: Arc<Mutex<RecvBuffer>>,
    shutdown_rx: tokio::sync::oneshot::Receiver<()>,
    port_tx: std::sync::mpsc::Sender<u16>,
) {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind 127.0.0.1:0");
        let port = listener.local_addr().expect("local_addr").port();
        port_tx.send(port).expect("send port");

        let state = ServiceState {
            config: Arc::new(config),
            recv_buffer,
        };
        let auto_rig = MockAutoRigService {
            state: state.clone(),
        };
        let motion = MockTextToMotionService {
            state: state.clone(),
        };
        let mesh = MockMeshGenerationService { state };

        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

        let result = tonic::transport::Server::builder()
            .add_service(
                proto::auto_rigging_service_server::AutoRiggingServiceServer::new(auto_rig),
            )
            .add_service(
                proto::text_to_motion_service_server::TextToMotionServiceServer::new(motion),
            )
            .add_service(
                proto::mesh_generation_service_server::MeshGenerationServiceServer::new(mesh),
            )
            .serve_with_incoming_shutdown(incoming, async {
                let _ = shutdown_rx.await;
            })
            .await;
        if let Err(e) = result {
            eprintln!("phase5 mock server exited with error: {e}");
        }
    });
}

#[derive(Clone)]
struct ServiceState {
    config: Arc<MockServerConfig>,
    recv_buffer: Arc<Mutex<RecvBuffer>>,
}

impl ServiceState {
    // tonic decodes the wire bytes into a typed message before our handler runs,
    // so we cannot keep the literal bytes the client sent. Re-encoding via prost
    // is bit-identical thanks to the deterministic encoding used by both the
    // Rust and Python protobuf libraries (proto must avoid map<>).
    fn record_re_encoded_request<M: Message>(&self, service: &str, method: &str, message: &M) {
        let re_encoded = message.encode_to_vec();
        let mut buffer = self.recv_buffer.lock().expect("recv_buffer poisoned");
        buffer.record(service, method, re_encoded);
    }

    fn load_fixture_response<M: Message + Default>(
        &self,
        fixture_filename: &str,
    ) -> Result<M, tonic::Status> {
        match self.config.as_ref() {
            MockServerConfig::EchoDefault => Ok(M::default()),
            MockServerConfig::DeterministicFromFixtures(root) => {
                let path = root.join("proto").join(fixture_filename);
                let bytes = std::fs::read(&path).map_err(|e| {
                    tonic::Status::internal(format!(
                        "fixture {} not readable: {e}",
                        path.display()
                    ))
                })?;
                M::decode(bytes.as_slice()).map_err(|e| {
                    tonic::Status::internal(format!(
                        "fixture {} failed to decode: {e}",
                        path.display()
                    ))
                })
            }
        }
    }

    fn is_deterministic(&self) -> bool {
        matches!(
            self.config.as_ref(),
            MockServerConfig::DeterministicFromFixtures(_)
        )
    }
}

struct MockAutoRigService {
    state: ServiceState,
}

#[tonic::async_trait]
impl proto::auto_rigging_service_server::AutoRiggingService for MockAutoRigService {
    async fn generate_rig(
        &self,
        request: tonic::Request<proto::RiggingRequest>,
    ) -> Result<tonic::Response<proto::RiggingResponse>, tonic::Status> {
        let inner = request.into_inner();
        self.state
            .record_re_encoded_request(SERVICE_AUTO_RIG, METHOD_GENERATE_RIG, &inner);

        if self.state.is_deterministic() {
            let response: proto::RiggingResponse =
                self.state.load_fixture_response(RIGGING_RESPONSE_FIXTURE)?;
            return Ok(tonic::Response::new(response));
        }
        Ok(tonic::Response::new(echo_rigging_response(&inner)))
    }

    async fn get_rigging_status(
        &self,
        _: tonic::Request<proto::RiggingStatusRequest>,
    ) -> Result<tonic::Response<proto::RiggingStatusResponse>, tonic::Status> {
        Ok(tonic::Response::new(proto::RiggingStatusResponse {
            ready: true,
            model_name: "Phase5MockUniRig".into(),
            gpu_memory_mb: 0,
        }))
    }
}

struct MockTextToMotionService {
    state: ServiceState,
}

#[tonic::async_trait]
impl proto::text_to_motion_service_server::TextToMotionService for MockTextToMotionService {
    async fn generate_motion(
        &self,
        request: tonic::Request<proto::MotionRequest>,
    ) -> Result<tonic::Response<proto::MotionResponse>, tonic::Status> {
        let inner = request.into_inner();
        self.state
            .record_re_encoded_request(SERVICE_MOTION, METHOD_GENERATE_MOTION, &inner);

        if self.state.is_deterministic() {
            let response: proto::MotionResponse =
                self.state.load_fixture_response(MOTION_RESPONSE_FIXTURE)?;
            return Ok(tonic::Response::new(response));
        }
        Ok(tonic::Response::new(echo_motion_response(&inner)))
    }

    async fn get_server_status(
        &self,
        _: tonic::Request<proto::StatusRequest>,
    ) -> Result<tonic::Response<proto::StatusResponse>, tonic::Status> {
        Ok(tonic::Response::new(proto::StatusResponse {
            ready: true,
            active_model: "Phase5MockLightT2M".into(),
            gpu_memory_mb: 0,
        }))
    }
}

struct MockMeshGenerationService {
    state: ServiceState,
}

#[tonic::async_trait]
impl proto::mesh_generation_service_server::MeshGenerationService for MockMeshGenerationService {
    async fn generate_mesh(
        &self,
        request: tonic::Request<proto::MeshRequest>,
    ) -> Result<tonic::Response<proto::MeshResponse>, tonic::Status> {
        let inner = request.into_inner();
        self.state
            .record_re_encoded_request(SERVICE_MESH, METHOD_GENERATE_MESH, &inner);

        if self.state.is_deterministic() {
            let response: proto::MeshResponse =
                self.state.load_fixture_response(MESH_RESPONSE_FIXTURE)?;
            return Ok(tonic::Response::new(response));
        }
        Ok(tonic::Response::new(echo_mesh_response(&inner)))
    }

    async fn get_mesh_service_status(
        &self,
        _: tonic::Request<proto::MeshStatusRequest>,
    ) -> Result<tonic::Response<proto::MeshStatusResponse>, tonic::Status> {
        Ok(tonic::Response::new(proto::MeshStatusResponse {
            ready: true,
            t2i_model: "Phase5MockSDXL".into(),
            i2m_model: "Phase5MockTRELLIS".into(),
            gpu_memory_mb: 0,
        }))
    }
}

fn echo_rigging_response(req: &proto::RiggingRequest) -> proto::RiggingResponse {
    proto::RiggingResponse {
        rigged_glb_data: req.glb_data.clone(),
        metadata: Some(proto::RiggingMetadata {
            joint_count: 1,
            bone_count: 1,
            generation_time_ms: 0.0,
        }),
        skeleton_joints: vec![proto::SkeletonJoint {
            name: "root".into(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            tail_x: 0.0,
            tail_y: 0.1,
            tail_z: 0.0,
            parent_index: -1,
        }],
    }
}

fn echo_motion_response(req: &proto::MotionRequest) -> proto::MotionResponse {
    proto::MotionResponse {
        curves: vec![proto::AnimationCurve {
            bone_name: "root".into(),
            property_type: proto::PropertyType::TranslationX as i32,
            keyframes: vec![proto::CurveKeyframe {
                time: 0.0,
                value: 0.0,
                tangent_in_dt: 0.0,
                tangent_in_dv: 0.0,
                tangent_out_dt: 0.0,
                tangent_out_dv: 0.0,
                interpolation: proto::InterpolationType::Linear as i32,
            }],
        }],
        generation_time_ms: 0.0,
        model_used: format!(
            "phase5_echo:{}",
            req.prompt.chars().take(16).collect::<String>()
        ),
    }
}

fn echo_mesh_response(_req: &proto::MeshRequest) -> proto::MeshResponse {
    proto::MeshResponse {
        glb_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
        metadata: Some(proto::MeshMetadata {
            vertex_count: 0,
            face_count: 0,
            generation_time_ms: 0.0,
            intermediate_image_png: vec![],
        }),
    }
}
