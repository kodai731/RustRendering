use std::sync::mpsc;
use std::thread;

use super::request::{
    GrpcRequest, GrpcResponse, RawAnimationCurve, RawCurveKeyframe, TextToMotionRequest,
};

pub struct GrpcThreadHandle {
    sender: Option<mpsc::Sender<GrpcRequest>>,
    receiver: mpsc::Receiver<GrpcResponse>,
    join_handle: Option<thread::JoinHandle<()>>,
}

pub mod proto {
    tonic::include_proto!("animation_ml");
}

impl GrpcThreadHandle {
    pub fn spawn(endpoint: &str) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<GrpcRequest>();
        let (res_tx, res_rx) = mpsc::channel::<GrpcResponse>();
        let endpoint = endpoint.to_string();

        let join_handle = thread::Builder::new()
            .name("grpc-ml".to_string())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Failed to create tokio runtime");

                rt.block_on(run_grpc_loop(&endpoint, req_rx, res_tx));
            })
            .expect("Failed to spawn gRPC thread");

        Self {
            sender: Some(req_tx),
            receiver: res_rx,
            join_handle: Some(join_handle),
        }
    }

    pub fn send(&self, request: GrpcRequest) {
        if let Some(ref sender) = self.sender {
            let _ = sender.send(request);
        }
    }

    pub fn try_recv(&self) -> Option<GrpcResponse> {
        self.receiver.try_recv().ok()
    }
}

impl Drop for GrpcThreadHandle {
    fn drop(&mut self) {
        if let Some(ref sender) = self.sender {
            let _ = sender.send(GrpcRequest::Shutdown);
        }
        self.sender.take();

        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

async fn run_grpc_loop(
    endpoint: &str,
    req_rx: mpsc::Receiver<GrpcRequest>,
    res_tx: mpsc::Sender<GrpcResponse>,
) {
    let mut motion_client: Option<MotionGrpcClient> = None;
    #[cfg(feature = "text-to-mesh")]
    let mut mesh_client: Option<MeshGrpcClient> = None;

    while let Ok(request) = req_rx.recv() {
        match request {
            GrpcRequest::Shutdown => break,

            GrpcRequest::CheckStatus => {
                handle_check_status(endpoint, &mut motion_client, &res_tx).await;
            }

            GrpcRequest::GenerateMotion(req) => {
                handle_generate_motion(endpoint, &mut motion_client, &res_tx, req).await;
            }

            #[cfg(feature = "text-to-mesh")]
            GrpcRequest::CheckMeshStatus => {
                handle_check_mesh_status(endpoint, &mut mesh_client, &res_tx).await;
            }

            #[cfg(feature = "text-to-mesh")]
            GrpcRequest::GenerateMesh(req) => {
                handle_generate_mesh(endpoint, &mut mesh_client, &res_tx, req).await;
            }
        }
    }
}

type MotionGrpcClient =
    proto::text_to_motion_service_client::TextToMotionServiceClient<tonic::transport::Channel>;

#[cfg(feature = "text-to-mesh")]
type MeshGrpcClient =
    proto::mesh_generation_service_client::MeshGenerationServiceClient<tonic::transport::Channel>;

async fn ensure_connected(
    endpoint: &str,
    client: &mut Option<MotionGrpcClient>,
    res_tx: &mpsc::Sender<GrpcResponse>,
) -> bool {
    if client.is_some() {
        return true;
    }

    match tonic::transport::Channel::from_shared(endpoint.to_string()) {
        Ok(channel_builder) => match channel_builder.connect().await {
            Ok(channel) => {
                *client = Some(
                    proto::text_to_motion_service_client::TextToMotionServiceClient::new(channel),
                );
                true
            }
            Err(e) => {
                let _ = res_tx.send(GrpcResponse::Error {
                    message: format!("Failed to connect to {}: {}", endpoint, e),
                });
                false
            }
        },
        Err(e) => {
            let _ = res_tx.send(GrpcResponse::Error {
                message: format!("Invalid endpoint '{}': {}", endpoint, e),
            });
            false
        }
    }
}

#[cfg(feature = "text-to-mesh")]
async fn ensure_mesh_connected(
    endpoint: &str,
    client: &mut Option<MeshGrpcClient>,
    res_tx: &mpsc::Sender<GrpcResponse>,
) -> bool {
    if client.is_some() {
        return true;
    }

    match tonic::transport::Channel::from_shared(endpoint.to_string()) {
        Ok(channel_builder) => match channel_builder.connect().await {
            Ok(channel) => {
                *client = Some(
                    proto::mesh_generation_service_client::MeshGenerationServiceClient::new(
                        channel,
                    )
                    .max_decoding_message_size(64 * 1024 * 1024),
                );
                true
            }
            Err(e) => {
                let _ = res_tx.send(GrpcResponse::Error {
                    message: format!("Failed to connect to {}: {}", endpoint, e),
                });
                false
            }
        },
        Err(e) => {
            let _ = res_tx.send(GrpcResponse::Error {
                message: format!("Invalid endpoint '{}': {}", endpoint, e),
            });
            false
        }
    }
}

async fn handle_check_status(
    endpoint: &str,
    client: &mut Option<MotionGrpcClient>,
    res_tx: &mpsc::Sender<GrpcResponse>,
) {
    if !ensure_connected(endpoint, client, res_tx).await {
        return;
    }

    let c = client
        .as_mut()
        .expect("ensure_connected returned true so client is Some");
    let request = tonic::Request::new(proto::StatusRequest {});

    match c.get_server_status(request).await {
        Ok(response) => {
            let status = response.into_inner();
            let _ = res_tx.send(GrpcResponse::ServerStatus {
                ready: status.ready,
                active_model: status.active_model,
                gpu_memory_mb: status.gpu_memory_mb,
            });
        }
        Err(e) => {
            *client = None;
            let _ = res_tx.send(GrpcResponse::Error {
                message: format!("GetServerStatus failed: {}", e),
            });
        }
    }
}

async fn handle_generate_motion(
    endpoint: &str,
    client: &mut Option<MotionGrpcClient>,
    res_tx: &mpsc::Sender<GrpcResponse>,
    req: TextToMotionRequest,
) {
    if !ensure_connected(endpoint, client, res_tx).await {
        return;
    }

    let c = client
        .as_mut()
        .expect("ensure_connected returned true so client is Some");
    let proto_request = proto::MotionRequest {
        prompt: req.prompt,
        duration_seconds: req.duration_seconds,
        target_fps: req.target_fps,
        skeleton_type: proto::SkeletonType::VrmHumanoid as i32,
        bone_mappings: vec![],
    };

    match c.generate_motion(tonic::Request::new(proto_request)).await {
        Ok(response) => {
            let motion = response.into_inner();
            let curves = convert_proto_curves(&motion.curves);

            let _ = res_tx.send(GrpcResponse::MotionGenerated {
                curves,
                generation_time_ms: motion.generation_time_ms,
                model_used: motion.model_used,
            });
        }
        Err(e) => {
            *client = None;
            let _ = res_tx.send(GrpcResponse::Error {
                message: format!("GenerateMotion failed: {}", e),
            });
        }
    }
}

#[cfg(feature = "text-to-mesh")]
async fn handle_check_mesh_status(
    endpoint: &str,
    client: &mut Option<MeshGrpcClient>,
    res_tx: &mpsc::Sender<GrpcResponse>,
) {
    if !ensure_mesh_connected(endpoint, client, res_tx).await {
        return;
    }

    let c = client
        .as_mut()
        .expect("ensure_mesh_connected returned true so client is Some");
    let request = tonic::Request::new(proto::MeshStatusRequest {});

    match c.get_mesh_service_status(request).await {
        Ok(response) => {
            let status = response.into_inner();
            let _ = res_tx.send(GrpcResponse::MeshServerStatus {
                ready: status.ready,
            });
        }
        Err(e) => {
            *client = None;
            let _ = res_tx.send(GrpcResponse::Error {
                message: format!("GetMeshServiceStatus failed: {}", e),
            });
        }
    }
}

#[cfg(feature = "text-to-mesh")]
async fn handle_generate_mesh(
    endpoint: &str,
    client: &mut Option<MeshGrpcClient>,
    res_tx: &mpsc::Sender<GrpcResponse>,
    req: super::request::TextToMeshRequest,
) {
    if !ensure_mesh_connected(endpoint, client, res_tx).await {
        return;
    }

    let c = client
        .as_mut()
        .expect("ensure_mesh_connected returned true so client is Some");
    let proto_mode = match req.input_mode {
        super::request::MeshInputMode::TextOnly => proto::MeshInputMode::TextToMesh,
        super::request::MeshInputMode::Image => proto::MeshInputMode::ImageRefinedToMesh,
    };

    let proto_model_type = match req.model_type {
        super::request::MeshModelType::Trellis => proto::MeshModelType::Trellis,
        super::request::MeshModelType::Hunyuan3D => proto::MeshModelType::Hunyuan3d,
        super::request::MeshModelType::CharacterGen => proto::MeshModelType::CharacterGen,
    };

    let proto_request = proto::MeshRequest {
        prompt: req.prompt,
        params: Some(proto::MeshGenerationParams {
            target_faces: req.target_faces as i32,
            seed: req.seed as i32,
            image_size: 0,
            image_inference_steps: 0,
        }),
        input_image_png: req.input_image_png.unwrap_or_default(),
        input_mode: proto_mode as i32,
        model_type: proto_model_type as i32,
    };

    match c.generate_mesh(tonic::Request::new(proto_request)).await {
        Ok(response) => {
            let resp = response.into_inner();
            let metadata = resp.metadata.unwrap_or_default();
            let image = if metadata.intermediate_image_png.is_empty() {
                None
            } else {
                Some(metadata.intermediate_image_png)
            };
            let _ = res_tx.send(GrpcResponse::MeshGenerated {
                glb_data: resp.glb_data,
                vertex_count: metadata.vertex_count as u32,
                face_count: metadata.face_count as u32,
                generation_time_ms: metadata.generation_time_ms,
                intermediate_image_png: image,
            });
        }
        Err(e) => {
            *client = None;
            let _ = res_tx.send(GrpcResponse::Error {
                message: format!("GenerateMesh failed: {}", e),
            });
        }
    }
}

fn convert_proto_curves(proto_curves: &[proto::AnimationCurve]) -> Vec<RawAnimationCurve> {
    proto_curves
        .iter()
        .map(|c| RawAnimationCurve {
            bone_name: c.bone_name.clone(),
            property_type: c.property_type,
            keyframes: c
                .keyframes
                .iter()
                .map(|kf| RawCurveKeyframe {
                    time: kf.time,
                    value: kf.value,
                    tangent_in_dt: kf.tangent_in_dt,
                    tangent_in_dv: kf.tangent_in_dv,
                    tangent_out_dt: kf.tangent_out_dt,
                    tangent_out_dv: kf.tangent_out_dv,
                    interpolation: kf.interpolation,
                })
                .collect(),
        })
        .collect()
}
