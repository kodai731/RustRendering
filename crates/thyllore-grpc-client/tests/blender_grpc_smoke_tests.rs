//! Phase 3 — Blender <-> gRPC smoke test.
//!
//! Mirrors the pattern used by `tests/gltf_export_tests.rs` for graceful
//! skipping when Blender is unavailable.
//!
//! Run: cargo test -p thyllore-grpc-client --features auto-rig --test blender_grpc_smoke_tests -- --nocapture

#![cfg(feature = "auto-rig")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use thyllore_grpc_client::proto;
use thyllore_grpc_client::proto::auto_rigging_service_server::{
    AutoRiggingService, AutoRiggingServiceServer,
};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root (two levels above grpc-client crate)")
        .to_path_buf()
}

#[derive(Default, Clone)]
struct CapturedRequest {
    glb_size: usize,
    num_sample_points: i32,
    metadata: Vec<(String, String)>,
}

struct MockAutoRigService {
    captured: Arc<Mutex<CapturedRequest>>,
}

#[tonic::async_trait]
impl AutoRiggingService for MockAutoRigService {
    async fn generate_rig(
        &self,
        request: tonic::Request<proto::RiggingRequest>,
    ) -> Result<tonic::Response<proto::RiggingResponse>, tonic::Status> {
        let metadata: Vec<(String, String)> = request
            .metadata()
            .iter()
            .filter_map(|kv| match kv {
                tonic::metadata::KeyAndValueRef::Ascii(key, value) => Some((
                    key.as_str().to_string(),
                    value.to_str().unwrap_or("").to_string(),
                )),
                tonic::metadata::KeyAndValueRef::Binary(_, _) => None,
            })
            .collect();

        let inner = request.into_inner();
        {
            let mut captured = self.captured.lock().expect("captured lock");
            captured.glb_size = inner.glb_data.len();
            captured.num_sample_points = inner
                .params
                .as_ref()
                .map(|p| p.num_sample_points)
                .unwrap_or(0);
            captured.metadata = metadata;
        }

        Ok(tonic::Response::new(proto::RiggingResponse {
            rigged_glb_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
            metadata: Some(proto::RiggingMetadata {
                joint_count: 1,
                bone_count: 1,
                generation_time_ms: 1.23,
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
        }))
    }

    async fn get_rigging_status(
        &self,
        _: tonic::Request<proto::RiggingStatusRequest>,
    ) -> Result<tonic::Response<proto::RiggingStatusResponse>, tonic::Status> {
        Ok(tonic::Response::new(proto::RiggingStatusResponse {
            ready: true,
            model_name: "MockUniRig".into(),
            gpu_memory_mb: 0,
        }))
    }
}

struct MockServerHandle {
    port: u16,
    captured: Arc<Mutex<CapturedRequest>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    join: Option<thread::JoinHandle<()>>,
}

impl MockServerHandle {
    fn captured(&self) -> CapturedRequest {
        self.captured.lock().expect("captured lock").clone()
    }
}

impl Drop for MockServerHandle {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

fn start_mock_tonic_server() -> MockServerHandle {
    let captured = Arc::new(Mutex::new(CapturedRequest::default()));
    let captured_for_service = captured.clone();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let (port_tx, port_rx) = std::sync::mpsc::channel::<u16>();

    let join = thread::Builder::new()
        .name("mock-tonic-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime");

            rt.block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind 127.0.0.1:0");
                let port = listener.local_addr().expect("local_addr").port();
                port_tx.send(port).expect("send port");

                let service = MockAutoRigService {
                    captured: captured_for_service,
                };
                let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);

                tonic::transport::Server::builder()
                    .add_service(AutoRiggingServiceServer::new(service))
                    .serve_with_incoming_shutdown(incoming, async {
                        let _ = shutdown_rx.await;
                    })
                    .await
                    .expect("tonic serve");
            });
        })
        .expect("spawn mock server thread");

    let port = port_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("receive port from server thread");

    MockServerHandle {
        port,
        captured,
        shutdown_tx: Some(shutdown_tx),
        join: Some(join),
    }
}

fn read_blender_path() -> Option<String> {
    let paths_file = workspace_root().join(".claude/local/paths.md");
    let content = std::fs::read_to_string(&paths_file).ok()?;
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
    let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let s = canonical.to_string_lossy().to_string();
    PathBuf::from(s.strip_prefix(r"\\?\").unwrap_or(&s))
}

#[test]
fn blender_grpc_smoke_auto_rig() {
    let blender_path = match read_blender_path() {
        Some(p) => p,
        None => {
            eprintln!("Skipping: BlenderPath not configured in .claude/local/paths.md");
            return;
        }
    };

    let root = workspace_root();
    let script_path = root.join("blender_addon/tests/smoke_auto_rig.py");
    if !script_path.exists() {
        eprintln!("Skipping: blender_addon/tests/smoke_auto_rig.py not found");
        return;
    }

    let stubs_pb2 = root.join("blender_addon/grpc_client/stubs/animation_ml_pb2.py");
    if !stubs_pb2.exists() {
        eprintln!(
            "Skipping: gRPC Python stubs not generated. \
             Run: pwsh -File scripts/gen_grpc_stubs.ps1"
        );
        return;
    }

    let server = start_mock_tonic_server();
    let port = server.port;

    let output_glb = std::env::temp_dir().join("blender_grpc_smoke_output.glb");
    let _ = std::fs::remove_file(&output_glb);

    let script_abs = canonicalize_no_prefix(&script_path);
    let output_abs = canonicalize_no_prefix(&output_glb);
    let pythonpath = canonicalize_no_prefix(&root);

    let result = Command::new(&blender_path)
        .args([
            "--background",
            "--python",
            &script_abs.to_string_lossy(),
            "--",
            &port.to_string(),
            &output_abs.to_string_lossy(),
        ])
        .env("PYTHONPATH", pythonpath.as_os_str())
        .output()
        .expect("Failed to spawn Blender");

    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    eprintln!("Blender stdout:\n{}", stdout);

    assert!(
        result.status.success(),
        "Blender gRPC smoke failed:\nstderr:\n{}",
        stderr,
    );

    assert!(
        output_glb.exists(),
        "Blender did not produce output GLB at {}",
        output_glb.display(),
    );

    let captured = server.captured();
    assert!(
        captured.glb_size > 0,
        "Mock server received empty glb_data — Blender export may have failed",
    );
    assert_eq!(
        captured.num_sample_points, 1024,
        "num_sample_points was not propagated through grpc_client",
    );

    std::fs::remove_file(&output_glb).ok();
}
