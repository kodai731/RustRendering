use std::sync::mpsc;
use std::thread;

use anyhow::Result;

use thyllore_ml_core::copilot::v1::rawfuture::{RawFutureRequest, RawFutureSession, MAX_HORIZON};
use thyllore_ml_core::{
    InferenceActorId, InferenceModelKind, InferenceRequest, InferenceRequestKind, InferenceResult,
    InferenceResultKind,
};

pub struct InferenceThreadHandle {
    sender: Option<mpsc::Sender<InferenceRequest>>,
    receiver: mpsc::Receiver<InferenceResult>,
    join_handle: Option<thread::JoinHandle<()>>,
    max_steps: Option<usize>,
}

impl InferenceThreadHandle {
    pub fn spawn(
        model_path: &str,
        actor_id: InferenceActorId,
        model_kind: InferenceModelKind,
    ) -> Result<Self> {
        let (req_tx, req_rx) = mpsc::channel::<InferenceRequest>();
        let (res_tx, res_rx) = mpsc::channel::<InferenceResult>();

        let InferenceModelKind::CurveCopilot = model_kind;
        let session = RawFutureSession::from_onnx_path(model_path)?;
        let max_steps = Some(MAX_HORIZON);

        let join_handle = thread::Builder::new()
            .name(format!("inference-actor-{}", actor_id))
            .spawn(move || {
                run_inference_loop(session, req_rx, res_tx);
            })?;

        Ok(Self {
            sender: Some(req_tx),
            receiver: res_rx,
            join_handle: Some(join_handle),
            max_steps,
        })
    }

    pub fn max_steps(&self) -> Option<usize> {
        self.max_steps
    }

    pub fn send(&self, request: InferenceRequest) -> Result<()> {
        if let Some(ref sender) = self.sender {
            sender.send(request)?;
        }
        Ok(())
    }

    pub fn try_recv(&self) -> Option<InferenceResult> {
        self.receiver.try_recv().ok()
    }
}

impl Drop for InferenceThreadHandle {
    fn drop(&mut self) {
        self.sender.take();

        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_inference_loop(
    mut session: RawFutureSession,
    receiver: mpsc::Receiver<InferenceRequest>,
    sender: mpsc::Sender<InferenceResult>,
) {
    while let Ok(request) = receiver.recv() {
        match execute_inference(&mut session, &request) {
            Ok(result_kind) => {
                let response = InferenceResult {
                    request_id: request.request_id,
                    actor_id: request.actor_id,
                    kind: result_kind,
                };
                if sender.send(response).is_err() {
                    break;
                }
            }
            Err(e) => {
                log_error!("Inference error for actor {}: {:?}", request.actor_id, e);
            }
        }
    }
}

fn execute_inference(
    session: &mut RawFutureSession,
    request: &InferenceRequest,
) -> Result<InferenceResultKind> {
    let InferenceRequestKind::CurveCopilotPredict {
        context,
        future,
        reveal_mask,
        fps,
    } = &request.kind;

    let mean_curve = session.predict_mean_curve(RawFutureRequest {
        context,
        future,
        reveal_mask,
        fps: *fps,
    })?;

    log!(
        "CurveCopilot raw output: {} dense values (fps={:.1})",
        mean_curve.len(),
        fps
    );

    Ok(InferenceResultKind::CurveCopilotPredict { mean_curve })
}
