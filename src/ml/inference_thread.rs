use std::sync::mpsc;
use std::thread;

use anyhow::Result;

use thyllore_ml_core::copilot::input::BoneContextInput;
use thyllore_ml_core::copilot::session::{CurveCopilotRequest, Session};
use thyllore_ml_core::{
    InferenceActorId, InferenceRequest, InferenceRequestKind, InferenceResult, InferenceResultKind,
};

pub struct InferenceThreadHandle {
    sender: Option<mpsc::Sender<InferenceRequest>>,
    receiver: mpsc::Receiver<InferenceResult>,
    join_handle: Option<thread::JoinHandle<()>>,
    max_steps: Option<usize>,
}

impl InferenceThreadHandle {
    pub fn spawn(model_path: &str, actor_id: InferenceActorId) -> Result<Self> {
        let (req_tx, req_rx) = mpsc::channel::<InferenceRequest>();
        let (res_tx, res_rx) = mpsc::channel::<InferenceResult>();

        let session = Session::from_onnx_path(model_path)?;
        let max_steps = session.max_steps();

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
    mut session: Session,
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
    session: &mut Session,
    request: &InferenceRequest,
) -> Result<InferenceResultKind> {
    match &request.kind {
        InferenceRequestKind::CurvePredict { input } => {
            let output = session.run_curve_predict(input)?;
            Ok(InferenceResultKind::CurvePredict { output })
        }

        InferenceRequestKind::CurveCopilotPredict {
            context,
            property_type_id,
            topology_features,
            bone_name_tokens,
            query_times,
            curve_window,
            bone_context_keyframes,
            bone_context_topology,
            bone_context_rest_positions,
            bone_context_mask,
        } => {
            let steps = session.run_curve_copilot(CurveCopilotRequest {
                context,
                property_type_id: *property_type_id,
                topology_features,
                bone_name_tokens,
                query_times,
                curve_window,
                bone_context: BoneContextInput {
                    keyframes: bone_context_keyframes,
                    topology: bone_context_topology,
                    rest_positions: bone_context_rest_positions,
                    mask: bone_context_mask,
                },
            })?;

            log!(
                "CurveCopilot raw output: {} steps, query_times={:?}",
                steps.len(),
                query_times
            );

            Ok(InferenceResultKind::CurveCopilotPredict { steps })
        }
    }
}
