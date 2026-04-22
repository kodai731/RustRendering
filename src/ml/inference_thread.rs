use std::sync::mpsc;
use std::thread;

use anyhow::Result;
use ort::session::Session;

use thyllore_ml_core::copilot::input::{build_curve_copilot_tensors, build_curve_predict_tensor};
use thyllore_ml_core::copilot::output::{parse_curve_copilot_output, parse_curve_predict_output};
use thyllore_ml_core::{
    InferenceActorId, InferenceRequest, InferenceRequestKind, InferenceResult, InferenceResultKind,
};

pub struct InferenceThreadHandle {
    sender: Option<mpsc::Sender<InferenceRequest>>,
    receiver: mpsc::Receiver<InferenceResult>,
    join_handle: Option<thread::JoinHandle<()>>,
}

impl InferenceThreadHandle {
    pub fn spawn(model_path: &str, actor_id: InferenceActorId) -> Result<Self> {
        let (req_tx, req_rx) = mpsc::channel::<InferenceRequest>();
        let (res_tx, res_rx) = mpsc::channel::<InferenceResult>();

        let session = Session::builder()?
            .with_intra_threads(2)?
            .with_inter_threads(1)?
            .commit_from_file(model_path)?;

        let join_handle = thread::Builder::new()
            .name(format!("inference-actor-{}", actor_id))
            .spawn(move || {
                run_inference_loop(session, req_rx, res_tx);
            })?;

        Ok(Self {
            sender: Some(req_tx),
            receiver: res_rx,
            join_handle: Some(join_handle),
        })
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
            let tensor = build_curve_predict_tensor(input)?;
            let outputs = session.run(ort::inputs![tensor])?;
            let output = parse_curve_predict_output(&outputs)?;
            Ok(InferenceResultKind::CurvePredict { output })
        }

        InferenceRequestKind::CurveCopilotPredict {
            context,
            property_type_id,
            topology_features,
            bone_name_tokens,
            query_times,
            curve_window,
        } => {
            let tensors = build_curve_copilot_tensors(
                context,
                *property_type_id,
                topology_features,
                bone_name_tokens,
                query_times,
                curve_window,
            )?;

            let outputs = session.run(ort::inputs![
                "context_keyframes" => tensors.context,
                "property_type" => tensors.property_type,
                "topology_features" => tensors.topology,
                "bone_name_tokens" => tensors.bone_name,
                "query_times" => tensors.query_times,
                "curve_window" => tensors.curve_window
            ])?;

            let steps = parse_curve_copilot_output(&outputs, query_times.len())?;

            log!(
                "CurveCopilot raw output: {} steps, query_times={:?}",
                steps.len(),
                query_times
            );

            Ok(InferenceResultKind::CurveCopilotPredict { steps })
        }
    }
}
