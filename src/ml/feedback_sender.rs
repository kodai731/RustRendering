use std::path::Path;
use std::sync::mpsc;
use std::thread;

use thyllore_ml_core::copilot::v2::forecast;
use thyllore_ml_core::copilot::v2::inference::CONTEXT_LENGTH;
use thyllore_ml_core::feedback::{
    build_feedback_record, channel_for_property_type, hash_model_file, send_feedback_batch,
    FeedbackEndpoint, FeedbackRecord, FeedbackRecordInputs, FeedbackResponse,
};
use thyllore_ml_core::unlock::{now_unix, UnlockGate};

use crate::animation::editable::PropertyType;

pub const FEEDBACK_ENDPOINT_ENV: &str = "THYLLORE_FEEDBACK_ENDPOINT";
pub const INGEST_TOKEN_ENV: &str = "THYLLORE_INGEST_TOKEN";

/// Fire-and-forget sender for full-mode feedback records. Thin mpsc wrapper
/// around `thyllore_ml_core::feedback` (the transport SSoT), mirroring
/// `InferenceThreadHandle`.
pub struct FeedbackSenderHandle {
    sender: Option<mpsc::Sender<FeedbackRecord>>,
    join_handle: Option<thread::JoinHandle<()>>,
    model_hash: String,
}

impl FeedbackSenderHandle {
    pub fn spawn_from_env(model_path: &str) -> Option<Self> {
        let Ok(url) = std::env::var(FEEDBACK_ENDPOINT_ENV) else {
            log_warn!(
                "CurveCopilot full mode: {} not set, feedback sending disabled",
                FEEDBACK_ENDPOINT_ENV
            );
            return None;
        };
        let Ok(ingest_token) = std::env::var(INGEST_TOKEN_ENV) else {
            log_warn!(
                "CurveCopilot full mode: {} not set, feedback sending disabled",
                INGEST_TOKEN_ENV
            );
            return None;
        };

        let model_hash = match hash_model_file(Path::new(model_path)) {
            Ok(hash) => hash,
            Err(e) => {
                log_warn!("CurveCopilot feedback: model hash failed: {}", e);
                String::new()
            }
        };

        log!("CurveCopilot feedback sender started (endpoint={})", url);
        let endpoint = FeedbackEndpoint {
            url,
            ingest_token,
            anon_id: format!("thyllore-engine-{}", now_unix()),
        };

        let (sender, receiver) = mpsc::channel::<FeedbackRecord>();
        let join_handle = thread::Builder::new()
            .name("curve-copilot-feedback".to_string())
            .spawn(move || run_sender_loop(endpoint, receiver))
            .ok()?;

        Some(Self {
            sender: Some(sender),
            join_handle: Some(join_handle),
            model_hash,
        })
    }

    pub fn model_hash(&self) -> &str {
        &self.model_hash
    }

    pub fn send(&self, record: FeedbackRecord) {
        if let Some(ref sender) = self.sender {
            let _ = sender.send(record);
        }
    }
}

impl Drop for FeedbackSenderHandle {
    fn drop(&mut self) {
        self.sender.take();

        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

fn run_sender_loop(endpoint: FeedbackEndpoint, receiver: mpsc::Receiver<FeedbackRecord>) {
    let gate = UnlockGate::from_build_env();

    while let Ok(record) = receiver.recv() {
        match send_feedback_batch(&endpoint, &[record]) {
            Ok(response) => log_feedback_response(&gate, &response),
            Err(e) => log_warn!("CurveCopilot feedback: send failed: {}", e),
        }
    }
}

fn log_feedback_response(gate: &UnlockGate, response: &FeedbackResponse) {
    match &response.unlock_token {
        Some(token) if gate.effective_context_length(Some(token), now_unix()) == CONTEXT_LENGTH => {
            log!(
                "CurveCopilot feedback: record accepted, unlock token verified (ctx64 unlocked, exp={:?})",
                response.exp
            );
        }
        Some(_) => log_warn!(
            "CurveCopilot feedback: unlock token rejected by UnlockGate \
             (no baked pubkey, invalid signature, or expired)"
        ),
        None => log_warn!("CurveCopilot feedback: response has no unlock_token"),
    }
}

/// Builds the full-mode record for one engine inference. `ground_truth` stays
/// `null`: a single forecast has no user-corrected curve to sample.
pub fn build_engine_feedback_record(
    model_hash: &str,
    property_type: PropertyType,
    origin_value: f32,
    context: &[f32],
    mean_curve: &[f32],
) -> FeedbackRecord {
    let continuity_offset = forecast::continuity_offset(mean_curve, origin_value);
    let context: Vec<f64> = context.iter().map(|&value| value as f64).collect();
    let prediction: Vec<f64> = mean_curve
        .iter()
        .map(|&value| (value + continuity_offset) as f64)
        .collect();
    let channel = channel_for_property_type(property_type);

    build_feedback_record(FeedbackRecordInputs {
        model_hash,
        channel_kind: &channel.kind,
        array_index: channel.array_index,
        scene_fps: forecast::DEPLOY_FPS as f64,
        deploy_fps: forecast::DEPLOY_FPS as f64,
        frame_step: 1.0,
        origin_value: origin_value as f64,
        context: &context,
        prediction: &prediction,
        ground_truth: None,
        signal: "ignored",
        ts: now_unix(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_record_is_origin_relative_and_anonymous() {
        let context = vec![1.0_f32, 2.0];
        let mean_curve = vec![3.0_f32, 4.0];
        let origin_value = 1.0_f32;

        let record = build_engine_feedback_record(
            "hash",
            PropertyType::TranslationY,
            origin_value,
            &context,
            &mean_curve,
        );

        assert_eq!(record.channel.kind, "location");
        assert_eq!(record.channel.array_index, 1);
        assert_eq!(record.context, vec![0.0, 1.0]);
        assert_eq!(record.ground_truth, None);
        assert_eq!(record.fps.frame_step, 1.0);

        let offset = forecast::continuity_offset(&mean_curve, origin_value);
        let expected: Vec<f64> = mean_curve
            .iter()
            .map(|&v| (v + offset - origin_value) as f64)
            .collect();
        for (actual, expected) in record.prediction.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-9);
        }
    }
}
