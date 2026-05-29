use crate::ecs::component::InferenceActorSetup;
use crate::ecs::resource::{ActorRuntime, InferenceActorState};
use crate::ml::{
    InferenceActorId, InferenceRequest, InferenceRequestId, InferenceRequestKind, InferenceResult,
    InferenceThreadHandle,
};

pub fn inference_actor_initialize(setup: &InferenceActorSetup, state: &mut InferenceActorState) {
    if !setup.enabled {
        return;
    }

    if state.actors.contains_key(&setup.actor_id) {
        return;
    }

    let handle = match InferenceThreadHandle::spawn(
        &setup.model_path,
        setup.actor_id,
        setup.model_kind.clone(),
    ) {
        Ok(h) => h,
        Err(e) => {
            log!(
                "Failed to spawn inference actor {}: {:?}",
                setup.actor_id,
                e
            );
            return;
        }
    };

    state.actors.insert(
        setup.actor_id,
        ActorRuntime {
            thread_handle: handle,
            enabled: true,
        },
    );

    log!("Initialized inference actor {}", setup.actor_id);
}

pub fn inference_actor_poll(state: &mut InferenceActorState) {
    let actor_ids: Vec<InferenceActorId> = state.actors.keys().copied().collect();

    for actor_id in actor_ids {
        if let Some(runtime) = state.actors.get(&actor_id) {
            while let Some(result) = runtime.thread_handle.try_recv() {
                state.pending_results.push(result);
            }
        }
    }
}

pub fn inference_actor_submit(
    state: &mut InferenceActorState,
    actor_id: InferenceActorId,
    kind: InferenceRequestKind,
) -> Option<InferenceRequestId> {
    let request_id = state.next_request_id;
    state.next_request_id += 1;

    let runtime = state.actors.get(&actor_id)?;

    if !runtime.enabled {
        return None;
    }

    let request = InferenceRequest {
        request_id,
        actor_id,
        kind,
    };

    match runtime.thread_handle.send(request) {
        Ok(()) => Some(request_id),
        Err(e) => {
            log!(
                "Failed to send inference request to actor {}: {:?}",
                actor_id,
                e
            );
            None
        }
    }
}

pub fn inference_actor_take_results(state: &mut InferenceActorState) -> Vec<InferenceResult> {
    std::mem::take(&mut state.pending_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::InferenceModelKind;

    const DUMMY_MODEL_PATH: &str = "assets/ml/dummy_curve_predictor.onnx";

    /// Check if ONNX Runtime library is available on this platform.
    /// ORT_DYLIB_PATH is set in .cargo/config.toml; if the library file
    /// does not exist (e.g. Windows DLL on Linux), ORT initialization will
    /// panic and poison global state, so we skip those tests.
    fn is_ort_available() -> bool {
        let ort_path = match std::env::var("ORT_DYLIB_PATH") {
            Ok(p) if !p.is_empty() => p,
            _ => return false,
        };
        std::path::Path::new(&ort_path).exists()
    }

    fn create_test_setup(actor_id: InferenceActorId, enabled: bool) -> InferenceActorSetup {
        InferenceActorSetup {
            actor_id,
            model_path: DUMMY_MODEL_PATH.to_string(),
            model_kind: InferenceModelKind::CurvePredictor,
            enabled,
        }
    }

    #[test]
    fn test_actor_roundtrip() {
        if !is_ort_available() {
            return;
        }

        let mut state = InferenceActorState::default();
        let setup = create_test_setup(1, true);

        inference_actor_initialize(&setup, &mut state);
        assert!(state.actors.contains_key(&1));

        let request_id = inference_actor_submit(
            &mut state,
            1,
            InferenceRequestKind::CurvePredict {
                input: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            },
        );
        assert!(request_id.is_some());

        std::thread::sleep(std::time::Duration::from_millis(500));
        inference_actor_poll(&mut state);

        let results = inference_actor_take_results(&mut state);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, request_id.unwrap());
        assert_eq!(results[0].actor_id, 1);

        match &results[0].kind {
            crate::ml::InferenceResultKind::CurvePredict { output } => {
                assert_eq!(output.len(), 3);
            }
            _ => panic!("Unexpected result kind"),
        }
    }

    #[test]
    fn test_actor_disabled_without_model() {
        if !is_ort_available() {
            return;
        }

        let mut state = InferenceActorState::default();
        let setup = InferenceActorSetup {
            actor_id: 10,
            model_path: "nonexistent_model.onnx".to_string(),
            model_kind: InferenceModelKind::CurvePredictor,
            enabled: true,
        };

        inference_actor_initialize(&setup, &mut state);
        assert!(!state.actors.contains_key(&10));
    }

    #[test]
    fn test_actor_disabled_flag() {
        let mut state = InferenceActorState::default();
        let setup = create_test_setup(20, false);

        inference_actor_initialize(&setup, &mut state);
        assert!(!state.actors.contains_key(&20));
    }

    #[test]
    fn test_multiple_actors() {
        if !is_ort_available() {
            return;
        }

        let mut state = InferenceActorState::default();

        let setup_a = create_test_setup(100, true);
        let setup_b = create_test_setup(200, true);

        inference_actor_initialize(&setup_a, &mut state);
        inference_actor_initialize(&setup_b, &mut state);
        assert!(state.actors.contains_key(&100));
        assert!(state.actors.contains_key(&200));

        inference_actor_submit(
            &mut state,
            100,
            InferenceRequestKind::CurvePredict {
                input: vec![1.0, 0.0, 0.0, 0.0, 0.0],
            },
        );
        inference_actor_submit(
            &mut state,
            200,
            InferenceRequestKind::CurvePredict {
                input: vec![0.0, 0.0, 0.0, 0.0, 1.0],
            },
        );

        std::thread::sleep(std::time::Duration::from_millis(500));
        inference_actor_poll(&mut state);

        let results = inference_actor_take_results(&mut state);
        assert_eq!(results.len(), 2);

        let actor_ids: Vec<_> = results.iter().map(|r| r.actor_id).collect();
        assert!(actor_ids.contains(&100));
        assert!(actor_ids.contains(&200));
    }

    #[test]
    fn test_actor_graceful_shutdown() {
        if !is_ort_available() {
            return;
        }

        let mut state = InferenceActorState::default();
        let setup = create_test_setup(300, true);

        inference_actor_initialize(&setup, &mut state);
        assert!(state.actors.contains_key(&300));

        state.actors.remove(&300);
        assert!(!state.actors.contains_key(&300));
    }

    #[test]
    fn test_actor_inference_latency() {
        if !is_ort_available() {
            return;
        }

        let mut state = InferenceActorState::default();
        let setup = create_test_setup(400, true);

        inference_actor_initialize(&setup, &mut state);

        let iterations = 100;
        let start = std::time::Instant::now();

        for i in 0..iterations {
            inference_actor_submit(
                &mut state,
                400,
                InferenceRequestKind::CurvePredict {
                    input: vec![i as f32, 0.0, 0.0, 0.0, 0.0],
                },
            );
        }

        let mut received = 0;
        let timeout = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while received < iterations && std::time::Instant::now() < timeout {
            inference_actor_poll(&mut state);
            let results = inference_actor_take_results(&mut state);
            received += results.len();

            if received < iterations {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

        assert_eq!(received, iterations);
        assert!(
            avg_ms < 16.0,
            "Average inference latency {:.2}ms exceeds 16ms",
            avg_ms
        );
    }
}
