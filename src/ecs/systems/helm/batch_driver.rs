//! Batch driver systems for helm regression testing.
//!
//! `run_before_helm` injects the next batch row's utterance into `HelmState.submitted_utterance`
//! before the helm phase runs. `run_after_helm` records results after dispatch and exits when
//! all rows are done.

use std::io::Write;

use crate::ecs::context::EcsContext;
use crate::ecs::resource::{CommandFeedback, HelmBatchState, HelmState};
use crate::ecs::UIEventQueue;

/// Read RSS from /proc/self/status (VmRSS in kB). Returns 0 on error.
fn read_rss_kb() -> usize {
    let contents = match std::fs::read_to_string("/proc/self/status") {
        Ok(c) => c,
        Err(_) => return 0,
    };
    for line in contents.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            let parts: Vec<&str> = value.split_whitespace().collect();
            if let Some(v) = parts.first() {
                if let Ok(kb) = v.parse::<usize>() {
                    return kb;
                }
            }
        }
    }
    0
}

/// Run before the helm phase: inject the next batch row's utterance.
pub fn run_before_helm(ctx: &mut EcsContext) {
    if !ctx.world.contains_resource::<HelmBatchState>() {
        return;
    }

    let has_submitted = {
        let state = ctx.world.resource::<HelmState>();
        state.submitted_utterance.is_some()
    };
    if has_submitted {
        return;
    }

    // If there's a pending confirmation, wait for it to resolve — do not inject.
    {
        let state = ctx.world.resource::<HelmState>();
        if state.pending.is_some() {
            return;
        }
    }

    let mut batch = ctx.world.resource_mut::<HelmBatchState>();

    if !batch.has_next() {
        return;
    }

    // Record UI event queue length before injection.
    {
        let ui_events = ctx.world.resource::<UIEventQueue>();
        batch.ui_events_before = ui_events.len();
    }

    let utterance = batch.peek_next().unwrap().utterance.clone();
    batch.injected_last_frame = true;

    let mut state = ctx.world.resource_mut::<HelmState>();
    state.submitted_utterance = Some(utterance);
}

/// Run after the helm phase: record results and check for completion.
pub fn run_after_helm(ctx: &mut EcsContext) {
    if !ctx.world.contains_resource::<HelmBatchState>() {
        return;
    }

    if finish_if_draining(ctx) {
        return;
    }

    let mut batch = ctx.world.resource_mut::<HelmBatchState>();
    if !batch.injected_last_frame {
        return;
    }

    // Get feedback, last_routed_tool, submitted_utterance, last_route_latency_ms, last_runtime_load_ms, and pending from HelmState.
    let (
        feedback,
        last_routed_tool,
        submitted_utterance,
        last_route_latency_ms,
        last_runtime_load_ms,
        has_pending,
    ): (
        Option<CommandFeedback>,
        Option<String>,
        Option<String>,
        Option<f32>,
        Option<f32>,
        bool,
    ) = {
        let mut state = ctx.world.resource_mut::<HelmState>();
        (
            state.feedback.take(),
            state.last_routed_tool.take(),
            state.submitted_utterance.take(),
            state.last_route_latency_ms.take(),
            state.last_runtime_load_ms.take(),
            state.pending.is_some(),
        )
    };

    // (1) Lazy-load delay: if feedback is None and submitted_utterance is still Some
    //     (deferred for next frame due to lazy-loading), do not record anything.
    if feedback.is_none() && submitted_utterance.is_some() {
        let mut state = ctx.world.resource_mut::<HelmState>();
        state.submitted_utterance = submitted_utterance;
        state.last_routed_tool = last_routed_tool;
        state.last_route_latency_ms = last_route_latency_ms;
        state.last_runtime_load_ms = last_runtime_load_ms;
        batch.injected_last_frame = true;
        return;
    }

    // (2) Confirm pending: if feedback is None and there's a pending confirmation,
    if feedback.is_none() && has_pending {
        let mut state = ctx.world.resource_mut::<HelmState>();
        state.confirm_response = Some(true);
        state.last_routed_tool = last_routed_tool;
        state.last_route_latency_ms = last_route_latency_ms;
        state.last_runtime_load_ms = last_runtime_load_ms;
        batch.injected_last_frame = true;
        return;
    }

    batch.injected_last_frame = false;

    // Compute UI events diff from stored before count.
    let ui_events_before = batch.ui_events_before;
    let ui_events_after = {
        let ui_events = ctx.world.resource::<UIEventQueue>();
        ui_events.len()
    };
    let ui_events_diff = ui_events_after - ui_events_before;

    let row = batch.peek_next().unwrap();

    // Determine decision tag.
    let decision = match &feedback {
        Some(CommandFeedback::Executed(_)) => "executed",
        Some(CommandFeedback::Report(_)) => "report",
        Some(CommandFeedback::DispatchError(_)) => "dispatch_error",
        Some(CommandFeedback::Unavailable(_)) => "unavailable",
        Some(CommandFeedback::Router(f)) => match f {
            crate::helm::systems::resolution::HelmFeedback::Rejected { .. } => "rejected",
            crate::helm::systems::resolution::HelmFeedback::ClarifyOptions(_) => "clarify",
            crate::helm::systems::resolution::HelmFeedback::MissingObjectName { .. } => {
                "missing_object"
            }
            crate::helm::systems::resolution::HelmFeedback::AmbiguousObjectName { .. } => {
                "ambiguous_object"
            }
            crate::helm::systems::resolution::HelmFeedback::NoCandidate => "no_candidate",
        },
        None => "none",
    };

    // tool_match: true if last_routed_tool's part before ':' matches expected_tool.
    // (3) Clarify as match: if feedback is ClarifyOptions, check first candidate's route id.
    // (4) Executed(name): if feedback is Executed(name), check name's part before ':' matches.
    let tool_match = match (&row.expected_tool, &last_routed_tool, &feedback) {
        (Some(expected), Some(routed), _) => {
            let actual_tool = routed.split(':').next().unwrap_or(routed.as_str());
            actual_tool == expected.as_str()
        }
        (Some(expected), None, Some(CommandFeedback::Executed(name))) => {
            let actual_tool = name.split(':').next().unwrap_or(name.as_str());
            actual_tool == expected.as_str()
        }
        (
            Some(expected),
            None,
            Some(CommandFeedback::Router(
                crate::helm::systems::resolution::HelmFeedback::ClarifyOptions(candidates),
            )),
        ) if !candidates.is_empty() => {
            let first_route = &candidates[0].0;
            let route_id = first_route.id();
            let actual_tool = route_id.split(':').next().unwrap_or(route_id.as_str());
            actual_tool == expected.as_str()
        }
        _ => false,
    };

    let result = serde_json::json!({
        "utterance": row.utterance,
        "decision": decision,
        "feedback": format!("{:?}", feedback),
        "ui_events": ui_events_diff,
        "expected_tool": row.expected_tool,
        "tool_match": tool_match,
        "last_routed_tool": last_routed_tool,
        "latency_ms": last_route_latency_ms,
        "lazy_load_ms": last_runtime_load_ms,
    });

    batch.results.push(result);
    batch.advance();

    // Check if all rows are done.
    if batch.is_done() {
        let matches: usize = batch
            .results
            .iter()
            .filter(|r| {
                r.get("tool_match")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            })
            .count();
        let total = batch.results.len();

        // Count breakdown of tool_match by decision type.
        let accept_matches: usize = batch
            .results
            .iter()
            .filter(|r| {
                r.get("tool_match")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                    && r.get("decision").and_then(|v| v.as_str()) == Some("executed")
            })
            .count();
        let clarify_matches: usize = batch
            .results
            .iter()
            .filter(|r| {
                r.get("tool_match")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                    && r.get("decision").and_then(|v| v.as_str()) == Some("clarify")
            })
            .count();

        // Collect latency_ms values for percentile calculation.
        let mut latencies: Vec<f32> = batch
            .results
            .iter()
            .filter_map(|r| {
                r.get("latency_ms")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32)
            })
            .collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let latency_p50 = percentile(&latencies, 0.5);
        let latency_p95 = percentile(&latencies, 0.95);
        let latency_max = latencies.last().copied().unwrap_or(0.0);

        // Collect lazy_load_ms values (from last_runtime_load_ms).
        let lazy_load_ms: f32 = batch
            .results
            .iter()
            .filter_map(|r| {
                r.get("lazy_load_ms")
                    .and_then(|v| v.as_f64())
                    .map(|f| f as f32)
            })
            .sum();

        // Read RSS at end.
        let rss_end_kb = read_rss_kb();
        let rss_start_kb = batch.rss_start_kb;

        // Write results to output file.
        let out_dir = batch.out.parent();
        if let Some(dir) = out_dir {
            let _ = std::fs::create_dir_all(dir);
        }
        let mut file = std::fs::File::create(&batch.out).expect("failed to create output file");
        for result in &batch.results {
            writeln!(file, "{}", serde_json::to_string(result).unwrap())
                .expect("failed to write result");
        }

        // Write bench summary JSONL.
        let bench_dir = std::path::PathBuf::from("log/helm_bench");
        let _ = std::fs::create_dir_all(&bench_dir);
        let unix_seconds = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let bench_path = bench_dir.join(format!("bench_{}.jsonl", unix_seconds));

        let bench_line = serde_json::json!({
            "rows": total,
            "tool_match": matches,
            "accept": accept_matches,
            "clarify": clarify_matches,
            "latency_ms_p50": latency_p50,
            "latency_ms_p95": latency_p95,
            "latency_ms_max": latency_max,
            "lazy_load_ms": lazy_load_ms,
            "rss_start_mb": rss_start_kb as f64 / 1024.0,
            "rss_end_mb": rss_end_kb as f64 / 1024.0,
        });

        let mut bench_file =
            std::fs::File::create(&bench_path).expect("failed to create bench file");
        writeln!(
            bench_file,
            "{}",
            serde_json::to_string(&bench_line).unwrap()
        )
        .expect("failed to write bench result");

        let mismatch = total - matches;
        println!(
            "[helm-batch] done: tool_match {}/{} (accept={}, clarify={}) -> {}",
            matches,
            total,
            accept_matches,
            clarify_matches,
            batch.out.display()
        );
        println!("[helm-batch] bench: {:?}", bench_line);

        batch.exit_code = if mismatch == 0 { 0 } else { 1 };
        batch.drain_frames_left = Some(DRAIN_FRAMES_AFTER_LAST_ROW);
    }
}

const DRAIN_FRAMES_AFTER_LAST_ROW: u32 = 3;

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    let position = args.iter().position(|arg| arg == flag)?;
    args.get(position + 1)
}

/// Counts down the drain frames after the last row; on zero writes the optional
/// `--batch-anim-dump` and exits. Returns true while draining.
fn finish_if_draining(ctx: &mut EcsContext) -> bool {
    let (remaining, exit_code) = {
        let batch = ctx.world.resource::<HelmBatchState>();
        match batch.drain_frames_left {
            Some(n) => (n, batch.exit_code),
            None => return false,
        }
    };

    if remaining > 0 {
        ctx.world.resource_mut::<HelmBatchState>().drain_frames_left = Some(remaining - 1);
        return true;
    }

    let args: Vec<String> = std::env::args().collect();
    if let Some(path) = flag_value(&args, "--batch-anim-dump") {
        if let Err(e) = crate::ecs::systems::batch_anim_dump_write(ctx.world, path) {
            eprintln!("[helm-batch] anim dump failed: {e}");
            std::process::exit(1);
        }
        println!("[helm-batch] anim dump -> {path}");
    }
    if let Some(path) = flag_value(&args, "--batch-export-camera") {
        if let Err(e) =
            crate::ecs::systems::export_active_camera_gltf(ctx.world, std::path::Path::new(path))
        {
            eprintln!("[helm-batch] camera export failed: {e}");
            std::process::exit(1);
        }
        println!("[helm-batch] camera export -> {path}");
    }
    std::process::exit(exit_code);
}

/// Calculate the p-th percentile of a sorted slice. Returns 0.0 if empty.
fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
