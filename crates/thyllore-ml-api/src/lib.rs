#![doc = include_str!("../README.md")]

/// ABI/schema marker bumped whenever the L3 PyO3 wheel's public surface changes
/// in a way that requires the Blender extension to be reinstalled.
///
/// The wheel exports this through `__abi_marker__`, and the L4 addon
/// (`blender_addon/__init__.py`) compares it against `EXPECTED_ABI_MARKER` at
/// register time. A mismatch makes the addon register a placeholder panel
/// instead of the operators, so users see a clear "update the addon" message
/// rather than crashes.
///
/// History:
/// - 1: rawfuture forecast surface — `PyRawFutureSession` (`from_onnx_path`,
///   `predict_mean_curve`, `build_forecast_preview`), `forecast_sample_offsets`,
///   `resolve_origin_frame`, `resolve_curve_copilot_model_path`, `capabilities`.
///   The pre-rawfuture multi-input copilot / curve-predict surface was removed.
pub const ABI_MARKER: u32 = 1;
