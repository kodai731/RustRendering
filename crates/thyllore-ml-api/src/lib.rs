#![doc = include_str!("../README.md")]

mod error;
mod request;
mod response;
mod traits;
mod wire;

pub use error::MlError;
pub use request::{CopilotRequest, CurvePredictRequest, SkeletonSnapshot};
pub use response::{CopilotResponse, CopilotStepPrediction, CurvePredictResponse, TopologyResult};
pub use traits::MlOps;
pub use wire::{
    CopilotRequestWire, CopilotResponseWire, CopilotStepWire, OP_RUN_CURVE_COPILOT,
};

/// ABI/schema marker bumped whenever this crate's public surface changes
/// in a way that requires the Blender extension to be reinstalled.
///
/// The L3 PyO3 wheel exports this through `__abi_marker__`, and the L4 addon
/// (`blender_addon/__init__.py`) compares it against `EXPECTED_ABI_MARKER`
/// at register time. A mismatch causes the addon to register a placeholder
/// panel instead of the regular operators, so users see a clear "update the
/// addon" message rather than crashes.
///
/// Bump rules:
/// - Adding a new variant to [`MlError`]: no bump (clients can match `_`).
/// - Adding a method to [`MlOps`] with a default implementation: no bump.
/// - Adding a field to a request/response struct: requires bump unless the
///   struct uses `#[serde(default)]` for the new field.
/// - Removing or renaming a method, field, or error variant: bump.
pub const ABI_MARKER: u32 = 1;
