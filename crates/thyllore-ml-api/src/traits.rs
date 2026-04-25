use crate::error::MlError;
use crate::request::{CopilotRequest, CurvePredictRequest, SkeletonSnapshot};
use crate::response::{CopilotResponse, CurvePredictResponse, TopologyResult};

/// Stable contract between [`thyllore_ml_core::MlCoreImpl`] (L1 implementation)
/// and the PyO3 wheel (L3 boundary adapter).
///
/// Trait methods are split into three groups:
///
/// 1. **Typed pipelines** (`run_curve_copilot`, `run_curve_predict`,
///    `compute_topology`) — high-level entries used by the corresponding
///    Blender operators. Their signatures are part of the L2 contract; adding
///    parameters or fields requires an `ABI_MARKER` bump.
/// 2. **Generic dispatch** (`call_op`) — opaque bytes-in / bytes-out channel
///    used to ship new features to the addon without rebuilding the wheel.
///    Payloads are bincode-serialized requests defined elsewhere; payload
///    schemas can evolve without touching this trait.
/// 3. **Discovery** (`capabilities`) — list of feature ids the implementation
///    actually supports. Operators check `capabilities()` in their `poll`
///    method and gray themselves out when their feature is missing.
///
/// Implementations should be cheap to call repeatedly: callers may hold a
/// `&dyn MlOps` for the entire Blender session.
pub trait MlOps: Send + Sync {
    fn run_curve_copilot(&self, request: CopilotRequest) -> Result<CopilotResponse, MlError>;

    fn run_curve_predict(
        &self,
        request: CurvePredictRequest,
    ) -> Result<CurvePredictResponse, MlError>;

    fn compute_topology(&self, skeleton: &SkeletonSnapshot)
        -> Result<TopologyResult, MlError>;

    fn call_op(&self, op_name: &str, payload: &[u8]) -> Result<Vec<u8>, MlError>;

    fn capabilities(&self) -> Vec<&'static str>;
}
