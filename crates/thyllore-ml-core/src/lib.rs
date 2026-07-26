pub mod copilot;
#[cfg(feature = "debug-log")]
pub mod debug_log;
pub mod degrade;
pub mod feedback;
pub mod mode;
pub mod model_path;
pub mod sentence_encoder;

#[cfg(feature = "python")]
mod pybindings;

pub use copilot::*;
pub use mode::CurveCopilotMode;

pub use thyllore_model_core::{Bone, BoneId, Skeleton, SkeletonId};
