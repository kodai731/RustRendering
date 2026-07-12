pub mod components;
pub mod copilot;
#[cfg(feature = "debug-log")]
pub mod debug_log;
pub mod degrade;
pub mod feedback;
pub mod license;
pub mod mode;
pub mod model_path;

#[cfg(feature = "python")]
mod pybindings;

pub use components::*;
pub use copilot::*;
pub use mode::CurveCopilotMode;

pub use thyllore_model_core::{Bone, BoneId, Skeleton, SkeletonId};
