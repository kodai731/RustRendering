//! Debug-only mirrors of render shaders. The renderers (render core / Vulkan
//! backend) must never depend on this crate; the app's debug dump systems may
//! use it to write full numerical traces (flame_field_trace) — offline
//! analysis tooling, never part of a render path.

pub mod dump_effect;
pub mod flame_fbm_mirror;
pub mod flame_field_trace;
pub mod flame_wave_mirror;
pub mod fringe_field;
