//! Debug-only mirrors of render shaders. This crate is a workspace member for
//! its test suite alone; the product (thyllore-animation / render core) must
//! never depend on it.

pub mod dump_effect;
pub mod flame_fbm_mirror;
pub mod flame_wave_mirror;
pub mod fringe_field;
