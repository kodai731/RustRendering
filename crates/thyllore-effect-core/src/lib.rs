pub mod debug;
pub mod flame;

pub use thyllore_texture_fit_core as flame_fit;

pub use thyllore_scene_core::{find_scalar_param, find_ui_param, ScalarParam, UiParam};

pub use debug::flame_wall_probe::{
    probe_flame_wall, WallProbeRay, WallProbeReport, WallProbeView, WALL_PROBE_GRID_COLS,
    WALL_PROBE_GRID_ROWS,
};
pub use flame::*;

pub use flame::analytic::field_manifest as flame_field_manifest;
pub use flame::analytic::pick as flame_pick;
pub use flame::analytic::radial as flame_radial;
pub use flame::analytic::shell as flame_shell;
pub use flame::analytic::wave as flame_wave;
pub use flame::bake::sdf as flame_sdf;
pub use flame::bake::texture_fit as flame_texture_fit;
pub use flame::plume as flame_plume;
pub use flame::trail as flame_trail;
