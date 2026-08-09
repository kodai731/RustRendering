pub mod backend;
mod billboard;
mod buffer_handle;
mod flame;
pub mod flame_fit;
mod flame_pick;
pub mod flame_plume;
mod flame_presets;
mod flame_radial;
pub mod flame_sdf;
mod flame_shell;
mod flame_texture_fit;
pub mod flame_trail;
mod flame_wall_probe;
pub mod flame_wave;
mod gizmo;
mod gizmo_data;
mod mesh;
mod projection;
mod render_data;
mod settings;
mod ubo;

pub use backend::RenderBackend;
pub use billboard::{BillboardMesh, BillboardTransform, BillboardVertex};
pub use buffer_handle::{BufferHandle, IndexBufferHandle, VertexBufferHandle};
pub use flame::{
    advance_flame_time, build_flame_inverse_model_matrix, build_flame_model_matrix,
    build_flame_ubo, build_flame_ubo_with_trail, build_warp_strain_params,
    default_height_falloff, default_radial_falloff,
    fit_flame_coefficients, flame_bounding_radius, integrate_emission_segment, profile_from_effect,
    read_env_warp_form_displacement,
    refresh_flame_coefficients, FlameCoefficients, FlameDebugView, FlameEffect, FlameProfile,
    FlameRenderSettings, FlameShadingMode, FlameUBO, HEIGHT_PRIMITIVE_COEFFICIENT_COUNT,
    RADIAL_COEFFICIENT_COUNT,
};
pub use flame_pick::{
    flame_bend_offset, flame_local_bounds, flame_local_bounds_corners, flame_support_scale,
    intersect_flame_bounds, intersect_flame_proxy, FlameLocalBounds,
};
pub use flame_presets::{apply_flame_preset, FLAME_PRESET_NAMES};
pub use flame_radial::{
    build_height_series, eroded_argument, envelope_fade, erosion_remap_scale,
    evaluate_gaussian_moments, evaluate_radial_density_factor, flame_radial_radius_scale,
    flame_radial_support_radius, ring_support_span, FlameRadialTaper,
};
pub use flame_shell::{
    flame_shell_outer_radius, flame_shell_radius_scale, flame_shell_support_scale,
    FLAME_SHELL_BASE_RADIUS, FLAME_SHELL_TAPER_TIP_SCALE,
};
pub use flame_texture_fit::*;
pub use flame_trail::*;
pub use flame_wall_probe::{
    probe_flame_wall, WallProbeRay, WallProbeReport, WallProbeView, WALL_PROBE_GRID_COLS,
    WALL_PROBE_GRID_ROWS,
};
pub use gizmo::{
    BoneDisplayStyle, ColorVertex, GizmoAxis, GizmoDraggable, GizmoPosition, GizmoSelectable,
    TransformGizmoHandle,
};
pub use gizmo_data::{BoneGizmoData, ConstraintGizmoData, GridMeshData, LightGizmoData};
pub use mesh::{DynamicMesh, GpuMeshRef, LineMesh, MeshScale, RenderInfo, FRAMES_IN_FLIGHT};
pub use projection::{DistanceAttenuation, ProjectionData};
pub use render_data::{MeshHandle, ObjectIndex, RenderData, SkeletonHandle};
pub use settings::{
    AutoExposure, BloomSettings, DepthOfField, Exposure, LensEffects, PhysicalCameraParameters,
    ToneMapOperator, ToneMapping,
};
pub use ubo::{FrameUBO, MaterialUBO, ObjectUBO};

pub type MeshId = usize;
pub type PipelineId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferMemoryType {
    DeviceLocal,
    HostVisible,
}
