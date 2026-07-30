pub mod backend;
mod billboard;
mod buffer_handle;
mod flame;
pub mod flame_fit;
mod flame_presets;
mod flame_radial;
pub mod flame_sdf;
mod flame_shell;
pub mod flame_trail;
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
    build_flame_ubo, build_flame_ubo_with_trail, default_height_falloff, default_radial_falloff,
    fit_flame_coefficients, integrate_emission_segment, profile_from_effect,
    refresh_flame_coefficients, FlameCoefficients, FlameEffect, FlameProfile, FlameRenderSettings,
    FlameShadingMode, FlameUBO, HEIGHT_PRIMITIVE_COEFFICIENT_COUNT, RADIAL_COEFFICIENT_COUNT,
};
pub use flame_presets::{apply_flame_preset, FLAME_PRESET_NAMES};
pub use flame_radial::{
    evaluate_gaussian_moments, evaluate_radial_density_factor, flame_radial_gaussian_scale,
    integrate_radial_emission, FlameRadialTaper, FLAME_RADIAL_BAND_COUNT,
};
pub use flame_shell::{
    flame_shell_outer_radius, flame_shell_radius_scale, generate_flame_shell_triangles,
    FLAME_SHELL_BASE_RADIUS, FLAME_SHELL_RING_SEGMENTS, FLAME_SHELL_STACKS,
    FLAME_SHELL_TAPER_TIP_SCALE,
};
pub use flame_trail::*;
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
