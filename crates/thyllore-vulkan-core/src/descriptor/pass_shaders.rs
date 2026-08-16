pub const FRAME_SET: u32 = 0;
pub const MATERIAL_SET: u32 = 1;
pub const OBJECT_SET: u32 = 2;
pub const FLAME_DESCRIPTOR_SET: u32 = 1;

pub const MODEL_SHADERS: [&str; 2] = ["assets/shaders/vert.spv", "assets/shaders/frag.spv"];
pub const GBUFFER_SHADERS: [&str; 2] = [
    "assets/shaders/gbufferVert.spv",
    "assets/shaders/gbufferFrag.spv",
];
pub const GRID_SHADERS: [&str; 2] = ["assets/shaders/gridVert.spv", "assets/shaders/gridFrag.spv"];
pub const GIZMO_SHADERS: [&str; 2] = [
    "assets/shaders/gizmoVert.spv",
    "assets/shaders/gizmoFrag.spv",
];
pub const BONE_SHADERS: [&str; 2] = ["assets/shaders/boneVert.spv", "assets/shaders/boneFrag.spv"];
pub const ONION_SKIN_GHOST_SHADERS: [&str; 2] = [
    "assets/shaders/gbufferVert.spv",
    "assets/shaders/onionSkinFrag.spv",
];
pub const ONION_SKIN_COMPOSITE_SHADERS: [&str; 2] = [
    "assets/shaders/tonemapVert.spv",
    "assets/shaders/onionSkinCompositeFrag.spv",
];
pub const FLAME_RESOLVE_SHADERS: [&str; 2] = [
    "assets/shaders/tonemapVert.spv",
    "assets/shaders/flameResolveFrag.spv",
];
pub const TONEMAP_SHADERS: [&str; 2] = [
    "assets/shaders/tonemapVert.spv",
    "assets/shaders/tonemapFrag.spv",
];
pub const BLOOM_DOWNSAMPLE_SHADERS: [&str; 2] = [
    "assets/shaders/tonemapVert.spv",
    "assets/shaders/bloomDownsampleFrag.spv",
];
pub const BLOOM_UPSAMPLE_SHADERS: [&str; 2] = [
    "assets/shaders/tonemapVert.spv",
    "assets/shaders/bloomUpsampleFrag.spv",
];
pub const DOF_SHADERS: [&str; 2] = [
    "assets/shaders/tonemapVert.spv",
    "assets/shaders/dofFrag.spv",
];
pub const AUTO_EXPOSURE_HISTOGRAM_SHADER: &str = "assets/shaders/autoExposureHistogram.spv";
pub const AUTO_EXPOSURE_AVERAGE_SHADER: &str = "assets/shaders/autoExposureAverage.spv";
pub const RAY_QUERY_SHADOW_SHADER: &str = "assets/shaders/rayQueryShadow.spv";
pub const COMPOSITE_SHADERS: [&str; 2] = [
    "assets/shaders/compositeVert.spv",
    "assets/shaders/compositeFrag.spv",
];
pub const BILLBOARD_SHADERS: [&str; 2] = [
    "assets/shaders/billboardVert.spv",
    "assets/shaders/billboardFrag.spv",
];

pub fn standard_graphics_shaders() -> Vec<&'static str> {
    [
        MODEL_SHADERS,
        GBUFFER_SHADERS,
        GRID_SHADERS,
        GIZMO_SHADERS,
        BONE_SHADERS,
        ONION_SKIN_GHOST_SHADERS,
    ]
    .concat()
}

pub fn frame_set_shaders() -> Vec<&'static str> {
    [standard_graphics_shaders(), FLAME_RESOLVE_SHADERS.to_vec()].concat()
}

pub fn bloom_shaders() -> Vec<&'static str> {
    [BLOOM_DOWNSAMPLE_SHADERS, BLOOM_UPSAMPLE_SHADERS].concat()
}
