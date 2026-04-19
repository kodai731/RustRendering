pub struct TextToMotionRequest {
    pub prompt: String,
    pub duration_seconds: f32,
    pub target_fps: i32,
    pub glb_data: Vec<u8>,
}

#[cfg(feature = "auto-rig")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshInputMode {
    TextOnly,
    Image,
}

#[cfg(feature = "auto-rig")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshModelType {
    Trellis,
    Hunyuan3D,
    CharacterGen,
    StdGen,
    Era3D,
}

#[cfg(feature = "auto-rig")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextToImageModelType {
    ServerDefault,
    Sdxl,
    Animagine,
}

#[cfg(feature = "auto-rig")]
impl MeshModelType {
    pub const IMAGE_VARIANTS: &[MeshModelType] = &[
        MeshModelType::Trellis,
        MeshModelType::Hunyuan3D,
        MeshModelType::CharacterGen,
        MeshModelType::StdGen,
        MeshModelType::Era3D,
    ];

    pub const TEXT_VARIANTS: &[MeshModelType] = &[
        MeshModelType::Trellis,
        MeshModelType::Hunyuan3D,
        MeshModelType::CharacterGen,
        MeshModelType::StdGen,
        MeshModelType::Era3D,
    ];

    pub fn variants_for_mode(mode: &MeshInputMode) -> &'static [MeshModelType] {
        match mode {
            MeshInputMode::TextOnly => Self::TEXT_VARIANTS,
            MeshInputMode::Image => Self::IMAGE_VARIANTS,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            MeshModelType::Trellis => "TRELLIS",
            MeshModelType::Hunyuan3D => "Hunyuan3D",
            MeshModelType::CharacterGen => "CharacterGen",
            MeshModelType::StdGen => "StdGen",
            MeshModelType::Era3D => "Era3D",
        }
    }
}

#[cfg(feature = "auto-rig")]
impl TextToImageModelType {
    pub const ALL_VARIANTS: &[TextToImageModelType] = &[
        TextToImageModelType::ServerDefault,
        TextToImageModelType::Sdxl,
        TextToImageModelType::Animagine,
    ];

    pub fn display_name(&self) -> &'static str {
        match self {
            TextToImageModelType::ServerDefault => "Server Default",
            TextToImageModelType::Sdxl => "SDXL",
            TextToImageModelType::Animagine => "Animagine",
        }
    }
}

#[cfg(feature = "auto-rig")]
pub struct TextToMeshRequest {
    pub prompt: String,
    pub target_faces: u32,
    pub seed: u32,
    pub input_mode: MeshInputMode,
    pub input_image_png: Option<Vec<u8>>,
    pub model_type: MeshModelType,
    pub t2i_model_type: TextToImageModelType,
}

#[cfg(feature = "auto-rig")]
pub struct RiggingRequest {
    pub glb_data: Vec<u8>,
    pub num_sample_points: u32,
}

#[cfg(feature = "auto-rig")]
pub struct SkeletonJointInfo {
    pub name: String,
    pub position: [f32; 3],
    pub tail: [f32; 3],
    pub parent_index: i32,
}

pub enum GrpcRequest {
    GenerateMotion(TextToMotionRequest),
    #[cfg(feature = "auto-rig")]
    GenerateMesh(TextToMeshRequest),
    #[cfg(feature = "auto-rig")]
    GenerateRig(RiggingRequest),
    CheckStatus,
    #[cfg(feature = "auto-rig")]
    CheckMeshStatus,
    #[cfg(feature = "auto-rig")]
    CheckRiggingStatus,
    Shutdown,
}

pub enum GrpcResponse {
    MotionGenerated {
        curves: Vec<RawAnimationCurve>,
        generation_time_ms: f32,
        model_used: String,
    },
    #[cfg(feature = "auto-rig")]
    MeshGenerated {
        glb_data: Vec<u8>,
        vertex_count: u32,
        face_count: u32,
        generation_time_ms: f32,
        intermediate_image_png: Option<Vec<u8>>,
    },
    ServerStatus {
        ready: bool,
        active_model: String,
        gpu_memory_mb: i32,
    },
    #[cfg(feature = "auto-rig")]
    MeshServerStatus {
        ready: bool,
    },
    #[cfg(feature = "auto-rig")]
    RigGenerated {
        rigged_glb_data: Vec<u8>,
        joint_count: u32,
        bone_count: u32,
        generation_time_ms: f32,
        skeleton_joints: Vec<SkeletonJointInfo>,
    },
    #[cfg(feature = "auto-rig")]
    RiggingServerStatus {
        ready: bool,
        model_name: String,
        gpu_memory_mb: i32,
    },
    Error {
        message: String,
    },
}

pub struct RawAnimationCurve {
    pub bone_name: String,
    pub property_type: i32,
    pub keyframes: Vec<RawCurveKeyframe>,
}

pub struct RawCurveKeyframe {
    pub time: f32,
    pub value: f32,
    pub tangent_in_dt: f32,
    pub tangent_in_dv: f32,
    pub tangent_out_dt: f32,
    pub tangent_out_dv: f32,
    pub interpolation: i32,
}
