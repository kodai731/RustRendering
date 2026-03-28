pub struct TextToMotionRequest {
    pub prompt: String,
    pub duration_seconds: f32,
    pub target_fps: i32,
}

#[cfg(feature = "text-to-mesh")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshInputMode {
    TextOnly,
    Image,
}

#[cfg(feature = "text-to-mesh")]
pub struct TextToMeshRequest {
    pub prompt: String,
    pub target_faces: u32,
    pub seed: u32,
    pub input_mode: MeshInputMode,
    pub input_image_png: Option<Vec<u8>>,
}

pub enum GrpcRequest {
    GenerateMotion(TextToMotionRequest),
    #[cfg(feature = "text-to-mesh")]
    GenerateMesh(TextToMeshRequest),
    CheckStatus,
    #[cfg(feature = "text-to-mesh")]
    CheckMeshStatus,
    Shutdown,
}

pub enum GrpcResponse {
    MotionGenerated {
        curves: Vec<RawAnimationCurve>,
        generation_time_ms: f32,
        model_used: String,
    },
    #[cfg(feature = "text-to-mesh")]
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
    #[cfg(feature = "text-to-mesh")]
    MeshServerStatus {
        ready: bool,
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
