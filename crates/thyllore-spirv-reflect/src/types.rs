use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderStage {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    Compute,
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
    Task,
    Mesh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescriptorKind {
    Sampler,
    CombinedImageSampler,
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    InputAttachment,
    AccelerationStructure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorCount {
    Fixed(u32),
    Unbounded,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedMember {
    pub name: String,
    pub offset: u32,
    pub size: u32,
    pub type_name: String,
    pub members: Vec<ReflectedMember>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedBlock {
    pub type_name: String,
    pub size: u32,
    pub members: Vec<ReflectedMember>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReflectedBinding {
    pub set: u32,
    pub binding: u32,
    pub name: String,
    pub kind: DescriptorKind,
    pub count: DescriptorCount,
    pub block: Option<ReflectedBlock>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShaderBinding {
    pub set: u32,
    pub binding: u32,
    pub kind: DescriptorKind,
    pub count: DescriptorCount,
}

impl ReflectedBinding {
    pub fn shader_binding(&self) -> ShaderBinding {
        ShaderBinding {
            set: self.set,
            binding: self.binding,
            kind: self.kind,
            count: self.count,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderReflection {
    pub stages: Vec<ShaderStage>,
    pub bindings: Vec<ReflectedBinding>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReflectError {
    #[error("SPIR-V magic number mismatch")]
    InvalidMagic,
    #[error("SPIR-V word stream is truncated")]
    Truncated,
    #[error("SPIR-V version {major}.{minor} is newer than the supported 1.0-1.6")]
    UnsupportedVersion { major: u32, minor: u32 },
    #[error("unknown SPIR-V execution model {0}")]
    UnknownExecutionModel(u32),
    #[error("SPIR-V type id {0} is not defined")]
    MissingType(u32),
    #[error("descriptor variable `{0}` has no Binding decoration")]
    MissingBinding(String),
    #[error("descriptor variable `{0}` has an unsupported type")]
    UnsupportedVariableType(String),
    #[error("block member of `{0}` has no Offset decoration")]
    MissingMemberOffset(String),
    #[error("array type id {0} has a non-constant length")]
    NonConstantArrayLength(u32),
    #[error("set {set} binding {binding} is declared with conflicting types across shader stages")]
    ConflictingBinding { set: u32, binding: u32 },
}
