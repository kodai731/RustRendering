mod glsl;
mod parser;
mod types;
mod verify;

pub use glsl::{scan_glsl_descriptor_declarations, GlslDescriptorClass, GlslDescriptorDeclaration};
pub use parser::{reflect_shader_bytes, reflect_shader_words};
pub use types::{
    DescriptorCount, DescriptorKind, ReflectError, ReflectedBinding, ShaderReflection, ShaderStage,
};
pub use verify::{verify_spirv_against_glsl, DeclarationMismatch};
