mod glsl;
mod naming;
mod parser;
mod types;
mod verify;

pub use glsl::{scan_glsl_descriptor_declarations, GlslDescriptorClass, GlslDescriptorDeclaration};
pub use naming::binding_const_name;
pub use parser::{reflect_shader_bytes, reflect_shader_words};
pub use types::{
    DescriptorCount, DescriptorKind, ReflectError, ReflectedBinding, ShaderBinding,
    ShaderReflection, ShaderStage,
};
pub use verify::{verify_spirv_against_glsl, DeclarationMismatch};
