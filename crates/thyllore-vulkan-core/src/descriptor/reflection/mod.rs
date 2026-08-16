mod parser;
mod set_table;
mod types;

pub use parser::{reflect_shader_bytes, reflect_shader_words};
pub use set_table::{
    default_descriptor_type, kind_accepts, shader_stage_flags, DescriptorSetTable, LayoutMismatch,
    MergedBinding,
};
pub use types::{
    DescriptorCount, DescriptorKind, ReflectError, ReflectedBinding, ShaderReflection, ShaderStage,
};
