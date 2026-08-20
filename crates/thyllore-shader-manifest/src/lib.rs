mod bindings_codegen;
mod codegen;
mod manifest;
mod naming;

pub use bindings_codegen::{generate_shader_bindings_rust, BindingCodegenError};
pub use codegen::generate_pass_manifest_rust;
pub use manifest::{ManifestError, PassDefinition, PassManifest, SetRole, StageKind, StageSource};
pub use naming::{is_shader_source, spirv_output_name};
