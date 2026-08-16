use crate::descriptor::pass_shaders::IMGUI_SHADERS;
use crate::descriptor::reflected_layout::ReflectedLayoutSpec;

pub const IMGUI_TEXTURE_BINDING: u32 = 0;

pub fn imgui_layout_spec() -> ReflectedLayoutSpec {
    ReflectedLayoutSpec::new(IMGUI_SHADERS.to_vec(), 0)
}
