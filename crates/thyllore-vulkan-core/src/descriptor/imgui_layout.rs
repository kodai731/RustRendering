use crate::descriptor::pass_manifest::IMGUI;
use crate::descriptor::reflected_layout::ReflectedLayoutSpec;

pub const IMGUI_TEXTURE_BINDING: u32 = 0;

pub fn imgui_layout_spec() -> ReflectedLayoutSpec {
    ReflectedLayoutSpec::local(&IMGUI)
}
