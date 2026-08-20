use crate::descriptor::pass_manifest::IMGUI;
use crate::descriptor::reflected_layout::ReflectedLayoutSpec;

pub fn imgui_layout_spec() -> ReflectedLayoutSpec {
    ReflectedLayoutSpec::local(&IMGUI)
}
