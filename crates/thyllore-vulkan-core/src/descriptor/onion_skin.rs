use crate::descriptor::pass_manifest::ONION_SKIN_COMPOSITE;
use crate::descriptor::reflected_layout::ReflectedLayoutSpec;

pub fn onion_skin_composite_layout_spec() -> ReflectedLayoutSpec {
    ReflectedLayoutSpec::local(&ONION_SKIN_COMPOSITE)
}
