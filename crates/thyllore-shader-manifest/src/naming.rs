const SHADER_EXTENSIONS: [&str; 4] = ["vert", "frag", "geom", "comp"];

pub fn is_shader_source(file_name: &str) -> bool {
    file_extension(file_name).is_some_and(|extension| SHADER_EXTENSIONS.contains(&extension))
}

pub fn spirv_output_name(source_file_name: &str) -> Option<String> {
    let extension = file_extension(source_file_name)?;
    let stem = &source_file_name[..source_file_name.len() - extension.len() - 1];

    let base_name = stem
        .trim_end_matches("Vertex")
        .trim_end_matches("vertex")
        .trim_end_matches("Fragment")
        .trim_end_matches("fragment")
        .trim_end_matches("Geometry")
        .trim_end_matches("geometry")
        .trim_end_matches("Compute")
        .trim_end_matches("compute");

    let stage_suffix = match extension {
        "vert" => "Vert",
        "frag" => "Frag",
        "geom" => "Geom",
        "comp" => "Comp",
        _ => return None,
    };

    if base_name.is_empty() {
        Some(format!("{}.spv", stage_suffix.to_ascii_lowercase()))
    } else {
        Some(format!("{base_name}{stage_suffix}.spv"))
    }
}

fn file_extension(file_name: &str) -> Option<&str> {
    let (_, extension) = file_name.rsplit_once('.')?;
    Some(extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_stage_word_and_appends_stage_suffix() {
        assert_eq!(
            spirv_output_name("vertex.vert").as_deref(),
            Some("vert.spv")
        );
        assert_eq!(
            spirv_output_name("fragment.frag").as_deref(),
            Some("frag.spv")
        );
        assert_eq!(
            spirv_output_name("gbufferVertex.vert").as_deref(),
            Some("gbufferVert.spv")
        );
        assert_eq!(
            spirv_output_name("imguiFragment.frag").as_deref(),
            Some("imguiFrag.spv")
        );
        assert_eq!(
            spirv_output_name("rayQueryShadow.comp").as_deref(),
            Some("rayQueryShadowComp.spv")
        );
        assert_eq!(
            spirv_output_name("histogramCompute.comp").as_deref(),
            Some("histogramComp.spv")
        );
    }

    #[test]
    fn rejects_non_shader_files() {
        assert_eq!(spirv_output_name("passes.toml"), None);
        assert_eq!(spirv_output_name("common.glsl"), None);
        assert!(!is_shader_source("include"));
        assert!(is_shader_source("dofFragment.frag"));
    }
}
