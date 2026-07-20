use std::fs;
use std::path::Path;

#[test]
fn test_shader_source_directory_exists() {
    assert!(
        Path::new("shaders").exists(),
        "Shader source directory should exist"
    );
}

#[test]
fn test_shader_output_directory_exists() {
    assert!(
        Path::new("assets/shaders").exists(),
        "Shader output directory should exist"
    );
}

#[test]
fn test_all_shader_sources_exist() {
    let shader_sources = [
        "shaders/vertex.vert",
        "shaders/fragment.frag",
        "shaders/gbufferVertex.vert",
        "shaders/gbufferFragment.frag",
        "shaders/compositeVertex.vert",
        "shaders/compositeFragment.frag",
        "shaders/gridVertex.vert",
        "shaders/gridFragment.frag",
        "shaders/gizmoVertex.vert",
        "shaders/gizmoFragment.frag",
        "shaders/imguiVertex.vert",
        "shaders/imguiFragment.frag",
        "shaders/rayQueryShadow.comp",
    ];

    for shader in &shader_sources {
        assert!(
            Path::new(shader).exists(),
            "Shader source should exist: {}",
            shader
        );
    }
}

#[test]
fn test_all_compiled_shaders_exist() {
    let compiled_shaders = [
        "assets/shaders/vert.spv",
        "assets/shaders/frag.spv",
        "assets/shaders/gbufferVert.spv",
        "assets/shaders/gbufferFrag.spv",
        "assets/shaders/compositeVert.spv",
        "assets/shaders/compositeFrag.spv",
        "assets/shaders/gridVert.spv",
        "assets/shaders/gridFrag.spv",
        "assets/shaders/gizmoVert.spv",
        "assets/shaders/gizmoFrag.spv",
        "assets/shaders/imguiVert.spv",
        "assets/shaders/imguiFrag.spv",
        "assets/shaders/rayQueryShadow.spv",
    ];

    for shader in &compiled_shaders {
        assert!(
            Path::new(shader).exists(),
            "Compiled shader should exist: {}",
            shader
        );
    }
}

#[test]
fn test_compiled_shaders_not_empty() {
    let compiled_shaders = [
        "assets/shaders/vert.spv",
        "assets/shaders/frag.spv",
        "assets/shaders/gbufferVert.spv",
        "assets/shaders/gbufferFrag.spv",
    ];

    for shader in &compiled_shaders {
        let metadata = fs::metadata(shader)
            .unwrap_or_else(|_| panic!("Failed to read shader metadata: {}", shader));

        assert!(
            metadata.len() > 0,
            "Compiled shader should not be empty: {}",
            shader
        );
    }
}

#[test]
fn test_shader_spv_header() {
    let shader = "assets/shaders/vert.spv";
    let data = fs::read(shader).expect("Failed to read shader file");

    assert!(data.len() >= 4, "Shader file should have at least 4 bytes");

    let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    assert_eq!(
        magic, 0x07230203,
        "Shader should have correct SPIR-V magic number"
    );
}

#[test]
fn test_vertex_shader_extension() {
    let vertex_shaders = [
        "shaders/vertex.vert",
        "shaders/gbufferVertex.vert",
        "shaders/compositeVertex.vert",
        "shaders/gridVertex.vert",
        "shaders/gizmoVertex.vert",
        "shaders/imguiVertex.vert",
    ];

    for shader in &vertex_shaders {
        assert!(
            shader.ends_with(".vert"),
            "Vertex shader should have .vert extension: {}",
            shader
        );
    }
}

#[test]
fn test_fragment_shader_extension() {
    let fragment_shaders = [
        "shaders/fragment.frag",
        "shaders/gbufferFragment.frag",
        "shaders/compositeFragment.frag",
        "shaders/gridFragment.frag",
        "shaders/gizmoFragment.frag",
        "shaders/imguiFragment.frag",
    ];

    for shader in &fragment_shaders {
        assert!(
            shader.ends_with(".frag"),
            "Fragment shader should have .frag extension: {}",
            shader
        );
    }
}

#[test]
fn test_compute_shader_extension() {
    let compute_shaders = ["shaders/rayQueryShadow.comp"];

    for shader in &compute_shaders {
        assert!(
            shader.ends_with(".comp"),
            "Compute shader should have .comp extension: {}",
            shader
        );
    }
}

#[test]
fn test_shader_count_matches() {
    let shader_sources_count = fs::read_dir("shaders")
        .expect("Failed to read shaders directory")
        .filter_map(|entry| entry.ok())
       .filter(|entry| {
            let path = entry.path();
            path.is_file()
                && (path.extension() == Some("vert".as_ref())
                    || path.extension() == Some("frag".as_ref())
                    || path.extension() == Some("comp".as_ref())
                    || path.extension() == Some("geom".as_ref()))
        })
        .count();

    let compiled_shaders_count = fs::read_dir("assets/shaders")
        .expect("Failed to read assets/shaders directory")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let path = entry.path();
            path.is_file() && path.extension() == Some("spv".as_ref())
        })
        .count();

    assert_eq!(
        shader_sources_count, compiled_shaders_count,
        "Number of shader sources should match compiled shaders"
    );
}
