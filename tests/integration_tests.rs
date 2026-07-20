use std::path::Path;

#[test]
fn test_project_structure() {
    let required_dirs = [
        "assets",
        "assets/models",
        "assets/textures",
        "assets/fonts",
        "assets/shaders",
        "shaders",
        "src",
        "vendor",
    ];

    for dir in &required_dirs {
        assert!(
            Path::new(dir).exists(),
            "Required directory should exist: {}",
            dir
        );
    }
}

#[test]
fn test_assets_directory_structure() {
    assert!(Path::new("assets/models").is_dir());
    assert!(Path::new("assets/textures").is_dir());
    assert!(Path::new("assets/fonts").is_dir());
    assert!(Path::new("assets/shaders").is_dir());
}

#[test]
fn test_font_files_exist() {
    let font_files = [
        "assets/fonts/Roboto-Regular.ttf",
        "assets/fonts/mplus-1p-regular.ttf",
    ];

    for font in &font_files {
        assert!(Path::new(font).exists(), "Font file should exist: {}", font);
    }
}

#[test]
fn test_vendor_directory_structure() {
    let vendor_dirs = [
        "vendor/imgui",
        "vendor/imgui-sys",
        "vendor/imgui-winit-support",
    ];

    for dir in &vendor_dirs {
        assert!(
            Path::new(dir).exists(),
            "Vendor directory should exist: {}",
            dir
        );
    }
}

#[test]
fn test_cargo_files_exist() {
    assert!(Path::new("Cargo.toml").exists(), "Cargo.toml should exist");
    assert!(Path::new("build.rs").exists(), "build.rs should exist");
}

#[test]
fn test_gitignore_exists() {
    assert!(Path::new(".gitignore").exists(), ".gitignore should exist");
}

#[test]
fn test_readme_exists() {
    assert!(Path::new("README.md").exists(), "README.md should exist");
}

#[test]
fn test_claude_md_exists() {
    assert!(Path::new("CLAUDE.md").exists(), "CLAUDE.md should exist");
}

#[test]
fn test_src_directory_structure() {
    let src_dirs = [
        "src/app",
        "src/vulkanr",
        "src/loader",
        "src/render",
        "src/scene",
        "src/platform",
        "src/debugview",
        "src/logger",
        "src/math",
    ];

    for dir in &src_dirs {
        assert!(
            Path::new(dir).exists(),
            "Source directory should exist: {}",
            dir
        );
    }
}

#[test]
fn test_main_files_exist() {
    assert!(Path::new("src/main.rs").exists(), "main.rs should exist");
    assert!(Path::new("src/lib.rs").exists(), "lib.rs should exist");
}

#[test]
fn test_log_directory_exists() {
    assert!(Path::new("log").exists(), "log directory should exist");
}

#[test]
fn test_cargo_config_exists() {
    assert!(
        Path::new(".cargo/config.toml").exists(),
        ".cargo/config.toml should exist"
    );
}
