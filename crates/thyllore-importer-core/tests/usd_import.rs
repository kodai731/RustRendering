//! Smoke test for the USD importer against a real asset.
//!
//! The asset path is machine-specific, so it is provided via the
//! `THYLLORE_USD_TEST_FILE` environment variable. The test is skipped when the
//! variable is unset, keeping CI and other machines green.

use thyllore_importer_core::usd::load_usd_file;

#[test]
fn imports_usd_asset_when_path_is_provided() {
    let Ok(path) = std::env::var("THYLLORE_USD_TEST_FILE") else {
        eprintln!("THYLLORE_USD_TEST_FILE not set; skipping USD import smoke test");
        return;
    };

    let result = load_usd_file(&path).expect("USD import should succeed");

    let total_vertices: usize = result
        .meshes
        .iter()
        .map(|m| m.vertex_data.vertices.len())
        .sum();
    let total_indices: usize = result
        .meshes
        .iter()
        .map(|m| m.vertex_data.indices.len())
        .sum();
    let total_bones: usize = result
        .animation_system
        .skeletons
        .iter()
        .map(|s| s.bones.len())
        .sum();
    let total_keyframes: usize = result
        .clips
        .iter()
        .flat_map(|c| c.channels.values())
        .map(|ch| ch.translation.len() + ch.rotation.len() + ch.scale.len())
        .sum();

    let morph = &result.morph_animation;
    let morphed_meshes = morph.targets.iter().filter(|t| !t.is_empty()).count();

    eprintln!(
        "USD import: meshes={} vertices={} indices={} skeletons={} bones={} clips={} keyframes={} skinned={} armature={}",
        result.meshes.len(),
        total_vertices,
        total_indices,
        result.animation_system.skeletons.len(),
        total_bones,
        result.clips.len(),
        total_keyframes,
        result.has_skinned_meshes,
        result.has_armature,
    );
    eprintln!(
        "USD import morph: empty={} morphed_meshes={} weight_keyframes={}",
        morph.is_empty(),
        morphed_meshes,
        morph.animations.len(),
    );

    assert!(!result.meshes.is_empty(), "expected at least one mesh");
    assert!(total_vertices > 0, "expected non-empty geometry");
    assert_eq!(
        total_indices % 3,
        0,
        "triangulated index buffer must be a multiple of 3"
    );

    if !morph.is_empty() {
        assert_eq!(
            morph.targets.len(),
            result.meshes.len(),
            "morph targets must be aligned per mesh"
        );
        assert_eq!(
            morph.base_vertices.len(),
            result.meshes.len(),
            "morph base vertices must be aligned per mesh"
        );
    }
}
