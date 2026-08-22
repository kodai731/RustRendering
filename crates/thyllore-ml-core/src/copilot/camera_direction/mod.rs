pub mod caption;
pub mod detokenize;
pub mod keyframes;
pub mod session;
pub mod text_encoder;

use crate::copilot::camera_direction::detokenize::detokenize_sequence;
use crate::copilot::camera_direction::session::{CameraDirectionOnnxPaths, CameraDirectionSession};
use crate::copilot::camera_direction::text_encoder::TextEncoder;

/// Generate camera poses from a natural-language utterance.
///
/// Convenience function that chains: build_movement_caption -> TextEncoder encode ->
/// CameraDirectionSession generate_tokens -> detokenize_sequence.
pub fn generate_camera_poses(
    paths: &CameraDirectionOnnxPaths,
    caption: &str,
) -> anyhow::Result<Vec<[[f32; 4]; 4]>> {
    let mut encoder = TextEncoder::from_paths(&paths.text_encoder, &paths.tokenizer)?;
    let cond_embeds = encoder.encode(caption)?;

    let mut session = CameraDirectionSession::from_paths(paths)?;
    let tokens = session.generate_tokens(&cond_embeds)?;

    detokenize_sequence(&tokens).map_err(|e| anyhow::anyhow!("detokenization failed: {e}"))
}
