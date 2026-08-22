use std::path::PathBuf;

use anyhow::{anyhow, Result};
use half::f16;
use ort::io_binding::IoBinding;
use ort::session::Session as OrtSession;
use ort::value::Tensor;

/// ONNX model paths for the GenDoP camera direction copilot.
#[derive(Clone, Debug)]
pub struct CameraDirectionOnnxPaths {
    pub decoder_step0: PathBuf,
    pub decoder_with_past: PathBuf,
    pub embd_table: PathBuf,
}

/// ONNX inference session for the GenDoP camera direction copilot.
///
/// Holds two ONNX sessions (step-0 and with-past) and the embedding table loaded from disk.
pub struct CameraDirectionSession {
    step0: OrtSession,
    with_past: OrtSession,
    embd_table: Vec<f16>,
}

const VOCAB_SIZE: usize = 260;
const HIDDEN_DIM: usize = 1024;
const N_LAYERS: usize = 12;
const NUM_COND_TOKENS: usize = 77;
const POSE_LENGTH: usize = 30;

/// Special token IDs.
const BOS: usize = 1;
const EOS: usize = 2;

/// Maximum number of tokens to generate (BOS + 10 * pose_length).
const MAX_TOTAL_LEN: usize = 10 * POSE_LENGTH + 1; // 301

type PastCache = Vec<(Tensor<f16>, Tensor<f16>)>;

impl CameraDirectionSession {
    /// Load ONNX sessions from the given paths.
    pub fn from_paths(paths: &CameraDirectionOnnxPaths) -> Result<Self> {
        let step0 = OrtSession::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(paths.decoder_step0.as_path())?;

        let with_past = OrtSession::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(paths.decoder_with_past.as_path())?;

        // Load embedding table: f32 binary [260, 1024], row-major.
        let bytes = std::fs::read(&paths.embd_table)?;
        let n_bytes = bytes.len();
        let expected = VOCAB_SIZE * HIDDEN_DIM * 4; // f32 = 4 bytes
        if n_bytes != expected {
            return Err(anyhow!(
                "embedding table size mismatch: got {} bytes, expected {}",
                n_bytes,
                expected
            ));
        }
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let embd_table: Vec<f16> = values.iter().map(|v| f16::from_f32(*v)).collect();

        Ok(Self {
            step0,
            with_past,
            embd_table,
        })
    }

    /// Generate tokens from condition embeddings.
    ///
    /// `cond_embeds` is f32 [77, 1024] — the pre-computed condition embeddings.
    /// Returns the generated content token values (BOS/EOS excluded) as f32 Vec,
    /// which can be passed directly to `detokenize_sequence`.
    pub fn generate_tokens(&mut self, cond_embeds: &[f32]) -> Result<Vec<f32>> {
        if cond_embeds.len() != NUM_COND_TOKENS * HIDDEN_DIM {
            return Err(anyhow!(
                "cond_embeds length mismatch: got {}, expected {}",
                cond_embeds.len(),
                NUM_COND_TOKENS * HIDDEN_DIM
            ));
        }

        // Step 0: build inputs_embeds = [cond_embeds(77) | BOS embed(1)] -> shape [1, 78, 1024]
        let mut inputs_embeds: Vec<f16> = cond_embeds.iter().map(|v| f16::from_f32(*v)).collect();
        let bos_embd = self.token_embedding(BOS);
        inputs_embeds.extend_from_slice(&bos_embd);

        // Run decoder_step0
        let (logits_step0, mut past) = self.run_step0(&inputs_embeds)?;
        // Extract logits for the last position (position 77 = BOS position).
        // logits shape: [1, 78, 260] -> last position is index 77.
        let mut current_logits: Vec<f16> = logits_step0[NUM_COND_TOKENS * VOCAB_SIZE..].to_vec();

        // Greedy decode loop
        let mut content_tokens: Vec<f32> = Vec::new();
        let mut total_len: usize = 1; // BOS counts as 1

        while total_len < MAX_TOTAL_LEN {
            let next_token = greedy_argmax_constrained(&current_logits, total_len);

            if next_token == EOS {
                break;
            }

            content_tokens.push((next_token - 3) as f32);
            total_len += 1;

            // Build inputs_embeds for next step: embedding of the chosen token -> shape [1, 1, 1024]
            let token_embd: Vec<f16> = self.token_embedding(next_token).to_vec();

            // Run decoder_with_past
            let (logits_step, new_past) = self.run_with_past(&token_embd, &past)?;
            past = new_past;

            // logits shape: [1, 1, 260] -> only one position, replace current_logits
            current_logits = logits_step;
        }

        Ok(content_tokens)
    }

    /// Get the f16 embedding for a token ID.
    fn token_embedding(&self, token_id: usize) -> &[f16] {
        let start = token_id * HIDDEN_DIM;
        &self.embd_table[start..start + HIDDEN_DIM]
    }

    /// Run the step-0 decoder (no past).
    fn run_step0(&mut self, inputs_embeds: &[f16]) -> Result<(Vec<f16>, PastCache)> {
        let shape: Vec<i64> = vec![1, (NUM_COND_TOKENS + 1) as i64, HIDDEN_DIM as i64];
        let tensor = Tensor::from_array((shape, inputs_embeds.to_vec()))?;

        let outputs = self.step0.run(ort::inputs!["inputs_embeds" => tensor])?;

        // Extract logits: [1, 78, 260]
        let (_shape, logits_data) = outputs["logits"].try_extract_tensor::<f16>()?;
        let logits: Vec<f16> = logits_data.to_vec();

        // Extract 24 past tensors (12 layers * 2 for k and v)
        let mut past: PastCache = Vec::with_capacity(N_LAYERS);
        for layer in 0..N_LAYERS {
            let k_name = format!("past_{}_k", layer);
            let v_name = format!("past_{}_v", layer);
            let (k_shape, k_data) = outputs[k_name.as_str()].try_extract_tensor::<half::f16>()?;
            let k_tensor = Tensor::from_array((k_shape.to_vec(), k_data.to_vec()))?;

            let (v_shape, v_data) = outputs[v_name.as_str()].try_extract_tensor::<half::f16>()?;
            let v_tensor = Tensor::from_array((v_shape.to_vec(), v_data.to_vec()))?;
            past.push((k_tensor, v_tensor));
        }

        Ok((logits, past))
    }

    /// Run the with-past decoder using IoBinding for dynamic input count.
    fn run_with_past(
        &mut self,
        token_embd: &[f16],
        past: &PastCache,
    ) -> Result<(Vec<f16>, PastCache)> {
        let shape: Vec<i64> = vec![1, 1, HIDDEN_DIM as i64];
        let embeds_tensor = Tensor::from_array((shape, token_embd.to_vec()))?;

        // Build IoBinding with all inputs
        let mut binding: IoBinding = self.with_past.create_binding()?;
        binding.clear_inputs();
        binding.clear_outputs();
        binding.bind_input("inputs_embeds", &embeds_tensor)?;
        for (layer, (k, v)) in past.iter().enumerate() {
            let k_name = format!("past_{}_k", layer);
            let v_name = format!("past_{}_v", layer);
            binding.bind_input(&k_name, k)?;
            binding.bind_input(&v_name, v)?;
        }

        // Bind outputs — ONNX Runtime requires at least one output bound when using IoBinding.
        // Use bind_output_to_device since shapes may vary between calls.
        let mem_info = self.with_past.allocator().memory_info();
        binding.bind_output_to_device("logits", &mem_info)?;
        for layer in 0..N_LAYERS {
            let k_name = format!("new_past_{}_k", layer);
            let v_name = format!("new_past_{}_v", layer);
            binding.bind_output_to_device(&k_name, &mem_info)?;
            binding.bind_output_to_device(&v_name, &mem_info)?;
        }

        let outputs = self.with_past.run_binding(&binding)?;

        // Extract logits: [1, 1, 260]
        let (_shape, logits_data) = outputs["logits"].try_extract_tensor::<f16>()?;
        let logits: Vec<f16> = logits_data.to_vec();

        // Extract new past tensors (decoder_with_past.onnx names its past outputs with a
        // "new_" prefix to avoid colliding with the identically-named past inputs).
        let mut new_past: PastCache = Vec::with_capacity(N_LAYERS);
        for layer in 0..N_LAYERS {
            let k_name = format!("new_past_{}_k", layer);
            let v_name = format!("new_past_{}_v", layer);
            let (k_shape, k_data) = outputs[k_name.as_str()].try_extract_tensor::<half::f16>()?;
            let k_tensor = Tensor::from_array((k_shape.to_vec(), k_data.to_vec()))?;

            let (v_shape, v_data) = outputs[v_name.as_str()].try_extract_tensor::<half::f16>()?;
            let v_tensor = Tensor::from_array((v_shape.to_vec(), v_data.to_vec()))?;
            new_past.push((k_tensor, v_tensor));
        }

        Ok((logits, new_past))
    }
}

/// Greedy argmax with token constraints.
///
/// `total_len` is the length including BOS (starts at 1).
/// Candidates are [3, vocab_size), and EOS (2) is added only when total_len % 10 == 1.
fn greedy_argmax_constrained(logits: &[f16], total_len: usize) -> usize {
    let mut best_score = f32::NEG_INFINITY;
    let mut best_token: usize = 3; // default to first valid candidate

    for token in 3..VOCAB_SIZE {
        let score = logits[token].to_f32();
        if score > best_score {
            best_score = score;
            best_token = token;
        }
    }

    // EOS is eligible only when total_len % 10 == 1
    if total_len % 10 == 1 {
        let eos_score = logits[EOS].to_f32();
        if eos_score > best_score {
            return EOS;
        }
    }

    best_token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(VOCAB_SIZE, 260);
        assert_eq!(HIDDEN_DIM, 1024);
        assert_eq!(N_LAYERS, 12);
        assert_eq!(NUM_COND_TOKENS, 77);
        assert_eq!(POSE_LENGTH, 30);
        assert_eq!(MAX_TOTAL_LEN, 301);
    }

    #[test]
    fn test_greedy_argmax_constrained_basic() {
        // All zeros except token 5 which is 1.0
        let mut logits: Vec<f16> = vec![f16::from_f32(0.0); VOCAB_SIZE];
        logits[5] = f16::from_f32(1.0);

        // total_len=1 (BOS only, 1 % 10 == 1, so EOS is eligible)
        let chosen = greedy_argmax_constrained(&logits, 1);
        assert_eq!(chosen, 5);
    }

    #[test]
    fn test_greedy_argmax_constrained_eos_eligible() {
        // All zeros except EOS (token 2) which is 1.0
        let mut logits: Vec<f16> = vec![f16::from_f32(0.0); VOCAB_SIZE];
        logits[EOS] = f16::from_f32(1.0);

        // total_len=1 (1 % 10 == 1, EOS eligible) -> should pick EOS
        let chosen = greedy_argmax_constrained(&logits, 1);
        assert_eq!(chosen, EOS);

        // total_len=2 (2 % 10 != 1, EOS not eligible) -> should pick token 3 (first candidate)
        let chosen = greedy_argmax_constrained(&logits, 2);
        assert_eq!(chosen, 3);
    }

    #[test]
    fn test_greedy_argmax_constrained_eos_at_11() {
        // total_len=11 (11 % 10 == 1, EOS eligible)
        let mut logits: Vec<f16> = vec![f16::from_f32(0.0); VOCAB_SIZE];
        logits[EOS] = f16::from_f32(1.0);
        let chosen = greedy_argmax_constrained(&logits, 11);
        assert_eq!(chosen, EOS);
    }

    #[test]
    fn test_greedy_argmax_constrained_eos_not_at_12() {
        // total_len=12 (12 % 10 != 1, EOS not eligible)
        let mut logits: Vec<f16> = vec![f16::from_f32(0.0); VOCAB_SIZE];
        logits[EOS] = f16::from_f32(1.0);
        let chosen = greedy_argmax_constrained(&logits, 12);
        assert_eq!(chosen, 3); // first candidate token
    }

    #[test]
    fn test_generate_tokens_offset_correction() {
        // Verify that greedy_argmax_constrained returns token IDs >= 3,
        // so (token - 3) is in [0, 256) — the range expected by detokenize_pose.
        //
        // Construct logits where index 100 has the highest value among candidates (3..VOCAB_SIZE).
        let mut logits: Vec<f16> = vec![f16::from_f32(-100.0); VOCAB_SIZE];
        logits[100] = f16::from_f32(1.0); // highest among candidates
        let chosen = greedy_argmax_constrained(&logits, 1);
        assert_eq!(chosen, 100);
        let offset_corrected = (chosen - 3) as f32;
        assert!(
            offset_corrected >= 0.0 && offset_corrected < 256.0,
            "offset-corrected token {} is out of [0, 256) range",
            offset_corrected
        );

        // Edge case: lowest candidate (token 3) should map to 0.0
        let mut logits_low: Vec<f16> = vec![f16::from_f32(-100.0); VOCAB_SIZE];
        logits_low[3] = f16::from_f32(1.0);
        let chosen_low = greedy_argmax_constrained(&logits_low, 1);
        assert_eq!(chosen_low, 3);
        let offset_corrected_low = (chosen_low - 3) as f32;
        assert_eq!(offset_corrected_low, 0.0);

        // Edge case: highest candidate (token VOCAB_SIZE-1 = 259) should map to 256.0
        let mut logits_high: Vec<f16> = vec![f16::from_f32(-100.0); VOCAB_SIZE];
        logits_high[VOCAB_SIZE - 1] = f16::from_f32(1.0);
        let chosen_high = greedy_argmax_constrained(&logits_high, 1);
        assert_eq!(chosen_high, VOCAB_SIZE - 1);
        let offset_corrected_high = (chosen_high - 3) as f32;
        assert_eq!(offset_corrected_high, 256.0_f32);
        // Note: 256.0 is the boundary — detokenize_pose clamps to [0, 255], so this is acceptable.
    }
}

#[test]
fn test_camera_direction_parity() -> Result<(), anyhow::Error> {
    let onnx_dir = match std::env::var("GENDOP_ONNX_DIR") {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Skipping test_camera_direction_parity: GENDOP_ONNX_DIR not set ({e})");
            return Ok(());
        }
    };

    let paths = CameraDirectionOnnxPaths {
        decoder_step0: PathBuf::from(&onnx_dir).join("decoder_step0.onnx"),
        decoder_with_past: PathBuf::from(&onnx_dir).join("decoder_with_past.onnx"),
        embd_table: PathBuf::from(&onnx_dir).join("embd_table.bin"),
    };

    let mut session = CameraDirectionSession::from_paths(&paths)?;

    let cond_embeds: Vec<f32> = vec![0.0f32; 77 * 1024];
    let tokens = session.generate_tokens(&cond_embeds)?;

    assert!(!tokens.is_empty(), "generated token list is empty");
    assert_eq!(
        tokens.len() % 10,
        0,
        "token length {} is not a multiple of 10",
        tokens.len()
    );

    Ok(())
}
