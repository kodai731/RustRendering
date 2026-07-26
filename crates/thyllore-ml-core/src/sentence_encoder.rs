//! Sentence embeddings from an ONNX encoder, for the orchestrator's route index.
//!
//! Mirrors `scripts/orchestrator_eval/contextual_embedder.py` step for step —
//! the `query: ` prefix E5 was trained with, special tokens on, mean pooling over
//! the attention mask, then L2 normalization. Cosine similarity is a dot product
//! only after that last step, and the router's exemplar index is exported by the
//! Python side, so a difference anywhere in the sequence would compare vectors
//! from two different spaces.
//!
//! Knows nothing about routes or tools: it turns text into a vector, and
//! `orchestrator::systems::router` decides what that vector means.

use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

pub const E5_QUERY_PREFIX: &str = "query: ";

const TOKENIZER_FILENAME: &str = "tokenizer.json";
const ONNX_RELATIVE_PATH: &str = "onnx/model.onnx";
const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
const TOKEN_TYPE_IDS: &str = "token_type_ids";
const OUTPUT_HIDDEN_STATE: &str = "last_hidden_state";
const FALLBACK_PAD_TOKEN_ID: i64 = 1;
const PAD_TOKEN: &str = "<pad>";
const PROBE_TEXT: &str = "sample utterance";

struct TokenizedBatch {
    token_ids: Vec<i64>,
    attention: Vec<i64>,
    rows: usize,
    width: usize,
}

pub struct SentenceEncoder {
    tokenizer: Tokenizer,
    session: Session,
    prefix: String,
    pad_token_id: i64,
    expects_token_type_ids: bool,
    dimensions: usize,
}

impl SentenceEncoder {
    /// Loads `tokenizer.json` and `onnx/model.onnx` from a model directory.
    ///
    /// Encodes one probe utterance before returning, which is what establishes the
    /// embedding width. A directory holding a model of the wrong shape then fails
    /// here rather than at the first user utterance.
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let tokenizer_path = model_dir.join(TOKENIZER_FILENAME);
        let onnx_path: PathBuf = model_dir.join(ONNX_RELATIVE_PATH);

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| anyhow::anyhow!("{}: {error}", tokenizer_path.display()))?;
        let session = Session::builder()?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("{}", onnx_path.display()))?;

        let pad_token_id = tokenizer
            .token_to_id(PAD_TOKEN)
            .map_or(FALLBACK_PAD_TOKEN_ID, i64::from);
        let expects_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == TOKEN_TYPE_IDS);

        let mut encoder = Self {
            tokenizer,
            session,
            prefix: E5_QUERY_PREFIX.to_string(),
            pad_token_id,
            expects_token_type_ids,
            dimensions: 0,
        };
        encoder.dimensions = encoder.encode(PROBE_TEXT)?.len();
        ensure!(
            encoder.dimensions > 0,
            "{} produced an empty embedding",
            onnx_path.display()
        );
        Ok(encoder)
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn encode(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut embeddings = self.encode_batch(&[text])?;
        Ok(embeddings.remove(0))
    }

    /// One forward pass over the whole batch, padded to its longest sequence.
    ///
    /// The mask keeps padding out of the pooled mean, so a batch produces the same
    /// vectors as the utterances encoded one at a time.
    pub fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        ensure!(!texts.is_empty(), "cannot encode an empty batch");
        let batch = self.tokenize(texts)?;

        let shape = vec![batch.rows as i64, batch.width as i64];
        let mut inputs = ort::inputs![
            INPUT_IDS => Tensor::from_array((shape.clone(), batch.token_ids.clone()))?,
            ATTENTION_MASK => Tensor::from_array((shape.clone(), batch.attention.clone()))?
        ];
        if self.expects_token_type_ids {
            inputs.push((
                TOKEN_TYPE_IDS.into(),
                Tensor::from_array((shape, vec![0_i64; batch.rows * batch.width]))?.into(),
            ));
        }

        let outputs = self.session.run(inputs)?;
        let (hidden_shape, hidden) = outputs[OUTPUT_HIDDEN_STATE].try_extract_tensor::<f32>()?;
        let width = *hidden_shape
            .last()
            .context("last_hidden_state must be rank 3")? as usize;

        Ok(pool_and_normalize(hidden, &batch, width))
    }

    fn tokenize(&self, texts: &[&str]) -> Result<TokenizedBatch> {
        let prefixed: Vec<String> = texts
            .iter()
            .map(|text| format!("{}{text}", self.prefix))
            .collect();
        let encoded = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|error| anyhow::anyhow!("tokenization failed: {error}"))?;

        let width = encoded
            .iter()
            .map(|item| item.get_ids().len())
            .max()
            .unwrap_or(0);
        ensure!(
            width > 0,
            "every utterance in the batch tokenized to nothing"
        );

        let rows = encoded.len();
        let mut token_ids = vec![self.pad_token_id; rows * width];
        let mut attention = vec![0_i64; rows * width];
        for (row, item) in encoded.iter().enumerate() {
            for (column, &id) in item.get_ids().iter().enumerate() {
                token_ids[row * width + column] = i64::from(id);
                attention[row * width + column] = 1;
            }
        }

        Ok(TokenizedBatch {
            token_ids,
            attention,
            rows,
            width,
        })
    }
}

fn pool_and_normalize(hidden: &[f32], batch: &TokenizedBatch, width: usize) -> Vec<Vec<f32>> {
    (0..batch.rows)
        .map(|row| {
            let mut pooled = vec![0.0_f32; width];
            let mut counted = 0.0_f32;

            for column in 0..batch.width {
                if batch.attention[row * batch.width + column] == 0 {
                    continue;
                }
                counted += 1.0;
                let start = (row * batch.width + column) * width;
                for (target, value) in pooled.iter_mut().zip(&hidden[start..start + width]) {
                    *target += value;
                }
            }

            normalize(&mut pooled, counted.max(f32::MIN_POSITIVE));
            pooled
        })
        .collect()
}

fn normalize(vector: &mut [f32], divisor: f32) {
    for value in vector.iter_mut() {
        *value /= divisor;
    }
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    for value in vector.iter_mut() {
        *value /= length.max(f32::MIN_POSITIVE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_row(attention: Vec<i64>) -> TokenizedBatch {
        let width = attention.len();
        TokenizedBatch {
            token_ids: vec![0; width],
            attention,
            rows: 1,
            width,
        }
    }

    #[test]
    fn pooling_averages_only_the_attended_positions() {
        let hidden = vec![3.0, 0.0, 1.0, 0.0, 99.0, 99.0];
        let pooled = pool_and_normalize(&hidden, &single_row(vec![1, 1, 0]), 2);

        assert_eq!(pooled.len(), 1);
        assert!((pooled[0][0] - 1.0).abs() < 1e-6, "{:?}", pooled[0]);
        assert!(pooled[0][1].abs() < 1e-6);
    }

    #[test]
    fn pooled_vectors_are_unit_length() {
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let pooled = pool_and_normalize(&hidden, &single_row(vec![1, 1]), 2);

        let length = pooled[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6, "got {length}");
    }

    #[test]
    fn an_all_zero_row_does_not_produce_nan() {
        let pooled = pool_and_normalize(&[0.0, 0.0], &single_row(vec![1]), 2);
        assert!(pooled[0].iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rows_pool_independently_of_each_other() {
        let hidden = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let batch = TokenizedBatch {
            token_ids: vec![0; 4],
            attention: vec![1, 0, 1, 1],
            rows: 2,
            width: 2,
        };

        let pooled = pool_and_normalize(&hidden, &batch, 2);
        assert!((pooled[0][0] - 1.0).abs() < 1e-6);
        assert!((pooled[1][1] - 1.0).abs() < 1e-6);
    }
}
