"""Sentence embeddings from a contextual encoder on plain onnxruntime.

Option B of the router design. Serves as the reference point that tells whether
option C (pooling Gemma's embedding table, no context) is close enough to
justify saving the extra resident weights.

Exposes the same encode_batch / fit_corpus surface as StaticEmbedder so the
evaluation driver treats both identically.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

E5_QUERY_PREFIX = "query: "


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def pad_to_batch(sequences: list[list[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(sequence) for sequence in sequences)
    token_ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(sequences), width), dtype=np.int64)
    for row, sequence in enumerate(sequences):
        token_ids[row, : len(sequence)] = sequence
        attention[row, : len(sequence)] = 1
    return token_ids, attention


class ContextualEmbedder:
    def __init__(
        self,
        model_dir: str | Path,
        onnx_file: str = "model.onnx",
        prefix: str = E5_QUERY_PREFIX,
        intra_op_threads: int = 1,
    ):
        self.model_dir = Path(model_dir)
        self.prefix = prefix
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_dir / "onnx" / onnx_file),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {tensor.name for tensor in self.session.get_inputs()}
        self.pad_id = self.tokenizer.token_to_id("<pad>") or 1

    def fit_corpus(self, texts: list[str]) -> None:
        return

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = [self.tokenizer.encode(self.prefix + text).ids for text in texts]
        token_ids, attention = pad_to_batch(encoded, self.pad_id)

        feeds = {"input_ids": token_ids, "attention_mask": attention}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(token_ids)

        hidden = self.session.run(["last_hidden_state"], feeds)[0]
        mask = attention[:, :, None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1e-12)
        return normalize_rows(pooled)
