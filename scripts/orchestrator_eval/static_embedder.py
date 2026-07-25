"""Sentence embeddings built from the Gemma input embedding table alone.

168M of Gemma 3 270M's 268M parameters are the 262144 x 640 vocabulary table.
Argument resolution keeps that table resident anyway, so pooling it costs no
additional weights and no transformer forward. It sees no context, which is
exactly what this evaluation is meant to quantify.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from tokenizers import Tokenizer

EMBEDDING_TENSOR_NAME = "model.embed_tokens.weight"
SIF_SMOOTHING = 1e-3
POOLING_STRATEGIES = ("mean", "unit_mean", "sif")


def read_external_tensor_location(model_dir: Path) -> tuple[Path, int, tuple[int, int]]:
    graph = onnx.load(str(model_dir / "onnx" / "model.onnx"), load_external_data=False).graph
    tensor = next(
        (init for init in graph.initializer if init.name == EMBEDDING_TENSOR_NAME), None
    )
    if tensor is None:
        raise ValueError(f"{EMBEDDING_TENSOR_NAME} not found in {model_dir}")

    entries = {entry.key: entry.value for entry in tensor.external_data}
    if "location" not in entries:
        raise ValueError(f"{EMBEDDING_TENSOR_NAME} is not stored as external data")

    return (
        model_dir / "onnx" / entries["location"],
        int(entries.get("offset", 0)),
        (tensor.dims[0], tensor.dims[1]),
    )


def load_embedding_table(model_dir: Path) -> np.ndarray:
    data_path, offset, shape = read_external_tensor_location(model_dir)
    return np.memmap(data_path, dtype=np.float32, mode="r", offset=offset, shape=shape)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class StaticEmbedder:
    def __init__(self, model_dir: str | Path, pooling: str = "unit_mean"):
        if pooling not in POOLING_STRATEGIES:
            raise ValueError(f"unknown pooling {pooling}")

        self.model_dir = Path(model_dir)
        self.pooling = pooling
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.table = load_embedding_table(self.model_dir)
        self.token_weights: dict[int, float] = {}
        self.common_component: np.ndarray | None = None

    def dimension(self) -> int:
        return int(self.table.shape[1])

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _lookup(self, token_ids: list[int]) -> np.ndarray:
        rows = np.asarray(self.table[token_ids], dtype=np.float32)
        if self.pooling == "mean":
            return rows
        return normalize_rows(rows)

    def _pool(self, token_ids: list[int]) -> np.ndarray:
        if not token_ids:
            return np.zeros(self.dimension(), dtype=np.float32)

        rows = self._lookup(token_ids)
        if self.pooling != "sif":
            return rows.mean(axis=0)

        weights = np.array(
            [self.token_weights.get(token_id, 1.0) for token_id in token_ids],
            dtype=np.float32,
        )
        return (rows * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-12)

    def fit_corpus(self, texts: list[str]) -> None:
        if self.pooling != "sif":
            return

        tokenized = [self.tokenize(text) for text in texts]
        counts = Counter(token_id for ids in tokenized for token_id in ids)
        total = max(sum(counts.values()), 1)
        self.token_weights = {
            token_id: SIF_SMOOTHING / (SIF_SMOOTHING + count / total)
            for token_id, count in counts.items()
        }

        self.common_component = None
        pooled = np.stack([self._pool(ids) for ids in tokenized])
        _, _, right = np.linalg.svd(pooled - pooled.mean(axis=0), full_matrices=False)
        self.common_component = right[0]

    def _remove_common_component(self, vectors: np.ndarray) -> np.ndarray:
        if self.common_component is None:
            return vectors
        projection = vectors @ self.common_component
        return vectors - projection[:, None] * self.common_component[None, :]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        pooled = np.stack([self._pool(self.tokenize(text)) for text in texts])
        return normalize_rows(self._remove_common_component(pooled))
