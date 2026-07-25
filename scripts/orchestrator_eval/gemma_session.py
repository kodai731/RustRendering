"""Constrained greedy decoding for Gemma 3 270M on plain onnxruntime.

This is the Python rehearsal of the decode loop that src/orchestrator/ will own
in Rust: the same onnx-community export, the same llguidance grammar, the same
KV cache wiring. It exists to decide whether the Rust work is worth starting,
and later to emit teacher-forcing vectors for the Rust golden tests.

Nothing about the model is hardcoded — layer count, kv head count, head_dim,
eos ids, chat template and tensor names all come from the model directory, so
swapping the model for a Gemma 4 fallback does not silently break the loop.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import llguidance
import numpy as np
import onnxruntime as ort
from jinja2 import Environment
from llguidance.numpy import (
    allocate_token_bitmask,
    apply_token_bitmask_inplace,
    fill_next_token_bitmask,
)
from tokenizers import Tokenizer


@dataclass
class GenerationResult:
    text: str
    token_count: int
    prefill_seconds: float
    decode_seconds: float
    mask_seconds: float
    stop_reason: str
    grammar_error: str | None = None

    def milliseconds_per_token(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.decode_seconds * 1000.0 / self.token_count

    def milliseconds_per_mask(self) -> float:
        if self.token_count == 0:
            return 0.0
        return self.mask_seconds * 1000.0 / self.token_count


def raise_template_exception(message: str) -> None:
    raise ValueError(message)


def read_token_text(token_config_value) -> str:
    if isinstance(token_config_value, dict):
        return token_config_value["content"]
    return token_config_value


class GemmaOnnxSession:
    def __init__(self, model_dir: str | Path, intra_op_threads: int = 1):
        self.model_dir = Path(model_dir)
        self.config = self._read_json("config.json")
        tokenizer_config = self._read_json("tokenizer_config.json")
        generation_config = self._read_json("generation_config.json")

        self.layer_count = self.config["num_hidden_layers"]
        self.kv_head_count = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.bos_token = read_token_text(tokenizer_config["bos_token"])
        self.eos_token_ids = self._read_eos_token_ids(generation_config)

        tokenizer_json = (self.model_dir / "tokenizer.json").read_text()
        self.tokenizer = Tokenizer.from_str(tokenizer_json)
        self.grammar_tokenizer = llguidance.LLTokenizer(
            tokenizer_json, eos_token=self.eos_token_ids[0]
        )

        template_environment = Environment()
        template_environment.globals["raise_exception"] = raise_template_exception
        self.chat_template = template_environment.from_string(
            (self.model_dir / "chat_template.jinja").read_text()
        )

        self.session = self._create_session(intra_op_threads)
        self.past_key_names = [
            name for name in self._input_names() if name.startswith("past_key_values.")
        ]
        self.present_names = [
            name for name in self._output_names() if name.startswith("present.")
        ]

    def _read_json(self, filename: str) -> dict:
        return json.loads((self.model_dir / filename).read_text())

    def _read_eos_token_ids(self, generation_config: dict) -> list[int]:
        configured = generation_config.get("eos_token_id", self.config["eos_token_id"])
        if isinstance(configured, list):
            return configured
        return [configured]

    def _create_session(self, intra_op_threads: int) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(self.model_dir / "onnx" / "model.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )

    def _input_names(self) -> list[str]:
        return [tensor.name for tensor in self.session.get_inputs()]

    def _output_names(self) -> list[str]:
        return [tensor.name for tensor in self.session.get_outputs()]

    def render_prompt(self, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.chat_template.render(
            messages=messages, bos_token=self.bos_token, add_generation_prompt=True
        )

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids

    def build_grammar(self, json_schema: dict) -> str:
        return llguidance.LLMatcher.grammar_from_json_schema(json_schema)

    def _empty_past(self) -> dict:
        empty = np.zeros((1, self.kv_head_count, 0, self.head_dim), dtype=np.float32)
        return {name: empty for name in self.past_key_names}

    def _run_step(self, token_ids: list[int], past: dict, total_length: int) -> tuple[np.ndarray, dict]:
        feeds = dict(past)
        feeds["input_ids"] = np.array([token_ids], dtype=np.int64)
        feeds["attention_mask"] = np.ones((1, total_length), dtype=np.int64)

        outputs = self.session.run(None, feeds)
        output_names = self._output_names()
        named = dict(zip(output_names, outputs))

        next_past = {
            past_name: named[present_name]
            for past_name, present_name in zip(self.past_key_names, self.present_names)
        }
        return named["logits"], next_past

    def generate_constrained(
        self, prompt: str, grammar: str, max_new_tokens: int = 64
    ) -> GenerationResult:
        matcher = llguidance.LLMatcher(self.grammar_tokenizer, grammar)
        if matcher.is_error():
            return GenerationResult("", 0, 0.0, 0.0, 0.0, "grammar_error", matcher.get_error())

        prompt_token_ids = self.tokenize(prompt)
        bitmask = allocate_token_bitmask(1, self.grammar_tokenizer.vocab_size)

        prefill_start = time.perf_counter()
        logits, past = self._run_step(prompt_token_ids, self._empty_past(), len(prompt_token_ids))
        prefill_seconds = time.perf_counter() - prefill_start

        generated_ids: list[int] = []
        total_length = len(prompt_token_ids)
        mask_seconds = 0.0
        stop_reason = "max_tokens"
        decode_start = time.perf_counter()

        while len(generated_ids) < max_new_tokens:
            next_token_logits = np.array(logits[0, -1:, :], dtype=np.float32)

            mask_start = time.perf_counter()
            fill_next_token_bitmask(matcher, bitmask)
            mask_seconds += time.perf_counter() - mask_start
            apply_token_bitmask_inplace(next_token_logits, bitmask)

            next_token_id = int(np.argmax(next_token_logits[0]))
            if next_token_id in self.eos_token_ids:
                stop_reason = "eos"
                break

            if not matcher.consume_token(next_token_id):
                stop_reason = "matcher_rejected"
                break

            generated_ids.append(next_token_id)
            if matcher.is_stopped():
                stop_reason = "grammar_complete"
                break

            total_length += 1
            logits, past = self._run_step([next_token_id], past, total_length)

        decode_seconds = time.perf_counter() - decode_start

        return GenerationResult(
            text=self.tokenizer.decode(generated_ids),
            token_count=len(generated_ids),
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            mask_seconds=mask_seconds,
            stop_reason=stop_reason,
        )
