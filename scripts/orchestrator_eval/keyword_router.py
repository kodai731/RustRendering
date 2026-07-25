"""Deterministic keyword routing, driven by the same table as the engine build.

The rule table is `src/orchestrator/data/keyword_router_rules.json`, which
`src/orchestrator/systems/keyword_router.rs` embeds with `include_str!`. Reading that
one file from both sides is what makes the measured numbers describe the shipped
behaviour instead of a Python re-implementation that drifted.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from route_schema import RouteSpec

RULE_TABLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "orchestrator"
    / "data"
    / "keyword_router_rules.json"
)

FULLWIDTH_FIRST = 0xFF01
FULLWIDTH_LAST = 0xFF5E
FULLWIDTH_TO_ASCII_OFFSET = 0xFEE0
IDEOGRAPHIC_SPACE = "　"


def to_half_width(character: str) -> str:
    if character == IDEOGRAPHIC_SPACE:
        return " "
    code = ord(character)
    if FULLWIDTH_FIRST <= code <= FULLWIDTH_LAST:
        return chr(code - FULLWIDTH_TO_ASCII_OFFSET)
    return character


def normalize_utterance(utterance: str) -> str:
    widened = "".join(to_half_width(character) for character in utterance)
    return " ".join(widened.lower().split())


@dataclass(frozen=True)
class KeywordRoutingOutcome:
    normalized: str
    speed: str | None
    decided_route: str | None
    candidates: tuple[str, ...]


class KeywordRouter:
    def __init__(self, routes: list[RouteSpec], table_path: Path = RULE_TABLE_PATH):
        table = json.loads(table_path.read_text(encoding="utf-8"))
        self.interrogative_trailing = table["interrogative_trailing"]
        self.interrogative_contains = table["interrogative_contains"]
        self.speed_modifiers = table["speed_modifiers"]
        self.rules = table["rules"]

        self.route_kinds = {route.route_id: route.kind for route in routes}
        self._verify_rule_routes_exist()

    def _verify_rule_routes_exist(self) -> None:
        unknown = [rule["route"] for rule in self.rules if rule["route"] not in self.route_kinds]
        if unknown:
            raise ValueError(f"rules reference unknown routes: {unknown}")

    def is_interrogative(self, normalized: str) -> bool:
        if any(normalized.endswith(marker) for marker in self.interrogative_trailing):
            return True
        return any(marker in normalized for marker in self.interrogative_contains)

    def extract_speed(self, normalized: str) -> str | None:
        for entry in self.speed_modifiers:
            if any(term in normalized for term in entry["terms"]):
                return entry["preset"]
        return None

    def narrow_candidates(self, normalized: str) -> tuple[str, ...]:
        if not self.is_interrogative(normalized):
            return tuple(self.route_kinds)
        return tuple(
            route_id for route_id, kind in self.route_kinds.items() if kind == "query"
        )

    def find_matching_rule(
        self, normalized: str, candidates: tuple[str, ...], interrogative: bool
    ) -> str | None:
        for rule in self.rules:
            if rule["route"] not in candidates:
                continue
            if rule.get("requires_interrogative", False) and not interrogative:
                continue
            if any(term in normalized for term in rule.get("forbidden", [])):
                continue
            if all(
                any(term in normalized for term in group)
                for group in rule.get("required", [])
            ):
                return rule["route"]
        return None

    def route(self, utterance: str) -> KeywordRoutingOutcome:
        normalized = normalize_utterance(utterance)
        interrogative = self.is_interrogative(normalized)
        candidates = self.narrow_candidates(normalized)
        return KeywordRoutingOutcome(
            normalized=normalized,
            speed=self.extract_speed(normalized),
            decided_route=self.find_matching_rule(normalized, candidates, interrogative),
            candidates=candidates,
        )
