"""Utterance normalization, mirroring src/orchestrator/systems/normalize.rs.

Deliberately narrow: full-width to half-width, ASCII lowercase, whitespace
collapse. No stemming and no Japanese segmentation, because the polarity terms are
derived against this exact output and anything cleverer would make a derived term
mean something different in Rust than it did when it was measured.
"""

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
