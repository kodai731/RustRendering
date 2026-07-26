"""Polarity tie-break: keyword evidence applied to a near-tie, not to routing.

The rule table this replaces returned a route outright, which bypassed the
similarity threshold and let a wrong keyword match execute unguarded. A tie-break
can only reorder routes the encoder already ranked first and second, so its worst
case is a swap between two routes that were within `margin` of each other anyway.

A group is a set of routes that name opposite ends or degrees of one axis — the
confusions the held-out evaluation actually shows: start/end, next_key/prev_key,
pause/stop, undo/redo, show/hide, slow/normal/fast. Categorical enums are
deliberately excluded even though the schema shape looks identical: the encoder
separates walk from jump reliably, so a tie-break there can only ever do harm,
and it did — a rare conjugation fragment (`生成し`) overturned a correct
`generate_motion:run` across a 0.93 margin before these groups were narrowed.

Terms are not hand-written: `derive_polarity_terms.py` keeps only the terms one
route's exemplars use and no other route's do, so the table cannot encode
anything the exemplars do not already say.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

TABLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "orchestrator"
    / "data"
    / "polarity_groups.json"
)

POLARITY_GROUPS = {
    "seek_bound": ("seek_time:start", "seek_time:end"),
    "seek_key_direction": ("seek_time:next_key", "seek_time:prev_key"),
    "playback_halt": ("pause_animation", "stop_animation"),
    "edit_history": ("undo", "redo"),
    "visibility": ("set_object_visibility:show", "set_object_visibility:hide"),
    "playback_degree": (
        "set_playback_speed:slow",
        "set_playback_speed:normal",
        "set_playback_speed:fast",
    ),
}

LATIN_TOKEN = re.compile(r"[a-z0-9]+")
CJK_RUN = re.compile(r"[぀-ヿ一-鿿]+")
CJK_IDEOGRAPH = re.compile(r"[一-鿿]")

MIN_LATIN_TOKEN_LENGTH = 3
CJK_GRAM_LENGTHS = (2, 3)

# A pole is never named by a conjunction, a determiner, a pronoun or an auxiliary:
# English expresses direction and degree with prepositions, adverbs and
# comparatives. Those word classes stay, so `before`, `ahead`, `back` and `forward`
# remain available; the classes below cannot carry a pole yet still pass the
# exclusivity test whenever one exemplar was the only one to use them. Held-out had
# `but` deciding pause-over-stop, which is the same accident as `生成し` overturning
# a correct `generate_motion:run` — right answer, reason that will not repeat.
#
# A general-purpose stopword list is the wrong tool here: NLTK's English list drops
# `before`, `after`, `over` and `under`, which are exactly this domain's poles.
CONJUNCTIONS = "and but nor yet because although while whereas unless whether"
DETERMINERS = "the this that these those any all some each every both another such"
PRONOUNS = "you your yours his her hers its our ours they them their theirs there here"
INTERROGATIVES = "what which who whom whose"
AUXILIARIES = (
    "is are was were been being have has had having "
    "does did doing don dont doesn didn can could will would should must"
)

LATIN_FUNCTION_WORDS = frozenset(
    f"{CONJUNCTIONS} {DETERMINERS} {PRONOUNS} {INTERROGATIVES} {AUXILIARIES}".split()
)


def extract_candidate_terms(utterance: str) -> set[str]:
    terms = {
        token
        for token in LATIN_TOKEN.findall(utterance)
        if len(token) >= MIN_LATIN_TOKEN_LENGTH and token not in LATIN_FUNCTION_WORDS
    }

    for run in CJK_RUN.findall(utterance):
        terms.update(CJK_IDEOGRAPH.findall(run))
        for length in CJK_GRAM_LENGTHS:
            terms.update(run[start : start + length] for start in range(len(run) - length + 1))
    return terms


def count_term_support(utterances: list[str]) -> dict[str, int]:
    support: dict[str, int] = {}
    for utterance in utterances:
        for term in extract_candidate_terms(utterance.lower()):
            support[term] = support.get(term, 0) + 1
    return support


def derive_group_terms(
    members: tuple[str, ...], support_by_route: dict[str, dict[str, int]], min_support: int
) -> dict[str, list[str]]:
    """A term qualifies only if no other route in the whole corpus ever uses it.

    Contrasting a route against its siblings alone is not enough: it admits any
    filler word the siblings happened not to use, and those dominate. `and`,
    `animation` and `生成し` all passed the sibling test, and `生成し` then
    overturned a correct `generate_motion:run` on a 0.93 margin. Requiring
    corpus-wide exclusivity is what makes a term mean the route rather than the
    phrasing one exemplar happened to take.
    """
    derived: dict[str, list[str]] = {}
    for route in members:
        others = [other for other in support_by_route if other != route]
        derived[route] = [
            term
            for term, count in sorted(support_by_route[route].items())
            if count >= min_support
            and all(term not in support_by_route[other] for other in others)
        ]
    return derived


def derive_table(exemplars: list[dict], min_support: int) -> dict:
    exemplars_by_route: dict[str, list[str]] = {}
    for row in exemplars:
        exemplars_by_route.setdefault(row["route"], []).append(row["utterance"])

    support_by_route = {
        route: count_term_support(utterances)
        for route, utterances in exemplars_by_route.items()
    }

    return {
        "min_support": min_support,
        "groups": {
            name: derive_group_terms(members, support_by_route, min_support)
            for name, members in POLARITY_GROUPS.items()
        },
    }


@dataclass(frozen=True)
class TieBreak:
    winner: str
    swapped: bool


class PolarityTieBreaker:
    """Promotes the runner-up when the utterance names its pole and not the top's.

    There is no similarity margin to tune. A margin was measured across
    0.00–1.01 and never prevented a wrong swap that the two structural
    conditions had not already prevented, so it would have been a threshold
    carrying no decision. What bounds the mechanism instead is that the pair must
    be declared opposite poles of one axis, and that the deciding term must be one
    no other route's exemplars use.
    """

    def __init__(self, table: dict):
        self.terms_by_route: dict[str, list[str]] = {}
        self.group_of_route: dict[str, str] = {}
        self.members_by_axis: dict[str, list[str]] = {}

        for group_name, members in table["groups"].items():
            self.members_by_axis[group_name] = list(members)
            for route, terms in members.items():
                self.terms_by_route[route] = terms
                self.group_of_route[route] = group_name

    def count_evidence(self, normalized: str, route: str) -> int:
        return sum(term in normalized for term in self.terms_by_route.get(route, ()))

    def shares_axis(self, first: str, second: str) -> bool:
        group = self.group_of_route.get(first)
        return group is not None and group == self.group_of_route.get(second)

    def resolve(self, normalized: str, ranked: list[tuple[str, float]]) -> TieBreak:
        top, runner_up = ranked[0][0], ranked[1][0]
        if not self.shares_axis(top, runner_up):
            return TieBreak(top, swapped=False)

        if self.count_evidence(normalized, runner_up) > self.count_evidence(normalized, top):
            return TieBreak(runner_up, swapped=True)
        return TieBreak(top, swapped=False)


def load_tie_breaker(table_path: Path = TABLE_PATH) -> PolarityTieBreaker:
    return PolarityTieBreaker(json.loads(table_path.read_text(encoding="utf-8")))
