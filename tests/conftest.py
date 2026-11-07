"""Shared scripted scorer for the gate tests.

Kept in conftest rather than duplicated: the script IS the test double for a model, so both
the gate tests and the sidecar tests have to agree on exactly what it says is free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from inkmark.canon import tokenize
from inkmark.keyring import WriterKey
from inkmark.providers import Slot

KEY = WriterKey(secret=b"4" * 32, key_id="gate-test", created="", year=2026)

# Each entry is a synonym GROUP mapped to conditional log-probabilities. Probabilities attach
# to the group rather than to the word on the page, because that is what a real model does:
# after the mark swaps "walked" for "strolled", the position must still look like a plateau, or
# the script is claiming a model that forgets what it just read. Words in no group are spikes:
# one plausible continuation, which is what content feels like.
SCRIPT: dict[str, dict[str, float]] = {
    "walked": {"walked": -1.2, "strolled": -1.3, "marched": -4.0},
    "station": {"station": -1.0, "depot": -1.05, "terminal": -3.5},
    "grey": {"grey": -0.9, "gray": -0.95, "dull": -5.0},
    "slowly": {"slowly": -0.4, "quietly": -3.2},
    "crowd": {"crowd": -0.2, "throng": -6.0},
}

GROUP_OF: dict[str, dict[str, float]] = {
    word: spec for spec in SCRIPT.values() for word in spec
}

SENTENCE = "she walked slowly toward the grey station and the crowd parted"


@dataclass
class FakePass:
    """Whatever a scorer hands back: enough to place candidates, nothing more."""

    tokens: list
    text: str

    def logprob_of(self, index: int, members: tuple[str, ...]) -> tuple[float, ...]:
        word = self.tokens[index].surface.lower()
        spec = GROUP_OF.get(word)
        if spec is None:
            return tuple(-math.inf if m != word else -0.01 for m in members)
        # After "he" the field collapses onto one word. This only becomes reachable once an
        # earlier slot has been rewritten, which is exactly the case the loop must survive.
        prev = self.tokens[index - 1].surface.lower() if index else ""
        if prev == "he":
            return tuple(-math.inf if m != word else -0.02 for m in members)
        return tuple(spec.get(m, -8.0) for m in members)

    def anchors(self, reachable):
        indices, norms = [], []
        for i, token in enumerate(self.tokens):
            if token.kind == "word" and token.norm not in reachable:
                indices.append(i)
                norms.append(token.norm)
        return indices, norms


class ScriptedScorer:
    model_id = "scripted/1"

    def score(self, text: str) -> FakePass:
        return FakePass(tokens=tokenize(text), text=text)


class ScriptedProvider:
    """Proposals straight from the script, so the gate is tested apart from any lexicon."""

    requires_gate = True

    def __init__(self, script):
        self.script = script

    def slots(self, text, tokens=None):

        toks = list(tokens) if tokens is not None else tokenize(text)
        out = []
        for i, token in enumerate(toks):
            spec = next((g for g in self.script.values() if token.surface.lower() in g), None)
            if not spec:
                continue
            members = tuple(sorted(m for m, lp in spec.items() if lp > -6.0))
            if token.surface.lower() not in members or len(members) < 2:
                continue
            out.append(
                Slot(
                    class_id=f"sc:{members[0]}",
                    tier="lm",
                    start=token.start,
                    end=token.end,
                    token_index=i,
                    n_tokens=1,
                    members=members,
                    observed=members.index(token.surface.lower()),
                    matched=(token.surface,),
                )
            )
        return out


