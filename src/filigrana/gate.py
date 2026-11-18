"""The free-choice gate: a bounded test, applied to candidates somebody else proposed.

Anthropic's explainer says the watermark only acts where either choice would be equally
good. A writer has no logits, so `filigrana` recovers that condition after the fact: score the
draft with a small causal model and keep a position only when the writer's word sits on a
**plateau** -- within `epsilon` nats of the best alternative, with at least one other
admissible candidate in the same band.

Two things this deliberately does not do, both learned the hard way:

* **It does not invent the candidate set.** A model's global top-k continuations of "The
  engineer" are `said`, `,`, `'s` -- plausible next tokens, none of them substitutable for
  `walked`. Proposals come from a provider (`CuratedTable`, `WordNetProvider`); the gate only
  decides which proposals are genuinely free in context. Getting this backwards made the
  first version discover zero slots on real prose, which is at least a loud failure.
* **It does not "improve" your text.** If the model rates the writer's word below the whole
  band, the position is skipped rather than upgraded. Otherwise embedding quietly rewrites the
  draft toward a 124M-parameter model's taste, and the mark becomes style transfer with
  statistics attached.

Because the old word and the new one are inside one epsilon-wide band, the log-probability
cost of every accepted edit is *bounded by* `epsilon` per edit. The quality cost is a number
you set, not one you discover after publishing.

Verification keeps its shape: the key's demand is computed from *anchor words* (see
`keystream`) and the provider's member list, never from a logit. With a sidecar the model is
not needed to check a copy at all. Re-deriving slots without one does need the same model,
and changing a word at position i shifts the distribution at i+1, so discovery is stabilised
over a few passes and **retention** is reported rather than assumed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .providers import Slot

WORDISH = re.compile(r"^[A-Za-z][A-Za-z’'-]{2,}$")

#: A scored surface that is not a substitutable word: subword tail, byte fallback, control
#: token. Accepting these would mark mojibake and call it a watermark.
NON_WORD = ("Ġ", "▁", "=", "<", "[", "]", "http", "www")

#: `logprob_of(token_index, members)` -> conditional log-probability of each member at that
#: position given the text before it, aligned with `members`. `-inf` means the model cannot
#: place that surface there at all (multi-subword words, unknown spellings).
LogprobOf = Callable[[int, tuple[str, ...]], tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class GateSlot:
    """A proposal that survived the band test. `members` is the band, not the proposal."""

    slot: Slot
    members: tuple[str, ...]
    observed: int
    mismatch: float
    mass: float

    @property
    def class_id(self) -> str:
        return self.slot.class_id

    @property
    def tier(self) -> str:
        return self.slot.tier

    @property
    def start(self) -> int:
        return self.slot.start

    @property
    def end(self) -> int:
        return self.slot.end

    @property
    def token_index(self) -> int:
        return self.slot.token_index

    @property
    def n(self) -> int:
        return len(self.members)


@dataclass(frozen=True, slots=True)
class GateReport:
    slots: tuple[GateSlot, ...]
    unencodable: int
    dominated: int
    thin: int

    def summary(self) -> str:
        return (
            f"{len(self.slots)} free choices kept | dropped {self.dominated} where your word "
            f"dominates, {self.thin} with too narrow a band, {self.unencodable} the model "
            "cannot place"
        )


class FreeChoiceGate:
    """Keeps proposals whose alternatives are near-equal in probability.

    `epsilon` is in nats against the best member of the proposal; `min_band` is the floor
    under which a candidate is too unlikely to belong in the band at all. Past `max_members`
    the tail of a distribution is where meaning quietly starts to move.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.35,
        min_band: float = 0.002,
        max_members: int = 4,
        min_members: int = 2,
        min_word_len: int = 4,
        forbid: frozenset[str] = frozenset(),
    ) -> None:
        if not 0 < min_members <= max_members:
            raise ValueError("need min_members <= max_members")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive; 0 admits no plateau at all")
        self.epsilon = epsilon
        self.min_band = min_band
        self.max_members = max_members
        self.min_members = min_members
        self.min_word_len = min_word_len
        self.forbid = forbid

    def apply(self, tokens: Sequence, slots: Sequence[Slot], logprob_of: LogprobOf) -> GateReport:
        kept: list[GateSlot] = []
        dominated = thin = unencodable = 0
        floor = math.log(self.min_band)
        for slot in slots:
            token = tokens[slot.token_index]
            if token.kind != "word" or len(token.surface) < self.min_word_len:
                thin += 1
                continue
            if not WORDISH.match(token.surface) or token.norm in self.forbid:
                thin += 1
                continue
            if any(part in token.surface for part in NON_WORD):
                thin += 1
                continue
            scores = tuple(logprob_of(slot.token_index, slot.members))
            usable = [
                (member, lp)
                for member, lp in zip(slot.members, scores)
                if math.isfinite(lp) and lp >= floor
            ]
            if len(usable) < self.min_members:
                unencodable += 1
                continue
            best = max(lp for _m, lp in usable)
            band = tuple(sorted(m for m, lp in usable if best - lp <= self.epsilon))
            if len(band) < self.min_members:
                thin += 1
                continue
            observed = token.norm
            if observed not in band:
                dominated += 1
                continue
            band = band[: self.max_members]
            p_observed = next(lp for m, lp in usable if m == observed)
            kept.append(
                GateSlot(
                    slot=slot,
                    members=band,
                    observed=band.index(observed),
                    mismatch=1.0 - math.exp(min(0.0, p_observed - best)),
                    mass=sum(math.exp(lp) for _m, lp in usable),
                )
            )
        return GateReport(tuple(kept), unencodable, dominated, thin)


__all__ = ["NON_WORD", "WORDISH", "FreeChoiceGate", "GateReport", "GateSlot", "LogprobOf"]
