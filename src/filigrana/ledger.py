"""The ledger: the one walk that both the embedder and the detector run.

Everything about reproducibility lives here. Embedding and detection must agree, slot for
slot, on the class, the candidate order and the key's demand -- and they only can if they share
the traversal instead of each reimplementing it.

Two modes:

``embed``  the final value of a slot is the key's demand, unless the writer vetoes it.
``detect`` every slot's value is what the text actually says.

The context each demand is keyed off is `anchor_before`: the handful of words immediately
preceding the slot, in whatever form they end up in. Because the walk runs left to right,
every anchor word is already final when its slot is decided, so embed and a verifier reading a
scraped copy compute the same string without either seeing the other's draft -- and losing a
sentence costs only the slots within `window` words of it, not the rest of the document.
"""

from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

from .canon import Token, tokenize
from .keystream import keyed_choice
from .providers import CuratedTable, default_table

DEFAULT_WINDOW = 4


@dataclass(frozen=True, slots=True)
class Entry:
    class_id: str
    tier: str
    start: int
    end: int
    members: tuple[str, ...]
    observed: int
    wanted: int
    final: int
    bit: int | None
    vetted: bool
    anchor: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return len(self.members)

    @property
    def on_key(self) -> bool:
        return self.final == self.wanted


def anchor_before(
    tokens, index: int, window: int, reachable: Set[str] = frozenset()
) -> tuple[str, ...]:
    """The `window` word tokens before `index` that the watermark itself cannot rewrite.

    Skipping reachable words is not a refinement, it is what makes the anchor reproducible:
    an ordinary word near a slot is guaranteed to still say the same thing after embedding,
    while the slot's own word is exactly what just changed. Anchoring on raw neighbours made
    embed and detect disagree anywhere two slots fell within one window of each other -- 12 of
    84 slots in `tests/test_statistics.py`, which is a silent, damning failure.

    `reachable` must therefore be the *same set* at both ends. The curated path passes the
    table's own member tokens, the gated path the bands it just discovered, and sidecar
    verification reads it back out of the sidecar rather than re-deriving it.

    Word-level and model-free throughout: a verifier needs the text, the key, and nothing
    else.
    """
    out: list[str] = []
    for token in reversed(tokens[:index]):
        if token.kind == "word" and token.norm not in reachable:
            out.append(token.norm)
            if len(out) >= window:
                break
    return tuple(reversed(out))


def walk(
    text: str,
    key,
    table: CuratedTable | None = None,
    *,
    window: int = DEFAULT_WINDOW,
    bit_source=None,
    mode: str = "detect",
    veto=None,
    tokens: list[Token] | None = None,
) -> list[Entry]:
    """Traverse `text` and return every slot with the key's demand at that slot.

    ``bit_source(slot_index)`` supplies the payload bit a slot carries, so each payload bit
    collects its own independent votes instead of every slot fighting over the same one.
    ``veto(slot, wanted)`` returning True keeps the writer's original word and knowingly
    spends one statistical vote to do it.
    """
    table = table or default_table()
    toks = tokens if tokens is not None else tokenize(text)
    if not toks:
        return []

    entries: list[Entry] = []
    for index, slot in enumerate(table.slots(text, toks)):
        bit = bit_source(index) if bit_source is not None else None
        anchor = anchor_before(toks, slot.token_index, window, table.reachable)
        wanted = keyed_choice(
            key.secret,
            slot.class_id,
            slot.n,
            anchor,
            window=window,
            key_id=key.key_id,
            bit=bit,
        )
        vetted = mode == "embed" and veto is not None and veto(slot, wanted)
        final = slot.observed if (mode != "embed" or vetted) else wanted
        entries.append(
            Entry(
                class_id=slot.class_id,
                tier=slot.tier,
                start=slot.start,
                end=slot.end,
                members=slot.members,
                observed=slot.observed,
                wanted=wanted,
                final=final,
                bit=bit,
                vetted=vetted,
                anchor=anchor,
            )
        )
    return entries


def word_count(text: str) -> int:
    return sum(1 for t in tokenize(text) if t.kind == "word")


__all__ = ["DEFAULT_WINDOW", "Entry", "anchor_before", "walk", "word_count"]
