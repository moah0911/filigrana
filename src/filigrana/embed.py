"""Embedding: propose the writer's key as a set of edits, never silently rewrite a draft.

`embed()` returns a result, it does not touch the filesystem, and by default the caller
sees every proposed change before anything is written. That is not politeness: the
automated-synonym version of this technique measurably made 100% of passages less fluent
and gave 17% of them a new grammatical error, so an unreviewed pass is a way to damage
writing at scale.

Embedding then re-reads its own output in detect mode and reports *drift* -- slots whose
ledger position moved because a substitution changed the surrounding token stream. Drift is
the failure mode that would make a mark unverifiable in practice, so it is measured rather
than assumed away.
"""


from dataclasses import dataclass

from .canon import relabel
from .ledger import DEFAULT_WINDOW, Entry, walk, word_count
from .providers import CuratedTable, default_table


def apply_diff(text: str, entries) -> str:
    """Splice keyed choices into `text` in one left-to-right pass.

    Shared with the model-gated path so a substitution can only ever mean one thing: the
    writer's casing survives, the rest of the string is untouched, and nothing outside a slot
    is ever rewritten.
    """
    pieces: list[str] = []
    cursor = 0
    for entry in entries:
        if entry.final == entry.observed:
            continue
        pieces.append(text[cursor : entry.start])
        pieces.append(_render(text[entry.start : entry.end], entry.members[entry.final]))
        cursor = entry.end
    pieces.append(text[cursor:])
    return "".join(pieces)


@dataclass(frozen=True, slots=True)
class Edit:
    class_id: str
    tier: str
    start: int
    end: int
    source: str
    replacement: str
    bit: int | None

    def render(self) -> str:
        arrow = f"{self.source} -> {self.replacement}"
        return f"[{self.tier}] {self.class_id:<22} {arrow}"


@dataclass(frozen=True, slots=True)
class EmbedResult:
    text: str
    edits: tuple[Edit, ...]
    entries: tuple[Entry, ...]
    key_id: str = ""
    words: int = 0
    slots: int = 0
    vetoed: int = 0
    drift: int = 0
    on_key: int = 0
    reachable: tuple[str, ...] = ()

    @property
    def agreement(self) -> float:
        return self.on_key / self.slots if self.slots else 0.0

    def sidecar(self, *, window: int = DEFAULT_WINDOW) -> dict:
        """The per-slot ledger. Ship it with a published article and verification stops
        needing to guess which choices were free -- and needs no model to check any of them.
        """
        from .sidecar import build_sidecar

        return build_sidecar(
            self.entries,
            key_id=self.key_id,
            window=window,
            reachable=self.reachable,
        )

    def summary(self) -> str:
        pct = 100 * len(self.edits) / max(self.words, 1)
        lines = [
            f"{self.words} words, {self.slots} eligible choices",
            f"{len(self.edits)} edits ({pct:.2f}% of words), {self.vetoed} declined by you",
            f"{self.on_key}/{self.slots} slots now agree with the key",
        ]
        if self.drift:
            lines.append(f"! {self.drift} slots drifted after substitution; re-detect to confirm")
        return "\n".join(lines)


def embed(
    text: str,
    key,
    *,
    table: CuratedTable | None = None,
    window: int = DEFAULT_WINDOW,
    n_bits: int = 0,
    payload: int | None = None,
    veto=None,
    disprefer=None,
) -> EmbedResult:
    table = table or default_table()
    if n_bits and payload is None:
        raise ValueError("n_bits set requires a payload")
    if payload is not None and n_bits and payload >= (1 << n_bits):
        raise ValueError("payload does not fit in n_bits")

    from .codec import bit_source as _bit_source

    # The raise above is what makes `payload` definite here, but say it so the type checker
    # can see the invariant instead of trusting that it holds.
    demand_bit = _bit_source(payload, n_bits) if n_bits and payload is not None else None
    entries = walk(
        text,
        key,
        table,
        window=window,
        mode="embed",
        veto=veto if disprefer is None else _style_guard(veto, disprefer),
        bit_source=demand_bit,
    )

    edits = tuple(
        Edit(e.class_id, e.tier, e.start, e.end, text[e.start : e.end], e.members[e.final], e.bit)
        for e in entries
        if e.final != e.observed
    )
    out = apply_diff(text, entries)

    # Re-read our own output the way a verifier will, with no sidecar. If the ledger moved,
    # the mark is weaker than it looks and the writer deserves to know before publishing.
    check = walk(out, key, table, window=window, mode="detect", bit_source=demand_bit)
    drift = abs(len(check) - len(entries)) + sum(
        1
        for before, after in zip(entries, check)
        if (before.class_id, before.final) != (after.class_id, after.observed)
    )
    on_key = sum(1 for e in check if e.on_key)
    return EmbedResult(
        key_id=key.key_id,
        text=out,
        edits=tuple(edits),
        entries=tuple(entries),
        reachable=tuple(sorted(table.reachable)),
        words=word_count(text),
        slots=len(entries),
        vetoed=sum(1 for e in entries if e.vetted),
        drift=drift,
        on_key=on_key,
    )


def _style_guard(veto, disprefer):
    """Keep the writer's word when the key would push it toward machine-sounding prose.

    Deliberately expensive: a vetoed slot is one the detector cannot count, so anti-slop
    trades measured statistical strength for style. `tools/detector_interference.py` reports
    the exchange rate instead of pretending it is free.
    """
    def guarded(slot, wanted):
        if disprefer(slot.members[wanted]) and not disprefer(slot.members[slot.observed]):
            return True
        return veto(slot, wanted) if veto is not None else False

    return guarded


def _render(source: str, replacement: str) -> str:
    """Preserve the leading capital of the slot, and the rest of the phrase lowercase."""
    first, _, _rest = source.partition(" ")
    new_first = relabel(first, replacement.split(" ")[0])
    tail = " ".join(replacement.split(" ")[1:])
    return f"{new_first} {tail}" if tail else new_first


__all__ = ["Edit", "EmbedResult", "embed"]
