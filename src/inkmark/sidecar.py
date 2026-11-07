"""The sidecar: publish what was free, so verifying it later needs no model.

A sidecar is not a copy of the article. It records, per slot, the class, the alternatives the
writer could have chosen, the choice made, and the short anchor of surrounding words that keys
the demand. Verification then needs neither the writer's draft nor the model that judged which
positions were free -- just the suspect text, the writer's key, and a hash. That is the
difference between a mark a publisher can check in ten milliseconds on a shared host and one
that requires them to install torch and match your model revision.

Matching is by anchor rather than by offset or index, because both of those move when someone
excerpt, reflow or re-paragraph the text. A slot whose neighbourhood was edited is *dropped*
rather than scored as a disagreement: an excerpt of your essay should look weaker, not
fabricated, and reporting it the other way is how provenance tools earn a bad reputation.
"""

from __future__ import annotations

from .canon import tokenize
from .decisions import Decision
from .detect import score_entries
from .keystream import keyed_choice
from .ledger import DEFAULT_WINDOW, Entry, anchor_before, word_count

FORMAT = "inkmark-sidecar/2"


def build_sidecar(
    entries,
    *,
    key_id: str,
    reachable,
    window: int = DEFAULT_WINDOW,
    model: str | None = None,
    provider: str = "CuratedTable",
    gate: dict | None = None,
    retained: float | None = None,
) -> dict:
    record = {
        "format": FORMAT,
        "key_id": key_id,
        "window": window,
        # Required, not defaulted: anchors skip the words the watermark could rewrite, so a
        # ledger that does not say which those were cannot be re-checked at all. Costs a few
        # hundred strings and buys verification that never has to guess which table, tier or
        # model was in use when the mark was made.
        "reachable": sorted(set(reachable) | {w for e in entries for w in " ".join(e.members).split()}),
        "provider": provider,
        "slots": [
            {
                "class": e.class_id,
                "tier": e.tier,
                "members": list(e.members),
                "final": e.final,
                "anchor": list(e.anchor),
                "bit": e.bit,
                "vetted": e.vetted,
            }
            for e in entries
        ],
    }
    if model:
        record["model"] = model
    if gate:
        record["gate"] = gate
    if retained is not None:
        record["retained"] = round(retained, 4)
    return record


def locate(text: str, sidecar: dict) -> list[Entry]:
    """Re-derive entries from a suspect copy using only the published ledger.

    Left-to-right, one cursor, so two identical slots in the text cannot both claim the same
    record and a reordered paragraph simply fails to match rather than scoring as evidence.
    """
    tokens = tokenize(text)
    window = sidecar.get("window", DEFAULT_WINDOW)
    reachable = frozenset(sidecar.get("reachable", ()))
    cursor = 0
    found: list[Entry] = []
    for record in sidecar.get("slots", ()):
        members = tuple(record["members"])
        anchor = tuple(record.get("anchor", ()))
        hit = _match(tokens, members, anchor, cursor, reachable, window)
        if hit is None:
            continue
        position, observed, width = hit
        cursor = position + width
        found.append(
            Entry(
                class_id=record["class"],
                tier=record.get("tier", "lm"),
                start=tokens[position].start,
                end=tokens[position + width - 1].end,
                members=members,
                observed=observed,
                wanted=-1,
                final=record["final"],
                bit=record.get("bit"),
                vetted=bool(record.get("vetted")),
                anchor=anchor,
            )
        )
    return found


def _match(tokens, members, anchor, cursor, reachable, window):
    """Next slot after `cursor` whose surface is one of `members` with exactly `anchor` before.

    The anchor is matched as literal words, which is what makes verification need no table, no
    lexicon and no model -- and leaves no second definition of "anchor" that could disagree
    with the one embedding used.

    Members are matched as *token sequences*, because the table is full of multi-word
    alternatives ("in spite of", "a number of"). Matching only the head word silently dropped
    every such slot from verification -- 11 of 108 in the fixture here -- which is the worst
    kind of bug in this project: the mark still tests strong, it just quietly stops counting
    the evidence that is there.
    """
    spans = [
        (index, member, tuple(part for part in member.split() if part))
        for index, member in enumerate(members)
    ]
    for i in range(cursor, len(tokens)):
        if tokens[i].kind != "word":
            continue
        if anchor_before(tokens, i, window, reachable) != anchor:
            continue
        for index, _member, parts in spans:
            if not parts or i + len(parts) > len(tokens):
                continue
            if tuple(tok.norm for tok in tokens[i : i + len(parts)]) == parts:
                return i, index, len(parts)
    return None


def verify(text: str, key, sidecar: dict, *, n_hypotheses: int = 1) -> Decision:
    """Check a suspect copy against a published ledger. No model, no downloads.

    `wanted` is recomputed from the anchor, so this is a real test of the key and not a
    restatement of what the sidecar already said.
    """
    from dataclasses import replace

    entries = []
    for entry in locate(text, sidecar):
        wanted = keyed_choice(
            key.secret,
            entry.class_id,
            entry.n,
            entry.anchor,
            window=sidecar.get("window", DEFAULT_WINDOW),
            key_id=key.key_id,
            bit=entry.bit,
        )
        entries.append(replace(entry, wanted=wanted))
    decision = score_entries(
        entries, words=word_count(text), key=key, n_hypotheses=n_hypotheses, sidecar=sidecar
    )
    recorded = len(sidecar.get("slots", ()))
    caveats = list(decision.caveats)
    caveats.insert(
        1,
        f"Verified from the published ledger: {decision.slots} of {recorded} slots at "
        "publication could still be located in this copy. Missing slots are treated as absent, "
        "not as disagreement, so an excerpt reads weaker rather than guilty.",
    )
    if sidecar.get("provider") not in (None, "CuratedTable"):
        caveats.append(
            f"Slots were discovered by {sidecar['provider']} gated on {sidecar.get('model', 'a model')}; "
            "re-discovering them without this sidecar needs that same model."
        )
    return _rebuild(decision, tuple(caveats))


def _rebuild(decision: Decision, caveats: tuple[str, ...]) -> Decision:
    from dataclasses import replace as _replace

    return _replace(decision, caveats=caveats)


__all__ = ["FORMAT", "build_sidecar", "locate", "verify"]
