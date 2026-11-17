"""Detection: one statistic, computed from the suspect text and the writer's key.

The test generalises the usual binary coin-flip form to classes of different sizes. A slot
with ``n`` members matches the key's demand by chance with probability ``1/n``, so the null
is a sum of non-identical Bernoullis rather than a fair coin:

    expected  = sum(1 / n_i)
    variance  = sum((1 / n_i) * (1 - 1 / n_i))
    z         = (hits - expected) / sqrt(variance)

which reduces to the familiar ``(2h - n) / sqrt(n)`` when every class is a pair. Using the
plain binomial on a table with 3-member classes would overstate the evidence, and
overstating evidence about a person is the failure this project cannot afford.
"""

from __future__ import annotations

from dataclasses import replace

from .decisions import Decision, normal_tail
from .ledger import DEFAULT_WINDOW, walk, word_count
from .providers import CuratedTable, default_table

MIN_SLOTS = 8  # below this, coin flips simply cannot carry evidence, whatever the p-value says


def score_entries(entries, *, words: int, key, n_hypotheses: int = 1, sidecar=None) -> Decision:
    """The single statistic, shared by the curated and the model-gated paths.

    Two implementations of "how surprising is this agreement" is how a project ends up
    quoting two different p-values for the same essay.
    """
    expected = sum(1.0 / e.n for e in entries)
    variance = sum((1.0 / e.n) * (1 - 1.0 / e.n) for e in entries)
    hits = sum(1 for e in entries if e.on_key)
    z = (hits - expected) / variance**0.5 if variance > 0 else 0.0

    tiers: dict[str, int] = {}
    for entry in entries:
        tiers[entry.tier] = tiers.get(entry.tier, 0) + 1

    caveats: list[str] = [
        ("Detects an intact or lightly edited copy. Measured prior art puts full paraphrase "
        "and translation at zero survival -- the words carry the mark, so rewriting the words "
        "removes it."),
    ]
    if sidecar is None:
        caveats.insert(
            0,
            "Computed from the text alone: no sidecar, so the verifier re-derived which "
            "choices were free. A sidecar from publication time is strictly stronger.",
        )
    if len(entries) < MIN_SLOTS:
        caveats.append(
            f"Only {len(entries)} key-correlated choices in {words} words; below {MIN_SLOTS} "
            "the statistic is noise and must not be quoted."
        )
    if variance <= 0:
        caveats.append("No eligible choices at all; nothing to test.")
    if getattr(key, "revoked", False):
        caveats.append("key_revoked")
    if sidecar is not None:
        recorded = len(sidecar.get("slots", []))
        caveats.insert(
            0,
            f"Verified against the sidecar's publication-time ledger: {len(entries)} of the "
            f"{recorded} recorded slots re-derive from this copy. No equivalence table or "
            "model was needed.",
        )
    if n_hypotheses > 1:
        caveats.append(
            f"{n_hypotheses} keys were tested against this text, so the corrected p-value is "
            "the one to quote: scanning a registry of a million writers at p<1e-6 per test "
            "produces about one false accusation per scan."
        )

    return Decision(
        slots=len(entries),
        hits=hits,
        expected=expected,
        variance=variance,
        z=z,
        p_value=normal_tail(z) if variance > 0 else 1.0,
        key_id=key.key_id,
        fingerprint=key.fingerprint,
        tier_counts=tiers,
        words=words,
        n_hypotheses=n_hypotheses,
        caveats=tuple(caveats),
    )


def scan(
    text: str,
    key,
    *,
    table: CuratedTable | None = None,
    window: int = DEFAULT_WINDOW,
    n_hypotheses: int = 1,
    sidecar: dict | None = None,
) -> Decision:
    """Tier-1 detection over the curated table. See `scan_lm` for the model-gated path."""
    table = table or default_table()
    entries = walk(text, key, table, window=window, mode="detect")
    return score_entries(
        entries, words=word_count(text), key=key, n_hypotheses=n_hypotheses, sidecar=sidecar
    )


def scan_against(text: str, keys, **kwargs) -> Decision:
    """Best-of-several-keys, with the multiplicity correction applied instead of hidden.


    Returns the strongest-looking result but always reports how many keys were tested,
    because a registry lookup is exactly where uncorrected p-values start accusing people.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("no keys given")
    best = max(
        (scan(text, key, n_hypotheses=len(keys), **kwargs) for key in keys),
        key=lambda d: d.z,
    )
    if any(getattr(key, "revoked", False) for key in keys):
        best = replace(best, caveats=(*best.caveats, "key_revoked"))
    return best


__all__ = ["MIN_SLOTS", "scan", "scan_against", "score_entries"]
