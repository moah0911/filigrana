"""Stage 3 measurement: what the model gate buys, and what it costs.

Four configurations, same articles, same key schedule:

  curated                the hand-audited table alone (the 8.3 / 1000 words baseline)
  curated+gate           the table, but only positions the model calls a plateau
  wordnet+gate           WordNet proposals filtered by the model
  wordnet alone          refused by design; shown here to prove why

For each we report rho (slots per 1,000 words), retention (how many slots still re-derive
from the marked copy with no sidecar), edits per 1,000 words, the mean conditional log-prob
shift (the style cost), and detection at p<1e-6.

The two numbers that decide whether any of this is usable are retention and the log-prob
shift. Density that does not survive its own embedding is not density, and a mark that costs
prose is a mark writers will not put on their work.

    uv run python tools/measure_lm_gate.py --corpus corpus --articles 10 --words 1075
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from filigrana.detect import scan
from filigrana.embed import embed
from filigrana.gate import FreeChoiceGate
from filigrana.keyring import WriterKey
from filigrana.lexical import WordNetProvider
from filigrana.lm import embed_gated, scan_gated
from filigrana.providers import default_table
from filigrana.sidecar import verify


def sample_articles(corpus: Path, count: int, words: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    files = sorted(corpus.glob("*.txt"))
    out = []
    for _ in range(count):
        parts = rng.choice(files).read_text(encoding="utf-8", errors="replace").split()
        start = rng.randrange(0, max(1, len(parts) - words))
        out.append(" ".join(parts[start : start + words]))
    return out


def row(label, zs, retentions, edits, words, slots, shifts, proposed=None, extra=None) -> dict:
    detected = sum(z >= 4.754 for z in zs) / max(len(zs), 1)
    return {
        "config": label,
        "rho": round(1000 * statistics.mean(slots) / words, 1),
        # The funnel. `rho_proposed` is what the lexicon offers, `rho_gated` is what survives
        # the band test, `rho` is what is left after embedding. Collapsing these three into one
        # number is how a project discovers too late that its filter is the bottleneck.
        "rho_proposed": round(1000 * statistics.mean(proposed) / words, 1) if proposed else None,
        "edits_per_1000w": round(1000 * statistics.mean(edits) / words, 1),
        "retention": round(statistics.mean(retentions), 3) if retentions else None,
        "mean_logprob_shift": round(statistics.mean(shifts), 4) if shifts else None,
        "mean_z": round(statistics.mean(zs), 2),
        "detected_p1e6": round(detected, 3),
        **(extra or {}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--articles", type=int, default=10)
    ap.add_argument("--words", type=int, default=1075)
    ap.add_argument("--epsilons", default="0.35,0.7")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", default="bench/lm_gate.json")
    args = ap.parse_args(argv)

    texts = sample_articles(Path(args.corpus), args.articles, args.words, args.seed)
    keys = [WriterKey.generate() for _ in texts]
    rows = []

    for tier in ("ab", "all"):
        table = default_table(tier)
        results = [embed(t, k, table=table) for t, k in zip(texts, keys)]
        decisions = [scan(r.text, k, table=table) for r, k in zip(results, keys)]
        rows.append(
            row(
                f"curated:{tier}",
                [d.z for d in decisions],
                [1.0] * len(results),
                [len(r.edits) for r in results],
                args.words,
                [r.slots for r in results],
                [0.0] * len(results),
                extra={
                    "p_value_via_sidecar": [
                        f"{verify(r.text, k, r.sidecar()).p_value:.2g}"
                        for r, k in zip(results, keys)
                    ][:3]
                },
            )
        )

    from filigrana.lm import LocalModelScorer

    scorer = LocalModelScorer(args.model)
    for epsilon in (float(x) for x in args.epsilons.split(",")):
        gate = FreeChoiceGate(epsilon=epsilon)
        for name, provider in (
            ("curated+gate", default_table("all")),
            ("wordnet+gate", WordNetProvider()),
        ):
            results, decisions, zs, ret, edits, shifts, slots, proposed = [], [], [], [], [], [], [], []
            for text, key in zip(texts, keys):
                scoring = scorer.score(text)
                proposed.append(len(provider.slots(text, scoring.tokens)))
                marked = embed_gated(text, key, provider, scorer, gate=gate)
                decision = scan_gated(marked.text, key, provider, scorer, gate=gate)
                zs.append(decision.z)
                ret.append(marked.retained)
                edits.append(marked.edits_applied)
                shifts.append(marked.logprob_shift)
                slots.append(marked.slots)
            rows.append(
                row(
                    f"{name}:eps{epsilon}",
                    zs,
                    ret,
                    edits,
                    args.words,
                    slots,
                    shifts,
                    proposed,
                    {"model": args.model, "epsilon": epsilon},
                )
            )
            print(
                f"{name:<14} eps {epsilon:<5} prop {rows[-1]['rho_proposed']:>6} rho {rows[-1]['rho']:>6}  retained "
                f"{rows[-1]['retention']:>6.1%}  edits/1k {rows[-1]['edits_per_1000w']:>6}  "
                f"dlogp {rows[-1]['mean_logprob_shift']:>+8}  z {rows[-1]['mean_z']:>6}  "
                f"det@1e-6 {rows[-1]['detected_p1e6']:>5}"
            )

    print()
    for r in rows:
        if not r["config"].startswith("curated:"):
            continue
        print(
            f"{r['config']:<14} eps none    rho {r['rho']:>6}  retained {1.0:>6.1%}  "
            f"edits/1k {r['edits_per_1000w']:>6}  dlogp {0.0:>+8}  z {r['mean_z']:>6}  "
            f"det@1e-6 {r['detected_p1e6']:>5}"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"model": args.model, "articles": args.articles, "words": args.words, "rows": rows}, indent=1),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")
    print("Compare rho against docs/CAPACITY.md. If retention is far below 1, the honest")
    print("capacity number is the gated rho, not the number of positions the lexicon knows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
