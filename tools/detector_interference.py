"""Stage 6: does marking your own prose make it look machine-generated to someone else?

Paired comparison on the same articles -- original, marked, marked with the style guard --
scored on `filigrana.slop` metrics. Because every article appears in all three conditions the
comparison is paired, so we report the sign of the difference per article, not just means.

The honest framing: this measures a *proxy*. To settle the question you need matched marked
and unmarked copies run through Originality/Pangram/GPTZero, which nobody can do from an open
repository without an account and a budget. If you can run that, `--export-pairs` writes the
matched pairs so a third party can.

    uv run python tools/detector_interference.py --corpus corpus --articles 150
"""


import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from filigrana.embed import embed
from filigrana.keyring import WriterKey
from filigrana.providers import default_table
from filigrana.slop import is_slop, profile

KEYS = (
    ("burstiness", "sentence-length sd (lower reads more machine)"),
    ("ttr", "type-token ratio (lower reads more machine)"),
    ("long_rate", "share of long words"),
    ("cliches_per_1k", "cliche hits per 1000 words"),
    ("connective_rate", "formal sentence openers"),
    ("slop_rate", "filler words from the slop list"),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--articles", type=int, default=150)
    ap.add_argument("--words", type=int, default=1075)
    ap.add_argument("--tier", default="all")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--export-pairs", default=None, help="write matched JSON for a real detector")
    ap.add_argument(
        "--out", default=None, help="write the aggregate numbers as JSON next to the other bench files"
    )
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    table = default_table(args.tier)
    files = sorted(Path(args.corpus).glob("*.txt"))
    if not files:
        raise SystemExit("no corpus/*.txt")

    deltas: dict[str, list[float]] = {k: [] for k, _ in KEYS}
    guarded: dict[str, list[float]] = {k: [] for k, _ in KEYS}
    spent = 0
    total_slots = 0
    pairs = []

    for _ in range(args.articles):
        raw = rng.choice(files).read_text(encoding="utf-8", errors="replace")
        words = raw.split()
        start = rng.randrange(0, max(1, len(words) - args.words))
        article = " ".join(words[start : start + args.words])
        key = WriterKey.generate()

        plain = profile(article)
        marked = embed(article, key, table=table)
        careful = embed(article, key, table=table, disprefer=is_slop)
        mprof, cprof = profile(marked.text), profile(careful.text)

        total_slots += marked.slots
        spent += sum(1 for e in careful.entries if e.vetted)
        for name, _ in KEYS:
            deltas[name].append(getattr(mprof, name) - getattr(plain, name))
            guarded[name].append(getattr(cprof, name) - getattr(plain, name))
        if args.export_pairs:
            pairs.append({"original": article, "marked": marked.text})

    print(f"{args.articles} articles x {args.words} words, tier {args.tier}\n")
    print(f"{'metric':<16}{'marked vs original':>26}{'style guard':>26}   interpretation")
    for name, note in KEYS:
        d, g = deltas[name], guarded[name]
        wins = sum(1 for x in d if x > 0)
        print(
            f"{name:<16}{statistics.mean(d):>+26.5f}{statistics.mean(g):>+26.5f}"
            f"   up in {wins}/{len(d)} articles"
        )
    print(f"\nslots spent by the style guard: {spent}/{total_slots} "
          f"({100 * spent / max(total_slots, 1):.1f}% of the statistical strength)")
    print("The guard pays votes to keep prose human; the columns above are what it buys.")
    print("Not a commercial detector. See src/filigrana/slop.py before quoting any number here.")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "articles": args.articles,
                    "words": args.words,
                    "tier": args.tier,
                    "seed": args.seed,
                    "slots_spent_by_style_guard": spent,
                    "total_slots": total_slots,
                    "rows": [
                        {
                            "metric": name,
                            "marked_minus_original": round(statistics.mean(deltas[name]), 5),
                            "with_style_guard": round(statistics.mean(guarded[name]), 5),
                            "rose_in": sum(1 for x in deltas[name] if x > 0),
                        }
                        for name, _note in KEYS
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")

    if args.export_pairs:
        Path(args.export_pairs).write_text(json.dumps(pairs), encoding="utf-8")
        print(f"wrote {args.export_pairs} ({len(pairs)} matched pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

