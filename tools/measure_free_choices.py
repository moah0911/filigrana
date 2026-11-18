"""Stage 0: measure the free-choice budget of real prose.

Everything else in this repository is downstream of two numbers:

  rho   key-correlated slots per 1000 words, for a table a human would actually approve
  Delta achieved bias per slot, and whether the null really is N(0,1)

Signal grows as Delta*sqrt(N), so length is not a convenience, it is the constraint. This
script is deliberately dependency-free so anyone can re-run it and disagree with the table.

    uv run python tools/measure_free_choices.py --corpus corpus/ --tiers ab all

Null calibration is the important half. For each chunk we score many random keys; if the
resulting z values are not mean-zero and unit-variance, the detector is lying to writers and
the false-positive rate nobody can quote is the least of our problems.
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
from filigrana.keyring import WriterKey
from filigrana.ledger import walk, word_count
from filigrana.providers import default_table

Z_1e6 = 4.754  # one-sided p < 1e-6
Z_1p = 2.326  # one-sided p < 1%


def chunk_words(text: str, size: int) -> list[str]:
    """Roughly `size` words per chunk, cut at sentence ends so slots are not amputated."""
    words = text.split()
    out, buf, since_break = [], [], 0
    for w in words:
        buf.append(w)
        since_break += 1
        if since_break >= size and w.endswith((".", "!", "?", '"')):
            out.append(" ".join(buf))
            buf, since_break = [], 0
    if len(buf) > size // 4:
        out.append(" ".join(buf))
    return out


def random_key(rng: random.Random) -> WriterKey:
    return WriterKey(
        secret=bytes(rng.getrandbits(8) for _ in range(32)),
        key_id=f"probe-{rng.getrandbits(32):08x}",
        created="2026-01-01",
        year=2026,
    )


def measure(chunks: list[str], keys: list[WriterKey], table, tier: str) -> dict:
    counts = [len(walk(c, keys[0], table)) for c in chunks]
    words = [word_count(c) for c in chunks]
    total_words = sum(words) or 1
    zs = [scan(c, key, table=table).z for c in chunks for key in keys]
    return {
        "tier": tier,
        "chunks": len(chunks),
        "words": total_words,
        "slots": sum(counts),
        "rho_per_1000w": round(1000 * sum(counts) / total_words, 2),
        "slots_per_chunk_mean": round(statistics.mean(counts), 2) if counts else 0,
        "slots_per_chunk_sd": round(statistics.pstdev(counts), 2) if counts else 0,
        "null_z_mean": round(statistics.mean(zs), 4) if zs else None,
        "null_z_sd": round(statistics.pstdev(zs), 4) if len(zs) > 1 else None,
        "null_z_max": round(max(zs), 3) if zs else None,
        "fpr_at_1pct": round(sum(z >= Z_1p for z in zs) / len(zs), 5) if zs else None,
        "fpr_at_1e6": round(sum(z >= Z_1e6 for z in zs) / len(zs), 5) if zs else None,
        "words_for_1e6": round(Z_1e6**2 / max(1000 * sum(counts) / total_words, 1e-9) * 1000),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--tiers", default="ab,all")
    ap.add_argument("--lengths", default="215,1075")
    ap.add_argument("--keys", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", default="bench/density.json")
    args = ap.parse_args(argv)

    files = sorted(Path(args.corpus).glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files in {args.corpus}/ -- see corpus/README.md")
    prose = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
    rng = random.Random(args.seed)
    keys = [random_key(rng) for _ in range(args.keys)]

    report: dict = {"files": [f.name for f in files], "n_keys": args.keys, "runs": []}
    for length in (int(x) for x in args.lengths.split(",")):
        chunks = [c for c in chunk_words(prose, length) if word_count(c) > length // 2]
        for tier in args.tiers.split(","):
            row = measure(chunks, keys, default_table(tier), tier)
            row["chunk_words"] = length
            report["runs"].append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"corpus: {len(files)} files, {word_count(prose):,} words\n")
    header = ("tier", "chunk", "rho/1k", "slots", "null z", "sd", "max z", "fpr@1%", "fpr@1e-6", "words→1e-6")
    print(f"{header[0]:>4} {header[1]:>6} {header[2]:>7} {header[3]:>6} {header[4]:>7} "
          f"{header[5]:>5} {header[6]:>6} {header[7]:>7} {header[8]:>9} {header[9]:>9}")
    for row in report["runs"]:
        print(
            f"{row['tier']:>4} {row['chunk_words']:>6} {row['rho_per_1000w']:>7} "
            f"{row['slots']:>6} {row['null_z_mean']:>7} {row['null_z_sd']:>5} "
            f"{row['null_z_max']:>6} {row['fpr_at_1pct']:>7} {row['fpr_at_1e6']:>9} "
            f"{row['words_for_1e6']:>9}"
        )
    print(f"\nwrote {args.out}")
    print("null z should be mean~0, sd~1. If sd is well above 1 the slots are correlated "
          "and every published p-value is too optimistic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
