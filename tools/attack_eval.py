"""How much editing does the mark survive, measured mechanically.

No LLM here on purpose: the attacks below are the ones a content farm actually runs --
partial quotes, reflow into a template, a house-style pass, a proofreader swapping some
words. A paraphrase model would score zero and we already know that from the literature;
the interesting question is the shape of the degradation before it, and whether an adversary
who *knows* the scheme can cheaply undo it.

    uv run python tools/attack_eval.py --corpus corpus --sentences 1200
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from filigrana.detect import scan
from filigrana.embed import embed
from filigrana.keyring import WriterKey
from filigrana.ledger import word_count
from filigrana.providers import default_table

Z = 4.754


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?]) +", text) if word_count(s) > 4]


def article(corpus: Path, words: int, rng: random.Random) -> str:
    files = sorted(corpus.glob("*.txt"))
    text = rng.choice(files).read_text(encoding="utf-8", errors="replace")
    parts = text.split()
    start = rng.randrange(0, max(1, len(parts) - words))
    return " ".join(parts[start : start + words])


def a_verbatim(t: str, rng) -> str:
    return t


def a_reflow(t: str, rng) -> str:
    """Wrap at whitespace the way a template or CMS does. Character-slicing would be a

    corruption attack, not a serialization one, and it belongs in a different column."""
    lines, current = [], ""
    for word in t.split():
        if len(current) + len(word) + 1 > 72:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    lines.append(current)
    return "\n".join(lines)


def a_print_typeset(t: str, rng) -> str:
    """The honest version of reflow: a PDF or print pass hyphenates and breaks words."""
    out = []
    for word in t.split():
        if rng.random() < 0.06 and len(word) > 8:
            cut = rng.randrange(4, len(word) - 3)
            out.append(word[:cut] + "-\n" + word[cut:])
        else:
            out.append(word)
    return " ".join(out)


def a_lower(t: str, rng) -> str:
    return t.lower()


def a_partial_middle(t: str, rng) -> str:
    words = t.split()
    return " ".join(words[len(words) // 4 : 3 * len(words) // 4])


def _drop_sentences(t: str, rate: float, rng):
    s = sentences(t)
    return " ".join(x for x in s if rng.random() > rate)


def a_word_swap(t: str, rng) -> str:
    """A proofreader who changes ordinary words. Tests anchor locality, not the slots."""
    words = t.split(" ")
    out = []
    for i, w in enumerate(words):
        if rng.random() < 0.10 and re.fullmatch(r"[A-Za-z]{5,}", w or ""):
            out.append("".join(rng.sample("abcdefghijkmnop", min(len(w), 10))))
        else:
            out.append(w)
    return " ".join(out)


def a_house_style(t: str, rng) -> str:
    """The informed adversary: normalise locale spellings and contractions to one form."""
    pairs = [
        ("colour", "color"), ("behaviour", "behavior"), ("favour", "favor"),
        ("recognise", "recognize"), ("organise", "organize"), ("analyse", "analyze"),
        ("realise", "realize"), ("judgement", "judgment"), ("towards", "toward"),
        ("amongst", "among"), ("afterwards", "afterward"), ("don't", "do not"),
        ("can't", "cannot"), ("won't", "will not"), ("isn't", "is not"),
    ]
    out = t
    for a, b in pairs:
        out = re.sub(rf"\b{re.escape(a)}\b", b, out, flags=re.IGNORECASE)
    return out


ATTACKS = {
    "verbatim": a_verbatim,
    "reflow_72col": a_reflow,
    "lowercased": a_lower,
    "middle_50pct_quote": a_partial_middle,
    "sentences_cut_10pct": lambda t, r: _drop_sentences(t, 0.10, r),
    "sentences_cut_30pct": lambda t, r: _drop_sentences(t, 0.30, r),
    "sentences_cut_50pct": lambda t, r: _drop_sentences(t, 0.50, r),
    "proofreader_10pct": a_word_swap,
    "print_hyphenation": a_print_typeset,
    "house_style_normaliser": a_house_style,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--articles", type=int, default=120)
    ap.add_argument("--words", type=int, default=1075)
    ap.add_argument("--tiers", default="ab,all")
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--out", default="bench/attacks.json")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    if not any(corpus.glob("*.txt")):
        raise SystemExit("no corpus/*.txt")
    rng = random.Random(args.seed)
    report: dict = {"articles": args.articles, "words": args.words, "runs": []}

    for tier in args.tiers.split(","):
        table = default_table(tier)
        keys = [WriterKey.generate() for _ in range(6)]
        marked = []
        for _ in range(args.articles):
            key = rng.choice(keys)
            text = embed(article(corpus, args.words, rng), key, table=table)
            marked.append((key, text.text, text.slots))
        for name, attack in ATTACKS.items():
            zs, slot_ratio = [], []
            for key, text, slots in marked:
                hurt = attack(text, rng)
                decision = scan(hurt, key, table=table)
                zs.append(decision.z)
                slot_ratio.append(decision.slots / max(slots, 1))
            detected = sum(z >= Z for z in zs) / len(zs)
            row = {
                "tier": tier,
                "attack": name,
                "mean_z": round(statistics.mean(zs), 2),
                "median_z": round(statistics.median(zs), 2),
                "detected_p1e6": round(detected, 4),
                "slots_retained": round(statistics.mean(slot_ratio), 3),
            }
            report["runs"].append(row)
            print(f"{tier:>4} {name:<22} z {row['mean_z']:>7} med {row['median_z']:>7} "
                  f"det {row['detected_p1e6']:>7} slots {row['slots_retained']:>6}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(__import__("json").dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("House-style normalising is the informed attack: it only works where tier c is on,")
    print("which is the honest reason not to enable it for prose you publish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
