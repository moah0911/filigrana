# Capacity

What a personal watermark can actually support, measured on 648,423 words of public-domain
prose. Reproduce with:

```console
uv run python tools/measure_free_choices.py --corpus corpus --lengths 215,1075 --keys 10
uv run python tools/attack_eval.py --corpus corpus --articles 120 --words 2500
uv run python tools/measure_lm_gate.py --corpus corpus --articles 60 --words 1075
uv run python tools/detector_interference.py --corpus corpus --articles 60 --out bench/interference.json
```

Every number below is from this repository's own runs unless attributed. Machine-readable
copies live in `bench/`: `density.json`, `attacks.json`, `lm_gate.json`, `interference.json`.
The gated run needs `filigrana[lm]` and a local model; the rest need nothing but the corpus.

## The two numbers that decide everything

Signal is `z = Δ·√N`. `N` is the count of key-correlated choices in the text and `Δ` is how
far from chance each one lands. Both are empirical, and `N` is the binding one:

| table | ρ slots per 1,000 words | words for p<1e-6 at full strength |
|---|---|---|
| tier `a`+`b` (safe) | **8.27** | **2,733** |
| tier `all` (+ locale & contractions) | **21.75** | **1,039** |

Measured over 2,805 chunks × 10 foreign keys, `N` is key-independent so the density holds for
anyone's key.

Prior art reports ρ ≈ 60 per 1,000 words at article length
([TDS, §3.2.2](https://towardsdatascience.com/text-watermarking-in-python-catch-whoever-copies-your-writing/)).
We are 3–7× lower on purpose: their table reaches `cost→price`-class substitutions that made
**100% of passages less fluent and 17.3% gain a grammatical error**. Our rule is that a class
is admissible only if every member substitutes for every other in *every* context. Density is
not a property of English, it is a property of how much damage you are willing to do.

The corpus bias runs both ways and is documented in `corpus/README.md`: 19th-century prose
over-uses `towards`/`amongst`/`-ise` and under-uses contractions.

## The statistic is calibrated, which is the part you have to take on faith otherwise

Null distribution of `z` under foreign keys on unmarked prose:

| chunk | tier | mean z | sd | max z | measured FPR@1% | FPR@1e-6 |
|---|---|---|---|---|---|---|
| 215 w | `ab` | 0.001 | 0.91 | 2.67 | 0.05% | 0 |
| 215 w | `all` | 0.001 | 0.99 | 3.36 | 0.48% | 0 |
| 1,075 w | `ab` | 0.006 | 0.98 | 3.16 | 0.84% | 0 |
| 1,075 w | `all` | 0.003 | 0.99 | 3.16 | 0.94% | 0 |

Mean ~0 and sd ~1 means the model behaves; ~28,000 scored text/key pairs produced no single
false reach of `p<1e-6`. Slight conservatism at 1% is welcome, not a bug.

Short chunks are *under*-powered rather than trigger-happy: at 215 words `z` physically
cannot reach the threshold, because there are ~1.9 slots. A couple of coin flips are not
evidence, and this is why `detect` refuses to look confident below 8 slots instead of printing
a small p.

## What survives, at 2,500 words

Detection rate at `p<1e-6` across 120 marked articles, per tier:

| transformation | tier `ab` | tier `all` | slots retained |
|---|---|---|---|
| verbatim copy | 37% | **97%** | 1.00 |
| whitespace reflow, lowercasing | 37% | **97%** | 1.00 |
| print hyphenation | 28% | 93% | 0.99 |
| proofreader swaps 10% of ordinary words | 9% | 84% | 0.93–0.96 |
| house-style normaliser (informed adversary) | 13% | 77% | 1.00 |
| 10% of sentences deleted | 18% | 89% | 0.90 |
| 30% of sentences deleted | 1% | 64% | 0.69 |
| middle-50% quote | 1% | 62% | 0.50 |
| 50% of sentences deleted | 0% | 21% | 0.50 |
| full paraphrase / translation | ~0% | ~0% | — |

Reflow and case are **exactly** a no-op (identical `z` to verbatim), which is the point of
`canon.py` and `tests/test_determinism.py`: those are serialization, not attacks.

Paraphrase and translation rows are quoted from prior art rather than measured, because an
honest in-band number for them is 0 and we have no way to beat it. **The only claim worth
making is verbatim or lightly edited republication of long-form prose.**

Two findings worth naming:

- **The house-style attack is real but shallow.** Normalising 15 known variant pairs costs a
  tier-`all` article 20 points of detection (97% → 77%) while touching *no* slots
  (retention 1.00) — it flips demands rather than deleting them. An adversary would have to
  enumerate the entire table to do better, which is an argument for keeping the table partly
  private, and for not shipping tier `c` on prose you expect to be contested.
- **Locality is what saved this design.** Keying each demand off *previous slot decisions*
  looked elegant and failed measurement: a 30%-sentence cut dropped a marked article to
  30/53 slots, i.e. chance, because one deletion shifts the chain for the whole rest of the
  document. Keying instead off nearby words that no class can reach keeps damage local: same
  attack now retains 40/53. Pinned by `tests/test_statistics.py::test_light_editing_...`.

## Tier 2 (decoding an identity) is a different and much harder problem

A slot carrying a payload bit can vote for `bit=0` or `bit=1`. For two-member classes the two
demands coincide about half the time, and that slot carries **no information** — measured, 28
of 53 slots in the test fixture. Effective payload rate is therefore roughly ρ/2, and each
bit needs ~28 independent votes to reach `p<1e-6`:

| payload | votes needed | informative slots | words at tier `all` |
|---|---|---|---|
| 16 bits (short id) | ~450 | ~900 | **~41,000** |
| 24 bits (id + date) | ~670 | ~1,340 | **~62,000** |
| 45-bit wire format | ~1,260 | ~2,520 | **~116,000** |

Repetition buys reliability, not capacity. So the deployable design is the inverse of the
intuitive one:

- **identity goes in the key's fingerprint** — free, works at tier-1 length, and is exactly
  the question a writer asks about a scraped copy of their own essay;
- **the 24-bit id + date payload exists** (`codec.encode_identity`) for the stranger-who-finds-
  unattributed-text case, and it should be quoted as book-to-novel length or not at all;
- **a handle should never be spelled into the bits.** At 6 bits/character, an 8-character
  handle is ~50 bits on top of everything else.

If your threat model is "someone republished my work", you do not need tier 2. That is the
most useful sentence in this document.

## Does marking your own prose make it look machine-generated?

Structurally plausible — a key-correlated choice pattern is what an LLM watermark also is —
and nobody has published the measurement. 60 matched articles, paired comparison, proxy
metrics from `filigrana/slop.py` (**not** a commercial detector; the real answer needs
Originality/Pangram/GPTZero accounts, which is why `--export-pairs` exists):

| metric | marked − original | with `--anti-slop` | rose in |
|---|---|---|---|
| burstiness (sentence-length sd) | −0.045 | −0.045 | 13/60 |
| type-token ratio | +0.005 | +0.005 | 55/60 (diversity *up*) |
| long-word rate | +0.0007 | +0.0001 | 50/60 |
| cliché hits per 1k | −0.079 | −0.079 | 11/60 |
| formal connective openers | +0.0000 | 0.0000 | 0/60 |
| filler-word rate | +0.00066 | **−0.00012** | 39/60 |

Machine copy: [bench/interference.json](bench/interference.json). Keys are `secrets`-derived,
so re-runs drift column-by-column (this run: clichés −0.079 vs an earlier −0.094); what
repeats across runs is the *direction* of every row.

Reading it honestly: the mark barely perturbs style, lexical diversity goes *up*, and cliché
hits go slightly down. The one direction that looks bad is filler words, and the style guard
removes it for a price of **50 of 1,581 slots — 3.2% of statistical strength**. That is a
good trade, and it is the one thing in this project that might make a writer actually want a
mark instead of resenting it.

Still unproven: whether commercial detectors, which are closed models, react differently. If
they flag marked prose, the fix is not a better disclaimer.

## The model-gated path: a precision instrument, not a capacity multiplier

The seam (`providers.CandidateProvider`) lets any proposer hand candidates to `gate.py`,
which keeps a slot only where the model itself is nearly indifferent between the
alternatives — a probability plateau. Measured at the same length as the curated numbers
above: gpt2, 60 articles × 1,075 words, ε = band half-width in nats. `edits` counts what
was *applied across rounds*, not what was pending at convergence:

| proposer | proposals ρ | kept ρ | applied edits ρ | retained | Δ logprob/token | mean z | det @1e-6 |
|---|---|---|---|---|---|---|---|
| curated, ε=0.35 | 24.5 | 0.6 | 0.2 | 41.7% | −0.0001 | 0.50 | 0 |
| wordnet, ε=0.35 | 51.9 | 0.5 | 0.7 | 31.9% | −0.0006 | 0.25 | 0 |
| curated, ε=0.7 | 24.5 | 1.1 | 0.5 | 65.0% | −0.0003 | 0.84 | 0 |
| wordnet, ε=0.7 | 51.9 | 1.0 | 1.3 | 48.4% | −0.0014 | 0.48 | 0 |
| curated:ab, no model | — | 8.4 | 4.0 | 100% | ~0 | 3.00 | 0 |
| curated:all, no model | — | 24.5 | 12.4 | 100% | ~0 | 5.06 | 66.7% |

Machine copy: [bench/lm_gate.json](bench/lm_gate.json). Re-runs drift by a slot or two per
column — candidates with |Δ| right at ε sit on float32 rounding — but the scale and
direction of every number here repeat.

Findings, in the order that matters:

1. **The plateau is rare in real prose.** 96–98% of proposals fail the band test: 24.5
   (curated) or 51.9 (wordnet) editorial candidates per 1,000 words funnel to 0.5–1.1 kept.
   Doubling ε recovers only ~0.5 slots, which is the signature of positions that are not
   near-ties to begin with.
2. **Discovery is unstable.** Only 32–65% of gated slots re-derive from the marked copy
   alone — each accepted word shifts the next pass's plateaus. The sidecar is therefore
   *mandatory* for the gated path, and band stability should become a requirement (keep a
   slot only when two consecutive passes both call it free), not a reported number.
3. **The style cost is now measured, and it is small.** 0.2–1.3 applied edits per 1,000
   words, ≤ 0.0014 nats of mean log-prob shift per token. The instrument does exactly what
   it claims: prose damage near zero.
4. **But none of it reaches a threshold.** Mean z at 1,075 words is 0.25–0.84 with zero
   detections at p<1e-6. Since z ∝ √N, reaching the z≈5.06 the untouched table gets at
   1,039 words takes **~39,000 words** (curated proposals, ε=0.7) to **~120,000** (wordnet).
   That is not paragraph length; it is a book, again.

Verdict: the gate earns its place as the *low-damage* embedder — the mode for a writer who
wants a mark that costs nothing measurable in style — while the curated tier-`all` table
(24.5 slots/1,000 at 100% retention) remains what reach looks like. Next steps, in order:
require band stability across passes (which should also restore retention), re-run the ε
grid with it, and try a better-scaled scorer than gpt2-small before calling the seam a dead
end.

## The rules this document buys

1. Below **~8 slots**, nothing is quotable. At tier `a`+`b` that is ~1,000 words.
2. Target **2,500+ words** and tier `all` for anything you expect to be contested; expect
   tier `ab` alone to need **~4,000** to cover an editor's 25% of damage.
3. **Never enable tier `c` for prose a house style will flatten.**
4. **Never claim more than verbatim-or-lightly-edited republication.**
5. Use the **manifest** for the legal conversation and the **mark** for the paste. Neither is
   proof of ownership; see [THREAT-MODEL.md](THREAT-MODEL.md).
