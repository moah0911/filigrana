# inkmark

A writer-held statistical watermark for prose you wrote **by hand**.

`inkmark` lets you bias the equally-good choices in your own writing — `toward` vs
`towards`, `however` vs `nevertheless`, `a number of` vs `several` — according to a key only
you hold. Six months later, when a content farm has republished your essay as plain text,
you can run your key over their copy and get a p-value for whether your word choices are in
it.

Nothing is added to the text. There are no hidden characters, no metadata, no invisible
codepoints. A reader sees an essay; someone with your key sees statistics.

```console
$ git clone https://github.com/YOU/inkmark && cd inkmark && uv sync   # not on PyPI yet
$ alias inkmark='uv run inkmark'
$ inkmark key new                            # scrypt-encrypted keyring, 0600
$ inkmark audit essay.md                     # can this article even carry a mark?
$ inkmark embed essay.md                     # preview the diff, writes nothing
$ inkmark embed essay.md --anti-slop --out essay.marked.md --sidecar essay.sidecar.json
$ inkmark publish essay.marked.md --handle ada --published 2026-11-06 --domain example.com
$ inkmark detect scraped-copy.html           # p-value plus caveats, never a verdict
```

A real run, on the 144-word article in [`examples/essay.md`](examples/essay.md). Both output
files are checked in, so you can diff `essay.md` against `essay.marked.md` yourself — the
visible changes are `a lot of`→`lots of`, `afterwards`→`afterward`, `grey`→`gray`,
`Perhaps`→`Maybe`.

```console
$ inkmark embed examples/essay.md --tier all --anti-slop
144 words, 23 eligible choices
10 edits (6.94% of words), 2 declined by you
21/23 slots now agree with the key

$ inkmark detect examples/essay.marked.md --tier all
agreement  21/23 slots on the key's side (chance would give 11.3)
statistics z = 4.04, p = 2.66e-05
strength   15.2 bits of evidence against chance

$ inkmark audit examples/essay.md --tier all      # what this piece can support, up front
best possible z at full strength: 4.88  (p = 5.38e-07)
  verbatim copy              z ->   4.88   p -> 5.38e-07
  25% of slots edited away   z ->   4.22   p -> 1.2e-05
  half the text quoted       z ->   3.45   p -> 0.000282
  a quarter of slots left    z ->   2.44   p -> 0.00737
```

Against a **foreign** key the same marked text scores 12/23, z = 0.28, p = 0.39 — chance. One
draw is not evidence of calibration; [docs/CAPACITY.md](docs/CAPACITY.md) has the null over
~28,000 foreign key/text pairs, where `z` sits at mean −0.018, sd 1.00, max 3.05, with zero
false reaches of `p<1e-6` in 648k words of unmarked prose.

---

## Read this first: the idea is not new

Three things you should know before you build anything on this README.

**Someone has already shipped a version of this.** [`vumichien/text-watermarking-toolkit`](https://github.com/vumichien/text-watermarking-toolkit)
(MIT, Aug 2026) and its [write-up in Towards Data Science](https://towardsdatascience.com/text-watermarking-in-python-catch-whoever-copies-your-writing/)
implement keyed word-choice watermarking for your own writing, with a model-free detector
and a calibrated null. It is the best introduction to the mechanism available, and it
measured the failure modes honestly — numbers we reproduce and extend in
[docs/CAPACITY.md](docs/CAPACITY.md).

**The mechanism is older than LLMs.** Keyed synonym substitution is Topkara, Topkara &
Atallah, *ACM MM&Sec 2006*; syntactic equivalence classes go back to Atallah's *NL Watermark*
(IH 2001); context-aware lexical substitution is Yang et al., AAAI 2022. Model-free detection
by Bernoulli test is Yang et al., arXiv:2305.08883. `inkmark` claims no new algorithm.

**What `inkmark` is actually for** is the engineering those works do not do: an installable
library with a tested determinism contract, a key and identity model with rotation and
revocation (because a personal watermark identifies a *person*, which Anthropic's explicitly
does not), multiplicity-corrected p-values, an auditable equivalence table where every class
is provably substitutable in both directions, and published capacity numbers instead of
optimism.

## What it is for, and what it is not

For: *"someone republished my work and I want more than my word against theirs."*

Not for:

- **Proof of ownership.** A watermark shows your *version* was copied. It cannot show you
  owned the underlying text. Genius planted a watermark in 301 songs, found it in Google's
  index, and still lost the case — because it licensed the lyrics. See
  [docs/THREAT-MODEL.md](docs/THREAT-MODEL.md).
- **Surviving a rewrite.** The mark lives in the words. Paraphrase it and it is gone —
  measured at 0% by prior art, and we expect the same. Translation, likewise.
- **Any short text.** A tweet has maybe one free choice. One coin flip is not evidence.
- **Accusing people.** The detector answers a narrow statistical question about one key and
  one text. That is why `Decision` has no truth value, and why there is no hosted verifier.

## The two tiers, and why they are different problems

| | Question | Cost in bits | Viable length |
|---|---|---|---|
| **tier 1 — attribution** | *you* test your own key against a suspect copy | **0** — identity is the key's public fingerprint | ~1,000+ words |
| **tier 2 — discovery** | a stranger decodes *who wrote this* with no candidate | 45 (writer id 24 + date 13 + CRC 8) | book-length; see CAPACITY.md |

Most designs conflate these. For the scraper problem you already know who you are, so you
never need to decode anything — you just test your key. That is why tier 1 is the default and
why `inkmark` works at essay length where a full handle-plus-date payload does not.

## How the watermark gets its "free choice"

Production LLM watermarks bias a model's logits during sampling. A human writer has no
logits. So `inkmark` manufactures the choice from a hand-audited table of **equivalence
classes**:

```python
{"id": "b.although", "tier": "b", "members": ["although", "though"],
 "not_after": ["even"], "avoid_clause_final": true}
```

A class is admissible only if every member can replace every other in *every* context. That
bar is much higher than "these are synonyms", and it is the whole project: `help`/`aid` is
excluded because `foreign aid` cannot become `foreign help`; `although`/`though` is kept but
guarded, because `I liked it though` has no `although` version. Automated synonym tables get
this wrong — the prior art measured **100% of marked passages becoming less fluent and 17.3%
gaining a grammatical error**. `inkmark` therefore previews a diff and lets you veto any
edit, rather than rewriting your draft.

Each slot's demanded member is `HMAC-SHA256(secret, scheme ‖ key_id ‖ class ‖ anchor ‖
payload bit) mod n_candidates`, where the anchor is the handful of preceding words that
**no class can reach** — words the watermark itself will never rewrite, which is what makes
them safe to key off.

An earlier revision chained each demand off the *previous slot decisions* instead. It looks
elegant and it is wrong: one deleted sentence shifts the chain for the entire rest of the
document, and measurement showed a marked article scoring 30/53 slots — pure chance — after
30% sentence deletion. Anchoring bounds the damage to the neighbourhood of the edit, and the
same article keeps 40+/53. That regression is `tests/test_statistics.py`, not a footnote.

SynthID's own reference hash explicitly disclaims cryptographic security; ours needs it,
because a leaked key here lets someone plant your mark on prose you never wrote.

## Tiers of risk in the table itself

| tier | what | density | cost |
|---|---|---|---|
| `a` | orthographic — `toward/towards`, `among/amongst` | low | invisible |
| `b` | low-salience lexical and connective — `maybe/perhaps`, `often/frequently` | medium | slight register shift |
| `c` | locale (`colour/color`, `-ise/-ize`) and register (`don't/do not`) | high | **mixing forms in one article is a visible inconsistency** |

Default is `ab`. Tier `c` is where the capacity is, and that is the real trade: a watermark
*is* intra-text inconsistency, because the mark has to vary. Anyone telling you a text
watermark is free has not looked at the output.

## Ethics, in code rather than in prose

- **`Decision` has no truth value.** `bool(decision)` raises `TypeError`. There is no
  `is_mine` field and no `--unsafe-verdicts` escape hatch.
- **Multiplicity correction is automatic.** `detect --key a --key b --key c` reports the
  Sidak-corrected p-value, because testing a registry of a million writers at p<1e-6 per test
  buys roughly one false accusation per scan.
- **Caveats are never empty**, and short text says so itself: below 8 slots the statistic is
  noise and the output tells you not to quote it.
- **Keys stay local.** `inkmark detect` performs no network I/O. Any third-party "inkmark
  checker" is untrusted by design — it could both forge your mark and fingerprint your whole
  corpus.
- **Verification is delegable but scoped.** `key.for_domain("example.com")` gives a publisher
  enough to verify you on their site, not enough to derive your master key or scan the rest of
  your body of work.

## What it costs, measured

Over 648,423 words of public-domain prose, 2,805 chunks and ~28,000 key/text pairs
(full tables in [docs/CAPACITY.md](docs/CAPACITY.md)):

| | tier `ab` (safe) | tier `all` (+ locale, contractions) |
|---|---|---|
| free choices per 1,000 words | **8.3** | **21.7** |
| words to reach `p<1e-6` | 2,733 | 1,039 |
| detected after 30% of sentences deleted | 1% | 64% |
| detected after a house-style normaliser | 13% | 77% |
| detected after full paraphrase or translation | ~0% | ~0% |
| null `z` under a foreign key | mean −0.018, sd 1.00, max 3.05 | mean +0.004, sd 0.99, max 3.40 |
| false positives at `p<1e-6` in 28,000 trials | 0 | 0 |

Three consequences worth stating plainly:

- **Prior art's ρ ≈ 60 is achievable and costs prose.** Their own numbers: 100% of passages
  less fluent, 17.3% with a new grammatical error. Our 8.3 is the price of a table where every
  class is provably substitutable both directions. Density is a measure of how much damage you
  accept, not a property of English.
- **Tier 2 (decoding an identity out of unattributed text) needs ~41,000 words for 16 bits.**
  Roughly half of all two-way slots carry no bit at all because the key demands the same word
  either way. Identity belongs in the key fingerprint, which is free, not in the payload.
- **Serialization is free and rewriting is fatal.** Reflow and case changes cost exactly
  nothing (identical `z`). A paraphrase costs everything, and nobody has found a way around
  that. Claim only verbatim-or-lightly-edited republication.

## Layout

```
canon.py            markup masking, offsets preserved, NFC (never NFKC)
keystream.py        HMAC-SHA256 keyed choice over an anchor of unreachable words
ledger.py           the ONE traversal both embed and detect share -- determinism lives here
providers.py        equivalence-class scanning, guards, longest-match-first; also the
                    proposal seam (`CandidateProvider`) the model-gated path plugs into
embed.py            diff proposal + veto + style guard + self-verification for drift
detect.py           hits vs sum(1/n) over variable-size classes, multiplicity-corrected
decisions.py        the output type is the safety mechanism: bool() raises
sidecar.py          publish what was free, so verification needs no table, lexicon or model
codec.py            tier-2 wire format and per-bit voting
keyring.py          scrypt-encrypted keyring, fingerprints, domain subkeys, rotation
registry.py         self-hosted id -> handle file; we run no service
manifest.py         detached signed provenance (HMAC by default, Ed25519 via inkmark[sign])
gate.py             the free-choice gate: keep a proposal only on a probability plateau
lm.py               model-gated embedding under torch, optional: proposals filtered, never invented
lexical.py          WordNet proposals, sense-blind by construction, refused without a gate
slop.py             transparent proxy metrics for the AI-detector collision question
cli.py              key | embed | detect | audit | publish | verify | registry | table
data/equivalence.json   135 hand-audited classes across three risk tiers
integrations/synthid.py writer-derived keys for the HF logits path
tools/                  measure_free_choices, attack_eval, measure_lm_gate, detector_interference
bench/                  the JSON those runs produced (density, attacks, lm_gate, interference)
```

## Status

`0.1.0`, alpha, 92 tests, ruff + mypy + coverage-gated CI, full type annotations (`py.typed`).
Shipped and measured: key generation with scrypt-encrypted storage and scoped verifier
subkeys; tier-1 embedding with reviewed per-slot diffs; drift self-check; detection with
variable-arity statistics, multiplicity correction and mandatory caveats; a
serialization-invariance suite; the tier-2 payload codec; the registry format; detached
manifests (HMAC always; Ed25519 through the `inkmark[sign]` extra); the anti-slop style
guard; and the capacity report over 648k words.

**The LM-assisted free-choice gate is in this tree and measured:** `gate.py` adds the
plateau test, `lm.py` the torch-helper embedding path, `lexical.py` the WordNet proposer.
Measured at article length it keeps ~1 slot per 1,000 words at near-zero style cost
(≤ 0.0014 nats/token) — a precision instrument, not a capacity multiplier; the curated
tables remain what reach looks like. The numbers live in [docs/CAPACITY.md](docs/CAPACITY.md)
and [bench/lm_gate.json](bench/lm_gate.json).

Not shipped, in the order that matters:

1. **Inference beyond a first plateau pass**: higher density without paying in prose quality
   is bounded by how rare the plateau condition is in real prose (measured; see CAPACITY).
2. **Ed25519 manifests as the default rather than the extra** (`inkmark[sign]` already
   installs the audited library; the default stays stdlib-HMAC until there is a reason).
3. **A real detector-interference study** on matched pairs through Originality/Pangram/
   GPTZero. The proxy says the collision is mild; `--export-pairs` and
   [bench/interference.json](bench/interference.json) are what the repository contributes to
   that study, not an answer.
4. **A public-but-unforgeable in-band mark.** Bounded by the impossibility results and by
   ~5,000 contiguous untouched tokens of cost. If someone claims to have shipped this, ask
   them what length it works at.

MIT. Not affiliated with Anthropic; the explainer this was read against is
[anthropic.com/news/claude-text-watermark](https://www.anthropic.com/news/claude-text-watermark).
