# Threat model

Who `filigrana` defends against, who it does not, and what a score is allowed to be used for.

## The actor we are writing this for

A person who publishes long-form prose and finds it republished elsewhere as plain text —
a content farm, an aggregator, an LLM-training scrape, or an editor's "we lost your byline"
conversation. They hold a key, they have their original, and they want more than their own
word against theirs.

That scenario is the reason for three design decisions that would otherwise look like
over-engineering: the key never leaves the machine; verification is delegable but scoped; and
the output is a p-value with caveats rather than a verdict.

## What the mark survives

| Transformation | Effect | Category |
|---|---|---|
| Copy-paste, verbatim | none — full signal | — |
| Markdown → HTML → EPUB → plain text, whitespace reflow, smart quotes, NFC, case changes | none by construction, tested in `tests/test_determinism.py` | **serialization**: a bug if it ever fails |
| 10% of sentences deleted | proportional loss of slots | attack |
| 30% of sentences deleted | still detectable in a 2,500-word article with tier `all` on (64% measured); see CAPACITY.md | attack |
| A proofreader swapping ordinary words | local: only slots whose anchor window moved | attack |
| Partial quote (middle 50%) | halved slots; below ~8 slots the output refuses to look confident | attack |
| Metadata stripped, sanitizer run, Unicode normalization | **no effect — there is no metadata and nothing hidden to strip** | the whole point |

## What the mark does not survive

- **Rewriting.** The mark lives in word choices. A paraphrase re-flips every coin. Prior art
  measures 0% survival after full paraphrase and 0% after translation, and we have no reason
  to believe otherwise. Anyone claiming a paraphrase-resistant *in-band* text watermark is
  selling something else.
- **An informed adversary who normalises your text.** If they know you use tier `c`
  (locale and contraction classes), a single house-style pass to one consistent dialect flips
  enough demands to cut measured detection from 97% to 77% while deleting no slots at all.
  This is the strongest argument for keeping tier `c` off, and the reason it is not the
  default.
- **Short text.** Below about eight key-correlated choices the statistic is noise, and
  `detect` says so in its caveats rather than printing a confident number.

## What a detection result does *not* mean

1. **Not ownership.** A watermark shows that text derived from *your marked version* was
   copied. It cannot show you owned the underlying content. Genius planted a Morse-code
   apostrophe watermark in 301 songs, found it in Google's results for 116 of them, and lost
   the case anyway — because the lyrics were licensed, not owned. See §9 of the TDS write-up
   and the 2020 dismissal.
2. **Not authorship of unmarked text.** A high p-value is evidence of *absence of your mark*,
   never evidence that someone else wrote it, that a human wrote it, or that a machine did.
3. **Not identity, without tier 2.** Tier 1 answers "does this key explain these choices?".
   Only the payload path can name a writer, and it needs far more text.
4. **Not a verdict at any p-value**, especially not a scanned one. Testing a registry of
   10⁶ writers at a per-test 1e-6 threshold yields roughly one false accusation per scan;
   `scan_against` applies the correction and prints it, and the corrected number is the only
   one worth quoting.

## Adversary capabilities we assume away

- **The key holder is honest.** If the writer's key leaks, anyone can plant their mark on
  prose they never wrote. That is a *forgery* vector, not just a privacy leak, and it is why
  the key model records rotation and revocation, and why the fingerprint — not the secret —
  is what you publish.
- **The verifier is not the writer.** Giving a publisher your master key lets them test any
  text you have ever written. `WriterKey.for_domain("example.com")` gives a scoped subkey
  instead: enough to verify on one site, not enough to fingerprint your corpus or derive the
  master.
- **No hosted verifier exists, and none should.** Any third-party "filigrana checker" is
  untrusted by design: it can both forge marks and fingerprint writers across the corpus.
  Detection performs no network I/O.

## Why this is not an AI detector

Anthropic's watermark answers "was a model involved?" and deliberately identifies nobody.
`filigrana` answers "whose key explains these choices?" and deliberately identifies one person.
The two are orthogonal: a marked human essay says nothing about model involvement, and marked
LLM output says nothing about which human wrote the prompt. Do not conflate them, and do not
use `filigrana` to accuse someone of using AI.

## Public verification is an open problem, not a missing feature

Letting anyone verify without letting anyone forge requires asymmetric watermarking —
accumulators plus NIZK proofs (Fairoze et al., eprint 2023/1661) or pseudorandom
error-correcting codes — and the impossibility results bound what is achievable: robust +
public + distortion-free cannot all hold given sampling access (Zhang et al., *Watermarks in
the Sand*; Christ-Gunn-Zamir, eprint 2024/1739). The practical cost is worse: the Fairoze
construction needs roughly 5,000 contiguous untouched tokens. That is a book chapter, not an
article, so `filigrana` does not ship it and does not pretend otherwise.
