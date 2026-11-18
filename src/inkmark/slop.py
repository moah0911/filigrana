"""A transparent, dependency-free proxy for what commercial AI detectors key on.

**This is not an AI detector and must not be presented as one.** Originality, Pangram and
GPTZero are closed models on proprietary corpora; their scores cannot be reproduced here and
their internals cannot be inspected. What we can do without pretending is measure the surface
properties those tools are publicly reported to lean on, which is enough to answer the one
question that matters to a writer: does adding my personal watermark make my prose look more
machine-generated to a third party?

The concern is structural, not hypothetical. A watermark makes word choices correlate with a
key, and so does a rigged sampler. A detector trained to tell "statistically nudged" from
"natural" has no obligation to distinguish the two, and the writer is the one who loses the
contract.

Metrics are reported separately rather than blended into a fake authority score:

* burstiness    standard deviation of sentence lengths, reported as *lower* in LLM output.
* ttr           type-token ratio over content words: lexical diversity.
* long_rate     share of words over 10 characters, a cheap formality proxy.
* cliches       hits per 1000 words on patterns named in Anthropic's own explainer
                ("this isn't X, it's Y", the adverb "quietly") plus the list that circulates
                in writer communities.
* connective_rate share of sentences opened with a formal connective.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

CONTENT_STOP = frozenset(
    "the a an and or but if then than that this those these of to in on at by for with from "
    "is are was were be been being it its as not so too very he she they we you i his her".split()
)

CLICHE_PATTERNS = [
    r"\bnot (?:just |merely |only )?[\w ,]{2,40}, (?:but|it'?s)\b",
    r"\bquietly\b", r"\bdelve(?:s|d)?\b", r"\btapestry\b", r"\btestament to\b",
    r"\bin a world where\b", r"\bit'?s worth noting\b", r"\bnuanced\b", r"\bseamless(?:ly)?\b",
    r"\bcomprehensive\b", r"\bcrucial\b", r"\bpivotal\b", r"\bunderscores?\b",
    r"\bfoster(?:s|ing)?\b", r"\bleverage(?:s|d)?\b", r"\bmultifaceted\b", r"\blandscape\b",
    r"\bparadigm\b", r"\bshowcase(?:s|d)?\b", r"\bembod(?:y|ies) of\b", r"\benduring legacy\b",
]

#: Words that read as machine-generated filler where nothing requires them. Several are
#: members of inkmark's own equivalence classes -- `nevertheless`, `frequently`,
#: `hence`, `owing to` -- which is precisely the collision this module exists to manage.
SLOP_WORDS = frozenset(
    """moreover furthermore nevertheless subsequently approximately utilize delve tapestry

    nuanced comprehensive seamless crucial pivotal landscape paradigm showcase testament
    underscores foster leverage multifaceted hence owing notwithstanding wherein whereby
    elucidate augment holistically""".split()
)

FORMAL_OPENERS = re.compile(
    r"^(?:However|Moreover|Furthermore|Nevertheless|Additionally|Subsequently|Hence)\b"
)
_WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
_SENTENCE = re.compile(r"[^.!?]+[.!?]+|[^.!?\n]+")


@dataclass(frozen=True, slots=True)
class StyleProfile:
    words: int
    burstiness: float
    ttr: float
    long_rate: float
    cliches_per_1k: float
    connective_rate: float
    slop_rate: float

    def as_row(self, label: str) -> str:
        return (
            f"{label:<18}{self.words:>6}{self.burstiness:>10.3f}{self.ttr:>9.4f}"
            f"{self.long_rate:>10.4f}{self.cliches_per_1k:>14.3f}"
            f"{self.connective_rate:>16.4f}{self.slop_rate:>10.4f}"
        )


def profile(text: str) -> StyleProfile:
    sentences = [s.group().strip() for s in _SENTENCE.finditer(text) if s.group().strip()]
    lengths = [len(_WORD.findall(s)) for s in sentences]
    all_words = _WORD.findall(text.lower())
    content = [w for w in all_words if w not in CONTENT_STOP]
    mean = sum(lengths) / len(lengths) if lengths else 0.0
    burst = math.sqrt(sum((n - mean) ** 2 for n in lengths) / len(lengths)) if lengths else 0.0
    hits = sum(len(re.findall(pat, text, flags=re.IGNORECASE)) for pat in CLICHE_PATTERNS)
    openers = sum(1 for s in sentences if FORMAL_OPENERS.match(s))
    words = max(len(all_words), 1)
    return StyleProfile(
        words=words,
        burstiness=burst,
        ttr=len(set(content)) / len(content) if content else 0.0,
        long_rate=sum(1 for w in all_words if len(w) > 10) / words,
        cliches_per_1k=1000 * hits / words,
        connective_rate=openers / len(sentences) if sentences else 0.0,
        slop_rate=sum(1 for w in all_words if w in SLOP_WORDS) / words,
    )


def is_slop(surface: str) -> bool:
    """Whether a candidate member is filler. Multi-word members key on their first word."""
    lowered = surface.lower()
    if lowered in SLOP_WORDS:
        return True
    first = lowered.split()[0] if lowered else ""
    # Multi-word members key on their first word; bare members count by direct membership
    # ('owing' is itself listed, so bare 'owing' counts too). This list feeds the measured
    # slop_rate in bench/interference.json -- change it only with a re-run of that bench.
    return first in SLOP_WORDS and lowered != first


def slop_predicate(surface: str) -> bool:
    """The predicate handed to `embed(disprefer=...)`."""
    return is_slop(surface)


__all__ = ["SLOP_WORDS", "StyleProfile", "is_slop", "profile", "slop_predicate"]
