"""WordNet as a candidate proposer -- broad, sense-blind, and only safe behind the gate.

The curated table is precise and thin: 8.3 free choices per 1,000 words. WordNet is the
obvious way to get density, and it is also the obvious way to write "the crowd traipsed"
where the author meant "marched", so this provider is deliberately built to be filtered:

* it proposes lemmas from every synset that matches a *guessed* part of speech, without doing
  word-sense disambiguation. Wrong-sense candidates are expected and are not the bug;
* the caller must pair it with a scorer, because a wrong sense is precisely the thing that
  gets a low conditional probability in real context. `FreeChoiceGate` prunes to candidates
  inside one epsilon-wide band, which is where sense errors die and style choices survive;
* regular inflection only. `walked` proposes `strolled` and `marched`; `went` proposes
  nothing, because mapping an irregular form back through WordNet lemmas and re-inflecting it
  produces "goed". Irregular verbs are a real chunk of prose and this module does not pretend.

Using this provider *without* the gate is refused rather than merely discouraged: the same
class of mistake is what makes automated-synonym watermarking measure 17% new grammatical
errors in the published experiments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .providers import Slot

_ADJ_SUFFIXES = ("ous", "ive", "ful", "ent", "ant", "able", "ible", "al", "ic", "ary")
_MAX_PROPOSALS = 6


#: Which WordNet part-of-speech tag a suffix proves. None means the shape says nothing, and
#: guessing here is what produced "marchlanded" from a noun: without a required POS the
#: re-inflection is untyped arithmetic on words.
SUFFIX_POS = {"ed": "v", "ing": "v", "ly": "r", "s": ("n", "v"), "": None}


@dataclass(frozen=True, slots=True)
class Inflection:
    lemma: str
    suffix: str

    def form(self, other_lemma: str) -> str:
        return other_lemma + self.suffix

    def inflect(self, other_lemma: str) -> str | None:
        """Re-inflect a sibling lemma the way the observed word was inflected, or refuse.

        WordNet stores dictionary forms and no conjugations, so this is a suffix machine:
        right for the regular hundreds, wrong for `speak`/`spoke`, wrong for `bunch`+`s`.
        Irregular forms cannot be fixed from a lemma list at all -- they get pruned by the
        band filter, which is the design, not an apology: a model gives `speaked` a
        probability of essentially zero in every context. What is fixed here is the
        self-evident nonsense: doubled suffixes and plural spelling.
        """
        if self.suffix in ("ed", "ing", "ly") and other_lemma.endswith(self.suffix):
            return None  # `quick` + `ly` is fine; `quickly` + `ly` is `quicklyly`
        if self.suffix == "s" and other_lemma.endswith(("s", "x", "z", "ch", "sh")):
            other_lemma += "e"  # bunch -> bunches, not bunchs
        candidate = other_lemma + self.suffix
        return candidate if re.fullmatch(r"[a-z]{4,}", candidate) else None

    @property
    def required_pos(self) -> tuple[str, ...]:
        return tuple(SUFFIX_POS.get(self.suffix) or ())


def parse_inflection(surface: str) -> Inflection | None:
    """Recover lemma+suffix for regular inflections only.

    Returns None for anything irregular, abbreviated, or capitalised as a name; a wrong
    guess here is worse than no candidate at all.
    """
    word = surface.lower()
    if not re.fullmatch(r"[a-z]{5,}", word):
        return None
    if word.endswith("ing"):
        for lemma in (word[:-3], word[:-3] + "e", word[:-3] + word[-4]):
            if len(lemma) >= 3 and _lemma_known(lemma):
                suffix = "ing"
                return Inflection(lemma, suffix)
        return None
    if word.endswith("ed"):
        for lemma in (word[:-2], word[:-1]):
            if len(lemma) >= 3 and _lemma_known(lemma):
                return Inflection(lemma, "ed")
        return None
    if word.endswith("ly"):
        lemma = word[:-2]
        if _lemma_known(lemma):
            return Inflection(lemma, "ly")
        return None
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        for lemma in (word[:-1], word[:-2] if word.endswith(("ies", "es")) else word[:-1]):
            if len(lemma) >= 3 and _lemma_known(lemma):
                return Inflection(lemma, "s")
        return None
    if _lemma_known(word):
        return Inflection(word, "")
    return None


@lru_cache(maxsize=4096)
def _lemma_known(lemma: str) -> bool:
    return bool(_synsets(lemma))


@lru_cache(maxsize=4096)
def _synsets(lemma: str) -> tuple[str, ...]:
    """Synset names for a lemma, keyed by the POS we would guess for it."""
    wordnet = _wordnet()
    if wordnet is None:
        return ()
    return tuple(sorted(s.name() for s in wordnet.synsets(lemma)))


def _wordnet():
    try:
        from nltk.corpus import wordnet
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "WordNetProvider needs 'inkmark[lexical]' and a WordNet corpus: "
            "uv pip install nltk && python -c \"import nltk; nltk.download('wordnet'); "
            "nltk.download('omw-1.4')\""
        ) from exc
    try:
        wordnet.synsets("verify")
    except LookupError as exc:  # pragma: no cover - corpus not downloaded
        raise RuntimeError(
            "WordNet lexicon is not downloaded. Run: python -c \"import nltk; "
            "nltk.download('wordnet'); nltk.download('omw-1.4')\""
        ) from exc
    return wordnet


class WordNetProvider:
    """Yields Slot proposals for words with same-POS WordNet siblings."""

    tier = "lm"

    def __init__(self, *, max_members: int = 4, min_word_len: int = 5) -> None:
        if max_members < 2:
            raise ValueError("a one-member proposal is not a choice")
        self.max_members = max_members
        self.min_word_len = min_word_len
        self.requires_gate = True

    def slots(self, text: str, tokens=None) -> list[Slot]:
        from .canon import tokenize

        toks = list(tokens) if tokens is not None else tokenize(text)
        out: list[Slot] = []
        for i, token in enumerate(toks):
            if token.kind != "word" or len(token.surface) < self.min_word_len:
                continue
            if not token.surface.islower() or not token.surface.isalpha():
                # Capitalised words are names and sentence starts often enough that a
                # sense-blind proposal would rewrite proper nouns.
                continue
            parsed = parse_inflection(token.surface)
            if parsed is None or not parsed.required_pos:
                # An uninflected word could be a noun, verb or adjective, and WordNet's
                # senses disagree about which. Proposing anyway re-inflects the wrong
                # part of speech onto it, so base forms are left alone.
                continue
            siblings = self._siblings(parsed)
            if len(siblings) < 2:
                continue
            members = tuple(sorted(siblings))[: self.max_members]
            if token.surface not in members:
                continue
            out.append(
                Slot(
                    class_id=f"wn:{parsed.lemma}",
                    tier=self.tier,
                    start=token.start,
                    end=token.end,
                    token_index=i,
                    n_tokens=1,
                    members=members,
                    observed=members.index(token.surface),
                    matched=(token.surface,),
                )
            )
        return out

    def _siblings(self, parsed: Inflection) -> set[str]:
        """English lemmas from every synset *in the POS the inflection proves*, most frequent
        senses first.

        Not one synset: `walk.v.01` has no sibling lemmas at all, and restricting to it gives
        an empty proposal. Not every synset either: that is how the first version produced
        `marchlanded` (a noun sense of march) and `paseoed` (a Spanish lemma). Constraining the
        part of speech kills the re-inflection nonsense, and the band test in the gate is what
        removes the remaining wrong *senses*, which a context-free lexicon cannot see.

        Senses stay in WordNet's own frequency order, so the first `max_members` are the
        commonest readings rather than whichever sort alphabetically.
        """
        wordnet = _wordnet()
        if wordnet is None:
            return set()
        base = parsed.form(parsed.lemma)  # the observed word itself, always a member
        forms = {base}
        for name in self._ordered_synsets(parsed):
            if len(forms) - 1 >= self.max_members - 1:
                break
            for lemma in wordnet.synset(name).lemma_names():
                lemma = lemma.lower().replace("_", " ")
                if not lemma.isascii() or not re.fullmatch(r"[a-z]{3,}", lemma):
                    continue
                if lemma == parsed.lemma:
                    continue
                candidate = parsed.inflect(lemma)
                if candidate is None:
                    continue
                forms.add(candidate)
        return forms if len(forms) > 1 else set()

    @staticmethod
    def _ordered_synsets(parsed: Inflection) -> list[str]:
        """Synsets restricted to the POS the suffix proves, in WordNet sense order."""
        names = _synsets(parsed.lemma)
        allowed = set(parsed.required_pos)
        # Synset names are `lemma.pos.number`, so the POS is the second-to-last component.
        # Reading `rsplit(".", 1)[-1]` here returns the *number*, which silently matched no
        # sense at all -- and, before that, silently matched every sense. Both directions of
        # the same off-by-one, and both invisible in the output.
        return [name for name in names if name.split(".")[1:2] and name.split(".")[1] in allowed]


__all__ = ["Inflection", "WordNetProvider", "parse_inflection"]
