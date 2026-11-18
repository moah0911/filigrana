"""Equivalence classes: the hand-audited source of free choices.

This table is the only reason a watermark can touch human text at all, so it lives in data
rather than code -- it needs auditing, versioning, and disagreement. Everything else in
`filigrana` treats it as a given.

A class is admissible only if every member is substitutable for every other member in
every context the class appears in: symmetric, not merely "usually synonymous". Direction
matters, which is why `help`/`aid` is excluded -- `foreign aid` cannot become `foreign
help`. Automated synonym tables get this wrong and measure ~17% of passages gaining a
grammatical error.

Members are stored lowercase; the sorted tuple is the canonical index order that embedder
and detector agree on, so an index never depends on which form the writer happened to use.

Tiers, because risk is not uniform:
  a  orthographic variant; no reader notices either way
  b  low-salience lexical or connective variant, slight register shift
  c  locale- or register-marked (BrE/AmE, contractions). Highest density, but mixing
     forms inside one article is a visible inconsistency, so it is off by default.

Guards (`after`, `not_after`, `before`, `not_before`, `avoid_clause_final`) exist because
most unsafe substitutions are context-specific rather than word-specific.
"""


import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib.resources import files

from .canon import Token, tokenize

_CLAUSE_END = re.compile(r"[.,;:!?]")
_NUMERAL = re.compile(r"\d")
_TIER_LETTERS = "abc"


def normalise_tier(value: str) -> str:
    """Canonical tier spec: sorted, de-duplicated a/b/c letters, or ``all``.

    Accepting ``"ba"`` and returning ``"ab"`` keeps the CLI forgiving without letting a
    typo like ``"xyz"`` silently produce a table with no classes at all.
    """

    text = value.strip().lower()
    if text == "all":
        return "all"
    letters = frozenset(text)
    if not letters or not letters <= frozenset(_TIER_LETTERS):
        raise ValueError(f"tier must be a, b, c, ab, ac, bc, abc, or all, not {value!r}")
    return "".join(letter for letter in _TIER_LETTERS if letter in letters)


class AmbiguousSurface(Exception):
    """A word belongs to more than one class, so no deterministic single answer exists."""



@dataclass(frozen=True, slots=True)
class EquivalenceClass:
    id: str
    members: tuple[str, ...]
    tier: str
    note: str = ""
    after: tuple[str, ...] = ()
    not_after: tuple[str, ...] = ()
    before: tuple[str, ...] = ()
    not_before: tuple[str, ...] = ()
    avoid_clause_final: bool = False
    locale: bool = False
    register: bool = False

    def __post_init__(self) -> None:
        if len(self.members) < 2:
            raise ValueError(f"class {self.id!r} needs at least two members")
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"class {self.id!r} has repeated members")
        if self.members != tuple(sorted(self.members)):
            raise ValueError(f"class {self.id!r} members must arrive sorted")

    @property
    def first_tokens(self) -> frozenset[str]:
        return frozenset(member.split()[0] for member in self.members)

    @property
    def max_tokens(self) -> int:
        return max(len(member.split()) for member in self.members)

    def token_spans(self) -> dict[tuple[str, ...], int]:
        """Member surface forms as token tuples, mapped to their canonical index."""
        spans: dict[tuple[str, ...], int] = {}
        for index, member in enumerate(self.members):
            parts = tuple(member.lower().split())
            if parts:
                spans[parts] = index
        return spans


@dataclass(frozen=True, slots=True)
class Slot:
    """One key-correlated decision, with its position in the token stream."""

    class_id: str
    tier: str
    start: int
    end: int
    token_index: int
    n_tokens: int
    members: tuple[str, ...]
    observed: int
    matched: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.members)


class CuratedTable:
    """Finds equivalence-class slots in prose, left to right, longest match first.

    Non-overlapping and deterministic: the same input string always yields the same slot
    list, which is what lets a verifier re-derive the ledger from a scraped copy.
    """

    def __init__(self, classes: Iterable[EquivalenceClass]) -> None:
        self.classes: dict[str, EquivalenceClass] = {}
        self._by_first: dict[str, list[EquivalenceClass]] = {}
        self.reachable: set[str] = set()
        for cls in classes:
            if cls.id in self.classes:
                raise ValueError(f"duplicate class id {cls.id!r}")
            self.classes[cls.id] = cls
            for first in cls.first_tokens:
                self._by_first.setdefault(first, []).append(cls)
            for member in cls.members:
                self.reachable.update(member.lower().split())
        # Longest first so `a lot of` wins over a bare `a`-initial member.
        for bucket in self._by_first.values():
            bucket.sort(key=lambda c: (-c.max_tokens, c.id))

    def anchors(self, tokens) -> tuple[list[int], list[str]]:
        """Token indices and norms for words the watermark can never rewrite.

        These are the only words safe to key a demand off, because after embedding they are
        guaranteed to still say the same thing.
        """
        indices, norms = [], []
        for i, token in enumerate(tokens):
            if token.kind == "word" and token.norm not in self.reachable:
                indices.append(i)
                norms.append(token.norm)
        return indices, norms

    @classmethod
    def load(cls, tier: str = "ab") -> "CuratedTable":
        raw = json.loads(
            files("filigrana.data").joinpath("equivalence.json").read_text(encoding="utf-8")
        )
        if not isinstance(raw, dict) or not isinstance(raw.get("classes"), list):
            raise ValueError("equivalence table must be a JSON object with a 'classes' list")
        canonical = normalise_tier(tier)
        allowed = frozenset(_TIER_LETTERS) if canonical == "all" else frozenset(canonical)
        classes = []
        for item in raw["classes"]:
            if item["tier"] not in allowed:
                continue
            classes.append(
                EquivalenceClass(
                    id=item["id"],
                    members=tuple(sorted(item["members"])),
                    tier=item["tier"],
                    note=item.get("note", ""),
                    after=tuple(item.get("after", ())),
                    not_after=tuple(item.get("not_after", ())),
                    before=tuple(item.get("before", ())),
                    not_before=tuple(item.get("not_before", ())),
                    avoid_clause_final=bool(item.get("avoid_clause_final")),
                    locale=bool(item.get("locale")),
                    register=bool(item.get("register")),
                )
            )
        return cls(classes)

    def slots(self, text: str, tokens: Sequence[Token] | None = None) -> list[Slot]:
        """All choice points in `text`, in ascending offset order."""
        toks = list(tokens) if tokens is not None else tokenize(text)
        found: list[Slot] = []
        i = 0
        while i < len(toks):
            slot = self._match_at(text, toks, i)
            if slot is None:
                i += 1
                continue
            found.append(slot)
            i += self._advance(toks, slot)
        return found

    def _advance(self, toks: Sequence[Token], slot: Slot) -> int:
        consumed = 0
        for token in toks:
            if token.start >= slot.start and token.start < slot.end:
                consumed += 1
        return max(consumed, 1)

    def _match_at(self, text: str, toks: Sequence[Token], i: int) -> Slot | None:
        candidates = self._by_first.get(toks[i].norm)
        if not candidates:
            return None
        for cls in candidates:
            for span, index in cls.token_spans().items():
                if len(span) > len(toks) - i:
                    continue
                if tuple(t.norm for t in toks[i : i + len(span)]) != span:
                    continue
                if not self._guards_ok(text, toks, i, len(span), cls):
                    continue
                start, end = toks[i].start, toks[i + len(span) - 1].end
                return Slot(
                    class_id=cls.id,
                    tier=cls.tier,
                    start=start,
                    end=end,
                    token_index=i,
                    n_tokens=len(span),
                    members=cls.members,
                    observed=index,
                    matched=tuple(t.surface for t in toks[i : i + len(span)]),
                )
        return None

    def _guards_ok(
        self, text: str, toks: Sequence[Token], i: int, length: int, cls: EquivalenceClass
    ) -> bool:
        prev = toks[i - 1].norm if i > 0 else ""
        nxt = toks[i + length].norm if i + length < len(toks) else ""
        if cls.after and prev not in cls.after:
            return False
        if cls.not_after and prev in cls.not_after:
            return False
        if cls.before:
            if nxt not in cls.before:
                return False
        if cls.not_before and nxt in cls.not_before:
            return False
        if cls.avoid_clause_final:
            if not nxt:  # member is the last word in the document
                return False
            following_char = text[toks[i + length - 1].end : toks[i + length - 1].end + 1]
            if _CLAUSE_END.match(following_char):  # clause-final, where the classes diverge
                return False
        return True

    def audit(self) -> list[str]:
        """Surfaces claimed by more than one class -- refuses to guess at those."""
        seen: dict[str, list[str]] = {}
        for cls in self.classes.values():
            for member in cls.members:
                seen.setdefault(member, []).append(cls.id)
        return [f"{word}: {ids}" for word, ids in sorted(seen.items()) if len(ids) > 1]

    @property
    def class_count(self) -> int:
        return len(self.classes)

    @property
    def member_count(self) -> int:
        return sum(len(c.members) for c in self.classes.values())


def default_table(tier: str = "ab") -> CuratedTable:
    return CuratedTable.load(tier)


__all__ = [
    "AmbiguousSurface",
    "CuratedTable",
    "EquivalenceClass",
    "Slot",
    "default_table",
    "normalise_tier",
]
