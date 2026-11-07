"""Canonical text form: the single place where serialization differences get absorbed.

Text-alone detection only works if a word found in Markdown, in HTML, and in a plain-text
paste yields the same equivalence class and the same context. Offsets are preserved by
masking markup with spaces instead of deleting it, so substitutions always write back into
whatever string the caller handed us.

NFC, deliberately not NFKC: NFKC folds ligatures and halfwidth forms, which rewrites words
(``ﬁne`` -> ``fine``) and would silently move slot boundaries.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Words and numerals are both tokens: numeral adjacency ("about ten") has to be visible to
# guards. Punctuation is not a token except as a clause end, read off the raw string.
WORD_RE = re.compile(r"[A-Za-z][A-Za-z’'’\-]*")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_TOKEN_SCAN = re.compile(r"\d+(?:[.,]\d+)?|[A-Za-z][A-Za-z’'’\-]*")

_HTML_TAG_RE = re.compile(r"<[^<>]{0,400}>")
_MD_FENCE_RE = re.compile(r"^([ \t]*)(```+|~~~+)[^\n]*$", re.MULTILINE)
_MD_INLINE_CODE_RE = re.compile(r"(`+)(?:[^`\\\n]|\\.)*?\1")
_MD_LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]*)\)")
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__|\*|_)(?=\S)")


@dataclass(frozen=True, slots=True)
class Token:
    surface: str
    start: int
    end: int
    kind: str = "word"  # "word" | "num"

    @property
    def norm(self) -> str:
        if self.kind == "num":
            return "#"
        return self.surface.lower().strip("-’'")


def mask_markup(text: str) -> str:
    """Blank out markup so every offset stays identical.

    Nothing inside these regions is prose: a substitution in code or in a URL changes
    behaviour rather than style, and the watermark must never touch it.
    """
    out = list(text)

    def blank(start: int, end: int) -> None:
        for i in range(max(start, 0), min(end, len(out))):
            if out[i] not in "\n\t":
                out[i] = " "

    for m in _HTML_TAG_RE.finditer(text):
        blank(m.start(), m.end())

    open_fence: re.Match | None = None
    for m in _MD_FENCE_RE.finditer(text):
        if open_fence is None:
            open_fence = m
        elif m.group(2)[:3] == open_fence.group(2)[:3]:
            blank(open_fence.start(), m.end())
            open_fence = None
    if open_fence is not None:  # unterminated fence runs to end of text
        blank(open_fence.start(), len(text))

    for m in _MD_INLINE_CODE_RE.finditer(text):
        blank(m.start(), m.end())

    for m in _MD_LINK_RE.finditer(text):
        # Keep the link label (it is prose); blank the brackets and the URL.
        blank(m.start(), m.start(2))
        blank(m.end(2), m.end())

    for m in _MD_EMPHASIS_RE.finditer(text):
        blank(m.start(), m.end())

    return "".join(out)


def tokenize(text: str) -> list[Token]:
    """Tokens over the markup-masked view, so offsets index the caller's string."""
    masked = mask_markup(text)
    tokens: list[Token] = []
    for m in _TOKEN_SCAN.finditer(masked):
        surface = m.group()
        kind = "num" if NUM_RE.fullmatch(surface) else "word"
        tokens.append(Token(surface, m.start(), m.end(), kind))
    return tokens


def canonical_form(text: str) -> str:
    """Whitespace-normalised lowercase token stream. Compare this to prove invariance."""
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(t.norm for t in tokenize(normalized))


def relabel(source: str, replacement: str) -> str:
    """Carry the source's casing pattern onto the replacement, for title-cased headings."""
    if not source or not replacement:
        return replacement
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[0].isupper():
        if source[1:].islower() or len(source) == 1:
            return replacement.capitalize()
    return replacement
