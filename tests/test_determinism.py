"""The determinism contract: a scraped copy must produce the same ledger as the original.

If this fails, nothing else in the project matters -- the writer would have a mark they
cannot verify. Every serialization here is one a content farm or a CMS actually produces.
"""


import re

import pytest

from inkmark.canon import canonical_form
from inkmark.keyring import WriterKey
from inkmark.ledger import walk
from inkmark.providers import default_table

ARTICLE = """

He said that the report was confidential and that nobody outside the team had read it.
Although it was early, the town had already moved on. We talked about it for about an hour,
and afterwards he went toward the station, grey and silent.
Perhaps the point is obvious; however, few people said so. She would rather leave than stay,
and he thought it somewhat cold. Frequently they met at this point in the winter, and in the
end they agreed. In spite of the noise the work went onward, and judgement varied among the
group. A lot of reasons to be careful exist, and several ways to fail as well.
""".strip()

TABLE = default_table("all")
KEY = WriterKey(secret=bytes(range(32)), key_id="test-2026-00", created="2026-01-01", year=2026)


def fingerprint(text: str) -> list[tuple[str, int]]:
    """(class id, final member index) per slot -- the whole evidence base of tier 1."""
    return [(e.class_id, e.observed) for e in walk(text, KEY, TABLE)]


def test_identical_text_is_deterministic():
    assert fingerprint(ARTICLE) == fingerprint(ARTICLE)
    assert len(ARTICLE) > 200


def test_slot_count_is_key_independent():
    other = WriterKey(secret=bytes(32), key_id="other", created="", year=2026)
    assert len(walk(ARTICLE, KEY, TABLE)) == len(walk(ARTICLE, other, TABLE))


@pytest.mark.parametrize(
    "transform",
    ["html", "markdown_wrap", "nbsp", "reflow", "curly_quotes", "smartcaps", "code_block"],
)
def test_serializations_preserve_the_ledger(transform: str):
    base = fingerprint(ARTICLE)
    assert base, "no slots found; the table or the fixture is wrong"
    variant = {
        "html": lambda t: "<article><p>" + "</p><p>".join(
            s.strip() for s in re.split(r"\n(?=\S)", t) if s.strip()
        ) + "</p></article>",
        "markdown_wrap": lambda t: "\n\n".join(
            f"**{s.strip()}**" for s in ARTICLE.split("\n") if s.strip()
        ),
        "nbsp": lambda t: re.sub(r"\s+", lambda m: "\u00a0" if len(m.group()) == 1 else m.group(), t),
        "reflow": lambda t: " ".join(t.split()),
        "curly_quotes": lambda t: t.replace('"', "\u201c"),
        "smartcaps": lambda t: t.upper(),
        "code_block": lambda t: t + "\n\n```python\ntoward the station; although however\n```\n",
    }[transform](ARTICLE)
    assert fingerprint(variant) == base


def test_markup_inside_code_is_never_a_slot():
    fenced = "toward the station\n\n```\ntoward towards although however\n```\n"
    ids = [e.class_id for e in walk(fenced, KEY, TABLE)]
    assert len(ids) == 1, "the watermark must not reach into code blocks"


def test_html_tags_do_not_become_slots():
    tagged = "<div class=toward><a href=/towards>toward the station</a></div>"
    assert [e.class_id for e in walk(tagged, KEY, TABLE)] == ["a.toward"]


ANCHORED = (
    "The engineer walked toward the depot and although the hour was early, he noticed that "
    "perhaps the matter was settled; however, few among the group said so, and afterwards the "
    "judgement changed, acknowledgement came late, and the work went onward regardless."
)


def test_demands_key_off_words_the_watermark_cannot_rewrite():
    """Anchor words are exactly those no class reaches, so embedding never disturbs them."""
    entries = walk(ANCHORED, KEY, TABLE)
    assert entries
    _indices, norms = TABLE.anchors(__import__("inkmark.canon", fromlist=["tokenize"]).tokenize(ANCHORED))
    assert all(norm not in TABLE.reachable for norm in norms)
    for entry in entries:
        assert entry.class_id.split(".")[0] in {"a", "b", "c"}


def test_deleting_a_far_sentence_leaves_distant_demands_unchanged():
    """The contract is locality, not invulnerability: damage must stay near the edit."""
    long_form = ANCHORED + " " + ANCHORED.replace("toward", "towards") + " " + ANCHORED
    before = [(e.class_id, e.wanted) for e in walk(long_form, KEY, TABLE)]
    after = [(e.class_id, e.wanted) for e in walk(long_form.replace(ANCHORED + " ", "", 1), KEY, TABLE)]
    tail_before, tail_after = before[-6:], after[-6:]
    assert len(after) < len(before)
    assert [c for c, _ in tail_before] == [c for c, _ in tail_after], "demands shifted document-wide"


def test_deleting_one_slot_disturbs_only_a_window():
    slots = [e.class_id for e in walk(ARTICLE, KEY, TABLE)]
    assert len(slots) > 4
    removed = ARTICLE.replace("Although it was early,", "It was early,", 1)
    survivors = [e.class_id for e in walk(removed, KEY, TABLE)]
    assert survivors[2:] == slots[3:]


def test_an_unterminated_fence_still_shields_its_words():
    # Forgetting to close a code fence is a real writer error; the words inside must
    # not become slots just because the markup is malformed.
    text = "```py\nshe walked slowly toward the station\n"
    assert walk(text, KEY, TABLE) == []
    assert canonical_form(text) == ""


def test_inline_code_never_becomes_a_slot():
    text = "the `walked slowly` path stayed on the record"
    assert walk(text, KEY, TABLE) == []
    assert canonical_form(text) == "the path stayed on the record"


def test_link_labels_are_prose_but_urls_are_not():
    text = "she [walked](https://example.org/w) toward the hills"
    entries = walk(text, KEY, TABLE)
    assert len(entries) == 1, "the label keeps its slot; the URL must not add one"
    assert entries[0].class_id == "a.toward"
    assert "https" not in canonical_form(text)


def test_relabel_carries_the_originals_case_pattern():
    from inkmark.canon import relabel

    assert relabel("Judgement", "judgment") == "Judgment"
    assert relabel("JUDGEMENT", "judgment") == "JUDGMENT"
    assert relabel("judgement", "judgment") == "judgment"
    assert relabel("A", "big") == "Big"
    assert relabel("", "judgment") == "judgment"
    assert relabel("plain", "Judgment") == "Judgment"  # no pattern to carry
