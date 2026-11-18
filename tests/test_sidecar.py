"""Model-free verification from a published ledger.

This is what makes the model path defensible: a model may decide which positions were free
at publish time, but the check afterwards must not depend on that model still existing, still
being installed, or still behaving the same way. The sidecar pins the band and the anchor, so
verification is a table lookup plus a hash.

The failure modes worth testing are the quiet ones: an excerpt that reads as *fabricated*
rather than merely weaker, and a reflowed copy whose slots stop matching because somebody
stored a character offset instead of a word.
"""

from __future__ import annotations

import json

import pytest

from conftest import KEY, SCRIPT, SENTENCE, ScriptedProvider, ScriptedScorer
from filigrana.embed import embed
from filigrana.keyring import WriterKey
from filigrana.lm import embed_gated
from filigrana.providers import default_table
from filigrana.sidecar import build_sidecar, locate, verify

# Distinct paragraphs, not one repeated: a mark is a *local* pattern, so repeating a single
# paragraph and then "shuffling" it would prove nothing about ordering, and would let a
# permutation of identical text pass a test that is supposed to fail it.
PARAGRAPHS = [
    ("She walked slowly toward the grey station, and the crowd parted before the engineers "
    "marched off into the rain. Nobody said that the report was anything other than "
    "confidential, and afterwards the judgement varied among them."),
    ("It was a dull morning, and although the enquiry had begun earlier, nothing had been "
    "written down. Perhaps the point was obvious; however, few said so, and she thought it "
    "somewhat cold. Frequently they met at this point in the winter."),
    ("In spite of the noise the work went onward, and acknowledgement came late. A lot of "
    "reasons existed to be careful, and several ways to fail as well. He would rather leave "
    "than stay, and the colour of the building did not change."),
    ("Among the group, the enquiry continued, and he did not recognise the behaviour of the "
    "board. She said that the record was complete, and that nobody had altered it, although "
    "the judgement of the committee was another matter entirely."),
    ("He walked toward the depot afterwards, grey and silent, and the crowd parted slowly. "
    "Perhaps the reason was obvious, and however few said so, the report remained what it was."),
    ("They talked about it for about an hour, and in the end they agreed on the matter. "
    "Frequently the work went onward in spite of the noise, and judgement varied among the "
    "engineers who had begun to worry about the winter."),
]
PROSE = " ".join(PARAGRAPHS * 3)


def _marked():
    return embed(PROSE, KEY, table=default_table("all"))


def test_curated_sidecar_verifies_without_a_model():
    marked = _marked()
    ledger = marked.sidecar()
    assert ledger["format"] == "filigrana-sidecar/2"
    decision = verify(marked.text, KEY, ledger)
    assert decision.slots == marked.slots
    assert decision.hits == decision.slots
    assert decision.p_value < 1e-6


def test_sidecar_verification_survives_reflow_and_case():
    marked = _marked()
    ledger = marked.sidecar()
    reflowed = "\n\n".join(
        " ".join(marked.text.split()[i : i + 9]) for i in range(0, len(marked.text.split()), 9)
    )
    decision = verify(reflowed, KEY, ledger)
    assert decision.slots >= 0.9 * marked.slots, "anchors are words, not offsets"
    assert decision.hits == decision.slots
    assert verify(marked.text.lower(), KEY, ledger).hits == decision.hits


def test_excerpt_reads_weaker_and_never_guilty():
    marked = _marked()
    words = marked.text.split()
    excerpt = " ".join(words[len(words) // 4 : 3 * len(words) // 4])
    decision = verify(excerpt, KEY, marked.sidecar())
    assert decision.slots < marked.slots
    assert decision.hits == decision.slots, "a lost slot must be absent, not a disagreement"
    assert any("could still be located" in c for c in decision.caveats)


def test_foreign_key_cannot_pass_sidecar_verification():
    marked = _marked()
    stranger = WriterKey(secret=b"9" * 32, key_id="stranger", created="", year=2026)
    quiet = verify(marked.text, stranger, marked.sidecar())
    assert quiet.slots == marked.slots, "slot locations do not depend on the key"
    assert abs(quiet.hits - quiet.expected) < quiet.slots / 2 + 1
    assert quiet.z < 3.5


def test_reordering_whole_paragraphs_loses_the_local_pattern():
    """Each record's anchor is the few words before it, so a permutation breaks the pairs that
    crossed a paragraph boundary and the mark degrades instead of falsely accusing."""
    marked = _marked()
    shuffled = " ".join(list(reversed(PARAGRAPHS)) * 3)
    decision = verify(shuffled, KEY, marked.sidecar())
    assert decision.slots < marked.slots, "a permutation must lose anchors, not preserve them"
    assert decision.z < 4.754, "a shuffled text must not be reported as this writer's"


def test_a_longer_marked_document_clears_the_strict_threshold():
    marked = _marked()
    decision = verify(marked.text, KEY, marked.sidecar())
    assert decision.slots >= 25
    assert decision.hits == decision.slots
    assert decision.p_value < 1e-6


def test_locate_consumes_each_record_once_in_document_order():
    marked = _marked()
    entries = locate(marked.text, marked.sidecar())
    offsets = [e.start for e in entries]
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


def test_gated_sidecar_names_its_model_and_is_checkable_without_it():
    marked = embed_gated(SENTENCE, KEY, ScriptedProvider(SCRIPT), ScriptedScorer())
    decision = verify(marked.text, KEY, marked.sidecar)
    assert decision.slots == marked.slots
    assert decision.hits == decision.slots
    assert any("scripted/1" in c for c in decision.caveats)


def test_sidecar_round_trips_through_json():
    marked = _marked()
    ledger = build_sidecar(
        marked.entries, key_id=KEY.key_id, reachable=default_table("all").reachable
    )
    assert verify(marked.text, KEY, json.loads(json.dumps(ledger))).hits == marked.slots


def test_rewriting_a_marked_word_out_of_its_class_drops_only_that_slot():
    marked = _marked()
    assert marked.slots >= 3
    for original, replacement in (("toward", "in the direction of"), ("grey", "dull coloured")):
        edited = marked.text.replace(original, replacement, 1)
        if edited == marked.text:
            continue
        decision = verify(edited, KEY, marked.sidecar())
        assert decision.slots >= marked.slots - 2, f"{original} loss cascaded"
        assert decision.hits == decision.slots
        return
    pytest.skip("none of the expected surfaces occurred")


def test_detect_with_a_sidecar_never_claims_text_alone():
    """The caveat list must tell the truth about which material the verdict rests on. It
    used to print "computed from the text alone" even when a sidecar had been consumed --
    a tool that lies about its own provenance in the same breath as its statistic cannot
    be trusted on anything else."""
    from filigrana.detect import scan

    marked = embed(
        "She walked slowly toward the grey station, and the crowd parted before the "
        "engineers marched off into the rain. Nobody said that the report was anything "
        "other than confidential, and afterwards the judgement varied among them.",
        KEY,
        table=default_table("all"),
    )
    ledger = json.dumps(marked.sidecar())
    ledger = json.loads(ledger)
    decision = scan(marked.text, KEY, table=default_table("all"), sidecar=ledger)
    text = " ".join(decision.caveats)
    assert "Computed from the text alone" not in text
    assert "Verified against the sidecar's publication-time ledger" in text
    alone = scan(marked.text, KEY, table=default_table("all"))
    assert "Computed from the text alone" in " ".join(alone.caveats)


def test_locate_drops_a_span_that_runs_off_the_end_of_the_copy():
    """A truncated copy must neither crash nor mis-score: a multi-word member whose span
    extends past the end of the text simply cannot match, and the slot is absent."""
    marked = _marked()
    ledger = json.loads(json.dumps(marked.sidecar()))
    full = locate(marked.text, ledger)
    entry = next(e for e in full if " " in e.members[e.observed])
    truncated = marked.text[: entry.start + len(entry.members[entry.observed].split()[0])]
    remaining = locate(truncated, ledger)
    assert entry.class_id not in {e.class_id for e in remaining}
    assert [e.class_id for e in remaining] == [
        e.class_id for e in full if e.end <= entry.start
    ], "records wholly before the cut still re-derive, in order"
    decision = verify(truncated, KEY, ledger)
    assert decision.hits == decision.slots, "a lost slot is absent, not a disagreement"


def test_locate_drops_a_slot_whose_anchor_no_longer_matches():
    marked = _marked()
    ledger = json.loads(json.dumps(marked.sidecar()))
    full = locate(marked.text, ledger)
    assert full
    broken = json.loads(json.dumps(marked.sidecar()))
    broken["slots"][0]["anchor"] = ["xyzzy", "plugh"]
    remaining = locate(marked.text, broken)
    assert [e.class_id for e in remaining] == [e.class_id for e in full[1:]], (
        "a slot whose anchor was edited is dropped, and the rest locate untouched"
    )


def test_a_sidecar_from_another_format_is_refused():
    with pytest.raises(ValueError, match="not a filigrana sidecar"):
        locate("some text", {"format": "filigrana-sidecar/1", "slots": []})
    with pytest.raises(ValueError, match="not a filigrana sidecar"):
        verify("some text", KEY, {"format": "filigrana-sidecar/1", "slots": []})


def test_a_malformed_slot_is_named_instead_of_crashing():
    with pytest.raises(ValueError, match="non-empty list of words"):
        locate(
            "some text",
            {
                "format": "filigrana-sidecar/2",
                "window": 4,
                "reachable": [],
                "slots": [{"class": "a.x", "members": [], "final": 0, "anchor": []}],
            },
        )


def test_swapping_a_marked_word_inside_its_class_reads_as_absent_not_guilty():
    """The published member index is part of the match. Two occurrences of a member can share
    an anchor once reachable words are skipped; binding to the wrong one scored a published
    word as present when the copy no longer contained it."""
    marked = _marked()
    ledger = marked.sidecar()
    full = locate(marked.text, ledger)
    entry = next(e for e in full if e.class_id == "a.toward" and e.n == 2)
    replacement = entry.members[(entry.observed + 1) % entry.n]
    edited = marked.text[: entry.start] + replacement + marked.text[entry.end :]
    decision = verify(edited, KEY, ledger)
    assert decision.slots == len(full) - 1, "the swapped slot is absent, not a disagreement"
    assert decision.hits == decision.slots
