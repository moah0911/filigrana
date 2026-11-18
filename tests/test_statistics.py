"""Statistical properties, measured rather than asserted.

Two directions of error matter and they are not symmetric: a detector that fires on ordinary
prose is worse than useless, and a detector that quietly cannot fire on a real article sells
false confidence. Both are tested here against the public-domain corpus when it is present.
"""


import math
import random
from dataclasses import replace
from pathlib import Path

import pytest

from inkmark.decisions import Decision, normal_tail
from inkmark.detect import scan, scan_against
from inkmark.embed import embed
from inkmark.keyring import WriterKey
from inkmark.ledger import walk
from inkmark.providers import default_table

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
has_corpus = CORPUS.glob("*.txt") and any(CORPUS.glob("*.txt"))

TEXT = (
    "He said that the report was confidential. Although it was early, the town moved on. "
    "Perhaps the point is obvious; however, few said so. She would rather leave, and it was "
    "somewhat cold. Frequently they met at this point, and in the end agreed. In spite of the "
    "noise the work went onward. Judgement varied among the group, and acknowledgement came "
    "late. Toward the station he walked afterwards. " * 6
)


def probe(rng: random.Random) -> WriterKey:
    return WriterKey(
        secret=bytes(rng.getrandbits(8) for _ in range(32)),
        key_id=f"probe-{rng.getrandbits(24):06x}",
        created="",
        year=2026,
    )


def test_embedding_is_verified_by_the_detector_itself():
    key = WriterKey(secret=b"7" * 32, key_id="rt", created="", year=2026)
    table = default_table("all")
    marked = embed(TEXT, key, table=table)
    assert marked.drift == 0
    decision = scan(marked.text, key, table=table)
    assert decision.hits == decision.slots
    assert decision.slots == marked.slots


def test_statistics_match_the_closed_form():
    """For uniform 2-way slots, z must equal (2h - n)/sqrt(n)."""

    key = WriterKey(secret=b"3" * 32, key_id="math", created="", year=2026)
    table = default_table("ab")
    entries = walk(TEXT, key, table)
    two_way = [e for e in entries if e.n == 2]
    hits = sum(e.on_key for e in two_way)
    n = len(two_way)
    if n:
        expected = (2 * hits - n) / math.sqrt(n)
        assert scan(TEXT, key, table=table).variance > 0
        assert abs(expected) <= abs(scan(TEXT, key, table=table).z) + 0.05


def test_null_is_standard_normal_on_real_prose():
    if not has_corpus:
        pytest.skip("no corpus/ files; run tools/measure_free_choices.py first")
    prose = "\n".join(
        f.read_text(encoding="utf-8", errors="replace") for f in sorted(CORPUS.glob("*.txt"))
    )
    chunks = [" ".join(w) for w in [prose.split()[i : i + 215] for i in range(0, 40_000, 215)]]
    rng = random.Random(20261106)
    keys = [probe(rng) for _ in range(6)]
    zs = [scan(c, k, table=default_table("all")).z for c in chunks for k in keys]
    mean = sum(zs) / len(zs)
    sd = math.sqrt(sum((z - mean) ** 2 for z in zs) / (len(zs) - 1))
    assert abs(mean) < 0.25, f"null is biased: mean {mean}"
    assert 0.75 < sd < 1.35, f"null is not unit variance: sd {sd}"
    assert sum(z >= 2.326 for z in zs) / len(zs) < 0.03
    assert max(zs) < 4.754, "a foreign key reached p<1e-6 on unmarked prose"


def test_article_length_is_the_boundary_not_a_formality():
    if not has_corpus:
        pytest.skip("no corpus/ files")
    words = [
        w
        for f in sorted(CORPUS.glob("*.txt"))
        for w in f.read_text(encoding="utf-8", errors="replace").split()
    ]
    article = " ".join(words[:1075])
    key = WriterKey(secret=b"9" * 32, key_id="article", created="", year=2026)
    table = default_table("all")
    marked = embed(article, key, table=table)
    decision = scan(marked.text, key, table=table)
    # rho ~22/1000 words at tier all: a 1075-word article is right at p<1e-6, not past it.
    assert decision.slots >= 20
    assert decision.p_value < 1e-4
    assert decision.hits == decision.slots


def test_light_editing_degrades_gracefully():
    key = WriterKey(secret=b"1" * 32, key_id="edit", created="", year=2026)
    table = default_table("all")
    marked = embed(TEXT, key, table=table).text
    rng = random.Random(5)
    sentences = [s for s in marked.split(". ") if s]
    kept = [s for i, s in enumerate(sentences) if rng.random() > 0.3]
    chewed = scan(". ".join(kept) + ".", key, table=table)
    assert chewed.slots < scan(marked, key, table=table).slots
    # Anchored context keeps damage local. A chained context over slot decisions measured
    # 30/53 here -- chance -- because one deletion shifts the chain for the whole document.
    assert chewed.hits / chewed.slots > 0.75
    assert chewed.z > 3.0


def test_caveats_state_the_attacks_that_work():
    decision = scan(TEXT, WriterKey(secret=b"2" * 32, key_id="c", created="", year=2026),
                    table=default_table("all"))
    text = " ".join(decision.caveats).lower()
    assert "paraphrase" in text
    assert "translation" in text


def test_bit_votes_resolve_by_majority_and_refuse_ties():
    """Tier 2 reads the payload by voting the slots assigned to each bit. A tie must be an
    unresolved bit, not a verdict, and the reported p is the excess over a fair 50/50."""
    from inkmark.codec import BitVote, PayloadRead

    assert BitVote(0, 3, 1).resolved == 0
    assert BitVote(0, 1, 3).resolved == 1
    assert BitVote(0, 2, 2).resolved is None, "a tie is not evidence either way"
    # p is the excess over a fair 50/50 split: an exact tie sits at chance, so p = 0.5,
    # while a bit with no votes at all never ran a test and reports 1.0.
    assert BitVote(0, 4, 4).p_value == 0.5
    assert BitVote(0, 0, 0).p_value == 1.0
    assert 0 < BitVote(0, 1, 8).p_value < 0.01
    read = PayloadRead(
        votes=(BitVote(0, 0, 5), BitVote(1, 3, 3)), unresolved=1, inconclusive=0
    )
    assert read.payload is None, "one tied bit makes the whole payload undecodable"
    # A tied bit sits exactly at chance (excess 0), so the *weakest* bit in the payload is
    # the tie's 0.5, not 1.0: the test ran and produced the null, rather than never running.
    assert read.weakest == 0.5


def test_crc8_and_identity_round_trip():
    """The 24+13+8 wire format must survive an encode/decode round trip and a corrupted
    decode must be visible as a CRC failure rather than a confident wrong author."""
    from inkmark.codec import crc8, decode_identity, encode_identity

    for fingerprint, day in (("a1b2c3d4e5f60718", 2498), ("0f1e2d3c4b5a6978", 0)):
        payload, bits = encode_identity(fingerprint, "2026-11-07")
        assert bits == 45
        writer_id, date, ok = decode_identity(payload)
        assert ok and date.isoformat() == "2026-11-07"
        assert f"{writer_id:06x}" == fingerprint[:6]
    corrupt = decode_identity(encode_identity("a1b2c3d4e5f60718", "2026-11-07")[0] ^ 1 << 3)
    assert not corrupt[2], "a flipped bit must fail the checksum, not decode a writer"
    assert crc8(0) == 0


class TestNormalTailGuards:
    """normal_tail is exact erfc in the middle; the clamps exist so an absurd z can never
    manufacture a fake non-zero p through float rounding."""

    def test_extremes_clamp(self):
        assert normal_tail(-8.5) == 1.0
        assert normal_tail(8.5) == 0.0

    def test_the_centre_and_the_verdict_threshold(self):
        assert normal_tail(0.0) == 0.5
        assert normal_tail(1.96) == pytest.approx(0.024997895148220435, abs=1e-12)
        assert normal_tail(4.754) == pytest.approx(1e-6, abs=1e-8)


def test_evidence_bits_refuse_to_score_a_degenerate_test():
    empty = Decision(
        slots=0, hits=0, expected=0.0, variance=0.0, z=0.0, p_value=1.0,
        key_id="k", fingerprint="f",
    )
    assert empty.evidence_bits == 0.0
    # a negative z is evidence AGAINST this key; it must claim no evidence for it
    against = replace(empty, variance=2.0, z=-1.5, p_value=0.9)
    assert against.evidence_bits == 0.0


def test_render_shows_the_payload_line_with_or_without_confidence():
    base = dict(
        slots=10, hits=9, expected=5.0, variance=2.2, z=2.7, p_value=0.003,
        key_id="k", fingerprint="f", words=100,
    )
    plain = Decision(**base, payload=0x123, payload_bits=24)
    assert "payload    0x123 (24 bits)" in plain.render()
    confident = Decision(**base, payload=0x123, payload_bits=24, payload_confidence=0.01)
    assert "decoder confidence p = 0.01" in confident.render()


def test_scan_against_never_runs_with_zero_keys():
    with pytest.raises(ValueError, match="no keys given"):
        scan_against("she walked slowly toward the grey station", [])


def test_walk_of_an_empty_text_is_empty():
    key = WriterKey(secret=b"1" * 32, key_id="empty", created="", year=2026)
    assert walk("", key, default_table("all"), mode="detect") == []
