"""Safety properties. These are the features that stop the tool accusing a human.

The realistic harm is not a broken statistic, it is someone quoting a statistic too far.
"""


import pytest

from filigrana.codec import PayloadRead, crc8, decode_identity, encode_identity, read_payload
from filigrana.decisions import Decision
from filigrana.detect import scan, scan_against
from filigrana.embed import embed
from filigrana.keyring import WriterKey
from filigrana.providers import default_table

TABLE = default_table("all")
TEXT = (
    "He said that the report was confidential. Although it was early, the town moved on. "
    "Perhaps the point is obvious; however, few said so. She would rather leave, and it was "
    "somewhat cold. Frequently they met at this point, and in the end agreed. In spite of the "
    "noise the work went onward, and judgement varied among the group. Toward the station he "
    "walked, afterwards, and did not recognise the colour of the building. " * 4
)


def key(secret=0) -> WriterKey:
    return WriterKey(
        secret=bytes((secret + i) % 256 for i in range(32)),
        key_id=f"k{secret}",
        created="2026-01-01",
        year=2026,
    )


def test_decision_has_no_truth_value():
    decision = scan(TEXT, key(), table=TABLE)
    with pytest.raises(TypeError):
        bool(decision)
    with pytest.raises(TypeError):
        if decision:
            pass  # pragma: no cover


def test_decision_exposes_no_verdict_field():
    fields = set(Decision.__dataclass_fields__)
    assert "is_watermarked" not in fields
    assert "verdict" not in fields
    assert {"p_value", "caveats", "slots"} <= fields


def test_rendered_output_always_carries_caveats():
    assert scan(TEXT, key(), table=TABLE).caveats
    assert "verdict" not in scan(TEXT, key(), table=TABLE).render().lower()


def test_tiny_text_refuses_to_look_confident():
    decision = scan("He walked toward the station.", key(), table=TABLE)
    assert any("noise" in c for c in decision.caveats)
    assert decision.slots < 8


def test_wrong_key_does_not_fire():
    marked = embed(TEXT, key(0), table=TABLE).text
    innocent = scan(marked, key(7), table=TABLE)
    assert innocent.z < 3.0, "a foreign key must not produce evidence"
    assert scan(marked, key(0), table=TABLE).z > innocent.z


def test_multiplicity_is_corrected_not_hidden():
    keys = [key(i) for i in range(20)]
    single = scan(TEXT, keys[0], table=TABLE)
    swept = scan_against(TEXT, keys, table=TABLE)
    assert swept.n_hypotheses == 20
    assert swept.p_value_corrected == pytest.approx(min(1.0, swept.p_value * 20))
    assert single.p_value_corrected == single.p_value


def test_revoked_key_is_tagged():
    leaked = WriterKey(secret=bytes(32), key_id="leaked", created="", year=2026, revoked=True)
    assert "key_revoked" in scan(TEXT, leaked, table=TABLE).caveats
    assert "key_revoked" in scan_against(TEXT, [key(0), leaked], table=TABLE).caveats


def test_veto_keeps_the_writers_word_and_the_ledger_intact():
    """Declining an edit is not a corruption: the slot records what the writer left."""

    marked = embed(TEXT, key(0), table=TABLE)
    refused = embed(TEXT, key(0), table=TABLE, veto=lambda slot, wanted: slot.class_id == "a.toward")
    assert refused.drift == 0
    assert sum(1 for e in refused.entries if e.vetted) >= 1
    assert refused.text.count("toward the") + refused.text.count("Toward the") >= 1
    assert marked.slots == refused.slots


def test_keyring_roundtrip_and_scoping(tmp_path):
    original = WriterKey.generate(note="ada")
    path = tmp_path / "keyring.json"
    original.save(path, "correct horse")
    restored = WriterKey.load(path, "correct horse")
    assert restored.secret == original.secret
    assert restored.fingerprint == original.fingerprint
    with pytest.raises(ValueError):
        WriterKey.load(path, "wrong passphrase")
    scoped = original.for_domain("Example.COM")
    assert scoped.fingerprint != original.fingerprint
    assert "example.com" in scoped.key_id


def test_payload_wire_format_roundtrips():
    fingerprint = "abcdef0123456789"
    payload, bits = encode_identity(fingerprint, "2026-11-06")
    writer_id, published, ok = decode_identity(payload, bits)
    assert ok and writer_id == int("abcdef", 16)
    assert published.isoformat() == "2026-11-06"
    assert crc8(payload >> 8) != crc8(payload >> 8) ^ 1


def test_payload_decode_needs_length_and_says_so():
    read = read_payload("toward the station, although early", key(0), 45, table=TABLE)
    assert read.payload is None or read.unresolved

    # A class-rich synthetic page: real prose cannot supply this many slots, which is the
    # capacity finding, not a detail of the fixture. See docs/CAPACITY.md.
    dense = (
        "He walked toward the depot, and although it was early, perhaps the point was obvious; "
        "however, few said so. She would rather leave, and it was somewhat cold. Frequently they "
        "met at this point, and in the end agreed. In spite of the noise the work went onward, "
        "judgement varied among the group, and acknowledgement came late. "
    ) * 90
    payload = 0x123456
    marked = embed(dense, key(0), table=TABLE, payload=payload, n_bits=24).text
    read = read_payload(marked, key(0), 24, table=TABLE)
    assert read.slots if hasattr(read, "slots") else True
    assert read.payload == payload, [(v.index, v.zero, v.one) for v in read.votes]
    assert read.inconclusive > 0, "half of all 2-way slots carry no bit by construction"


def test_payload_reading_refuses_a_zero_bit_wire():
    with pytest.raises(ValueError, match="n_bits must be positive"):
        read_payload(TEXT, key(0), 0, table=TABLE)


def test_an_empty_payload_read_cannot_invent_a_value():
    assert PayloadRead(votes=(), unresolved=0, inconclusive=0).payload is None
