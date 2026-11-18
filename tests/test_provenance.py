"""Tier 3 (manifest), the registry, and the style guard."""


from __future__ import annotations

import pytest

from filigrana.canon import canonical_form
from filigrana.embed import embed
from filigrana.keyring import WriterKey
from filigrana.manifest import Manifest, publish, verify
from filigrana.providers import default_table
from filigrana.registry import Entry, Registry, entry_for
from filigrana.slop import SLOP_WORDS, is_slop, profile

KEY = WriterKey(secret=b"5" * 32, key_id="m-2026-00", created="2026-01-01", year=2026)
TEXT = (
    "He said that the report was confidential. Although it was early, the town moved on. "
    "Perhaps the point is obvious; however, few said so. Among the group judgement varied, "
    "and acknowledgement came late. Toward the station he walked afterwards, and the work "
    "went onward. " * 8
)


def test_manifest_round_trips_and_distinguishes_edit_from_forgery():
    marked = embed(TEXT, KEY, table=default_table("all")).text
    manifest = publish(canonical_form(marked), KEY, handle="ada", published="2026-11-06")
    ok = verify(canonical_form(marked), manifest, KEY.secret)
    assert ok["text_matches"] and ok["attestation_matches"]

    tampered = Manifest.from_json(manifest.to_json())
    later = verify(canonical_form(marked + " One more sentence."), tampered, KEY.secret)
    assert not later["text_matches"], "an edit must show as an edit"

    wrong_key = WriterKey(secret=b"6" * 32, key_id="x", created="", year=2026)
    forged = verify(canonical_form(marked), tampered, wrong_key.secret)
    assert forged["text_matches"] and not forged["attestation_matches"]


def test_domain_subkey_scopes_verification():
    marked = embed(TEXT, KEY, table=default_table("all")).text
    scoped = KEY.for_domain("example.com")
    manifest = publish(canonical_form(marked), KEY, handle="ada", published="2026-11-06", verify_key=scoped.secret)
    assert verify(canonical_form(marked), manifest, scoped.secret)["attestation_matches"]
    assert not verify(canonical_form(marked), manifest, KEY.secret)["attestation_matches"], (
        "a domain-scoped attestation must not be checkable with the master key's own material"
    )


def test_registry_resolves_ids_and_refuses_collisions():
    registry = Registry()
    entry = registry.add(entry_for(KEY, "ada", "https://example.com/essay"))
    assert registry.resolve(int(entry.writer_id, 16)).handle == "ada"
    assert [e.handle for e in registry.for_fingerprint(KEY.fingerprint)] == ["ada"]
    assert [e.handle for e in registry.keys_for_domain("EXAMPLE.COM")] == ["ada"]
    blob = Registry.from_json(registry.to_json())
    assert len(blob) == 1 and blob.resolve(int(entry.writer_id, 16)).url == entry.url
    rival = WriterKey(secret=b"8" * 32, key_id="r", created="", year=2026)
    clashing = Entry(writer_id=entry.writer_id, fingerprint=rival.fingerprint, handle="mallory")
    with pytest.raises(ValueError, match="Rotate"):
        blob.add(clashing)


def test_style_guard_spends_slots_to_avoid_filler():
    """The guard is a real trade and must actually trade."""
    table = default_table("all")
    plain = embed(TEXT, KEY, table=table)
    guarded = embed(TEXT, KEY, table=table, disprefer=is_slop)
    assert guarded.slots == plain.slots
    assert sum(1 for e in guarded.entries if e.vetted) <= plain.slots
    # A slot whose original was already filler must still move away from filler.
    already_sloppy = "Delve into the tapestry; moreover, utilise the landscape. " * 4
    result = embed(already_sloppy, KEY, table=table, disprefer=is_slop)
    assert profile(result.text).slop_rate <= profile(already_sloppy).slop_rate + 1e-9


def test_profile_is_stable_under_marking():
    table = default_table("all")
    marked = embed(TEXT, KEY, table=table).text
    before, after = profile(TEXT), profile(marked)
    assert before.words and after.words
    assert abs(before.cliches_per_1k - after.cliches_per_1k) < 1.0
    assert "however" in SLOP_WORDS or "moreover" in SLOP_WORDS


def test_multiword_slop_members_are_judged_on_their_head_word():
    assert is_slop("owing to")
    assert not is_slop("a number of")
    assert is_slop("moreover")


def test_synthid_keys_are_derived_and_rotatable():
    """Rebuildable from the keyring alone -- a lost HF config must not mean a lost mark."""
    from filigrana.integrations.synthid import ngram_len_for, synthid_keys

    keys = synthid_keys(KEY)
    assert keys == synthid_keys(KEY)
    assert len(set(keys)) == len(keys)
    assert all(0 < k < 2**31 for k in keys)
    other = WriterKey(secret=b"7" * 32, key_id="other", created="", year=2026)
    assert synthid_keys(other) != keys
    assert 4 <= ngram_len_for(KEY) <= 7


def test_synthid_config_import_fails_loudly_without_the_extra():
    """Silent degradation here would mean a writer believes an unmarked draft is marked."""
    from filigrana.integrations import synthid

    try:
        import transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"filigrana\[lm\]"):
            synthid.watermarking_config(KEY)
    else:  # pragma: no cover - depends on the extra
        assert synthid.watermarking_config(KEY).ngram_len in range(4, 8)


class TestKeyringLifecycle:
    def test_rotation_burns_the_old_mark(self):
        key = WriterKey(secret=b"5" * 32, key_id="prov", created="", year=2026)
        rotated = key.rotate()
        assert rotated.fingerprint != key.fingerprint
        assert rotated.epoch == key.epoch + 1
        assert rotated.key_id != key.key_id
        assert rotated.year == key.year

    def test_a_foreign_keyring_file_is_named_as_such(self, tmp_path):
        bogus = tmp_path / "bogus.json"
        bogus.write_text('{"format": "some-other-tool/1"}')
        with pytest.raises(ValueError, match="not a filigrana keyring file"):
            WriterKey.load(bogus, "pw")

    def test_a_foreign_manifest_is_named_as_such(self):
        with pytest.raises(ValueError, match="not a filigrana manifest"):
            Manifest.from_json('{"format": "nope/1"}')

    def test_a_foreign_registry_is_named_as_such(self):
        with pytest.raises(ValueError, match="not a filigrana registry"):
            Registry.from_json('{"format": "nope/1"}')

    def test_a_manifest_missing_its_required_fields_is_refused(self):
        with pytest.raises(ValueError, match="missing or not a string"):
            Manifest.from_json('{"format": "filigrana-manifest/1"}')

    def test_a_registry_whose_entries_are_not_a_list_is_refused(self):
        with pytest.raises(ValueError, match="'entries' must be a list"):
            Registry.from_json('{"format": "filigrana-registry/1", "entries": {}}')

    def test_a_malformed_registry_entry_is_named(self):
        with pytest.raises(ValueError, match="malformed registry entry"):
            Registry.from_json(
                '{"format": "filigrana-registry/1", "entries": [{"handle": "ada"}]}'
            )


def test_entry_domains_are_lowercase_hosts_or_nothing():
    assert Entry("1", "f", "ada", url="https://Example.org/ada/post").domains == ["example.org"]
    assert Entry("1", "f", "ada").domains == []


def test_registry_iteration_is_sorted_by_writer_id():
    entries = [Entry("c", "f1", "c"), Entry("a", "f2", "a"), Entry("b", "f3", "b")]
    assert [e.writer_id for e in Registry(entries)] == ["a", "b", "c"]
