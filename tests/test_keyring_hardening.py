"""Keyring storage failures: the file is an identity, so every write has to be atomic.

A keyring is the one file in this project whose corruption is unrecoverable. A partial write
does not merely lose a preference; it destroys the secret that fingerprints, verification
subkeys, and every word choice derive from. These tests pin the failure paths, not just the
happy round trip already covered in test_safety.py.
"""

from __future__ import annotations

import json
import stat

import pytest

import filigrana.keyring as keyring
from filigrana.keyring import KeyringError, WriterKey

PASSPHRASE = "correct horse"


def _key(secret: bytes = b"1" * 32) -> WriterKey:
    return WriterKey(secret=secret, key_id="fil-2026-00-test", created="2026-01-01", year=2026)


def test_save_writes_0600_atomically_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "keyring.json"
    _key().save(path, PASSPHRASE)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".keyring.json.*.tmp"))
    assert WriterKey.load(path, PASSPHRASE).secret == b"1" * 32


def test_a_failed_replace_leaves_the_previous_keyring_intact(tmp_path, monkeypatch):
    path = tmp_path / "keyring.json"
    _key().save(path, PASSPHRASE)
    before = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("disk full")

    monkeypatch.setattr(keyring.os, "replace", fail_replace)
    with pytest.raises(KeyringError, match="cannot write"):
        _key(b"2" * 32).save(path, PASSPHRASE)

    assert path.read_bytes() == before, "a failed save must never truncate the old identity"
    assert not list(tmp_path.glob(".keyring.json.*.tmp")), "the temporary file must be cleaned up"


def test_an_empty_passphrase_is_refused_rather_than_encrypting_nothing(tmp_path):
    path = tmp_path / "keyring.json"
    with pytest.raises(KeyringError, match="non-empty passphrase"):
        _key().save(path, "")
    assert not path.exists()
    _key().save(path, PASSPHRASE)
    with pytest.raises(KeyringError, match="non-empty passphrase"):
        WriterKey.load(path, "")


def test_a_secret_that_is_not_32_bytes_is_refused(tmp_path):
    with pytest.raises(KeyringError, match="32 bytes"):
        _key(b"short").save(tmp_path / "keyring.json", PASSPHRASE)


def test_windows_is_refused_before_touching_the_filesystem(tmp_path, monkeypatch):
    path = tmp_path / "keyring.json"
    _key().save(path, PASSPHRASE)
    monkeypatch.setattr(keyring.os, "name", "nt")
    with pytest.raises(KeyringError, match="macOS and Linux only"):
        _key().save(tmp_path / "new.json", PASSPHRASE)
    with pytest.raises(KeyringError, match="macOS and Linux only"):
        WriterKey.load(path, PASSPHRASE)


def test_a_permissive_keyring_warns_instead_of_staying_silent(tmp_path):
    path = tmp_path / "keyring.json"
    _key().save(path, PASSPHRASE)
    path.chmod(0o644)
    with pytest.warns(RuntimeWarning, match="accessible to other users"):
        WriterKey.load(path, PASSPHRASE)


def test_a_keyring_that_is_not_json_is_named(tmp_path):
    path = tmp_path / "keyring.json"
    path.write_text("{not json")
    with pytest.raises(KeyringError, match="not valid JSON"):
        WriterKey.load(path, PASSPHRASE)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("salt", "zz" * 16, "salt is not valid hex"),
        ("salt", "00", "salt must be 16 bytes"),
        ("nonce", "00" * 16 + "00", "nonce must be 16 bytes"),
        ("tag", "00", "tag must be 32 bytes"),
        ("body", "", "body is empty"),
    ],
)
def test_malformed_hex_fields_are_refused(tmp_path, field, value, message):
    path = tmp_path / "keyring.json"
    _key().save(path, PASSPHRASE)
    blob = json.loads(path.read_text())
    blob[field] = value
    path.write_text(json.dumps(blob))
    with pytest.raises(KeyringError, match=message):
        WriterKey.load(path, PASSPHRASE)


def test_a_tampered_public_fingerprint_is_caught(tmp_path):
    path = tmp_path / "keyring.json"
    _key().save(path, PASSPHRASE)
    blob = json.loads(path.read_text())
    blob["fingerprint"] = "0" * 16
    path.write_text(json.dumps(blob))
    with pytest.raises(KeyringError, match="fingerprint does not match"):
        WriterKey.load(path, PASSPHRASE)


def test_revocation_is_recorded_in_the_key_file_and_survives_a_subkey(tmp_path):
    path = tmp_path / "keyring.json"
    revoked = _key().revoke()
    revoked.save(path, PASSPHRASE)
    restored = WriterKey.load(path, PASSPHRASE)
    assert restored.revoked
    assert restored.for_domain("example.com").revoked
