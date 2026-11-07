"""The Ed25519 tier of the manifest, when `inkmark[sign]` is installed.

Everything here skips cleanly without `cryptography`, which is the point of the tier: the
default HMAC path runs on stdlib alone, and the public-key path refuses to pretend with
hand-rolled curve arithmetic. Where the optional dependency exists its behaviour is pinned.
"""

import pytest

from inkmark.canon import canonical_form
from inkmark.manifest import (
    Manifest,
    generate_signing_pair,
    publish,
    verify,
    verify_ed25519,
)

pytest.importorskip("cryptography")

TEXT = "she walked slowly toward the gray station\n"


def _signed_manifest(private) -> Manifest:
    return publish(
        canonical_form(TEXT),
        KEY_SIGNER,
        handle="ada",
        published="2026-11-07",
        ed25519_private=private,
    )


from inkmark.keyring import WriterKey  # noqa: E402

KEY_SIGNER = WriterKey(secret=b"6" * 32, key_id="sign-test", created="", year=2026)


def test_a_public_key_signature_must_change_the_scheme_name():
    private, _public = generate_signing_pair()
    manifest = _signed_manifest(private)
    assert manifest.scheme == "ed25519+hmac"
    assert manifest.signature, "the signature is the publishable artefact"


def test_ed25519_verifies_under_the_right_key_and_rejects_strangers():
    private, public = generate_signing_pair()
    manifest = _signed_manifest(private)
    assert verify_ed25519(canonical_form(TEXT), manifest, public)
    _, stranger = generate_signing_pair()
    assert not verify_ed25519(canonical_form(TEXT), manifest, stranger)
    "an edited copy must not verify True even under the right key"
    edited = Manifest(
        digest=manifest.digest,
        meta=manifest.meta,
        attestation=manifest.attestation,
        scheme=manifest.scheme,
        signature="ff" * len(bytes.fromhex(manifest.signature)),
    )
    assert not verify_ed25519(canonical_form(TEXT), edited, public)


def test_ed25519_verifies_the_claimed_text_not_the_file_the_signature_came_from():
    """Text divergence is reported as text, not as forgery: text_matches and the
    attestation are separate fields, and the checked text is the one the reader has."""
    private, public = generate_signing_pair()
    manifest = _signed_manifest(private)
    assert verify_ed25519(canonical_form(TEXT + "\nmore\n"), manifest, public) is False
    check = verify(canonical_form(TEXT + "\nmore\n"), manifest, KEY_SIGNER.secret)
    # The attestation covers digest and meta together, so an edited text fails it as well;
    # the *distinguishable* fact is that the recomputed digest no longer matches the
    # manifest, which is what turns a tamper report into an edit report.
    assert check["text_matches"] is False
    assert check["attestation_matches"] is False


def test_one_copy_that_differs_cannot_lie_about_which_share_did():
    """Publish two texts under one key: each manifest verifies only its own text."""
    private, public = generate_signing_pair()
    other = publish(
        canonical_form("another article entirely"), KEY_SIGNER, handle="ada",
        published="2026-11-07", ed25519_private=private,
    )
    assert verify_ed25519(canonical_form(TEXT), _signed_manifest(private), public)
    assert not verify_ed25519(canonical_form(TEXT), other, public)
