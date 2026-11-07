"""Out-of-band provenance: the part that is strong, and the part that survives copying.

The in-band mark is fragile and survives copy-paste. The manifest is strong and does not:
strip the container and it is gone. They are complements, and `inkmark` ships both because a
writer needs both -- the manifest for the conversation with a lawyer or an editor, the mark
for the scraper who pasted your essay into a template.

Signing has two levels, and which one you get depends on whether `cryptography` is installed:

* **default, stdlib** -- an HMAC attestation under a nominated-verifier subkey. Anyone you
  hand that subkey to can check the manifest; nobody can forge it without the key; but you
  cannot publish verification to the world, because the checking key is the signing key.
* **`inkmark[sign]`** -- Ed25519 over the same bytes, so the public key goes on your site and
  anyone can verify. Not implemented with hand-rolled curve maths: an identity-bearing
  signature is exactly the place where writing your own crypto is indefensible, so this path
  refuses to pretend and requires the audited library.

Neither is C2PA. C2PA is the right long-term container -- it is what Anthropic attaches to
generated images -- and it needs a real issuer, a certificate chain, and timestamps from a
authority you trust. This format is deliberately `inkmark-manifest/1` and says so, so that
nobody mistakes it for the industry standard it hopes to become.
"""


import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime

FORMAT = "inkmark-manifest/1"

try:  # optional, and honestly so: no hand-written curve arithmetic in this repository
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _HAS_ED25519 = True
except ImportError:  # pragma: no cover - depends on the extra
    _HAS_ED25519 = False


def _canonical_digest(canonical_text: str) -> str:
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def _signing_bytes(digest: str, meta: dict) -> bytes:
    """The exact bytes a signature covers: hash first, then the claims, in fixed order."""

    parts = [FORMAT.encode(), b"\x00", digest.encode()]
    for field in ("handle", "published", "url", "license", "key_id", "fingerprint", "minted"):
        parts += b"\x01", field.encode(), b"=", str(meta.get(field, "")).encode()
    return b"".join(parts)


@dataclass(frozen=True, slots=True)
class Manifest:
    digest: str
    meta: dict
    attestation: str
    scheme: str
    signature: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "format": FORMAT,
                "sha256_canonical": self.digest,
                "scheme": self.scheme,
                "attestation": self.attestation,
                "ed25519": self.signature or None,
                **self.meta,
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, blob: str) -> "Manifest":
        data = json.loads(blob)
        if data.get("format") != FORMAT:
            raise ValueError("not an inkmark manifest")
        meta = {k: v for k, v in data.items() if k not in {"format", "sha256_canonical", "scheme", "attestation", "ed25519"}}
        return cls(
            digest=data["sha256_canonical"],
            meta=meta,
            attestation=data["attestation"],
            scheme=data["scheme"],
            signature=data.get("ed25519") or "",
        )


def publish(
    canonical_text: str,
    key,
    *,
    handle: str,
    published: str,
    url: str = "",
    license: str = "",
    verify_key: bytes | None = None,
    ed25519_private: bytes | None = None,
) -> Manifest:
    meta = {
        "handle": handle,
        "published": published,
        "url": url,
        "license": license,
        "key_id": key.key_id,
        "fingerprint": key.fingerprint,
        "minted": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    digest = _canonical_digest(canonical_text)
    payload = _signing_bytes(digest, meta)
    secret = verify_key if verify_key is not None else key.secret
    attestation = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    signature = ""
    if ed25519_private is not None and _HAS_ED25519:
        signature = Ed25519PrivateKey.from_private_bytes(ed25519_private).sign(payload).hex()
    return Manifest(
        digest=digest,
        meta=meta,
        attestation=attestation,
        scheme="ed25519+hmac" if signature else "hmac-sha256",
        signature=signature,
    )


def verify(canonical_text: str, manifest: Manifest, key: bytes) -> dict:
    """Recompute the attestation. Returns the claims plus whether they hold.

    `text_matches` and `attestation_matches` are reported separately on purpose: a manifest
    whose text no longer matches is not a failed signature, it is an edited article, and
    conflating the two is how provenance tools get a bad name.
    """
    digest = _canonical_digest(canonical_text)
    payload = _signing_bytes(digest, manifest.meta)
    expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
    out = {
        "text_matches": hmac.compare_digest(digest, manifest.digest),
        "attestation_matches": hmac.compare_digest(expected, manifest.attestation),
        "scheme": manifest.scheme,
        "claims": manifest.meta,
    }
    return out


def verify_ed25519(canonical_text: str, manifest: Manifest, public_key_bytes: bytes) -> bool:
    """Independent public-key check. Requires `inkmark[sign]`; there is no fallback, because a
    signature we could verify with stdlib alone would be a signature we could also forge."""
    if not manifest.signature or not _HAS_ED25519:
        return False
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(
            bytes.fromhex(manifest.signature),
            _signing_bytes(_canonical_digest(canonical_text), manifest.meta),
        )
        return True
    except (InvalidSignature, ValueError):  # bad signature bytes, or the signature rejects
        return False


def ed25519_available() -> bool:
    return _HAS_ED25519


def generate_signing_pair() -> tuple[bytes, bytes]:
    """Ed25519 keys, if the optional dependency is present. Never a private key from RNG soup."""
    if not _HAS_ED25519:
        raise RuntimeError("pip install 'inkmark[sign]' -- inkmark will not implement Ed25519 itself")
    private = Ed25519PrivateKey.generate()
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    return (
        private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
        private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
    )


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


__all__ = ["FORMAT", "Manifest", "generate_signing_pair", "publish", "timestamp", "verify", "verify_ed25519"]
