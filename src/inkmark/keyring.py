"""Writer keys: identity, storage, rotation, and the subkey that limits blast radius.

A personal watermark inverts Anthropic's privacy posture. Theirs deliberately identifies
nobody; ours deliberately identifies one person, which means the verification material is
a fingerprinting primitive: anyone who can test "is this this writer's text?" can run that
test across every essay the writer has ever published. So the key model is not a detail.

Two consequences encoded here:

* Export a *nominated verifier* subkey, not the master, when handing verification to a
  publisher or fact-checker. Scoped to a domain, revocable, and it does not let them
  derive the master.
* Rotation and revocation are recorded in the key file itself, because a leaked key here
  lets an adversary plant this writer's mark on prose they never wrote.
"""


import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

_FP_INFO = b"inkmark/1/fingerprint"
_SUBKEY_INFO = b"inkmark/1/domain"
_ENC_INFO = b"inkmark/1/enc"
_MAC_INFO = b"inkmark/1/mac"
_SCRYPT_N = 2**15
_SCRYPT_MEM = 128 * 1024 * 1024  # explicit, or CPython refuses on constrained hosts


def _derive(passphrase: str, salt: bytes) -> tuple[bytes, bytes]:
    material = hashlib.scrypt(
        passphrase.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=8, p=1, dklen=64,
        maxmem=_SCRYPT_MEM
    )
    return material[:32], material[32:]


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out += block
        counter += 1
    return bytes(out[:length])


@dataclass(frozen=True, slots=True)
class WriterKey:
    """The writer's master secret plus the public facts about it."""


    secret: bytes
    key_id: str
    created: str
    year: int
    epoch: int = 0
    revoked: bool = False
    note: str = ""

    @staticmethod
    def generate(*, year: int | None = None, note: str = "") -> "WriterKey":
        year = year or time.localtime().tm_year
        return WriterKey(
            secret=secrets.token_bytes(32),
            key_id=f"ink-{year}-00-{secrets.token_hex(4)}",
            created=time.strftime("%Y-%m-%d"),
            year=year,
            note=note,
        )

    @property
    def fingerprint(self) -> str:
        """Public identifier for a writer. Safe to publish; reveals nothing about `secret`."""
        return hmac.new(self.secret, _FP_INFO, hashlib.sha256).hexdigest()[:16]

    def for_domain(self, domain: str) -> "WriterKey":
        """Scoped verification subkey for one publisher.

        Enough to verify this writer's text on that domain, not enough to test the writer
        across the rest of the web, and not enough to derive the master.
        """
        sub = hmac.new(
            self.secret, _SUBKEY_INFO + domain.lower().encode("utf-8"), hashlib.sha256
        ).digest()
        return WriterKey(
            secret=sub,
            key_id=f"{self.key_id}/{domain.lower()}",
            created=self.created,
            year=self.year,
            epoch=self.epoch,
            note=f"scoped subkey of {self.key_id}",
        )

    def rotate(self) -> "WriterKey":
        """New key, new fingerprint. Old text stops verifying -- which is the point, and the cost."""
        return WriterKey(
            secret=secrets.token_bytes(32),
            key_id=f"ink-{self.year}-{self.epoch + 1:02d}-{secrets.token_hex(4)}",
            created=time.strftime("%Y-%m-%d"),
            year=self.year,
            epoch=self.epoch + 1,
            note=f"rotated from {self.key_id}",
        )

    def save(self, path: Path, passphrase: str) -> None:
        salt = secrets.token_bytes(16)
        enc_key, mac_key = _derive(passphrase, salt)
        nonce = secrets.token_bytes(16)
        plaintext = json.dumps(
            {
                "secret": self.secret.hex(),
                "key_id": self.key_id,
                "created": self.created,
                "year": self.year,
                "epoch": self.epoch,
                "note": self.note,
            }
        ).encode("utf-8")
        body = bytes(a ^ b for a, b in zip(plaintext, _keystream(enc_key, nonce, len(plaintext))))
        tag = hmac.new(mac_key, salt + nonce + body, hashlib.sha256).digest()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            json.dumps(
                {
                    "format": "inkmark-keyring/1",
                    "fingerprint": self.fingerprint,
                    "salt": salt.hex(),
                    "nonce": nonce.hex(),
                    "body": body.hex(),
                    "tag": tag.hex(),
                }
            ).encode("utf-8")
        )
        os.chmod(path, 0o600)

    @classmethod
    def load(cls, path: Path, passphrase: str) -> "WriterKey":
        blob = json.loads(path.read_bytes())
        if blob.get("format") != "inkmark-keyring/1":
            raise ValueError("not an inkmark keyring file")
        salt, nonce = bytes.fromhex(blob["salt"]), bytes.fromhex(blob["nonce"])
        body, tag = bytes.fromhex(blob["body"]), bytes.fromhex(blob["tag"])
        enc_key, mac_key = _derive(passphrase, salt)
        if not hmac.compare_digest(hmac.new(mac_key, salt + nonce + body, hashlib.sha256).digest(), tag):
            raise ValueError("bad passphrase or tampered keyring")
        data = json.loads(bytes(a ^ b for a, b in zip(body, _keystream(enc_key, nonce, len(body)))))
        return cls(
            secret=bytes.fromhex(data["secret"]),
            key_id=data["key_id"],
            created=data["created"],
            year=data["year"],
            epoch=data.get("epoch", 0),
            note=data.get("note", ""),
        )
