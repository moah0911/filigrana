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
import stat
import tempfile
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

# Cryptographic domain-separation strings, frozen from the pre-release name on purpose.
# They are part of the derivation contract, not user-facing branding: changing them would
# silently invalidate every fingerprint, subkey, and keyed word choice ever produced.
_FP_INFO = b"inkmark/1/fingerprint"
_SUBKEY_INFO = b"inkmark/1/domain"
_ENC_INFO = b"inkmark/1/enc"
_MAC_INFO = b"inkmark/1/mac"
_SCRYPT_N = 2**15
_SCRYPT_MEM = 128 * 1024 * 1024  # explicit, or CPython refuses on constrained hosts
_FORMAT = "filigrana-keyring/1"


class KeyringError(ValueError):
    """A keyring we refuse to trust, or a platform that cannot hold one safely."""


def _require_posix() -> None:
    """Windows has no 0600 guarantee and no directory fsync: refuse rather than pretend."""
    if os.name != "posix":
        raise KeyringError(
            "filigrana keyrings are supported on macOS and Linux only; Windows is unsupported"
        )


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


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
            key_id=f"fil-{year}-00-{secrets.token_hex(4)}",
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
            revoked=self.revoked,
            note=f"scoped subkey of {self.key_id}",
        )

    def revoke(self) -> "WriterKey":
        """Mark the key revoked. Detection reports that fact; it cannot un-publish old text."""
        return replace(self, revoked=True)

    def rotate(self) -> "WriterKey":
        """New key, new fingerprint. Old text stops verifying -- which is the point, and the cost."""
        return WriterKey(
            secret=secrets.token_bytes(32),
            key_id=f"fil-{self.year}-{self.epoch + 1:02d}-{secrets.token_hex(4)}",
            created=time.strftime("%Y-%m-%d"),
            year=self.year,
            epoch=self.epoch + 1,
            note=f"rotated from {self.key_id}",
        )

    def save(self, path: Path, passphrase: str) -> None:
        _require_posix()
        if not passphrase:
            raise KeyringError("a non-empty passphrase is required to save a keyring")
        if len(self.secret) != 32:
            raise KeyringError("writer secrets must be exactly 32 bytes")

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
                "revoked": self.revoked,
                "note": self.note,
            }
        ).encode("utf-8")
        body = bytes(a ^ b for a, b in zip(plaintext, _keystream(enc_key, nonce, len(plaintext))))
        tag = hmac.new(mac_key, salt + nonce + body, hashlib.sha256).digest()
        blob = json.dumps(
            {
                "format": _FORMAT,
                "fingerprint": self.fingerprint,
                "salt": salt.hex(),
                "nonce": nonce.hex(),
                "body": body.hex(),
                "tag": tag.hex(),
            }
        ).encode("utf-8")

        # Write beside the target, fsync, then rename over it. A crash mid-save must leave the
        # previous keyring intact: half a keyring is an identity you cannot verify or rotate.
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise KeyringError(f"cannot create {path.parent}: {exc.strerror or exc}") from None
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as handle:
                fd = -1  # the context manager owns it now
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = ""
            _fsync_dir(path.parent)
        except OSError as exc:
            raise KeyringError(f"cannot write {path}: {exc.strerror or exc}") from None
        finally:
            if fd >= 0:
                os.close(fd)
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: Path, passphrase: str) -> "WriterKey":
        _require_posix()
        if not passphrase:
            raise KeyringError("a non-empty passphrase is required to open a keyring")
        try:
            raw = Path(path).read_bytes()
        except OSError:
            raise
        try:
            blob = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise KeyringError("keyring is not valid JSON") from None
        if not isinstance(blob, dict):
            raise KeyringError("a keyring must be a JSON object")
        if blob.get("format") != _FORMAT:
            raise KeyringError("not a filigrana keyring file")

        _warn_if_permissive(Path(path))

        fields: dict[str, bytes] = {}
        for name in ("salt", "nonce", "body", "tag"):
            value = blob.get(name)
            if not isinstance(value, str):
                raise KeyringError(f"keyring {name} is missing or not a hex string")
            try:
                fields[name] = bytes.fromhex(value)
            except ValueError:
                raise KeyringError(f"keyring {name} is not valid hex") from None
        if len(fields["salt"]) != 16:
            raise KeyringError("keyring salt must be 16 bytes")
        if len(fields["nonce"]) != 16:
            raise KeyringError("keyring nonce must be 16 bytes")
        if len(fields["tag"]) != 32:
            raise KeyringError("keyring tag must be 32 bytes")
        if not fields["body"]:
            raise KeyringError("keyring body is empty")

        salt, nonce = fields["salt"], fields["nonce"]
        body, tag = fields["body"], fields["tag"]
        enc_key, mac_key = _derive(passphrase, salt)
        expected_tag = hmac.new(mac_key, salt + nonce + body, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_tag, tag):
            raise KeyringError("bad passphrase or tampered keyring")

        try:
            data = json.loads(
                bytes(a ^ b for a, b in zip(body, _keystream(enc_key, nonce, len(body))))
            )
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise KeyringError("keyring contents are not valid JSON") from None
        if not isinstance(data, dict):
            raise KeyringError("keyring contents must be a JSON object")
        try:
            secret = bytes.fromhex(data["secret"])
        except (KeyError, TypeError, ValueError):
            raise KeyringError("keyring secret is missing or not valid hex") from None
        if len(secret) != 32:
            raise KeyringError("keyring secret must be exactly 32 bytes")
        key_id, created = data.get("key_id"), data.get("created")
        year, epoch = data.get("year"), data.get("epoch", 0)
        revoked = data.get("revoked", False)
        note = data.get("note", "")
        if not isinstance(key_id, str) or not key_id:
            raise KeyringError("keyring key_id is missing or not a string")
        if not isinstance(created, str):
            raise KeyringError("keyring created field is missing or not a string")
        if not isinstance(year, int) or isinstance(year, bool):
            raise KeyringError("keyring year is missing or not an integer")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise KeyringError("keyring epoch must be a non-negative integer")
        if not isinstance(revoked, bool):
            raise KeyringError("keyring revoked flag must be a boolean")
        if not isinstance(note, str):
            raise KeyringError("keyring note must be a string")

        key = cls(
            secret=secret,
            key_id=key_id,
            created=created,
            year=year,
            epoch=epoch,
            revoked=revoked,
            note=note,
        )
        claimed = blob.get("fingerprint")
        if not isinstance(claimed, str) or len(claimed) != 16:
            raise KeyringError("keyring fingerprint is missing or malformed")
        if not hmac.compare_digest(key.fingerprint, claimed):
            raise KeyringError("keyring fingerprint does not match its secret")
        return key


def _warn_if_permissive(path: Path) -> None:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return
    if mode & 0o077:
        warnings.warn(
            f"keyring {path} is accessible to other users (mode {mode:03o}); chmod 600 it",
            RuntimeWarning,
            stacklevel=3,
        )
