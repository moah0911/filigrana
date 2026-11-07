"""The keyed keystream: which member of an equivalence class the writer's key demands.

Two properties, both load-bearing:

* Cryptographic. Unlike SynthID's ``accumulate_hash`` (whose own README says it makes no
  security guarantees), this is HMAC-SHA256. A key that identifies a *person* has to be
  unforgeable by inspection of published text, or anyone can plant your mark on prose you
  never wrote.
* Re-derivable from the suspect text alone. The context is the last ``window`` word tokens
  that the watermark itself can never touch -- words that are not members of any class.
  A chained context over *slot decisions* was tried first and fails hard: one deleted
  sentence shifts the chain forever and drops a marked article to chance (measured 30/53).
  Anchoring bounds the damage instead: an edit or deletion only disturbs slots whose
  neighbouring anchor words actually moved.
"""


import hashlib
import hmac

SCHEME = b"inkmark/1"


def _digest(key: bytes, payload: bytes) -> bytes:
    return hmac.new(key, payload, hashlib.sha256).digest()


def anchor_context(anchor: tuple[str, ...]) -> bytes:
    """Serialise the preceding untouched words into a stable byte string."""

    return b" ".join(word.encode("utf-8", "ignore") for word in anchor[-8:])


def keyed_choice(
    key: bytes,
    class_id: str,
    n_candidates: int,
    anchor: tuple[str, ...] = (),
    window: int = 4,
    *,
    key_id: str = "",
    bit: int | None = None,
) -> int:
    """Index into the sorted candidate list that the key demands at this slot.

    ``bit`` is the payload bit this slot is carrying (tier-2 multi-bit mode). Mixing it in
    means the same slot yields opposite demands for opposite bits, so a detector can test
    each payload bit by voting over the slots assigned to it.
    """
    message = b"".join(
        (
            SCHEME,
            b"\x00",
            key_id.encode("utf-8"),
            b"\x01",
            class_id.encode("utf-8"),
            b"\x02",
            anchor_context(tuple(anchor)[-window:]),
            b"\x03",
            str(bit if bit is not None else -1).encode("ascii"),
        )
    )
    value = int.from_bytes(_digest(key, message)[:8], "big")
    return value % n_candidates
