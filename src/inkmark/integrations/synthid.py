"""Seam 2: the writer who drafts with a model, and wants *their* key in it, not the vendor's.

This module deliberately does very little. Hugging Face Transformers already ships the
SynthID-Text watermark (`SynthIDTextWatermarkLogitsProcessor`) under Apache-2.0; rewriting it
would be the kind of hubris that gets attributed to hobby projects. What is missing is the
key management: the vendor configuration is one opaque list of integers, and a writer needs
that list derived from -- and recoverable from -- their `WriterKey`.

    from inkmark import WriterKey
    from inkmark.integrations.synthid import watermarking_config
    config = watermarking_config(key)
    out = model.generate(**inputs, watermarking_config=config, do_sample=True)

Then, and this is the part that makes this a warning rather than a feature:

* Detection needs a **trained** Bayesian classifier per key (~10k examples, representative of
  your own prose) or `synthid-text`'s Mean detector. There is no training-free public checker.
* The distortion-free configuration that a provider will actually deploy measures
  ``Delta ~ 0.05`` per token, which at article length gives ``z ~ 1.5`` -- see
  docs/CAPACITY.md. To get a usable per-article score you must run the distortionary delta,
  which is precisely the quality cost writers object to. So the logits path is *not* the
  strong option for a writer at desk length; tier 1 of this package exists because of that.
* It marks model-generated spans. It says nothing about the prose you typed, and if you paste
  your own paragraphs in between them, the mark only covers the model's words.

Kept import-guarded so the core stays stdlib-only.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..keyring import WriterKey

N_KEYS = 24  # the paper's own recommendation is 20-30
INFO = b"inkmark/1/synthid-keys"


def synthid_keys(key: WriterKey, count: int = N_KEYS, bound: int = 2**31 - 1) -> list[int]:
    """Derive a SynthID `keys` list from the writer's master secret.

    Deterministic, so a writer who loses their HF config can rebuild it from the keyring
    alone, and rotating the writer key rotates the watermark. Values are drawn from a full
    SHA-256 output rather than a single byte, so the 2**31 range is actually reachable.
    """
    out: list[int] = []
    counter = 0
    while len(out) < count:
        block = hmac.new(key.secret, INFO + counter.to_bytes(4, "big"), hashlib.sha256).digest()
        for offset in range(0, 32, 8):
            out.append(int.from_bytes(block[offset : offset + 8], "big") % bound)
            if len(out) == count:
                break
        counter += 1
    return out


def ngram_len_for(key: WriterKey, low: int = 4, high: int = 7) -> int:
    """A per-key n-gram length, so two writers cannot share a fingerprint by coincidence.

    Longer n-grams are more detectable and more brittle; keeping the choice inside a narrow
    band derived from the key avoids an argument about the right global value.
    """
    digest = hmac.new(key.secret, INFO + b"ngram", hashlib.sha256).digest()
    return low + digest[0] % (high - low + 1)


def watermarking_config(key: WriterKey, *, n_keys: int = N_KEYS):
    """A `SynthIDTextWatermarkingConfig` built from the writer's key.

    Raises ImportError rather than degrading silently -- a writer who thinks they watermarked
    their draft and did not is worse off than one who was told to install the extra.
    """
    try:
        from transformers import SynthIDTextWatermarkingConfig
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "the logits path needs 'inkmark[lm]' (transformers>=4.46). The tier-1 path in "
            "this package needs nothing and works on prose you wrote by hand."
        ) from exc
    return SynthIDTextWatermarkingConfig(keys=synthid_keys(key, n_keys), ngram_len=ngram_len_for(key))


def describe(key: WriterKey) -> str:
    return (
        f"synthid keys: {len(synthid_keys(key))} derived from {key.key_id} "
        f"(fingerprint {key.fingerprint}), ngram_len={ngram_len_for(key)}"
    )


__all__ = ["N_KEYS", "describe", "ngram_len_for", "synthid_keys", "watermarking_config"]
