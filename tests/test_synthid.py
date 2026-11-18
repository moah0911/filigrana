"""Seam 2's key derivation: writer key in, HF SynthID config out.

The derivation must be deterministic (a writer who loses the HF config rebuilds it from the
keyring alone) and key-specific (two writers never share a watermark by accident). The
config assembly needs the transformers extra; the derivation needs nothing.
"""

from __future__ import annotations

import pytest

from inkmark.integrations import synthid
from inkmark.keyring import WriterKey

KEY = WriterKey(secret=b"8" * 32, key_id="synthid-test", created="", year=2026)
OTHER = WriterKey(secret=b"9" * 32, key_id="synthid-other", created="", year=2026)


def test_keys_are_deterministic_and_bounded():
    first, again = synthid.synthid_keys(KEY), synthid.synthid_keys(KEY)
    assert first == again
    assert len(first) == synthid.N_KEYS
    assert all(0 <= k < 2**31 for k in first), "the paper's keys are 31-bit integers"


def test_key_count_is_honourable_and_rotation_rotates():
    assert len(synthid.synthid_keys(KEY, count=5)) == 5
    assert synthid.synthid_keys(KEY) != synthid.synthid_keys(OTHER)


def test_ngram_len_stays_in_the_published_band():
    for key in (KEY, OTHER):
        n = synthid.ngram_len_for(key)
        assert 4 <= n <= 7
        assert n == synthid.ngram_len_for(key)  # deterministic


def test_describe_names_the_key():
    text = synthid.describe(KEY)
    assert KEY.key_id in text
    assert KEY.fingerprint in text
    assert "ngram_len=" in text
    assert str(synthid.N_KEYS) in text  # "24 keys derived from ..."


def test_watermarking_config_matches_the_derivation():
    pytest.importorskip("transformers")
    try:
        from transformers import SynthIDTextWatermarkingConfig  # noqa: F401
    except ImportError:
        pytest.skip("this transformers build has no SynthIDTextWatermarkingConfig")
    config = synthid.watermarking_config(KEY)
    assert list(config.keys) == synthid.synthid_keys(KEY)
    assert config.ngram_len == synthid.ngram_len_for(KEY)
