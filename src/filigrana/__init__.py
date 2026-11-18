"""filigrana -- a writer-held statistical watermark for prose you wrote by hand.

    from filigrana import WriterKey, embed, scan

Two tiers, because they answer two different questions:

* tier 1 (default) -- you know who you are. `scan(text, your_key)` asks whether your key
  explains the word choices in a suspect copy. Identity rides in the key's fingerprint, so
  this costs no payload bits and works at essay length.
* tier 2 -- a stranger finds unattributed prose. `codec.read_payload` decodes a writer id
  and publish date out of the wording. Needs far more text; see docs/CAPACITY.md.
"""

from __future__ import annotations

from .codec import PayloadRead, bit_source, decode_identity, encode_identity, read_payload
from .decisions import CAUTION, Decision
from .detect import scan, scan_against
from .embed import Edit, EmbedResult, embed
from .keyring import WriterKey
from .providers import CuratedTable, EquivalenceClass, Slot, default_table

__version__ = "0.1.0"

__all__ = [
    "CAUTION",
    "CuratedTable",
    "Decision",
    "Edit",
    "EmbedResult",
    "EquivalenceClass",
    "PayloadRead",
    "Slot",
    "WriterKey",
    "__version__",
    "bit_source",
    "decode_identity",
    "default_table",
    "embed",
    "encode_identity",
    "read_payload",
    "scan",
    "scan_against",
]
