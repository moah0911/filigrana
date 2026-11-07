"""Tier 2: putting an identity in the wording itself.

Tier 1 answers "does this writer's key explain this text?" -- which is the question a
writer asks about a scraped copy, and it costs zero payload bits because identity lives in
the key's fingerprint. Tier 2 answers the harder question a *stranger* asks: find un-
attributed prose, decode who wrote it without being handed a candidate.

That needs bits, and bits are expensive: a slot carrying one payload bit can only vote for
one of two demands, and half the slots produce no vote at all because the two demands
happen to coincide. The wire format is deliberately small and self-checking.

    writer_id  24 bits   truncated key fingerprint, resolvable through a registry
    date       13 bits   days since 2020-01-01
    crc8        8 bits   over the 37 above, so a mis-decode is visible rather than confident
"""


import datetime as _dt
from dataclasses import dataclass

from .decisions import normal_tail
from .ledger import DEFAULT_WINDOW, walk
from .providers import CuratedTable, default_table

EPOCH = _dt.date(2020, 1, 1)
ID_BITS, DATE_BITS, CRC_BITS = 24, 13, 8
WIRE_BITS = ID_BITS + DATE_BITS + CRC_BITS


def crc8(value: int, bits: int = ID_BITS + DATE_BITS) -> int:
    crc = 0
    for i in reversed(range(bits)):
        crc ^= ((value >> i) & 1) << 7
        crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def writer_id_from(fingerprint: str) -> int:
    return int(fingerprint[:6], 16)


def encode_identity(fingerprint: str, published: _dt.date | str) -> tuple[int, int]:
    """Pack a writer id and publish date into the wire format. Returns (payload, n_bits)."""

    if isinstance(published, str):
        published = _dt.date.fromisoformat(published)
    days = (published - EPOCH).days
    if not 0 <= days < (1 << DATE_BITS):
        raise ValueError(f"publish date {published} outside the 13-bit window")
    writer_id = writer_id_from(fingerprint)
    head = (writer_id << DATE_BITS) | days
    return (head << CRC_BITS) | crc8(head), WIRE_BITS


def decode_identity(payload: int, bits: int = WIRE_BITS) -> tuple[int, _dt.date, bool]:
    head, crc = divmod(payload, 1 << CRC_BITS)
    writer_id, days = head >> DATE_BITS, head & ((1 << DATE_BITS) - 1)
    return writer_id, EPOCH + _dt.timedelta(days=days), crc == crc8(head)


def bit_source(payload: int, n_bits: int):
    """Slot i carries payload bit i % n_bits, most significant first."""
    def source(index: int) -> int:
        return (payload >> (n_bits - 1 - (index % n_bits))) & 1

    return source


@dataclass(frozen=True, slots=True)
class BitVote:
    index: int
    zero: int
    one: int

    @property
    def resolved(self) -> int | None:
        if self.zero == self.one:
            return None
        return 1 if self.one > self.zero else 0

    @property
    def p_value(self) -> float:
        total = self.zero + self.one
        if total < 2:
            return 1.0
        margin = abs(self.zero - self.one)
        # Excess over a fair 50/50 split of `total` votes.
        z = margin / (total**0.5)
        return normal_tail(z)


@dataclass(frozen=True, slots=True)
class PayloadRead:
    votes: tuple[BitVote, ...]
    unresolved: int
    inconclusive: int

    @property
    def payload(self) -> int | None:
        if self.unresolved:
            return None
        return int("".join(str(v.resolved) for v in self.votes), 2)

    @property
    def weakest(self) -> float:
        return max((v.p_value for v in self.votes), default=1.0)


def read_payload(
    text: str,
    key,
    n_bits: int,
    *,
    table: CuratedTable | None = None,
    window: int = DEFAULT_WINDOW,
) -> PayloadRead:
    """Vote each slot toward bit 0 and bit 1, then take per-bit majorities."""
    table = table or default_table()
    zero = walk(text, key, table, window=window, mode="detect", bit_source=lambda _i: 0)
    one = walk(text, key, table, window=window, mode="detect", bit_source=lambda _i: 1)
    tallies = [[0, 0] for _ in range(n_bits)]
    inconclusive = 0
    for index, entry in enumerate(zero):
        c0, c1 = entry.wanted, one[index].wanted
        if c0 == c1:
            inconclusive += 1  # the key demands the same word either way: no information
            continue
        if entry.observed == c0:
            tallies[index % n_bits][0] += 1
        elif entry.observed == c1:
            tallies[index % n_bits][1] += 1
    votes = tuple(BitVote(i, z, o) for i, (z, o) in enumerate(tallies))
    return PayloadRead(
        votes=votes,
        unresolved=sum(1 for v in votes if v.resolved is None),
        inconclusive=inconclusive,
    )


__all__ = [
    "EPOCH",
    "WIRE_BITS",
    "BitVote",
    "PayloadRead",
    "bit_source",
    "crc8",
    "decode_identity",
    "encode_identity",
    "read_payload",
    "writer_id_from",
]
