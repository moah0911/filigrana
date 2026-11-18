"""The registry: a text file you host, not a service we run.

Tier 2 decodes a 24-bit writer id out of the wording. That id means nothing without a
mapping to a name, and the moment you build the mapping centrally you have built a
deanonymisation service -- so `filigrana` does not. A registry is a static JSON document a
writer publishes on their own site, the same way they would publish a fingerprint or a PGP
key. Anyone may read it; nobody operates it.

Because resolution is a lookup rather than a decode, the payload stays small: 24 bits of id
instead of enough bits to spell a handle, which is the difference between book-length
requirements and merely long ones.
"""


import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import date

FORMAT = "filigrana-registry/1"


@dataclass(frozen=True, slots=True)
class Entry:
    writer_id: str  # hex, 24 bits, first 6 hex of the fingerprint
    fingerprint: str
    handle: str
    url: str = ""
    key_id: str = ""
    added: str = ""

    @property
    def domains(self) -> list[str]:
        if not self.url:
            return []
        rest = self.url.split("//", 1)[-1]
        return [rest.split("/", 1)[0].lower()]


class Registry:
    def __init__(self, entries: Iterable[Entry] | None = None) -> None:
        self._by_id: dict[str, Entry] = {}
        for entry in entries or ():
            self.add(entry)

    def add(self, entry: Entry) -> Entry:
        existing = self._by_id.get(entry.writer_id)
        if existing and existing.fingerprint != entry.fingerprint:
            raise ValueError(
                f"writer id {entry.writer_id} already belongs to fingerprint "
                f"{existing.fingerprint}; a collision means two keys claim one id. Rotate."
            )
        self._by_id[entry.writer_id] = entry
        return entry

    def resolve(self, writer_id: int) -> Entry | None:
        return self._by_id.get(f"{writer_id:06x}")

    def for_fingerprint(self, fingerprint: str) -> Iterator[Entry]:
        return (e for e in self._by_id.values() if e.fingerprint == fingerprint)

    def keys_for_domain(self, domain: str) -> list[Entry]:
        """Which writers to test against text from one site -- the scoped verification list."""

        domain = domain.lower()
        return [e for e in self._by_id.values() if domain in e.domains]

    def to_json(self) -> str:
        return json.dumps(
            {"format": FORMAT, "entries": [asdict(e) for e in sorted(self._by_id.values(), key=lambda x: x.writer_id)]},
            indent=1,
        )

    @classmethod
    def from_json(cls, blob: str) -> "Registry":
        try:
            data = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise ValueError(f"registry is not valid JSON: {exc.msg}") from None
        if not isinstance(data, dict):
            raise ValueError("a filigrana registry must be a JSON object")
        if data.get("format") != FORMAT:
            raise ValueError("not a filigrana registry")
        entries = data.get("entries")
        if not isinstance(entries, list):
            raise ValueError("registry 'entries' must be a list")
        try:
            return cls(Entry(**item) for item in entries)
        except TypeError as exc:
            raise ValueError(f"malformed registry entry: {exc}") from None

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Entry]:
        return iter(sorted(self._by_id.values(), key=lambda e: e.writer_id))


def entry_for(key, handle: str, url: str = "") -> Entry:
    from .codec import writer_id_from

    return Entry(
        writer_id=f"{writer_id_from(key.fingerprint):06x}",
        fingerprint=key.fingerprint,
        handle=handle,
        url=url,
        key_id=key.key_id,
        added=date.today().isoformat(),
    )


__all__ = ["FORMAT", "Entry", "Registry", "entry_for"]
