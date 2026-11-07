"""The output type is the safety mechanism.

`inkmark detect` produces evidence about a *person*, and the realistic harm is a false
accusation of plagiarism or a fabricated attribution. So the constraints live here rather
than in a disclaimer:

* no truthy path -- `bool(Decision(...))` raises, and `if decision:` in any caller is a
  bug that fails loudly at runtime;
* the number is a p-value about *this text under this key*, never a verdict;
* multiple hypotheses are corrected for, because scanning a registry of N writers at
  per-test p=1e-6 buys you roughly N times the false accusations;
* `caveats` is never empty, and short text says so itself.
"""


import math
from dataclasses import dataclass, field

_MIN_Z_FOR_VERDICT = 4.754  # one-sided p < 1e-6

CAUTION = (
    "This is a statistical test of whether one writer's key explains the word choices in "
    "this text. It is not proof of authorship, it cannot say who else might have written "
    "it, and it cannot show that the key holder owns the underlying text. Genius planted a "
    "watermark in 301 songs, found it in Google's index, and still lost the case because it "
    "licensed the lyrics. Treat this as one piece of evidence next to timestamps and drafts."
)


@dataclass(frozen=True, slots=True)
class Decision:
    slots: int
    hits: int
    expected: float
    variance: float
    z: float
    p_value: float
    key_id: str
    fingerprint: str
    tier_counts: dict[str, int] = field(default_factory=dict)
    words: int = 0
    n_hypotheses: int = 1
    payload: int | None = None
    payload_bits: int = 0
    payload_confidence: float | None = None
    caveats: tuple[str, ...] = ()

    def __bool__(self) -> bool:  # pragma: no cover - must never be reached
        raise TypeError(
            "Decision has no truth value. A watermark score is not a verdict; read "
            ".p_value and .caveats, and see docs/THREAT-MODEL.md before acting on it."
        )

    @property
    def p_value_corrected(self) -> float:
        """Sidak-corrected for the number of keys tested."""

        if self.n_hypotheses <= 1:
            return self.p_value
        return min(1.0, self.p_value * self.n_hypotheses)

    @property
    def evidence_bits(self) -> float:
        """log2 likelihood ratio in favour of this key over chance. Signed."""
        if self.variance <= 0:
            return 0.0
        return -math.log2(max(self.p_value, 5e-324)) if self.z > 0 else 0.0

    def render(self) -> str:
        lines = [
            f"key        {self.key_id}  fingerprint {self.fingerprint}"
            + ("  [REVOKED]" if "key_revoked" in self.caveats else ""),
            f"length     {self.words} words, {self.slots} key-correlated choices"
            + (f"  ({', '.join(f'{t}:{c}' for t, c in sorted(self.tier_counts.items()))})"
               if self.tier_counts else ""),
            (f"agreement  {self.hits}/{self.slots} slots on the key's side "
            f"(chance would give {self.expected:.1f})"),
            f"statistics z = {self.z:.2f}, p = {self.p_value:.3g}"
            + (f", p after {self.n_hypotheses} hypotheses = {self.p_value_corrected:.3g}"
               if self.n_hypotheses > 1 else ""),
            f"strength   {self.evidence_bits:.1f} bits of evidence against chance",
        ]
        if self.payload is not None:
            lines.append(
                f"payload    0x{self.payload:x} ({self.payload_bits} bits), "
                f"decoder confidence p = {self.payload_confidence:.3g}"
                if self.payload_confidence is not None
                else f"payload    0x{self.payload:x} ({self.payload_bits} bits)"
            )
        lines.append("")
        lines.extend(f"  ! {c}" for c in self.caveats)
        lines.append("")
        lines.append(f"  {CAUTION}")
        return "\n".join(lines)


def normal_tail(z: float) -> float:
    """One-sided upper tail of the standard normal, with continuity-free exact erfc."""
    if z <= -8:
        return 1.0
    if z >= 8:
        return 0.0
    return 0.5 * math.erfc(z / math.sqrt(2))


__all__ = ["CAUTION", "_MIN_Z_FOR_VERDICT", "Decision", "normal_tail"]
