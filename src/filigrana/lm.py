"""Model-gated embedding: proposals from a provider, filtered by what a model found free.

`LocalModelScorer` answers one question and nothing else -- given the text before a position,
how probable is each *proposed* surface there. It never proposes: candidate sets come from a
provider, because a model's most likely next tokens are continuations, not substitutes ("The
engineer said slowly toward the station" is fluent and is not the same sentence). One forward
pass suffices per text, since a causal model gives the conditional distribution at every
position at once.

    scorer = LocalModelScorer()                        # torch + transformers
    marked = embed_gated(essay, key, WordNetProvider(), scorer)
    decision = scan_gated(marked.text, key, WordNetProvider(), scorer)

Costs, stated up front and measured in tools/measure_lm_gate.py:

* Slot discovery without a sidecar needs the **same model**: a transformers upgrade can move
  which positions counted as free, so the model id is stamped into every sidecar and every
  decision. A model change is a verifiability event, which is why `verify_sidecar` (in
  `sidecar.py`) needs no model at all.
* Discovery is self-referential -- substituting at position i shifts the distribution at i+1
  -- so embedding iterates to a fixed point and reports **retention**, the share of slots that
  still re-derive from the marked copy. An unretained slot was never really marked.
* WordNet proposals are sense-blind. The band filter is what makes them usable, which is why
  `WordNetProvider` declares `requires_gate = True` and `embed_gated` refuses to run it
  without a scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

from .canon import mask_markup, tokenize
from .decisions import Decision
from .detect import score_entries
from .gate import FreeChoiceGate, GateReport
from .keystream import keyed_choice
from .ledger import DEFAULT_WINDOW, Entry, anchor_before, word_count
from .providers import Slot

DEFAULT_MODEL = "gpt2"

#: Left-context rows each later window keeps in front of the rows it actually scores.
WINDOW_CONTEXT = 64


def plan_windows(
    n_ids: int, max_positions: int, context: int = WINDOW_CONTEXT
) -> list[tuple[int, int, int]]:
    """Predicting-row windows for a text longer than the model's position table.

    GPT-2's position table is 1,024 subwords; a 1,075-word chunk tokenises to ~1,900, which
    is how `tools/measure_lm_gate.py` acquired an `IndexError` deep inside `wpe` the first
    time anybody ran the tool at the length `docs/CAPACITY.md` quotes. A "row" here is one
    position of the single full-sequence forward pass: row r holds the distribution over
    token ``r + 1``, so the rows worth scoring are ``0 .. n_ids - 2``.

    A model accepts at most `max_positions` positions per call, so a long text is scored in
    slices of that same encoding -- ids are sliced, never re-tokenized, which keeps subword
    boundaries identical from window to window and keeps every row index valid.

    Window ``(start, end, own)`` receives tokens ``[start, end)`` and scores rows
    ``own .. end - 2``; the caller slices ``lp[own - start : end - 1 - start]`` out of the
    window's log-probabilities. Row ``end - 1`` is produced by this window but left for the
    next one, which can see more rows before it.

    The plan tiles rows ``0 .. n_ids - 2`` disjointly and contiguously, every window is at
    most `max_positions` tokens wide, and the last one ends at `n_ids`.
    `test_plan_windows` pins all three, because a window planner with a seam in it is a
    silent hole in the mark.
    """
    if n_ids <= max_positions:
        return [(0, n_ids, 0)]
    out: list[tuple[int, int, int]] = []
    covered = 0  # the next scoring row that is not owned yet
    end = 0
    while covered <= n_ids - 2:
        start = max(0, end - context - 1) if out else 0
        end = min(start + max_positions, n_ids)
        out.append((start, end, covered))
        covered = end - 1
    return out


class LocalModelScorer:
    """Conditional log-probability of a proposed word at a proposed position.

    Constructed explicitly, never imported implicitly: tier 1 has to work on a laptop with no
    model download, and that is a property of the library, not a packaging detail.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, *, device: str = "cpu", dtype: str = "float32") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "the model-gated path needs 'filigrana[lm]' (torch + transformers). The curated "
                "path in filigrana.embed / filigrana.detect needs nothing at all."
            ) from exc
        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"{model_id}: the gate needs offset mapping to place slots in the writer's own "
                "text, and slow tokenizers do not provide it."
            )
        # transformers 5.x's stubs type `from_pretrained`'s first argument as the class it
        # should return, so a working call reads as an error. Any at this one line is the
        # honest annotation of a stub that disagrees with its own runtime.
        # transformers 5.x's stubs type `from_pretrained`'s first argument as the class it
        # should return, so this working call reads as an error; Any here says exactly that.
        model = cast(Any, AutoModelForCausalLM).from_pretrained(model_id, dtype=getattr(torch, dtype))
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        cfg = self.model.config
        max_positions = getattr(cfg, "n_positions", None) or getattr(
            cfg, "max_position_embeddings", None
        )
        if not max_positions:
            raise RuntimeError(
                f"{model_id}: its config advertises no context length, so the scorer cannot "
                "plan its windows. Score a model that states one."
            )
        self.max_positions = int(max_positions)
        self._id_cache: dict[str, int | None] = {}

    def token_id(self, surface: str, *, spaced: bool) -> int | None:
        """A single-subword id for `surface`, else None.

        Refusing multi-token words is not laziness: a two-subword substitute would need its
        joint probability, and treating the first token's probability as the word's would
        quietly overstate how free that position was. Density drops; the numbers stay honest.
        """
        key = (" " if spaced else "") + surface
        if key in self._id_cache:
            return self._id_cache[key]
        ids = self.tokenizer(key, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        value = ids[0] if len(ids) == 1 else None
        self._id_cache[key] = value
        return value

    def score(self, text: str) -> Pass:
        masked = mask_markup(text)
        enc = self.tokenizer(masked, return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        ids = enc["input_ids"].to(self.device)
        n_ids = ids.shape[1]
        # One full-sequence tokenisation, scored through `plan_windows` slices of *it*, so a
        # text longer than the position table still gives every row real left context
        # without a second definition of where the tokens are.
        owned: list[Any] = []  # tensors; torch is the optional-dep boundary, so Any by design
        with self.torch.no_grad():
            for start, end, own in plan_windows(n_ids, self.max_positions):
                logits = self.model(input_ids=ids[:, start:end]).logits[0]
                owned.append(self.torch.log_softmax(logits, dim=-1)[own - start : end - 1 - start])
            log_probs = self.torch.cat(owned, dim=0) if owned else self.torch.zeros(0, 0)
        # Byte-level BPE pulls the following space into the *preceding* piece, so a word's
        # character offset sits one inside the subword that starts it. Map characters to
        # whichever subword covers them; assuming the two starts agree finds zero slots.
        owner = [-1] * (len(masked) + 1)
        for pos, (start, end) in enumerate(offsets):
            for char in range(start, max(end, start + 1)):
                if char < len(owner) and owner[char] == -1:
                    owner[char] = pos
        return Pass(
            scorer=self,
            text=text,
            masked=masked,
            tokens=tokenize(text),
            log_probs=log_probs,
            owner=owner,
        )


@dataclass
class Pass:
    """One forward pass, indexed by our word tokens."""

    scorer: LocalModelScorer
    text: str
    masked: str
    tokens: list
    # A [rows, vocab] tensor on the optional-dep boundary: object-typed on purpose.
    log_probs: Any
    owner: list[int] = field(repr=False, default_factory=list)

    def logprob_of(self, index: int, members: tuple[str, ...]) -> tuple[float, ...]:
        token = self.tokens[index]
        row = self._predicting_row(token.start)
        if row is None:
            return tuple(-math.inf for _ in members)
        spaced = token.start == 0 or self.masked[token.start - 1].isspace()
        out = []
        for member in members:
            tid = self.scorer.token_id(member, spaced=spaced)
            out.append(-math.inf if tid is None else float(row[tid]))
        return tuple(out)

    def _predicting_row(self, char_start: int):
        pos = self.owner[char_start] if char_start < len(self.owner) else -1
        if pos <= 0:
            return None
        return self.log_probs[pos - 1]

    def mean_logprob(self) -> float:
        """Mean conditional log-probability of exactly this text. The style-cost number."""
        ids = self.scorer.tokenizer(self.masked, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            return 0.0
        # `log_probs` is already [sequence, vocab] with the batch axis dropped, so the
        # targets must be flattened the same way or gather compares 2-D against 3-D.
        rows = self.log_probs[: len(ids) - 1]
        targets = self.scorer.torch.tensor(ids[1:], device=rows.device).unsqueeze(-1)
        hit = rows.gather(-1, targets).squeeze(-1)
        return float(hit.mean())


def build_entries(
    scoring: Pass,
    gate_slots,
    key,
    *,
    window: int = DEFAULT_WINDOW,
    bit_source=None,
    veto=None,
    mode: str = "detect",
) -> list[Entry]:
    """Keyed demands for gated slots, anchored on words no proposal can rewrite."""
    reachable = frozenset(
        w for slot in gate_slots for member in slot.members for w in member.split()
    )
    entries: list[Entry] = []
    for position, slot in enumerate(gate_slots):
        bit = bit_source(position) if bit_source is not None else None
        anchor = anchor_before(scoring.tokens, slot.token_index, window, reachable)
        wanted = keyed_choice(
            key.secret, slot.class_id, slot.n, anchor, window=window, key_id=key.key_id, bit=bit
        )
        vetted = mode == "embed" and veto is not None and veto(slot, wanted)
        final = slot.observed if (mode != "embed" or vetted) else wanted
        entries.append(
            Entry(
                class_id=slot.class_id,
                tier=slot.tier,
                start=slot.start,
                end=slot.end,
                members=slot.members,
                observed=slot.observed,
                wanted=wanted,
                final=final,
                bit=bit,
                vetted=vetted,
                anchor=anchor,
            )
        )
    return entries


def reachable_words(gate_slots) -> frozenset[str]:
    return frozenset(w for slot in gate_slots for member in slot.members for w in member.split())


def _mean_logprob(scoring) -> float:
    """Style cost when the scorer can report one; 0.0 for a scripted test scorer.

    Reported as measured-or-absent rather than faked, because a zero in the output of a real
    run would be a lie and a zero from a script is simply not applicable.
    """
    return scoring.mean_logprob() if hasattr(scoring, "mean_logprob") else 0.0


def propose(provider, text: str, scoring, gate: FreeChoiceGate) -> GateReport:
    """Filter a provider's proposals through the band test.

    `scoring` is duck-typed rather than isinstance-checked: it only has to expose `.tokens`
    and `.logprob_of`, which is what the gate actually uses and what lets the gate be tested
    against a scripted scorer instead of a two-gigabyte model.
    """
    if getattr(provider, "requires_gate", False) and not hasattr(scoring, "logprob_of"):
        raise TypeError(
            f"{type(provider).__name__} proposes sense-blind and must be filtered by a gate"
        )
    slots: list[Slot] = provider.slots(text, scoring.tokens)
    return gate.apply(scoring.tokens, slots, scoring.logprob_of)


@dataclass(frozen=True, slots=True)
class GatedResult:
    text: str
    entries: tuple[Entry, ...]
    words: int
    rounds: int
    retained: float
    logprob_shift: float
    model_id: str
    provider: str
    gate: dict
    sidecar: dict
    # Substitutions written during convergence. The `.edits` property below counts what a
    # re-embed of the fixed point would still change -- zero at convergence -- so a
    # measurement tool must report this field instead: the cost happened across rounds,
    # not after the last one.
    edits_applied: int = 0

    @property
    def slots(self) -> int:
        return len(self.entries)

    @property
    def rho(self) -> float:
        return 1000 * self.slots / max(self.words, 1)

    @property
    def edits(self) -> int:
        return sum(1 for e in self.entries if e.final != e.observed)

    def summary(self) -> str:
        return (
            f"{self.words} words -> {self.slots} gated choices ({self.rho:.1f} per 1000 words)"
            f" via {self.provider} + {self.model_id}, stable after {self.rounds} pass(es)\n"
            f"  {100 * self.retained:.1f}% re-derive from the marked copy alone; mean log-prob"
            f" shift {self.logprob_shift:+.4f}/token\n"
            f"  {self.edits_applied} edits applied across rounds, "
            f"{sum(1 for e in self.entries if e.vetted)} declined"
        )


def embed_gated(
    text: str,
    key,
    provider,
    scorer: LocalModelScorer,
    *,
    gate: FreeChoiceGate | None = None,
    window: int = DEFAULT_WINDOW,
    max_rounds: int = 4,
    veto=None,
    bit_source=None,
) -> GatedResult:
    """Mark prose with proposals a model judged free, iterated until the ledger settles."""
    gate = gate or FreeChoiceGate()
    if getattr(provider, "requires_gate", False) and not hasattr(scorer, "score"):
        raise TypeError(
            f"{type(provider).__name__} proposes sense-blind and must be filtered by a gate; "
            "pass a LocalModelScorer, or use the curated table instead"
        )
    baseline = _mean_logprob(scorer.score(text))
    current = text
    entries: list[Entry] = []
    rounds = 0
    applied = 0
    from .embed import apply_diff

    for rounds in range(1, max_rounds + 1):
        scoring = scorer.score(current)
        report = propose(provider, current, scoring, gate)
        entries = build_entries(
            scoring,
            report.slots,
            key,
            window=window,
            veto=veto,
            bit_source=bit_source,
            mode="embed",
        )
        nxt = apply_diff(current, entries)
        applied += sum(1 for entry in entries if entry.final != entry.observed)
        if nxt == current:
            break
        current = nxt

    final = scorer.score(current)
    final_report = propose(provider, current, final, gate)
    final_entries = build_entries(
        final, final_report.slots, key, window=window, bit_source=bit_source
    )
    ledger = {(e.class_id, e.final, e.anchor) for e in entries}
    retained = (
        sum(1 for e in final_entries if (e.class_id, e.observed, e.anchor) in ledger) / len(ledger)
        if ledger
        else 0.0
    )
    gate_conf = {
        "epsilon": gate.epsilon,
        "min_band": gate.min_band,
        "max_members": gate.max_members,
        "min_members": gate.min_members,
    }
    from .sidecar import build_sidecar

    return GatedResult(
        text=current,
        entries=tuple(final_entries),
        words=word_count(current),
        rounds=rounds,
        retained=retained,
        logprob_shift=_mean_logprob(scorer.score(current)) - baseline,
        model_id=scorer.model_id,
        provider=type(provider).__name__,
        gate=gate_conf,
        edits_applied=applied,
        sidecar=build_sidecar(
            final_entries,
            key_id=key.key_id,
            window=window,
            reachable=reachable_words(final_report.slots),
            model=scorer.model_id,
            provider=type(provider).__name__,
            gate=gate_conf,
            retained=retained,
        ),
    )


def scan_gated(
    text: str,
    key,
    provider,
    scorer: LocalModelScorer,
    *,
    gate: FreeChoiceGate | None = None,
    window: int = DEFAULT_WINDOW,
    n_hypotheses: int = 1,
    bit_source=None,
) -> Decision:
    gate = gate or FreeChoiceGate()
    scoring = scorer.score(text)
    report = propose(provider, text, scoring, gate)
    entries = build_entries(scoring, report.slots, key, window=window, bit_source=bit_source)
    return score_entries(entries, words=word_count(text), key=key, n_hypotheses=n_hypotheses)


__all__ = [
    "DEFAULT_MODEL",
    "WINDOW_CONTEXT",
    "GatedResult",
    "LocalModelScorer",
    "Pass",
    "build_entries",
    "embed_gated",
    "plan_windows",
    "propose",
    "reachable_words",
    "scan_gated",
]
