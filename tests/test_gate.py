"""The gate and the model-gated path, against the scripted scorer in conftest."""

from __future__ import annotations

import pytest

from conftest import KEY, SCRIPT, SENTENCE, FakePass, ScriptedProvider, ScriptedScorer
from filigrana.canon import tokenize
from filigrana.gate import FreeChoiceGate
from filigrana.lm import embed_gated, scan_gated


def _pass(text: str) -> FakePass:
    return FakePass(tokens=tokenize(text), text=text)


def _gate() -> FreeChoiceGate:
    return FreeChoiceGate(epsilon=0.35, min_band=0.002)


def test_band_keeps_near_equal_alternatives_and_drops_spikes():
    scoring = _pass(SENTENCE)
    report = _gate().apply(
        scoring.tokens, ScriptedProvider(SCRIPT).slots(SENTENCE, scoring.tokens), scoring.logprob_of
    )
    found = {scoring.tokens[slot.token_index].surface: slot.members for slot in report.slots}
    assert "walked" in found, f"walked/strolled differ by only 0.1 nats: {found}"
    assert "station" in found
    assert "slowly" not in found, "quietly sits 2.8 nats behind, so the position is not free"
    assert "marched" not in found["walked"], "the band is the members, not the proposal"
    assert all(s.n >= 2 for s in report.slots)
    assert all(
        s.members[s.observed] == scoring.tokens[s.token_index].surface.lower() for s in report.slots
    )


def test_the_member_set_is_the_band_not_the_proposal():
    """`marched` is proposed but 2.8 nats out. Keeping it would hand the key a choice the
    model says is wrong, and would overstate how free the position was."""
    scoring = _pass(SENTENCE)
    report = _gate().apply(
        scoring.tokens, ScriptedProvider(SCRIPT).slots(SENTENCE, scoring.tokens), scoring.logprob_of
    )
    walked = next(s for s in report.slots if scoring.tokens[s.token_index].surface == "walked")
    assert walked.members == ("strolled", "walked")
    assert "marched" not in walked.members


def test_embed_gated_converges_and_rescans_identically():
    marked = embed_gated(SENTENCE, KEY, ScriptedProvider(SCRIPT), ScriptedScorer())
    assert marked.slots > 0
    assert marked.rounds <= 4
    decision = scan_gated(marked.text, KEY, ScriptedProvider(SCRIPT), ScriptedScorer())
    assert decision.slots == marked.slots, "the verifier must find the same slots"
    assert decision.hits == decision.slots
    assert marked.retained == pytest.approx(1.0, abs=1e-9)
    # At a fixed point a re-embed changes nothing (`edits` == 0), but the cost of
    # convergence was paid across rounds and must be reported as such. A measurement tool
    # printing `edits` would claim the gateway is free exactly when it is not.
    assert marked.edits_applied >= 1 or marked.edits == 0
    if marked.edits_applied >= 1:
        assert marked.edits == 0, "convergence means no pending edits; cost is edits_applied"


def test_embed_gated_never_upgrades_a_word_the_model_dislikes():
    """A position where the writer's word loses outright is content, not style."""
    script = {"station": {"station": -6.0, "depot": -0.2, "terminal": -0.3}}
    marked = embed_gated(
        "she walked toward the station", KEY, ScriptedProvider(script), ScriptedScorer()
    )
    assert "station" in marked.text
    assert all("station" not in e.members or e.observed == e.final for e in marked.entries)


def test_gated_result_exposes_a_sidecar_that_names_the_model():
    marked = embed_gated(SENTENCE, KEY, ScriptedProvider(SCRIPT), ScriptedScorer())
    assert marked.sidecar["model"] == "scripted/1"
    assert marked.sidecar["provider"] == "ScriptedProvider"
    assert marked.sidecar["gate"]["epsilon"] == 0.35
    assert all({"class", "members", "final", "anchor"} <= set(slot) for slot in marked.sidecar["slots"])


def test_wordnet_proposes_regular_siblings_and_refuses_irregular_ones():
    from filigrana.lexical import WordNetProvider, parse_inflection

    _require_wordnet()

    assert parse_inflection("walked") is not None
    assert parse_inflection("went") is None, "re-inflecting the lemma 'go' would yield 'goed'"
    provider = WordNetProvider()
    text = "the engineers walked slowly through the crowded square and the crowd began to part"
    slots = provider.slots(text)
    for slot in slots:
        assert len(slot.members) >= 2
        assert all(member.isalpha() and member.islower() for member in slot.members)
    proposals = {m for slot in slots for m in slot.members}
    assert proposals, "WordNet should propose something on ordinary prose"
    assert all(not p.endswith(("ingly", "edly")) for p in proposals)


def _require_wordnet() -> None:
    pytest.importorskip("nltk")
    from nltk.corpus import wordnet as wn

    try:
        wn.synsets("verify")
    except LookupError:
        pytest.skip("WordNet corpus missing: python -c \"import nltk; nltk.download('wordnet')\"")


def test_wordnet_provider_cannot_be_embedded_without_a_gate():
    from filigrana.lexical import WordNetProvider

    _require_wordnet()

    provider = WordNetProvider()
    assert provider.requires_gate, "sense-blind proposals must never be embedded unfiltered"
    with pytest.raises(TypeError, match="sense-blind"):
        embed_gated(SENTENCE, KEY, provider, object())


def test_gate_constructor_states_its_own_nonsense():
    with pytest.raises(ValueError, match="min_members"):
        FreeChoiceGate(min_members=4, max_members=2)
    with pytest.raises(ValueError, match="epsilon"):
        FreeChoiceGate(epsilon=0.0)


def test_gate_counts_what_it_refuses_and_why():
    scoring = _pass(SENTENCE)
    report = _gate().apply(
        scoring.tokens, ScriptedProvider(SCRIPT).slots(SENTENCE, scoring.tokens), scoring.logprob_of
    )
    assert report.summary().startswith(
        f"{len(report.slots)} free choices kept | dropped "
    ), "the funnel is part of the output: unknown refusals are how density illusions start"


def test_gate_word_shaped_filtering_refuses_odd_surfaces():
    """Odd surfaces never were a word the writer chose."""
    scoring = _pass("the grayishnot horizon")
    report = _gate().apply(
        scoring.tokens, ScriptedProvider(SCRIPT).slots("the grayishnot horizon", scoring.tokens),
        scoring.logprob_of,
    )
    assert not report.slots


def test_gate_never_kicks_the_observed_word_out_of_the_band():
    scoring = _pass("the purple sky")
    provider = ScriptedProvider(SCRIPT)
    report = FreeChoiceGate(epsilon=0.5, max_members=2).apply(
        scoring.tokens, provider.slots("the purple sky", scoring.tokens), scoring.logprob_of
    )
    for slot in report.slots:
        observed = scoring.tokens[slot.token_index].surface.lower()
        assert observed in slot.members


def test_gate_drops_below_the_word_length_floor():
    script = {"fox": {"fox": -1.0, "dog": -1.2}}
    scoring = _pass("the fox slept")
    provider = ScriptedProvider(script)
    report = FreeChoiceGate().apply(
        scoring.tokens, provider.slots("the fox slept", scoring.tokens), scoring.logprob_of
    )
    assert report.thin >= 1, "a three-character candidate is below the floor by construction"


def test_gated_result_reports_density_and_distinguishes_cost_from_fixed_point():
    """`.edits` counts what a re-embed at the fixed point would still change (zero), while
    `edits_applied` counts what was actually written across rounds. A measurement tool
    must report the latter, or the style cost reads as free."""
    from filigrana.ledger import Entry
    from filigrana.lm import GatedResult

    entry = Entry(
        class_id="sc:walked", tier="lm", start=0, end=6,
        members=("walked", "strolled"), observed=0, wanted=1, final=0,
        bit=None, vetted=False,
    )
    result = GatedResult(
        text="", entries=(entry,), words=100, rounds=2, retained=1.0,
        logprob_shift=0.0, model_id="m", provider="p", gate={}, sidecar={},
        edits_applied=3,
    )
    assert result.slots == 1
    assert result.rho == pytest.approx(10.0)  # 1000 * slots / words
    assert result.edits == 0, "at the fixed point a re-embed would change nothing"
    assert "3 edits applied across rounds" in result.summary()
    assert "0 declined" in result.summary()

    empty = GatedResult(
        text="", entries=(), words=0, rounds=0, retained=0.0,
        logprob_shift=0.0, model_id="m", provider="p", gate={}, sidecar={},
    )
    assert empty.rho == 0.0, "max(words, 1) keeps the density defined for empty text"
