"""The WordNet lexical provider and the suffix machine behind it.

WordNet stores lemmas, never forms, so `carry` + past has to be produced by rule. The rule
must reproduce the observed surface from its own lemma (`carries` -> carry -> `carries`) or
the writer's own word can never appear in its own proposal -- the y -> i family is exactly
where a naive append produced `carrys` and `happyly`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nltk")


@pytest.fixture(scope="module", autouse=True)
def _wordnet_corpus():
    from nltk.corpus import wordnet as wn

    try:
        wn.synsets("verify")
    except LookupError:
        pytest.skip("WordNet corpus missing: python -c \"import nltk; nltk.download('wordnet')\"")


from inkmark.lexical import Inflection, WordNetProvider, parse_inflection  # noqa: E402


class TestSuffixMachine:
    """parse_inflection plus the form/inflect round trip: the observed word must re-derive."""

    @pytest.mark.parametrize(
        ("surface", "lemma", "suffix"),
        [
            ("carries", "carry", "s"),
            ("studies", "study", "s"),
            ("carried", "carry", "ed"),
            ("studied", "study", "ed"),
            ("spied", "spy", "ed"),
            ("happily", "happy", "ly"),
            ("warily", "wary", "ly"),
            ("carrying", "carry", "ing"),  # -ing never swaps the y
            ("plays", "play", "s"),  # vowel-final y is untouched by the swap
            ("walked", "walk", "ed"),
        ],
    )
    def test_parse_and_self_form_round_trip(self, surface, lemma, suffix):
        parsed = parse_inflection(surface)
        assert parsed == Inflection(lemma=lemma, suffix=suffix)
        assert parsed.form(parsed.lemma) == surface

    def test_vowel_y_reinflects_by_plain_append(self):
        parsed = parse_inflection("plays")
        assert parsed.form("say") == "says"
        assert parsed.form("enjoy") == "enjoys"

    def test_consonant_y_carries_across_siblings(self):
        carries = parse_inflection("carries")
        assert carries.inflect("study") == "studies"
        assert carries.inflect("deny") == "denies"
        carried = parse_inflection("carried")
        assert carried.inflect("study") == "studied"
        assert carried.inflect("play") == "played"
        happily = parse_inflection("happily")
        assert happily.inflect("angry") == "angrily"
        assert happily.form("sad") == "sadly"

    def test_plural_spelling_still_gets_its_e(self):
        bunches = parse_inflection("bunches")
        assert bunches.inflect("box") == "boxes"

    def test_doubled_suffixes_are_refused_not_doubled(self):
        happily = parse_inflection("happily")
        assert happily.inflect("quickly") is None

    def test_short_words_stay_unparsed(self):
        # The [a-z]{5,} floor: 'says'/'days' are too short to guess about.
        assert parse_inflection("says") is None
        assert parse_inflection("days") is None

    def test_irregular_garbage_parses_then_fails_the_band_by_design(self):
        # 'speaked' cannot be fixed from a lemma list -- the module docstring is explicit
        # that the model's band filter is what prunes it. Pinning that it still parses as
        # speak+ed keeps that contract visible instead of silently 'fixed' into refusal.
        assert parse_inflection("speaked") == Inflection(lemma="speak", suffix="ed")

    def test_non_words_and_marks_are_refused(self):
        assert parse_inflection("zzzzz") is None
        assert parse_inflection("don't") is None
        assert parse_inflection("") is None
        assert parse_inflection("12345") is None

    def test_unparseable_inflections_exhaust_every_candidate_then_refuse(self):
        # Each loop must fall through to None when no candidate lemma is a WordNet word;
        # a wrong guess ('zzzz' as a stem) would poison a real proposal downstream.
        assert parse_inflection("zzzing") is None
        assert parse_inflection("zzzed") is None
        assert parse_inflection("zzzly") is None
        assert parse_inflection("zzzs") is None


class TestWordNetProvider:
    def test_one_member_max_is_refused(self):
        with pytest.raises(ValueError, match="not a choice"):
            WordNetProvider(max_members=1)

    def test_declares_itself_gate_required(self):
        # Sense-blind proposals must never be embedded unfiltered; embed_gated refuses.
        assert WordNetProvider().requires_gate is True

    def test_inflected_proposals_include_the_observed_surface(self):
        slots = WordNetProvider().slots("she carries the weight")
        assert slots, "'carries' should parse and propose"
        slot = slots[0]
        assert slot.class_id == "wn:carry"
        assert "carries" in slot.members
        assert slot.observed == slot.members.index("carries")

    def test_ed_forms_propose_too(self):
        slots = WordNetProvider().slots("she carried the weight")
        assert slots and slots[0].class_id == "wn:carry"
        assert "carried" in slots[0].members

    def test_base_forms_are_left_alone(self):
        # An uninflected word's part of speech is unknown; proposing anyway would
        # re-inflect the wrong POS ('carry' read as a noun, say).
        assert WordNetProvider().slots("she carry the weight") == []

    def test_capitalised_and_short_words_are_skipped(self):
        assert WordNetProvider().slots("Carries the weight") == []
        assert WordNetProvider().slots("it ends here") == []  # 'ends' is under the floor


def test_propose_refuses_a_gate_required_provider_without_a_scorer():
    # lm.propose is duck-typed: a provider that is sense-blind must be handed something
    # that can say which candidates the model finds plausible. `tokens` alone is not enough.
    from types import SimpleNamespace

    from inkmark.gate import FreeChoiceGate
    from inkmark.lm import propose

    scoring = SimpleNamespace(tokens=[])
    with pytest.raises(TypeError, match="sense-blind"):
        propose(WordNetProvider(), "she carried the weight", scoring, FreeChoiceGate())
