"""The style guard: what counts as filler, and how it is reported.

`slop_rate` numbers are committed in bench/interference.json, so the membership of the slop
list is a measured constant, not a style preference. These tests pin the exact behaviour the
artifacts were generated with, including the multi-word rule ('owing to' keys on its first
word, and 'owing' is itself a listed member).
"""

from inkmark.slop import SLOP_WORDS, StyleProfile, is_slop, profile, slop_predicate


class TestIsSlop:
    def test_listed_single_words_count(self):
        assert is_slop("moreover")
        assert is_slop("delve")
        assert is_slop("Underscores")  # case-insensitive

    def test_non_members_do_not_count(self):
        assert not is_slop("walk")
        assert not is_slop("station")

    def test_multiword_members_count_through_their_first_word(self):
        # 'owing to' and 'delve into' are candidate members of whole classes; the rule
        # keys on the first word so the writer's guard fires on the phrase, not the head.
        assert is_slop("owing to")
        assert is_slop("delve into")

    def test_bare_first_words_count_by_direct_membership(self):
        assert "owing" in SLOP_WORDS
        assert is_slop("owing")

    def test_predicate_shares_the_semantics(self):
        # slop_predicate is the exact callable handed to embed(disprefer=...).
        assert slop_predicate("moreover") is True
        assert slop_predicate("walked") is False


class TestStyleProfile:
    def test_as_row_has_fixed_columns(self):
        row = StyleProfile(
            words=100,
            burstiness=1.5,
            ttr=0.5,
            long_rate=0.1,
            cliches_per_1k=2.0,
            connective_rate=0.25,
            slop_rate=0.02,
        ).as_row("marked")
        assert row.startswith("marked".ljust(18))
        assert "100" in row
        assert "1.500" in row
        assert "0.5000" in row

    def test_profile_counts_what_it_says(self):
        text = (
            "Moreover, she was quietly delving into the tapestry of the landscape. "
            "It is not just a tool, but a weapon. We walked to the station."
        )
        p = profile(text)
        assert p.words > 0
        assert p.slop_rate > 0
        assert p.cliches_per_1k > 0  # 'quietly', 'delving', 'tapestry', the not-just-but frame
        assert p.connective_rate > 0  # the first sentence opens with Moreover

    def test_empty_text_degrades_to_zeros_not_crashes(self):
        p = profile("")
        assert p.words == 1  # max(len, 1) keeps division defined
        assert p.ttr == 0.0
        assert p.burstiness == 0.0
        assert p.connective_rate == 0.0
        assert p.slop_rate == 0.0
