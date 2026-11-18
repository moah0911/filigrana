"""The equivalence table: validation and the context guards that keep it safe.

A class is admissible only if every member can replace every other member everywhere the
class appears. The constructor refuses to even represent a class that breaks that promise,
and the guards refuse contexts where a safe-looking pair still diverges -- `although` and
`though` agree mid-clause but not clause-final, which is what `avoid_clause_final` is for.
"""

from __future__ import annotations

import pytest

from filigrana.providers import CuratedTable, EquivalenceClass, default_table, normalise_tier


class TestClassValidation:
    def test_one_member_is_not_a_choice(self):
        with pytest.raises(ValueError, match="at least two members"):
            EquivalenceClass(id="x", members=("only",), tier="b")

    def test_repeated_members_are_rejected(self):
        with pytest.raises(ValueError, match="repeated members"):
            EquivalenceClass(id="x", members=("same", "same"), tier="b")

    def test_unsorted_members_are_rejected(self):
        # The sorted tuple is the canonical index order embedder and detector agree on;
        # an index that depends on insertion order would not re-derive on a copy.
        with pytest.raises(ValueError, match="sorted"):
            EquivalenceClass(id="x", members=("zebra", "alpha"), tier="b")

    def test_duplicate_class_ids_are_rejected(self):
        twin = EquivalenceClass(id="twin", members=("aa", "ab"), tier="a")
        with pytest.raises(ValueError, match="duplicate class id"):
            CuratedTable([twin, twin])


class TestTheShippedTable:
    def test_tiers_are_canonicalised_before_they_select_classes(self):
        assert normalise_tier("BA") == "ab"
        assert normalise_tier("aab") == "ab"
        assert normalise_tier("all") == "all"
        assert default_table("ba").class_count == default_table("ab").class_count

    @pytest.mark.parametrize("tier", ["", "xyz", "d", "a-b"])
    def test_a_typo_cannot_silently_select_no_classes(self, tier):
        with pytest.raises(ValueError, match="tier must be"):
            normalise_tier(tier)
        with pytest.raises(ValueError, match="tier must be"):
            default_table(tier)

    def test_known_shared_surfaces_are_guard_disambiguated(self):
        # audit() lists surfaces claimed by more than one class; the shipped table has
        # exactly the approx pair, and `filigrana table` warns so both guards stay audited.
        problems = default_table("all").audit()
        assert problems, "if the approx pair stops sharing surfaces, update this pin"
        assert all(
            line.startswith(("about:", "approximately:", "around:")) for line in problems
        )

    def test_the_approx_pair_splits_on_numerals(self):
        # 'about' before a number is the approximating adverb; before 'a dozen' it is the
        # word-class reading; with no numeral after, neither class may fire at all.
        table = default_table("all")

        def one(text: str) -> list[str]:
            return [s.class_id for s in table.slots(text) if s.class_id.startswith("b.approx")]

        assert one("about 50 issues") == ["b.approx_num"]
        assert one("around 7 people") == ["b.approx_num"]
        assert one("about a dozen") == ["b.approx_word"]
        assert one("approximately zero progress") == ["b.approx_word"]
        assert one("we talked about it") == []

    def test_counts_are_consistent(self):
        table = default_table("all")
        assert table.class_count == len(table.classes)
        assert table.member_count == sum(len(c.members) for c in table.classes.values())


class TestContextGuards:
    """b.although = [although, though]: interchangeable mid-clause, not clause-final."""

    def _slots(self, text: str, class_id: str = "b.although"):
        table = default_table("all")
        return [s for s in table.slots(text) if s.class_id == class_id]

    def test_mid_clause_still_proposes(self):
        slots = self._slots("Although it rained, we walked to the station.")
        assert slots and slots[0].members == ("although", "though")
        assert slots[0].matched == ("Although",)

    def test_clause_final_is_refused(self):
        # 'I liked it though' is the adverb reading; 'although' cannot go there, so the
        # class must not fire where the two members would disagree.
        assert self._slots("We walked to the station, though.") == []

    def test_refuses_after_even(self):
        # 'even though' is a fixed connective with no 'even although' reading.
        assert self._slots("Even though it rained, we walked to the station.") == []
