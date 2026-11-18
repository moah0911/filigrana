"""`LocalModelScorer` — real-model tests, self-skipping where no model is cached.

Everything here needs `inkmark[lm]` *and* a model already in the local Hugging Face cache,
because a unit suite that downloads 500 MB is a test in name only. On a workstation with
gpt2 cached they pin the windowing arithmetic against a direct single pass; in CI they skip
and the pure `plan_windows` suite carries the invariants instead.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

SCORER_ARGS = {"model_id": "gpt2", "device": "cpu", "dtype": "float32"}


def _gpt2_cached_or_skip() -> None:
    roots = [
        root
        for root in (
            os.environ.get("HF_HOME", ""),
            os.environ.get("XDG_CACHE_HOME", ""),
            os.path.expanduser("~/.cache/huggingface"),
        )
        if root
    ]
    if not any((Path(root) / "hub" / "models--gpt2").exists() for root in roots):
        pytest.skip("gpt2 is not in the local Hugging Face cache")


_gpt2_cached_or_skip()


def test_short_text_matches_a_plain_single_pass():
    """For a text inside the position table the composed pass must be the same numbers a
    direct log_softmax over the whole input would give -- windowing may not change any
    answer for the lengths that never needed it."""
    import torch

    from inkmark.canon import mask_markup
    from inkmark.lm import LocalModelScorer

    scorer = LocalModelScorer(**SCORER_ARGS)
    text = "the engineers walked slowly through the crowded square"
    scoring = scorer.score(text)
    masked = mask_markup(text)
    enc = scorer.tokenizer(masked, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        direct = torch.log_softmax(scorer.model(input_ids=enc["input_ids"]).logits[0], dim=-1)
    owner = [-1] * (len(masked) + 1)
    for pos, (start, end) in enumerate(offsets):
        for char in range(start, max(end, start + 1)):
            if owner[char] == -1:
                owner[char] = pos
    for token in scoring.tokens:
        row = owner[token.start]
        if row <= 0:
            continue
        is_spaced = token.start == 0 or masked[token.start - 1].isspace()
        for member in ("walked", "strolled"):
            tid = scorer.token_id(member, spaced=is_spaced)
            if tid is None:
                continue
            pass_lp = float(scoring.log_probs[row - 1][tid])
            assert pass_lp == pytest.approx(float(direct[row - 1][tid]), abs=1e-6), (
                member,
                token.surface,
                pass_lp,
                float(direct[row - 1][tid]),
            )


def test_a_text_longer_than_the_position_table_scores_every_row_without_error():
    """The 1024-cap failure mode: a long article used to raise IndexError inside the model's
    position embeddings. It must instead return a pass shaped exactly like one full pass,
    with a late word still scoring through some window."""
    from inkmark.canon import mask_markup, tokenize
    from inkmark.lm import LocalModelScorer

    scorer = LocalModelScorer(**SCORER_ARGS)
    text = " ".join(
        "the engineers walked slowly through the crowded square and the grey dusk settled"
        for _ in range(120)
    )
    scoring = scorer.score(text)
    ids = scorer.tokenizer(mask_markup(text), add_special_tokens=False)["input_ids"]
    assert len(ids) > scorer.max_positions, "the fixture must actually cross the boundary"
    assert scoring.log_probs.shape[0] == len(ids) - 1

    index = len(tokenize(text)) - 10
    late = scoring.logprob_of(index, ("walked", "strolled"))
    assert any(math.isfinite(x) for x in late), (tokenize(text)[index].surface, late)


def test_logprob_of_the_first_word_is_impossible_everywhere():
    """The first word has no predicting row (the BOS piece owns row 0), so every member is
    -inf there. The gate must read that as 'no information', never as 'perfect evidence'."""
    from inkmark.lm import LocalModelScorer

    scorer = LocalModelScorer(**SCORER_ARGS)
    pas = scorer.score("she walked slowly toward the grey station and the crowd parted")
    assert pas.logprob_of(0, ("walked", "strolled")) == (-math.inf, -math.inf)


def test_mean_logprob_is_zero_for_a_fragment_and_negative_for_prose():
    from inkmark.lm import LocalModelScorer

    scorer = LocalModelScorer(**SCORER_ARGS)
    assert scorer.score("a").mean_logprob() == 0.0  # under two BPE pieces: nothing to condition on
    prose = scorer.score("she walked slowly toward the grey station and the crowd parted")
    value = prose.mean_logprob()
    assert math.isfinite(value) and value < 0


def test_propose_filters_a_gate_required_provider_through_a_real_pass():
    """WordNet proposals are sense-blind; `lm.propose` is the seam where a real model's
    band test turns them into a report. Whatever it keeps or drops, it must report."""
    from inkmark.gate import FreeChoiceGate
    from inkmark.lexical import WordNetProvider
    from inkmark.lm import LocalModelScorer, propose

    scorer = LocalModelScorer(**SCORER_ARGS)
    text = "she carried the weight of the whole argument"
    pas = scorer.score(text)
    report = propose(WordNetProvider(), text, pas, FreeChoiceGate())
    assert report.summary()
