"""The window planner: longertexts must not open a seam in the mark.

`plan_windows` is the only model-facing arithmetic in `filigrana` that has to be *exact* about
position arithmetic: a window covering the wrong rows would silently let gated slots score
themselves against no context at all, and nothing downstream would notice. These tests pin
the three invariants that make the composition correct, with no torch in sight.
"""

from filigrana.lm import WINDOW_CONTEXT, plan_windows


def _invariants(n: int, max_positions: int, context: int = WINDOW_CONTEXT):
    windows = plan_windows(n, max_positions, context)
    rows = []
    for start, end, own in windows:
        assert end - start <= max_positions, f"window wider than the position table: {start, end}"
        assert start < end, f"empty window in the plan for n={n}: {start, end}"
        assert own >= start, f"plans to score rows before its own input: {start, end, own}"
        # The rows the caller will slice: lp[own-start : end-1-start], i.e. scoring rows
        # own .. end-2, produced entirely inside this window's input.
        assert end - 1 - start >= own - start, f"slice crosses its own end: {start, end, own}"
        rows.extend(range(own, end - 1))
    assert rows == list(range(0, max(n - 1, 0))), (
        f"windows must tile scoring rows 0..{n - 2} disjointly and contiguously: {rows}"
    )
    if n > 1:
        assert windows[-1][1] == n, f"the last window must reach the end of the text: {windows}"
    return windows


def test_a_short_text_is_one_window():
    for n in (0, 1, 500, 1023):
        assert plan_windows(n, 1024) == [(0, n, 0)]


def test_exactly_the_position_table_is_one_window():
    assert plan_windows(1024, 1024) == [(0, 1024, 0)]


def test_one_past_the_table_needs_two_windows():
    windows = _invariants(1025, 1024)
    assert windows[0][:2] == (0, 1024)


def test_long_text_tiles_every_scoring_row():
    for n in (1025, 1900, 2048, 4001, 10_000):
        _invariants(n, 1024)


def test_later_windows_keep_real_left_context():
    """A scored row must have at least `context` rows before it inside its own window."""
    windows = plan_windows(5000, 1024)
    for (start, _end, own) in windows[1:]:
        assert own - start >= WINDOW_CONTEXT, (start, own)


def test_no_stub_tail_and_first_row_is_owned_by_window_one():
    """Seams sit at window boundaries; whichever way they round, the tiling must hold even
    when the text is a few rows longer than several whole windows."""
    for n in (1024 * 3 + 7, 1024 * 2 + WINDOW_CONTEXT // 2):
        _invariants(n, 1024)


def test_thin_context_still_tiles():
    _invariants(3000, 1024, context=8)  # restraint-by-construction, not only by default
