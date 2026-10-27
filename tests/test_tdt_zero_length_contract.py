"""The ``T == 0`` contract, enforced identically by both implementations of the objective.

Before 2026-10-26 the stdlib reference and the torch transcription disagreed about an encoder
that produced no positions. The reference returned ``-0.0`` for an empty target, raised
``TdtAlignmentImpossible`` when the terminal cell was simply unreachable, and raised an uncaught
``IndexError`` in exactly one case - ``0 in D`` with ``U == 1`` - because the sentinel rule makes
every intermediate cell inadmissible at ``T == 0``, so ``(0, 1)`` was the only cell that could
reach for a row ``validate`` had never required to exist. The torch path refused all of it.

These tests pin the agreed behaviour: ``T == 0`` is a caller error, always, in both programs.
They also carry the negative controls, because a guard that refuses everything passes just as
quietly as one that refuses the right thing.
"""

from __future__ import annotations

import math

import pytest

from nativevoice.training.errors import TdtContractError
from nativevoice.training.tdt_lattice import TdtSpec, is_feasible, loss

ZERO_CASES = [(D, U) for D in ((1,), (0, 1, 2)) for U in (0, 1, 2)]
ZERO_IDS = [f"D{''.join(map(str, D))}U{U}" for D, U in ZERO_CASES]

_NO_TORCH = ("NOT RUN: torch is absent from this environment; these checks run only in the "
             "isolated CPU-torch verification environment installed outside this repository")


def rows_for(T, U, D, value=-0.5):
    width = (U + 1) + len(D)
    return [[[value] * width for _ in range(U + 1)] for _ in range(T)]


def targets_for(U):
    return [[j + 1 for j in range(U)]]


@pytest.mark.parametrize("D,U", ZERO_CASES, ids=ZERO_IDS)
def test_reference_refuses_zero_encoder_positions(D, U):
    with pytest.raises(TdtContractError) as err:
        loss([rows_for(0, U, D)], targets_for(U), [[0, U]], TdtSpec(durations=D))
    assert err.value.details["T"] == 0 and err.value.details["U"] == U
    assert err.value.details["batch"] == 0


@pytest.mark.parametrize("D,U", ZERO_CASES, ids=ZERO_IDS)
def test_torch_refuses_the_same_inputs_the_same_way(D, U):
    """The parity claim: one contract, two programs, no case where they diverge."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    logits = torch.zeros((1, 0, U + 1, (U + 1) + len(D)), dtype=torch.float64)
    targets = torch.zeros((1, U), dtype=torch.int64)
    with pytest.raises(TdtContractError) as err:
        tdt_loss(logits, targets, torch.tensor([[0, U]]), TdtSpec(durations=D))
    assert "encoder row" in str(err.value) or "at least one" in str(err.value)


def test_refusal_is_about_the_declared_length_not_the_tensor_shape():
    """A caller that supplies a row but still declares T = 0 is refused too.

    Otherwise the contract would depend on how empty the tensor happens to look, and the two
    implementations would drift apart again on the first caller that padded.
    """
    D, U = (0, 1, 2), 1
    with pytest.raises(TdtContractError):
        loss([rows_for(1, U, D)], targets_for(U), [[0, U]], TdtSpec(durations=D))


def test_refusal_message_names_the_offending_input():
    D, U = (0, 1), 0
    with pytest.raises(TdtContractError) as err:
        loss([rows_for(0, U, D)], targets_for(U), [[0, U]], TdtSpec(durations=D))
    text = str(err.value)
    assert "at least one encoder position" in text
    assert "T=0" in text and "rows=0" in text


def test_the_row_shortage_guard_still_fires_after_the_new_clause():
    """The removed `T > 0` exemption must not have taken the real row check with it."""
    D, U = (0, 1, 2), 1
    with pytest.raises(TdtContractError) as err:
        loss([rows_for(2, U, D)], targets_for(U), [[3, U]], TdtSpec(durations=D))
    assert "fewer logit rows" in str(err.value)


def test_one_encoder_position_is_still_a_valid_lattice():
    """Non-vacuity: the smallest legal lattice scores, and the refusal is specific to T = 0."""
    for D in ((1,), (0, 1), (0, 1, 2)):
        value = loss([rows_for(1, 0, D)], [[]], [[1, 0]], TdtSpec(durations=D))[0]
        assert math.isfinite(value), D
        scored = loss([rows_for(1, 1, D)], [[1]], [[1, 1]], TdtSpec(durations=D))[0]
        assert math.isfinite(scored) and scored > 0.0, (D, scored)


def test_two_positions_still_score_and_are_not_empty_lattices():
    D = (0, 1, 2)
    one = loss([rows_for(2, 1, D)], [[1]], [[2, 1]], TdtSpec(durations=D))[0]
    assert math.isfinite(one) and one > 0.0


def test_is_feasible_stays_a_geometry_predicate_that_can_disagree_on_purpose():
    """``is_feasible(0, 0)`` is True while ``validate`` refuses ``T == 0``.

    That is the documented, deliberate boundary between a reachability question and a well-formed
    input question. If this test ever fails because someone "aligned" the two, the predicate has
    stopped meaning what its docstring says.
    """
    assert is_feasible(0, 0, TdtSpec(durations=(1,))) is True
    with pytest.raises(TdtContractError):
        loss([rows_for(0, 0, (1,))], [[]], [[0, 0]], TdtSpec(durations=(1,)))
    assert is_feasible(0, 1, TdtSpec(durations=(1,))) is False
    assert is_feasible(1, 1, TdtSpec(durations=(1,))) is True


def test_torch_and_reference_agree_on_the_smallest_legal_lattice():
    """The parity claim for inputs that are *not* refusals, so the guards above are not vacuous."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    D, U, T = (0, 1, 2), 2, 3
    rows = rows_for(T, U, D)
    reference = loss([rows], targets_for(U), [[T, U]], TdtSpec(durations=D))[0]
    logits = torch.tensor([rows], dtype=torch.float64)
    got = float(tdt_loss(logits, torch.tensor([targets_for(U)[0]], dtype=torch.int64),
                         torch.tensor([[T, U]]), TdtSpec(durations=D))[0])
    assert got == pytest.approx(reference, rel=1e-12, abs=1e-12), (got, reference)
