"""The declared-prefix target contract, shared by both implementations of the TDT loss.

A batch carries ``targets [B, U_max]`` and per-row ``lengths [B, 2]`` holding ``(T_b, U_b)``.
The declared ``U_b`` is the mask (decision O1): the lattice is bounded by it, so target values
beyond it belong to no hypothesis. Consequences this file pins down:

* ids beyond ``U_b`` are not validated as token content - a padded batch may hold blanks or
  out-of-vocabulary numbers there and both implementations accept it. Previously the torch path
  validated the whole tensor while the reference validated only the prefix, so the same padded
  data meant different things depending on which loss ran.
* the declared length may not exceed the available target width, and the target width may not
  exceed the joint output's prediction-axis capacity. Both refuse with ``TdtContractError``;
  before this change the torch path fell through to an uncaught ``IndexError``.
* ids *inside* the prefix are still validated, and still scored - the guard was scoped, not
  relaxed, and the negative controls here are the proof.
* the TDT mathematics and terminal semantics are unchanged. Since RD-1 the *declared* time length is
  authoritative in both implementations: ``T_b = 0`` is refused even when the tensor supplies rows,
  so neither program scores a zero-position utterance. Physical absence (a tensor with no rows) was
  already refused by both.

Deliberately out of scope, and unresolved (decision D-5): padding along the **frame** axis is not
inert for the Conformer skeleton - padded frames move encoder states at real positions, so this
file says nothing about the safety of variable-length batching through the full encoder. Only the
loss's treatment of target padding is covered here.
"""

from __future__ import annotations

import math

import pytest

from nativevoice.training.errors import TdtContractError
from nativevoice.training.tdt_lattice import TdtSpec, loss as ref_loss

D = (0, 1, 2)
U_MAX = 3
T_MAX = 5
AXIS = U_MAX + 1                     # prediction columns available in the joint output
VOCAB = AXIS                         # token slice: ids 0..VOCAB-1, 0 is the blank slot
W = VOCAB + len(D)                   # token slice first, duration slice last

_NO_TORCH = ("NOT RUN: torch is absent from this environment; these checks run only in the "
             "isolated CPU-torch verification environment installed outside this repository")

SPEC = TdtSpec(durations=D)

#: Distinct per-slot values. A uniform lattice would make every token id equally likely, and a
#: control that changes a real id would then "prove" nothing - the assertion would pass for the
#: wrong reason.
SLOT = [-0.5 + 0.25 * c for c in range(W)]


def rows(T, B=1):
    """``B`` utterances of logits: ``T`` rows x ``AXIS`` columns x ``W`` distinct slots."""
    return [[[list(SLOT) for _ in range(AXIS)] for _ in range(T)] for _ in range(B)]


def score_reference(T, U, ids, tensor_rows=T_MAX):
    """Declared length ``T`` against a tensor that may hold more rows than that."""
    return ref_loss(rows(tensor_rows), [ids], [[T, U]], SPEC)[0]


@pytest.fixture(scope="module")
def torch_env():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss
    return torch, tdt_loss


def score_torch(torch, tdt_loss, T, U, ids, B=1, tensor_rows=T_MAX):
    logits = torch.tensor(rows(tensor_rows, B), dtype=torch.float64)
    targets = torch.tensor(ids, dtype=torch.int64)
    lengths = torch.tensor([[T, U]] * B)
    return [float(v) for v in tdt_loss(logits, targets, lengths, SPEC)]


# ------------------------------------------------------------------ padding is unread

@pytest.mark.parametrize("padding", [[1, 2, 3], [1, 3, 2], [1, 1, 1], [1, 0, 0], [1, 99, 7]])
def test_reference_ignores_everything_beyond_the_declared_length(padding):
    """Values past ``U_b`` change nothing, including ids that would be illegal as tokens."""
    honest = score_reference(T_MAX, 1, [1])
    assert score_reference(T_MAX, 1, padding) == honest, padding


@pytest.mark.parametrize("padding", [[1, 2, 3], [1, 3, 2], [1, 1, 1], [1, 0, 0], [1, 99, 7]])
def test_torch_ignores_everything_beyond_the_declared_length(torch_env, padding):
    torch, tdt_loss = torch_env
    honest = score_torch(torch, tdt_loss, T_MAX, 1, [[1]])[0]
    padded = score_torch(torch, tdt_loss, T_MAX, 1, [padding])[0]
    assert padded == honest, padding


def test_blank_and_out_of_vocabulary_ids_in_padding_are_accepted_by_both(torch_env):
    """The divergence this change removes: torch used to refuse, the reference used to accept."""
    torch, tdt_loss = torch_env
    for ids in ([1, 0, 0], [1, 99, 99]):
        reference = score_reference(T_MAX, 1, ids)
        got = score_torch(torch, tdt_loss, T_MAX, 1, [ids])[0]
        assert got == pytest.approx(reference, rel=1e-12, abs=1e-12), ids


# ------------------------------------------------------------------ the prefix is still live

@pytest.mark.parametrize("ids", [[0], [0, 1], [7], [2, 7], [3, 0]])
def test_illegal_ids_inside_the_declared_prefix_are_refused_by_reference(ids):
    with pytest.raises(TdtContractError):
        score_reference(T_MAX, len(ids), ids)


@pytest.mark.parametrize("ids", [[0], [0, 1], [7], [2, 7], [3, 0]])
def test_illegal_ids_inside_the_declared_prefix_are_refused_by_torch(torch_env, ids):
    torch, tdt_loss = torch_env
    with pytest.raises(TdtContractError):
        score_torch(torch, tdt_loss, T_MAX, len(ids), [ids])


def test_prefix_tokens_are_scored_not_merely_validated():
    """Negative control for the scoping: a legal id inside the prefix must change the answer."""
    a = score_reference(T_MAX, 1, [1])
    b = score_reference(T_MAX, 1, [2])
    assert a != b, "the declared prefix stopped being read"
    assert math.isfinite(a) and math.isfinite(b)
    longer = score_reference(T_MAX, 2, [1, 2])
    assert longer != a, "the declared length stopped bounding the lattice"


# ------------------------------------------------------------------ dimension guards

@pytest.mark.parametrize("T,U,ids", [
    (T_MAX, 2, [1]),                       # declared U beyond the supplied width
    (T_MAX, 4, [1, 2, 3]),                 # declared U beyond width and beyond the axis
])
def test_declared_length_beyond_the_available_target_width_is_refused_by_reference(T, U, ids):
    with pytest.raises(TdtContractError) as err:
        ref_loss(rows(T), [ids], [[T, U]], SPEC)
    assert err.value.details["U"] == U


def test_declared_length_beyond_the_available_target_width_is_refused_by_torch(torch_env):
    """This case used to escape as a bare ``IndexError``; it is a contract breach now."""
    torch, tdt_loss = torch_env
    logits = torch.full((1, T_MAX, AXIS, W), -0.5, dtype=torch.float64)
    with pytest.raises(TdtContractError) as err:
        tdt_loss(logits, torch.tensor([[1]], dtype=torch.int64), torch.tensor([[T_MAX, 3]]), SPEC)
    assert "target width" in str(err.value)


def test_targets_wider_than_the_prediction_axis_are_refused_by_reference():
    ids = [1, 2, 3, 4]                                  # width 4 > capacity AXIS-1 == 3
    with pytest.raises(TdtContractError) as err:
        ref_loss(rows(T_MAX), [ids], [[T_MAX, 3]], SPEC)
    assert "prediction axis" in str(err.value)


def test_targets_wider_than_the_prediction_axis_are_refused_by_torch(torch_env):
    torch, tdt_loss = torch_env
    logits = torch.full((1, T_MAX, AXIS, W), -0.5, dtype=torch.float64)
    with pytest.raises(TdtContractError) as err:
        tdt_loss(logits, torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
                 torch.tensor([[T_MAX, 3]]), SPEC)
    assert "targets must be [B, U]" in str(err.value)


def test_the_widest_legal_target_tensor_is_still_accepted(torch_env):
    """Non-vacuity for the axis rule: capacity itself must be usable, not just below it."""
    torch, tdt_loss = torch_env
    reference = score_reference(T_MAX, U_MAX, [1, 2, 3])
    got = score_torch(torch, tdt_loss, T_MAX, U_MAX, [[1, 2, 3]])[0]
    assert got == pytest.approx(reference, rel=1e-12, abs=1e-12)


# ------------------------------------------------------------------ batches and empty hypotheses

BATCH_ROWS = [(T_MAX, U_MAX, [1, 2, 3]), (2, 1, [1, 0, 0]), (4, 2, [2, 3, 7]), (T_MAX, 0, [1, 2, 3])]


@pytest.mark.parametrize("T,U,ids", BATCH_ROWS)
def test_each_row_of_a_batch_scores_as_that_row_alone(torch_env, T, U, ids):
    """Unequal hypotheses in one batch, and per-row equivalence with the reference.

    Row 2's padding holds a blank and row 3's holds an out-of-vocabulary id: both are unread, so
    they must not affect the score. Row 4 declares an empty hypothesis, which is legal because
    ``T >= 1``.
    """
    torch, tdt_loss = torch_env
    logits = torch.tensor(rows(T_MAX, len(BATCH_ROWS)), dtype=torch.float64)
    targets = torch.tensor([r[2] for r in BATCH_ROWS], dtype=torch.int64)
    lengths = torch.tensor([[r[0], r[1]] for r in BATCH_ROWS])
    batch = [float(v) for v in tdt_loss(logits, targets, lengths, SPEC)]
    alone = score_torch(torch, tdt_loss, T, U, [ids])[0]
    reference = score_reference(T, U, ids)
    index = [i for i, r in enumerate(BATCH_ROWS) if r == (T, U, ids)][0]
    assert batch[index] == alone, (T, U, ids)
    assert batch[index] == pytest.approx(reference, rel=1e-12, abs=1e-12), (T, U, ids)


def test_empty_hypothesis_is_legal_when_positions_exist():
    """``U_b = 0`` is a pure blank chain and scores; ``T_b = 0`` stays refused (unchanged)."""
    value = score_reference(T_MAX, 0, [])
    assert math.isfinite(value) and value > 0.0
    with pytest.raises(TdtContractError):
        ref_loss(rows(1), [[]], [[0, 0]], SPEC)


def test_structural_guards_survive_the_relaxed_width_rule(torch_env):
    """Batch agreement and dtype are still checked for the whole tensor, prefix or not."""
    torch, tdt_loss = torch_env
    logits = torch.full((2, T_MAX, AXIS, W), -0.5, dtype=torch.float64)
    lengths = torch.tensor([[T_MAX, 3], [T_MAX, 3]])
    with pytest.raises(TdtContractError):
        tdt_loss(logits, torch.tensor([[1, 2, 3]], dtype=torch.int64), lengths, SPEC)   # 1 vs 2
    with pytest.raises(TdtContractError):
        tdt_loss(logits, torch.tensor([[1, 2, 3], [1, 2, 3]]).float(), lengths, SPEC)   # not integer


def test_error_types_are_shared_for_every_refused_target_shape(torch_env):
    """A single table, both implementations, one error type - the parity claim in one place."""
    torch, tdt_loss = torch_env
    parity_table = [
        ("declared U wider than the tensor", T_MAX, 3, [1, 2]),
        ("tensor wider than the axis", T_MAX, 3, [1, 2, 3, 4]),
        ("blank inside the prefix", T_MAX, 2, [0, 1]),
        ("out of vocabulary inside the prefix", T_MAX, 2, [1, 9]),
    ]
    for label, T, U, ids in parity_table:
        with pytest.raises(TdtContractError):
            tdt_loss(torch.tensor(rows(T_MAX), dtype=torch.float64),
                     torch.tensor([ids], dtype=torch.int64), torch.tensor([[T, U]]), SPEC)
        with pytest.raises(TdtContractError):
            ref_loss(rows(T_MAX), [ids], [[T, U]], SPEC)


def test_declared_zero_time_length_is_refused_by_the_reference_even_with_rows_present():
    """RD-1 on the reference side, and it holds without torch so it runs everywhere."""
    with pytest.raises(TdtContractError) as err:
        ref_loss(rows(T_MAX), [[1, 2, 3]], [[0, 1]], SPEC)
    assert "at least one encoder position" in str(err.value)
    assert err.value.details["T"] == 0


def test_declared_one_time_length_is_scored_by_the_reference():
    """The valid neighbour of the case above: the refusal is about ``T_b``, not about small ``T``."""
    value = score_reference(T=1, U=1, ids=[1])
    assert math.isfinite(value) and value > 0.0


def test_physically_empty_logits_is_refused_by_the_reference():
    with pytest.raises(TdtContractError) as err:
        ref_loss([[]], [[]], [[0, 0]], SPEC)
    assert err.value.details["T"] == 0


def test_declared_zero_time_length_is_refused_by_torch_even_with_rows_present(torch_env):
    """The divergence RD-1 closes: torch used to score this because its guard read the tensor."""
    torch, tdt_loss = torch_env
    logits = torch.tensor(rows(T_MAX), dtype=torch.float64)
    with pytest.raises(TdtContractError) as err:
        tdt_loss(logits, torch.tensor([[1, 2, 3]], dtype=torch.int64), torch.tensor([[0, 1]]), SPEC)
    assert "at least one encoder position" in str(err.value)
    assert err.value.details["T"] == 0 and err.value.details["batch"] == 0


def test_declared_one_time_length_is_scored_by_torch_and_matches_the_reference(torch_env):
    torch, tdt_loss = torch_env
    reference = score_reference(T=1, U=1, ids=[1])
    got = score_torch(torch, tdt_loss, 1, 1, [[1]])[0]
    assert got == pytest.approx(reference, rel=1e-12, abs=1e-12), (got, reference)


def test_physically_empty_logits_is_refused_by_torch(torch_env):
    torch, tdt_loss = torch_env
    empty = torch.zeros((1, 0, AXIS, W), dtype=torch.float64)
    with pytest.raises(TdtContractError):
        tdt_loss(empty, torch.zeros((1, 1), dtype=torch.int64), torch.tensor([[0, 1]]), SPEC)


def test_mixed_batch_with_one_invalid_declared_time_length_is_refused_by_both(torch_env):
    """One bad row poisons the batch, in both programs, with the same error type and the same
    offending row - the guarantee a caller needs before trusting ``lengths`` at all."""
    torch, tdt_loss = torch_env
    logits = torch.tensor(rows(T_MAX, 2), dtype=torch.float64)
    targets = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int64)
    lengths = torch.tensor([[T_MAX, 3], [0, 1]])
    with pytest.raises(TdtContractError) as err:
        tdt_loss(logits, targets, lengths, SPEC)
    assert err.value.details["batch"] == 1 and err.value.details["T"] == 0
    with pytest.raises(TdtContractError) as err2:
        ref_loss(rows(T_MAX, 2), [[1, 2, 3], [1]], [[T_MAX, 3], [0, 1]], SPEC)
    assert err2.value.details["batch"] == 1 and err2.value.details["T"] == 0


def test_mixed_batch_of_valid_rows_still_scores_row_by_row(torch_env):
    """Non-vacuity for the guard above: the same batch with legal ``T_b`` values scores normally."""
    torch, tdt_loss = torch_env
    logits = torch.tensor(rows(T_MAX, 2), dtype=torch.float64)
    targets = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int64)
    batch = [float(v) for v in tdt_loss(logits, targets, torch.tensor([[T_MAX, 3], [2, 1]]), SPEC)]
    alone_long = score_torch(torch, tdt_loss, T_MAX, 3, [[1, 2, 3]])[0]
    alone_short = score_torch(torch, tdt_loss, 2, 1, [[1, 2, 3]])[0]
    assert batch == [alone_long, alone_short]
    assert batch[0] == pytest.approx(score_reference(T_MAX, 3, [1, 2, 3]), rel=1e-12, abs=1e-12)
    assert batch[1] == pytest.approx(score_reference(2, 1, [1, 2, 3]), rel=1e-12, abs=1e-12)
