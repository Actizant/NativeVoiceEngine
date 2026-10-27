"""Contract guards for defects found in the pre-commit architecture review.

Every node pins a behaviour that was demonstrated to be wrong, with the probe recorded in the
docstring. Nothing here establishes anything about recognition quality, capacity or production
readiness: these are refusal and agreement checks on the scoring and shape layers.

Run in the project interpreter:

    python -B -m pytest -o addopts="" -q tests/test_scorer_contract_guards.py

The reference-side nodes execute here; the torch-side nodes are skipped in this environment and are
the ones that run in the isolated CPU-torch verification environment.
"""

import math

import pytest

from nativevoice.core.errors import ConfigurationError
from nativevoice.modeling.config import StudentConfig
from nativevoice.modeling.lengths import TWO_STAGES_PAD_ONE
from nativevoice.training.errors import TdtContractError
from nativevoice.training.tdt_lattice import TdtSpec, loss

NO_TORCH = "torch is absent from the project environment"

BASE = dict(
    label="contract guard fixture", d_model=8, attention_heads=2, feed_forward_mult=2,
    conv_kernel_size=3, block_count=1, dropout=0.0, normalization="layer_norm",
    position_encoding="none", relative_window=0, macaron="single", conv_activation="swish",
    mel_bins=80, subsample=TWO_STAGES_PAD_ONE, vocab_size=5, blank_index=2, durations=(0, 1, 2),
    prediction_dim=8, prediction_kind="embedding_only", prediction_layers=0,
    prediction_start_symbol=0, joint_dim=8)

SPEC = TdtSpec(durations=(0, 1, 2))                                    # token slice width 3, Nd 3


def config(**overrides):
    return StudentConfig(**{**BASE, **overrides})


def columns(*vectors):
    """The prefix columns of one encoder position."""
    return [list(v) for v in vectors]


def lattice(*rows):
    """Batch-of-one logits: one row per encoder position, each row a list of prefix columns."""
    return [list(rows)]


def one_position_two_prefix():
    return lattice(columns([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.4, 0.1, 0.2, 0.0, 0.0, 0.0]))


# ------------------------------------------------------------------ 1. the spec cannot be inherited


def test_the_reference_refuses_a_call_that_omits_the_spec():
    """DEFECT: ``loss(..., spec=TdtSpec())`` inherited a duration alphabet and a blank id by omission."""
    with pytest.raises(TypeError):
        loss(one_position_two_prefix(), [[1]], [[1, 1]])               # type: ignore[call-arg]


def test_declared_vocabulary_disagreement_is_refused_by_name():
    """DEFECT: the vocabulary was inferred as ``width - Nd``, so a spec with a shorter duration
    alphabet silently re-labelled the model's duration slots as token ids and still returned a
    finite loss. Width 9 with Nd 3 implies 6 tokens; a model declaring 5 must be refused."""
    wide = lattice(columns([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                            [0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.0, 0.4, 0.1]))
    with pytest.raises(TdtContractError) as caught:
        loss(wide, [[1, 2]], [[1, 2]], TdtSpec(durations=(0, 1, 2), vocab_size=5))
    details = caught.value.details
    assert details["tensor_width"] == 9, details
    assert details["n_durations"] == 3, details
    assert details["declared_vocab_size"] == 5, details
    assert details["implied_vocab_size"] == 6, details
    assert "duration slice" in str(caught.value), str(caught.value)


def test_declared_vocabulary_agreement_changes_nothing():
    wide = lattice(columns([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                            [0.2, 0.2, 0.2, 0.1, 0.1, 0.3, 0.0, 0.4, 0.1]))
    agreed = loss(wide, [[1, 2]], [[1, 2]], TdtSpec(durations=(0, 1, 2), vocab_size=6))
    undeclared = loss(wide, [[1, 2]], [[1, 2]], TdtSpec(durations=(0, 1, 2)))
    assert agreed == undeclared, "declaring a vocabulary that matches the tensor must be inert"


def test_blank_index_is_refused_against_the_declared_vocabulary():
    with pytest.raises(TdtContractError):
        TdtSpec(durations=(0, 1, 2), blank_index=5, vocab_size=5)


# --------------------------------------------------------------------- 2. non-finite logit slices


def test_a_nan_logit_is_refused_rather_than_producing_nan():
    """DEFECT: a NaN in a slice yielded NaN log-probabilities and the lattice then reported an
    unreachable alignment - a diagnosis that sends the reader to the wrong layer."""
    bad = lattice(columns([float("nan"), 0.2, 0.3, 0.0, 0.0, 0.0], [0.4, 0.1, 0.2, 0.0, 0.0, 0.0]))
    with pytest.raises(TdtContractError) as caught:
        loss(bad, [[1]], [[1, 1]], SPEC)
    assert "NaN" in str(caught.value), str(caught.value)
    assert caught.value.details["slice"] == "token"


def test_a_slice_with_no_finite_logit_is_named_as_a_malformed_input():
    """DEFECT: an all ``-inf`` slice made ``log_softmax`` return NaN silently."""
    bad = lattice(columns([float("-inf")] * 6, [0.4, 0.1, 0.2, 0.0, 0.0, 0.0]))
    with pytest.raises(TdtContractError) as caught:
        loss(bad, [[1]], [[1, 1]], SPEC)
    assert "no finite logit" in str(caught.value), str(caught.value)
    assert caught.value.details == {"batch": 0, "row": 0, "col": 0, "slice": "token"}


def test_minus_infinity_in_an_otherwise_finite_slice_is_still_legitimate():
    """Control against over-refusal: ``-inf`` means a class of probability zero and must keep working."""
    zero_class = lattice(columns([float("-inf"), 0.2, 0.3, 0.1, 0.0, 0.2],
                                 [0.4, 0.1, 0.2, 0.0, 0.3, 0.1]))
    near_zero = lattice(columns([-800.0, 0.2, 0.3, 0.1, 0.0, 0.2],
                                [0.4, 0.1, 0.2, 0.0, 0.3, 0.1]))
    a = loss(zero_class, [[1]], [[1, 1]], SPEC)[0]
    b = loss(near_zero, [[1]], [[1, 1]], SPEC)[0]
    assert math.isfinite(a), a
    assert abs(a - b) < 1e-9, (a, b)


# ------------------------------------------------------------------------ 3. the torch transcription


def test_torch_refuses_the_same_malformed_slices_with_the_same_type():
    """The two implementations must refuse the same inputs with the same type - pinned here."""
    torch = pytest.importorskip("torch", reason=NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss

    good = torch.tensor([[[[0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.4, 0.1, 0.2, 0.0, 0.0, 0.0]]]],
                        dtype=torch.float64)
    ids = torch.tensor([[1]], dtype=torch.int64)
    lengths = torch.tensor([[1, 1]], dtype=torch.int64)
    tdt_loss(good, ids, lengths, SPEC)

    nan = good.clone(); nan[0, 0, 0, 0] = float("nan")
    with pytest.raises(TdtContractError):
        tdt_loss(nan, ids, lengths, SPEC)

    allbad = good.clone(); allbad[0, 0, 0, :] = float("-inf")
    with pytest.raises(TdtContractError) as caught:
        tdt_loss(allbad, ids, lengths, SPEC)
    assert "no finite logit" in str(caught.value), str(caught.value)

    wide = torch.full((1, 1, 2, 9), 0.1, dtype=torch.float64)
    with pytest.raises(TdtContractError) as mismatch:
        tdt_loss(wide, torch.tensor([[1]], dtype=torch.int64), lengths,
                 TdtSpec(durations=(0, 1, 2), vocab_size=99))
    assert mismatch.value.details["declared_vocab_size"] == 99


def test_torch_and_reference_agree_on_a_finite_batch():
    torch = pytest.importorskip("torch", reason=NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss

    rows = lattice(columns([0.3, -0.2, 0.1, 0.4, 0.0, -0.5], [0.1, 0.5, -0.3, 0.2, 0.6, 0.0]))
    spec = TdtSpec(durations=(0, 1, 2), vocab_size=3)
    reference = loss(rows, [[1]], [[1, 1]], spec)[0]
    value = float(tdt_loss(torch.tensor(rows, dtype=torch.float64),
                           torch.tensor([[1]], dtype=torch.int64),
                           torch.tensor([[1, 1]], dtype=torch.int64), spec)[0].detach())
    assert abs(reference - value) < 1e-9, (reference, value)


# ------------------------------------------------------------------- 4. the encoder batch broadcast


def test_the_encoder_refuses_one_target_row_broadcast_over_a_batch():
    """DEFECT, probed before the fix: a ``[4, 40, 80]`` grid with a ``[1, 3]`` target tensor returned
    logits identical to the four-transcript call, because broadcasting over a size-1 batch dimension
    is legal in torch. Nothing refused it, so every row was scored against the same transcript."""
    torch = pytest.importorskip("torch", reason=NO_TORCH)
    from nativevoice.modeling.encoder import build_student

    torch.manual_seed(0)
    student = build_student(config()).double()
    grid = torch.randn(4, 40, 80, dtype=torch.float64)
    lengths = [40, 40, 40, 40]
    full, _, _ = student(grid, torch.tensor([[1, 2, 3]] * 4, dtype=torch.int64), frame_lengths=lengths)
    assert tuple(full.shape) == (4, 10, 4, 8), tuple(full.shape)

    with pytest.raises(ConfigurationError) as caught:
        student(grid, torch.tensor([[1, 2, 3]], dtype=torch.int64), frame_lengths=lengths)
    assert caught.value.details["grid_rows"] == 4
    assert caught.value.details["target_shape"] == [1, 3]


def test_the_encoder_still_accepts_a_well_formed_batch():
    """Control against over-refusal on the same guard."""
    torch = pytest.importorskip("torch", reason=NO_TORCH)
    from nativevoice.modeling.encoder import build_student

    torch.manual_seed(0)
    student = build_student(config()).double()
    grid = torch.randn(2, 40, 80, dtype=torch.float64)
    logits, per_row, padded = student(grid, torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                                      frame_lengths=[40, 30])
    assert per_row == (10, 8), per_row
    assert padded == 10 and tuple(logits.shape) == (2, 10, 3, 8), (padded, tuple(logits.shape))


def test_forward_returns_the_two_length_facts_apart():
    """R-9a: the old contract returned one number, the padded axis, and a caller that read it as a
    row's ``T_b`` silently scored fabricated positions. The probe measured 10.957882927 for a short
    row declared at the padded length against 5.774457903 declared correctly, and nothing refused
    it. ``forward`` now returns the per-row tuple and the padded axis separately."""
    torch = pytest.importorskip("torch", reason=NO_TORCH)
    from nativevoice.modeling.encoder import build_student

    torch.manual_seed(3)
    student = build_student(config()).double()
    grid = torch.randn(2, 40, 80, dtype=torch.float64)
    logits, per_row, padded = student(grid, torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                                      frame_lengths=[40, 20])
    assert per_row == (10, 5), per_row
    assert padded == 10, padded
    assert tuple(logits.shape)[1] == padded
    spec = TdtSpec(durations=(0, 1, 2), vocab_size=5)
    wrong = loss([logits[1, :padded].tolist()], [[3, 4]], [[padded, 2]], spec)[0]
    right = loss([logits[1, :per_row[1]].tolist()], [[3, 4]], [[per_row[1], 2]], spec)[0]
    assert wrong != right, "the two lengths must not be interchangeable"
    assert pytest.approx(right, rel=1e-9) == 5.774457903, right


def test_both_scorers_refuse_an_empty_batch():
    """R-9c: an empty batch used to yield an empty result. The reference returned ``[]`` and the
    torch path returned a zero-length tensor whose ``.mean()`` is NaN - a NaN that would reach a
    trainer's gradient with nothing naming the batch."""
    spec = TdtSpec(durations=(0, 1, 2), vocab_size=3)
    with pytest.raises(TdtContractError) as caught:
        loss([], [], [], spec)
    assert "empty batch" in str(caught.value), str(caught.value)


def test_a_single_row_with_an_empty_target_still_scores():
    """Control for the same guard: ``B >= 1`` with ``U = 0`` is a legitimate utterance with no
    tokens, and must keep scoring."""
    spec = TdtSpec(durations=(0, 1, 2), vocab_size=3)
    rows = [[[0.1, 0.2, 0.3, 0.4, 0.0, 0.2], [0.4, 0.1, 0.2, 0.0, 0.3, 0.1]]]
    assert len(loss([rows], [[]], [[1, 0]], spec)) == 1


def test_torch_refuses_an_empty_batch_with_the_same_type():
    torch = pytest.importorskip("torch", reason=NO_TORCH)
    from nativevoice.training.tdt_loss import tdt_loss

    spec = TdtSpec(durations=(0, 1, 2), vocab_size=3)
    with pytest.raises(TdtContractError) as caught:
        tdt_loss(torch.zeros(0, 3, 3, 6, dtype=torch.float64),
                 torch.zeros(0, 2, dtype=torch.int64),
                 torch.zeros(0, 2, dtype=torch.int64), spec)
    assert "empty batch" in str(caught.value), str(caught.value)
    scored = tdt_loss(torch.zeros(1, 3, 1, 6, dtype=torch.float64),
                      torch.zeros(1, 0, dtype=torch.int64),
                      torch.tensor([[3, 0]], dtype=torch.int64), spec)
    assert int(scored.shape[0]) == 1, scored.shape
