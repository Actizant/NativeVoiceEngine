"""Shape, contract and gradient-reachability tests for the student skeleton.

The config half runs anywhere. The encoder half needs ``torch``, which is absent from the
project environment, so those nodes report NOT RUN here and execute in the isolated CPU-torch
verification environment installed outside this repository.

Scope discipline, stated because it is the point: these tests assert tensor shapes, finite
losses, gradient reachability, refusals and the *materiality* of the architectural choices that
are still open. None of them asserts anything about speech. A randomly initialised encoder that
produces a finite loss is arithmetic, not recognition, and nothing here should be read as
capacity approval, accuracy, or a validated final architecture.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import subprocess
import sys

import pytest

from nativevoice.core.errors import ConfigurationError
from nativevoice.modeling.config import (
    CONV_ACTIVATIONS,
    MACARON_MODES,
    NORMALIZATIONS,
    OPEN_ARCHITECTURE_DECISIONS,
    PREDICTION_KINDS,
    POSITION_ENCODINGS,
    StudentConfig,
)
from nativevoice.modeling.encoder import ShapeContractError, build_student
from nativevoice.modeling.lengths import (
    ONE_STAGE_NO_PADDING,
    ONE_STAGE_PAD_ONE,
    SUBSAMPLE_VARIANTS,
    TWO_STAGES_NO_PADDING,
    TWO_STAGES_PAD_ONE,
    LengthSeamError,
    StackSpec,
    bridge,
    frequency_bins,
)
from nativevoice.training.errors import TdtContractError
from nativevoice.training.tdt_lattice import TdtSpec, loss

_NO_TORCH = ("NOT RUN: torch is absent from this environment; the encoder skeleton runs only in "
             "the isolated CPU-torch verification environment installed outside this repository")

FIXTURE_LABEL = "test fixture - not a capacity recommendation"
VARIANT_IDS = ["1stage-pad0", "1stage-pad1", "2stage-pad0", "2stage-pad1"]

FIXTURE = dict(
    label=FIXTURE_LABEL,
    d_model=8,
    attention_heads=2,
    feed_forward_mult=2,
    conv_kernel_size=3,
    block_count=2,
    dropout=0.0,
    normalization="layer_norm",
    position_encoding="relative_shaw_windowed",
    relative_window=3,
    macaron="macaron",
    conv_activation="glu",
    mel_bins=80,
    subsample=TWO_STAGES_NO_PADDING,
    vocab_size=5,
    blank_index=0,
    durations=(0, 1, 2),
    prediction_dim=8,
    prediction_kind="embedding_only",
    prediction_layers=0,
    prediction_start_symbol=4,
    joint_dim=8,
)


def fixture_config(**overrides):
    """A deliberately tiny configuration. Its label says what it is: a fixture, never a proposal."""
    return StudentConfig(**{**FIXTURE, **overrides})


def build_double(config):
    """Build the skeleton in float64, the dtype this repository's reference arithmetic uses.

    Feeding a float64 grid into float32 parameters is a genuine type error in torch rather than a
    silent cast, so the fixture states its dtype instead of letting the test discover it.
    """
    return build_student(config).double()


def full_row(grid):
    """``frame_lengths`` for these fixtures: one row, and every frame of it is audio.

    Step 3B made the declaration required, so the pre-existing callers state it rather than
    relying on an inferred default. Because nothing here is padded, the declared length equals the
    axis and every mask is inert - that is the compatibility these nodes still assert, and it is why
    they pass unchanged. Padded batches are ``tests/test_modeling_encoder_masks.py``.
    """
    return (int(grid.shape[1]),)


# --------------------------------------------------------------------- 1. config: no defaults

def test_config_refuses_every_attempt_to_omit_a_field():
    """No field may have a default, so no architecture can be produced by forgetting one."""
    missing = [f.name for f in dataclasses.fields(StudentConfig)
               if f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING]
    assert missing == [], missing
    with pytest.raises(TypeError):
        StudentConfig()                                             # type: ignore[call-arg]
    partial = {k: v for k, v in FIXTURE.items() if k != "d_model"}
    with pytest.raises(TypeError):
        StudentConfig(**partial)                                    # type: ignore[arg-type]


def test_config_label_must_say_what_the_configuration_is():
    for bad in ("", "   ", 7, None):
        with pytest.raises(ConfigurationError):
            fixture_config(label=bad)                               # type: ignore[arg-type]


@pytest.mark.parametrize("field,value", [
    ("d_model", 0), ("attention_heads", 0), ("attention_heads", 3),      # 8 % 3 != 0
    ("feed_forward_mult", 0), ("conv_kernel_size", 4), ("conv_kernel_size", 0),
    ("block_count", 0), ("mel_bins", 0), ("joint_dim", 0), ("prediction_dim", -1),
    ("vocab_size", 0), ("blank_index", -1), ("blank_index", 5),          # == vocab_size
    ("prediction_start_symbol", 9), ("prediction_start_symbol", -1),
    ("dropout", -0.1), ("dropout", 1.0), ("dropout", "0.1"),
    ("relative_window", -1), ("subsample", "two stages"), ("subsample", 3),
    ("normalization", "instance_norm"), ("position_encoding", "rotary"),
    ("macaron", "half"), ("conv_activation", "relu"), ("prediction_kind", "transformer"),
    ("prediction_layers", 2),                                            # with embedding_only
    ("durations", ()), ("durations", (0, 0, 1)), ("durations", (2, 0)), ("durations", (0, 1.5)),
    ("durations", (0, 0)), ("durations", (True, 1)),
])
def test_config_rejects_values_it_cannot_honestely_build(field, value):
    with pytest.raises(ConfigurationError):
        fixture_config(**{field: value})                            # type: ignore[arg-type]


def test_config_accepts_the_other_side_of_each_boundary():
    cfg = fixture_config(blank_index=4, prediction_start_symbol=0, dropout=0.0,
                         attention_heads=8, conv_kernel_size=5, durations=(1,),
                         prediction_kind="lstm", prediction_layers=1)
    assert cfg.n_durations == 1 and cfg.logit_width == 6
    assert cfg.d_model % cfg.attention_heads == 0


def test_logit_width_is_token_slice_then_duration_slice():
    """The head's slot order is the loss's contract, not a local preference."""
    cfg = fixture_config()
    assert cfg.logit_width == cfg.vocab_size + cfg.n_durations
    assert cfg.logit_width - cfg.n_durations == cfg.vocab_size      # the split the loss performs


def test_candidates_stay_enumerable_and_the_open_list_stays_long():
    for group in (NORMALIZATIONS, POSITION_ENCODINGS, MACARON_MODES, CONV_ACTIVATIONS, PREDICTION_KINDS):
        assert len(group) >= 2, group
    joined = " ".join(OPEN_ARCHITECTURE_DECISIONS)
    for still_open in ("positional information", "normalisation placement", "capacity",
                       "vocabulary size", "duration set", "A1", "macaron"):
        assert still_open in joined, still_open


def test_importing_the_package_needs_no_torch_runtime():
    """The audit rule in executable form: no module-level heavy import under src/.

    Run in a fresh interpreter so another test's torch import cannot mask a regression.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    probe = ("import sys, nativevoice.modeling;"
             "assert 'torch' not in sys.modules, 'package import pulled torch in';"
             "assert not hasattr(nativevoice.modeling, 'build_student')")
    done = subprocess.run([sys.executable, "-c", probe], cwd=str(root), env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


# --------------------------------------------------------------------- 2. encoder: shapes

@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_encoder_position_count_matches_the_bridge_for_every_variant(variant):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build_double(fixture_config(subsample=variant))
    grid = torch.zeros((1, 98, 80), dtype=torch.float64)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    _, _, positions = student(grid, ids, frame_lengths=full_row(grid))
    expected = bridge(16000, frontend=_StubFrontend(), variant=variant).positions
    assert positions == expected, (variant.label, positions, expected)
    assert int(grid.shape[1]) == 98 and frequency_bins(80, variant=variant) > 0


class _StubFrontend:
    sample_rate = 16000
    hop_samples = 160
    window_samples = 400

    def frame_count(self, n_samples):
        return 0 if n_samples < 400 else 1 + (n_samples - 400) // 160


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_emission_logits_have_the_slot_layout_the_loss_consumes(variant):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    cfg = fixture_config(subsample=variant)
    student = build_double(cfg)
    grid = torch.randn((1, 98, 80), dtype=torch.float64)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    logits, _, positions = student(grid, ids, frame_lengths=full_row(grid))
    assert tuple(logits.shape) == (1, positions, ids.shape[1] + 1, cfg.logit_width), tuple(logits.shape)
    scored = loss([logits[0].tolist()], [[1, 2]], [[positions, 2]], TdtSpec(durations=cfg.durations))
    assert scored[0] == scored[0] and scored[0] > 0.0, scored


def test_random_initialisation_gives_a_finite_loss_and_reaches_every_layer():
    """Gradient reachability, not training: the graph must connect grid to loss to weights."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    cfg = fixture_config()
    student = build_double(cfg)
    grid = torch.randn((1, 98, 80), dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    logits, _, positions = student(grid, ids, frame_lengths=full_row(grid))
    value = loss([logits[0].tolist()], [[1, 2]], [[positions, 2]],
                 TdtSpec(durations=cfg.durations))
    assert bool(torch.isfinite(logits).all()), "a random initialisation produced a non-finite logit"
    logits.sum().backward()
    assert grid.grad is not None and float(grid.grad.abs().sum()) > 0.0, "no gradient reached the grid"
    embed_grad = student.predictor.embed.weight.grad
    assert embed_grad is not None and float(embed_grad.abs().sum()) > 0.0, "the predictor is dead"
    seen = [name for name, p in student.named_parameters() if p.grad is not None]
    assert seen, "no parameter received a gradient"


def test_a_wrong_head_width_is_refused_by_the_loss_not_by_the_model():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    cfg = fixture_config()
    student = build_double(cfg)
    grid = torch.zeros((1, 98, 80), dtype=torch.float64)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    logits, _, positions = student(grid, ids, frame_lengths=full_row(grid))
    narrow = logits[..., :len(cfg.durations)]                       # token slice gone entirely
    with pytest.raises(TdtContractError):
        loss([narrow[0].tolist()], [[1, 2]], [[positions, 2]], TdtSpec(durations=cfg.durations))


def test_a_seam_mismatch_is_refused_before_the_loss_can_absorb_it():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build_double(fixture_config())
    grid = torch.zeros((1, 98, 80), dtype=torch.float64)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    _, _, positions = student(grid, ids, frame_lengths=full_row(grid))
    spec = bridge(16000, frontend=_StubFrontend(), variant=TWO_STAGES_NO_PADDING)
    assert spec.positions == positions
    assert spec.check_seam(positions) == positions
    with pytest.raises(LengthSeamError):
        spec.check_seam(positions + 1)


@pytest.mark.parametrize("variant,frames,expect", [
    (TWO_STAGES_NO_PADDING, 6, "refuse"),
    (TWO_STAGES_NO_PADDING, 7, 1),
    (ONE_STAGE_NO_PADDING, 2, "refuse"),
    (ONE_STAGE_NO_PADDING, 3, 1),
    (ONE_STAGE_PAD_ONE, 1, 1),
    (TWO_STAGES_PAD_ONE, 6, 2),
])
def test_the_stem_refuses_an_undefined_axis_instead_of_a_short_output(variant, frames, expect):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build_double(fixture_config(subsample=variant))
    grid = torch.zeros((1, frames, 80), dtype=torch.float64)
    ids = torch.tensor([[]], dtype=torch.int64)
    if expect == "refuse":
        with pytest.raises(ShapeContractError) as err:
            student(grid, ids, frame_lengths=full_row(grid))
        assert err.value.details["variant"] == variant.label
        assert err.value.details["frames"] == frames
    else:
        _, _, positions = student(grid, ids, frame_lengths=full_row(grid))
        assert positions == expect, (variant.label, frames, positions)


def test_positional_choice_is_material_and_explicit():
    """The config field is not decoration: swapping it changes both parameters and computation."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    head_dim = FIXTURE["d_model"] // FIXTURE["attention_heads"]
    window = FIXTURE["relative_window"]
    no_pos = build_double(fixture_config(position_encoding="none"))
    relative = build_double(fixture_config(position_encoding="relative_shaw_windowed"))
    expected_delta = FIXTURE["block_count"] * (2 * window + 1) * head_dim
    assert relative.parameters_total - no_pos.parameters_total == expected_delta, (
        relative.parameters_total, no_pos.parameters_total, expected_delta)
    torch.manual_seed(0)
    a = build_double(fixture_config(position_encoding="none"))
    torch.manual_seed(0)
    b = build_double(fixture_config(position_encoding="relative_shaw_windowed"))
    grid = torch.ones((1, 20, 80), dtype=torch.float64)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    out_a, _, _ = a(grid, ids, frame_lengths=full_row(grid))
    out_b, _, _ = b(grid, ids, frame_lengths=full_row(grid))
    assert not torch.allclose(out_a, out_b), "the positional choice did not change the computation"


@pytest.mark.parametrize("field,value", [
    ("normalization", "batch_norm"),
    ("conv_activation", "swish"),
    ("macaron", "single"),
    ("prediction_kind", "lstm"),
])
def test_each_alternative_candidate_still_produces_the_contracted_shape(field, value):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    overrides = {field: value}
    if field == "prediction_kind":
        overrides["prediction_layers"] = 1
    student = build_double(fixture_config(**overrides))
    grid = torch.randn((1, 98, 80), dtype=torch.float64)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    logits, _, positions = student(grid, ids, frame_lengths=full_row(grid))
    assert tuple(logits.shape) == (1, positions, 3, fixture_config().logit_width), tuple(logits.shape)
    assert bool(torch.isfinite(logits).all())


def test_declared_target_length_is_read_while_padding_beyond_it_is_not():
    """What the loss reads, and what it refuses to read.

    Corrected after a false claim in an earlier version of this test, which asserted that padding
    a short hypothesis with real tokens would be *scored as spoken*. Measurement says otherwise:
    the lattice is bounded by each row's declared ``U_b``, so target values beyond it are never
    read - ``tests/test_tdt_batch_length_contract.py`` now covers that in both implementations.
    What this test checks is the two facts that remain true from the model side: a longer declared
    hypothesis is a different hypothesis and changes the score, and changing only the unread
    padding does not, because the predictor is causal and the encoder never sees target ids.

    Separate and still open (D-5): padding along the *frame* axis. Step 3B masks the blocks, and
    ``tests/test_modeling_encoder_masks.py`` measures what that buys - exactly which variants and
    which normalisations now reproduce a row scored alone, and which still do not. Nothing in this
    tree claims batched encoding is safe in general.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    cfg = fixture_config()
    student = build_double(cfg)
    grid = torch.zeros((1, 98, 80), dtype=torch.float64)
    logits, _, positions = student(grid, torch.tensor([[1, 3]], dtype=torch.int64),
                                frame_lengths=full_row(grid))
    rows = logits[0].tolist()
    spec = TdtSpec(durations=cfg.durations)

    short_pad_two = loss([rows], [[1, 2]], [[positions, 1]], spec)[0]
    short_pad_three = loss([rows], [[1, 3]], [[positions, 1]], spec)[0]
    declared_two = loss([rows], [[1, 2]], [[positions, 2]], spec)[0]
    assert short_pad_two == short_pad_three, "unread padding changed the score"
    assert short_pad_two != declared_two, "the declared length was not read at all"


def test_the_same_seed_rebuilds_identically_and_no_state_is_persisted():
    """Determinism of the fixture only - a claim about seeding, not about behaviour."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    torch.manual_seed(11)
    first = build_double(fixture_config())
    torch.manual_seed(11)
    second = build_double(fixture_config())
    same = all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert same and first.parameters_total == second.parameters_total
