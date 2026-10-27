"""C4 - LayerNorm over channels - as the provisional normalisation direction, and its acceptance tests.

What selecting it changed in the architecture: nothing. The tree's existing ``layer_norm`` path already
computes a mean and a variance over the channel axis at each sequence position, at every normalisation
site, with no statistic pooled across the batch or across time - these nodes verify that as a permanent
property instead of leaving it as a claim in a report. What is recorded here is the ruling, that it is
provisional, and that a direction is not a default: ``normalization`` stays a required field and
``batch_norm`` stays enumerated, so the rejected alternative remains selectable and remains measured.

The contract these nodes hold, in the numbers the investigation produced: under ``layer_norm`` a row
reproduces itself at every valid position whatever the right padding is, and whoever else is in the
batch (float64, measured 5.55e-16 to 1.44e-15 across all four candidates in both modes; the arithmetic
floor, three orders inside the accepted tolerance), in training and in evaluation alike, because it
keeps no running statistics at all.
Under ``batch_norm`` the same comparisons come out at ``1.9e+0`` to ``3.7e+0`` for padding and
``6.9e-1`` to ``8.4e-1`` for batch composition, two padded batches move a clean row later by
``2.0e-1``, and a floor-length row cannot even be trained alone. Those figures are asserted here as
well, so the choice stays evidenced rather than remembered.

Bit-identity is claimed only where it is real. ``torch.equal`` is used at the stem and at a single
normalisation site, where a padded position genuinely cannot enter the arithmetic; the encoder as a
whole is held to the tolerance, because batched matrix products sum in a different order and two
identical rows inside one batch may differ in the last bits.

``torch`` is absent from the project environment, so the nodes that build a student report NOT RUN
there and execute in the isolated CPU-torch verification environment. Nothing in this file is evidence
about speech, convergence or readiness of any kind.
"""

from __future__ import annotations

import dataclasses

import pytest

from nativevoice.audio.features import FBankConfig  # the frozen frontend, asserted untouched
from nativevoice.core.errors import ConfigurationError
from nativevoice.modeling.config import (
    NORMALIZATIONS,
    OPEN_ARCHITECTURE_DECISIONS,
    StudentConfig,
)
from nativevoice.modeling.encoder import build_student
from nativevoice.modeling.lengths import (
    SUBSAMPLE_VARIANTS,
    TWO_STAGES_PAD_ONE,
    minimum_frames,
    stack_length,
)
from nativevoice.training.tdt_lattice import TdtSpec, loss

_NO_TORCH = ("NOT RUN: torch is absent from this environment; the encoder and its normalisation run "
             "only in the isolated CPU-torch verification environment installed outside this repository")

FIXTURE_LABEL = "test fixture - not a capacity recommendation"
ATOL64, RTOL = 1e-12, 0.0
FLOAT32_ATOL = 1e-5
#: The band these divergences actually land in: measured 5.55e-16 (one stage, no padding) up to
#: 1.44e-15 (one stage, padding 1) across all four candidates in both modes. ATOL64 stays the contract;
#: this tighter figure exists so that drift fails loudly instead of hiding inside the contract.
REPORTED_BAND = 1.5e-15
SHORT, LONG, WIDE_AXIS = 98, 400, 748
AXES = (LONG, WIDE_AXIS)
PAD_VALUES = (0.0, 3.0)
VARIANT_IDS = ["1stage-pad0", "1stage-pad1", "2stage-pad0", "2stage-pad1"]
PREVIOUSLY_LEAKY = (98, 101, 102)

FIXTURE = dict(
    label=FIXTURE_LABEL,
    d_model=8,
    attention_heads=2,
    feed_forward_mult=2,
    conv_kernel_size=5,
    block_count=2,
    dropout=0.0,
    normalization="layer_norm",
    position_encoding="relative_shaw_windowed",
    relative_window=3,
    macaron="macaron",
    conv_activation="glu",
    mel_bins=80,
    subsample=None,
    vocab_size=5,
    blank_index=0,
    durations=(0, 1, 2),
    prediction_dim=8,
    prediction_kind="embedding_only",
    prediction_layers=0,
    prediction_start_symbol=4,
    joint_dim=8,
)


def fixture_config(variant, **overrides):
    from nativevoice.modeling.lengths import TWO_STAGES_NO_PADDING
    return StudentConfig(**{**FIXTURE, "subsample": variant or TWO_STAGES_NO_PADDING, **overrides})


def build(variant, *, normalization="layer_norm", dtype="double", mode="eval", seed=3, **overrides):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    torch.manual_seed(seed)
    student = build_student(fixture_config(variant, normalization=normalization, **overrides))
    getattr(student, dtype)()
    return getattr(student, mode)()


def row_of(frames, seed=11, dtype="double"):
    import torch
    torch.manual_seed(seed)
    return torch.randn((frames, 80), dtype=getattr(torch, dtype))


def pad_to(row, axis, value):
    import torch
    out = torch.full((axis, 80), value, dtype=row.dtype)
    out[:row.shape[0]] = row
    return out


def partner(axis, seed, dtype="double"):
    import torch
    torch.manual_seed(seed)
    return torch.randn((axis, 80), dtype=getattr(torch, dtype))


def encode(student, grid, frame_lengths):
    import torch
    with torch.no_grad():
        return student.encode(grid, frame_lengths=frame_lengths)


def worst_difference(alone, batched_row, positions):
    import torch
    return float((alone[:positions] - batched_row[:positions]).abs().detach().max())


def sites_of(student, cls_name):
    import torch
    wanted = {"LayerNorm": torch.nn.LayerNorm, "BatchNorm": torch.nn.modules.batchnorm._BatchNorm}
    return {name: module for name, module in student.named_modules()
            if isinstance(module, wanted[cls_name])}


# ------------------------------------------------------------------ 1. the ruling is a record, not a default

def test_a_provisional_direction_is_recorded_and_still_not_a_default():
    """Selecting layer_norm must not smuggle in a default configuration or delete the alternative."""
    assert NORMALIZATIONS == ("layer_norm", "batch_norm"), NORMALIZATIONS
    field = next(f for f in dataclasses.fields(StudentConfig) if f.name == "normalization")
    assert field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    joined = " ".join(OPEN_ARCHITECTURE_DECISIONS)
    assert "PROVISIONALLY RULED to layer_norm" in joined
    assert "placement" in joined and "not ruled" in joined, "placement stays open"
    assert "R1 - stage count and padding" in joined and "no candidate is preferred" in joined
    for candidate in NORMALIZATIONS:
        fixture_config(None, normalization=candidate)
    with pytest.raises(ConfigurationError):
        fixture_config(None, normalization="instance_norm")


def test_the_choice_is_written_in_the_places_that_describe_it():
    """Documentation drift is a defect: the three files that state this must agree."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    encoder = (root / "src" / "nativevoice" / "modeling" / "encoder.py").read_text(encoding="utf-8")
    masking = (root / "src" / "nativevoice" / "modeling" / "masking.py").read_text(encoding="utf-8")
    assert "provisionally chosen direction" in encoder and "Selecting layer_norm is a\ndirection, not a default" in encoder
    assert "provisionally chosen ``layer_norm`` direction" in masking
    assert "Neither figure is a claim about speech" in masking


# ------------------------------------------------------------------- 2. what every site actually is

def test_every_normalisation_site_is_a_channel_wise_layernorm_with_no_pooled_statistic():
    """The structural half of the ruling: count the sites, check the axis, check what is absent."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    cfg = fixture_config(None)
    for macaron, expected_sites in (("macaron", 10), ("single", 8)):
        student = build(None, macaron=macaron)
        norms = sites_of(student, "LayerNorm")
        assert len(norms) == expected_sites, (macaron, len(norms))
        assert sites_of(student, "BatchNorm") == {}
        for name, module in norms.items():
            assert len(module.normalized_shape) == 1, (name, module.normalized_shape)
            width = module.normalized_shape[0]
            assert width in (cfg.d_model, 2 * cfg.d_model), (name, width)
            assert width == (2 * cfg.d_model if name.endswith("conv.norm") else cfg.d_model), name
            assert module.eps == 1e-5, (name, module.eps)


def test_layernorm_normalises_channels_at_one_position_and_cannot_see_another_position():
    """Per-position independence, asserted as bit identity - legitimate at this level.

    A statistic over channels at one position has no way to reach a second position, so changing the
    padded columns to absurd values must leave a real position exactly unchanged. That is the property
    the whole direction rests on, checked at the module rather than inferred from the config name.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    module = torch.nn.LayerNorm(8).double()          # channel-last, exactly as apply_norm calls it
    torch.manual_seed(5)
    base = torch.randn((1, 6, 8), dtype=torch.float64)
    quiet = module(base)
    for value in (0.0, 3.0, 1.0e3, -7.5):
        other = base.clone()
        other[:, 3:, :] = value
        moved = module(other)
        assert torch.equal(quiet[:, :3], moved[:, :3]), value
        assert not torch.equal(quiet[:, 3:], moved[:, 3:]), "pads must still change pads"
    row = quiet.detach()[:, 0]
    assert abs(float(row.mean(dim=-1))) <= 1e-15
    assert abs(float(row.var(dim=-1, unbiased=False)) - 1.0) <= 1e-4
    assert 0.9 < float(row.std(dim=-1)) < 1.1


# --------------------------------------------------------------------- 3. padding invariance, all four

@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("mode", ["train", "eval"], ids=["train", "eval"])
def test_right_padding_does_not_move_any_valid_position(variant, mode):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(variant, mode=mode)
    row = row_of(SHORT)
    positions = stack_length(SHORT, variant=variant)
    measured = []
    for axis in AXES:
        for pad in PAD_VALUES:
            grid = torch.stack([pad_to(partner(axis, 17), axis, 0.0), pad_to(row, axis, pad)])
            states = encode(student, grid, (axis, SHORT))
            alone = encode(student, row.unsqueeze(0), (SHORT,))
            measured.append(worst_difference(alone[0], states[1], positions))
    assert max(measured) <= REPORTED_BAND, (variant.label, mode, measured)
    assert max(measured) <= ATOL64, (variant.label, mode, measured)


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_a_floor_length_row_is_invariant_in_both_modes_and_trains_without_raising(variant):
    """The shortest row the stack admits, in training mode, which BatchNorm could not do at all."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    frames = minimum_frames(variant)
    assert stack_length(frames, variant=variant) == 1, variant.label
    for mode in ("train", "eval"):
        student = build(variant, mode=mode)
        row = row_of(frames)
        grid = torch.stack([pad_to(partner(LONG, 17), LONG, 0.0), pad_to(row, LONG, 3.0)])
        states = encode(student, grid, (LONG, frames))
        alone = encode(student, row.unsqueeze(0), (frames,))
        assert float((alone[0][0] - states[1][0]).abs().detach().max()) <= ATOL64, (variant.label, mode)
        assert bool(torch.isfinite(states).all()), (variant.label, mode)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    logits, _, _ = build(variant, mode="train")(row_of(frames).unsqueeze(0), ids,
                                             frame_lengths=(frames,))
    assert bool(torch.isfinite(logits).all()) and int(logits.shape[1]) == 1


@pytest.mark.parametrize("frames", PREVIOUSLY_LEAKY)
def test_the_lengths_that_used_to_be_the_stem_defect_are_exact_under_this_direction(frames):
    """Two-stage padding=1, the parity-sensitive case, re-checked with layer_norm in both modes."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = TWO_STAGES_PAD_ONE
    positions = stack_length(frames, variant=variant)
    row = row_of(frames)
    for mode in ("train", "eval"):
        student = build(variant, mode=mode)
        for axis in (LONG, WIDE_AXIS):
            grid = torch.stack([pad_to(partner(axis, 17), axis, 0.0), pad_to(row, axis, 3.0)])
            states = encode(student, grid, (axis, frames))
            alone = encode(student, row.unsqueeze(0), (frames,))
            assert worst_difference(alone[0], states[1], positions) <= ATOL64, (frames, axis, mode)


# ------------------------------------------------------------------ 4. batch-composition invariance

@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("mode", ["train", "eval"], ids=["train", "eval"])
def test_whoever_else_is_in_the_batch_does_not_move_the_row(variant, mode):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(variant, mode=mode)
    row = row_of(SHORT)
    positions = stack_length(SHORT, variant=variant)
    alone = encode(student, row.unsqueeze(0), (SHORT,))[0][:positions]
    seen = []
    for extra in (1, 3, 7):
        grids = [pad_to(partner(LONG, 400 + index), LONG, 0.0) for index in range(extra)]
        grids.append(pad_to(row, LONG, 3.0))
        states = encode(student, torch.stack(grids), (SHORT,) * extra + (SHORT,))
        seen.append(worst_difference(alone, states[-1], positions))
    for seed in (910, 911):
        grid = torch.stack([pad_to(partner(LONG, seed), LONG, 0.0), pad_to(row, LONG, 3.0)])
        states = encode(student, grid, (SHORT, SHORT))
        seen.append(worst_difference(alone, states[1], positions))
    assert max(seen) <= REPORTED_BAND, (variant.label, mode, seen)


def test_two_identical_rows_in_one_batch_agree_within_tolerance_not_necessarily_in_bits():
    """The honest form of the claim: bit identity lives at the stem, tolerance at the encoder.

    Batched matrix products reduce their sums in a different order than single-row ones, so the last
    bits of an encoder output are not a promise this tree makes - while the stem, which only convolves
    and masks, is checked with ``torch.equal`` above and here.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(None, mode="eval")
    row = row_of(SHORT)
    with torch.no_grad():
        alone = encode(student, row.unsqueeze(0), (SHORT,))
        pair = encode(student, torch.stack([pad_to(row, LONG, 0.0), pad_to(row, LONG, 0.0)]),
                      (SHORT, SHORT))
        stem_single = student.subsample(row.unsqueeze(0), (SHORT,))
        stem_pair = student.subsample(torch.stack([row, row]), (SHORT, SHORT))
    positions = int(alone.shape[1])
    assert float((alone[0][:positions] - pair[0][:positions]).abs().detach().max()) <= ATOL64
    # Two identical rows inside one batch: bit-identical, because nothing in a channel-wise statistic
    # or a per-row mask can see a neighbouring row. Measured, not assumed - and the encoder output
    # against a *different bucket axis* is held to the tolerance instead, which is where the last
    # bits genuinely move.
    assert torch.equal(pair[0][:positions], pair[1][:positions])
    assert torch.equal(stem_single[0], stem_pair[0])
    assert torch.equal(stem_pair[0], stem_pair[1])
    wide_axis = encode(student, torch.stack([pad_to(row, LONG, 0.0)]), (SHORT,))
    assert float((alone[0][:positions] - wide_axis[0][:positions]).abs().detach().max()) <= ATOL64


# ------------------------------------------------------------------------ 5. numerical behaviour

def test_float32_is_measured_and_holds_its_own_tolerance():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(None, dtype="float", mode="train")
    row32 = row_of(SHORT, dtype="float")
    assert row32.dtype == torch.float32
    positions = stack_length(SHORT, variant=student.config.subsample)
    grid = torch.stack([pad_to(partner(LONG, 17, "float"), LONG, 0.0), pad_to(row32, LONG, 3.0)])
    states = encode(student, grid, (LONG, SHORT))
    alone = encode(student, row32.unsqueeze(0), (SHORT,))
    moved = float((alone[0][:positions] - states[1][:positions]).abs().detach().max())
    assert moved <= FLOAT32_ATOL, moved
    assert moved > ATOL64, "float32 was reported as 2e-07 - it must not impersonate a float64 pass"


# --------------------------------------------------------------------- 6. gradients and structure

@pytest.mark.parametrize("mode", ["train", "eval"], ids=["train", "eval"])
def test_valid_frames_are_graded_and_padded_frames_are_not(mode):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(TWO_STAGES_PAD_ONE, mode=mode)
    grid = torch.stack([pad_to(partner(LONG, 17), LONG, 0.0), pad_to(row_of(SHORT), LONG, 3.0)])
    grid = grid.requires_grad_(True)
    ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.int64)
    logits, _, _ = student(grid, ids, frame_lengths=(LONG, SHORT))
    logits.sum().backward()
    assert float(grid.grad[1, SHORT:].abs().detach().sum()) == 0.0, mode
    assert float(grid.grad[1, :SHORT].abs().detach().sum()) > 0.0, mode
    assert [n for n, p in student.named_parameters() if p.grad is None] == [], mode


# --------------------------------------------- 7. the rejected alternative stays measured, not deleted

def test_batch_norm_still_fails_the_same_tests_that_layer_norm_passes():
    """The evidence for the ruling stays in the suite.

    Nothing here argues batch_norm is wrong in general - it is a legitimate operator elsewhere. It
    records that in *this* encoder, with these masks, the padding and composition divergences are
    orders of magnitude outside any tolerance, that its running statistics carry a padded batch into
    later inference, and that a floor-length row cannot be trained alone.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = SUBSAMPLE_VARIANTS[2]        # two stages, padding 0: the stem is inert, so BN is the only variable
    student = build(variant, normalization="batch_norm", mode="train")
    row = row_of(SHORT)
    positions = stack_length(SHORT, variant=variant)
    alone = encode(build(variant, normalization="batch_norm", mode="train"),
                   row.unsqueeze(0), (SHORT,))[0]
    batched = encode(student, torch.stack([pad_to(partner(LONG, 909), LONG, 0.0),
                                           pad_to(row, LONG, 3.0)]), (SHORT, SHORT))[1]
    wide = encode(student, torch.stack([pad_to(partner(WIDE_AXIS, 909), WIDE_AXIS, 0.0),
                                        pad_to(row, WIDE_AXIS, 3.0)]), (SHORT, SHORT))[1]
    assert worst_difference(alone, batched, positions) > 1e-1
    assert worst_difference(alone, wide, positions) > 1e-1

    learner = build(variant, normalization="batch_norm", mode="train")
    pristine = build(variant, normalization="batch_norm", mode="eval")
    before = encode(pristine, row.unsqueeze(0), (SHORT,))[0]
    grid = torch.stack([pad_to(partner(LONG, 909), LONG, 0.0), pad_to(row, LONG, 3.0)])
    for _ in range(2):
        encode(learner, grid, (SHORT, SHORT))
    learner.eval()
    after = encode(learner, row.unsqueeze(0), (SHORT,))[0]
    assert worst_difference(before, after, positions) > 1e-2, "buffer carry-over vanished: recheck"

    floor = minimum_frames(variant)
    with pytest.raises(ValueError):
        encode(build(variant, normalization="batch_norm", mode="train"),
               row_of(floor).unsqueeze(0), (floor,))


# ------------------------------------------------------------------- 8. contracts and interfaces

def test_the_direction_touches_no_contract_anybody_else_depends_on():
    """Output lengths, the emission slot layout, the loss seam and the frozen frontend are unchanged."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    cfg = fixture_config(None)
    for normalization in NORMALIZATIONS:
        student = build(None, normalization=normalization)
        grid = torch.stack([pad_to(partner(LONG, 17), LONG, 0.0), pad_to(row_of(SHORT), LONG, 3.0)])
        ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        logits, _, positions = student(grid, ids, frame_lengths=(LONG, SHORT))
        assert positions == stack_length(LONG, variant=cfg.subsample)
        assert tuple(logits.shape) == (2, positions, 3, cfg.logit_width), (normalization, logits.shape)
        rows = logits[1, :stack_length(SHORT, variant=cfg.subsample)].tolist()
        scored = loss([rows], [[1, 2]], [[stack_length(SHORT, variant=cfg.subsample), 2]],
                      TdtSpec(durations=cfg.durations))
        assert scored[0] == scored[0] and scored[0] > 0.0, normalization
        with pytest.raises(TypeError):
            student(grid, ids)                                        # type: ignore[misc]
    frozen = FBankConfig()
    assert (frozen.sample_rate, frozen.window_ms, frozen.hop_ms) == (16000, 25.0, 10.0)
    assert (frozen.window_samples, frozen.hop_samples, frozen.n_mels, frozen.fft_size) == (400, 160, 80, 1024)
    assert frozen.frame_count(16000) == 98 and frozen.frame_count(399) == 0
