"""Padding masks in the student encoder: what is inert, what is still not, measured.

Scope, in the owner's words: per-sample valid lengths arrive explicitly and every mask is derived
from them through the named ``StackSpec``; attention may only attend to a row's own keys; the
convolution module masks between its first pointwise and its depthwise, in every block; and each
subsampling stage zeroes the positions its row never produced before the next stage reads them. No
subsampling variant is chosen here and no TDT mathematics is touched.

What the measurements in these nodes support:

* under layer normalisation a row inside a ragged batch reproduces the same row scored alone at every
  one of its valid positions, for **all four** subsampling candidates, to ``1.4e-15`` - inside the
  provisional float64 tolerance ``atol=1e-12, rtol=0``, which is justified by that measured floor
  rather than chosen to cover anything - with global attention as well as with the Shaw window, at one,
  two and four blocks, for a row of length one, and whether the caller pads with zeros or with ``3.0``.
* the stem claim is not a sample: every length from each variant's floor to the padded axis is swept,
  and the assertion is that the set of diverging positions is empty.
* the same row's loss, through the stdlib reference and through the torch transcription, is the same
  number batched or alone.
* every mask is load-bearing: declared separately inert, the attention mask costs ``1.68e-1`` and the
  convolution mask ``2.29e-1`` at a valid position; un-masking a single block of two is enough to move
  the row; and running the same stem convolutions without the between-stage zeroing reopens the
  ``4.0e-1`` divergence at exactly the lengths where it used to be, for the two-stage ``padding=1``
  stack and for no other.
* the zeroing is a no-op when nothing is padded: the stem's output is then *bit-identical* to the
  single row, and the encoder beneath it agrees to ``8.9e-16`` - not bits, because a batched matrix
  product sums in a different order, which is also the origin of the noise floor quoted above.

What they do not support, and these nodes keep asserting so it cannot be quietly promoted:

* **batch normalisation**, a separate issue with a separate mechanism. Masking does not make it
  batch-invariant: in train mode the row moves by ``5.2e-1`` to ``9.3e-1`` against itself scored alone
  and by ``7.0e-1`` to ``9.2e-1`` against the same content in a different batch composition, and one
  padded batch's statistics update moves a clean row by ``1.1e-1`` afterwards. It pools over the length
  axis at *every* normalisation site in every block, so a partial change is not a fix. The subsampling
  defect above was arithmetic; this one is the choice of normalisation, and it stays open.
* nothing in this file is a claim about speech, training readiness or a frozen architecture, and
  float32 is not held to the float64 tolerance: its masked divergence is ``8.3e-7``, four orders above.
"""

from __future__ import annotations

import pytest

from nativevoice.core.errors import ConfigurationError, VoiceError
from nativevoice.modeling.config import StudentConfig
from nativevoice.modeling.encoder import ShapeContractError, build_student, derive_position_lengths
from nativevoice.modeling.lengths import (
    ONE_STAGE_NO_PADDING,
    ONE_STAGE_PAD_ONE,
    SUBSAMPLE_VARIANTS,
    TWO_STAGES_NO_PADDING,
    TWO_STAGES_PAD_ONE,
    minimum_frames,
    stack_length,
)
from nativevoice.modeling.masking import position_lengths
from nativevoice.training.tdt_lattice import TdtSpec, loss

_NO_TORCH = ("NOT RUN: torch is absent from this environment; the encoder and its masks run only "
             "in the isolated CPU-torch verification environment installed outside this repository")

#: Provisional and dtype-specific (D-B): float64 reference arithmetic only, not generalised.
ATOL = 1e-12
RTOL = 0.0
FLOAT32_ATOL = 1e-5

#: All four candidates now hold the contract, so the sweep runs over every one of them.
ALL_STEMS = SUBSAMPLE_VARIANTS
STEM_IDS = ["1stage-pad0", "1stage-pad1", "2stage-pad0", "2stage-pad1"]

FIXTURE_LABEL = "test fixture - not a capacity recommendation"
LONG, SHORT = 160, 98
PAD_VALUES = (0.0, 3.0)

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
    return StudentConfig(**{**FIXTURE, **overrides})


def build(variant=TWO_STAGES_NO_PADDING, *, dtype="double", **overrides):
    """A seeded student in the requested dtype, in eval mode, so no node depends on dropout.

    Every call reseeds, which is what lets a control student and a reference student be the same
    weights with different masks: the comparison is then only about the mask.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    torch.manual_seed(3)
    student = build_student(fixture_config(subsample=variant, **overrides))
    getattr(student, dtype)()
    return student.eval()


def batch_of_two(torch, *, pad=3.0, dtype="double", long=LONG, short=SHORT):
    """A two-row right-padded batch, the declaration for it, and the short row alone.

    Row 0 carries ``long`` frames of audio; row 1 carries ``short`` frames and then ``pad`` values
    out to the axis. ``frame_lengths`` is stated by the caller and is never read off ``grid.shape``.
    """
    torch.manual_seed(11)
    real = getattr(torch, dtype)
    first = torch.randn((long, 80), dtype=real)
    second = torch.randn((short, 80), dtype=real)
    padded = torch.full((long, 80), pad, dtype=real)
    padded[:short] = second
    return torch.stack([first, padded], dim=0), (long, short), second.unsqueeze(0)


def encode(student, grid, frame_lengths):
    import torch
    with torch.no_grad():
        return student.encode(grid, frame_lengths=frame_lengths)


def per_position(alone_row, batched_row):
    """Absolute difference per position, so a mean cannot average a boundary away."""
    return (alone_row - batched_row).abs().max(dim=-1).values


def row_and_padded(torch, frames, axis, *, pad):
    """One row's frames, and the same row sitting in an ``axis``-long grid filled with ``pad``.

    The pad value matters here: a nonzero pad tests the input-grid zeroing, a zero pad isolates the
    stem's own intermediate positions, because only with zeros does the convolution see what it
    would have seen alone. Nodes say which of the two they are measuring.
    """
    torch.manual_seed(11)
    row = torch.randn((frames, 80), dtype=torch.float64)
    padded = torch.full((axis, 80), pad, dtype=row.dtype)
    padded[:frames] = row
    return row, padded


def batch_with_long_row(torch, row, frames, axis, *, pad):
    """A two-row batch: a full-length foreign row, then ``row`` padded out to ``axis``."""
    torch.manual_seed(17)
    other = torch.randn((axis, 80), dtype=torch.float64)
    padded = torch.full((axis, 80), pad, dtype=torch.float64)
    padded[:frames] = row
    return torch.stack([other, padded], dim=0)


# ------------------------------------------------------------------ 1. the declaration is required

def test_frame_lengths_is_a_required_keyword_and_not_something_to_infer():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build()
    grid = torch.zeros((2, LONG, 80), dtype=torch.float64)
    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    with pytest.raises(TypeError):
        student(grid, ids)                                          # type: ignore[misc]
    with pytest.raises(TypeError):
        student(grid, ids, (LONG, LONG))                            # type: ignore[misc]
    _, _, positions = student(grid, ids, frame_lengths=(LONG, LONG))
    assert positions == stack_length(LONG, variant=TWO_STAGES_NO_PADDING)


@pytest.mark.parametrize("lengths,needle", [
    ((), "every row"),
    ((LONG,), "every row"),
    ((LONG, LONG, LONG), "every row"),
    ((LONG, "98"), "integer"),
    ((LONG, 98.0), "integer"),
    ((LONG, True), "integer"),
    ((LONG, -1), "non-negative"),
    ((LONG, LONG + 1), "padded frame axis"),
    ("160160", "not text"),
])
def test_a_declaration_that_cannot_be_the_truth_about_this_batch_is_refused(lengths, needle):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build()
    grid = torch.zeros((2, LONG, 80), dtype=torch.float64)
    with pytest.raises(VoiceError) as err:
        student.encode(grid, frame_lengths=lengths)
    assert needle in str(err.value), (needle, str(err.value))


def test_a_declared_zero_length_row_is_refused_rather_than_masked_into_the_batch():
    """RD-1's rule seen from the encoder side: a row with no position is a caller error.

    Which refusal fires depends on the named stack - an unpadded stack calls the operation
    undefined before it can be empty, a padded one reports an empty output - so this pins the
    shared property: no batch is built, and the row is named.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    for variant in SUBSAMPLE_VARIANTS:
        student = build(variant)
        grid = torch.zeros((2, LONG, 80), dtype=torch.float64)
        with pytest.raises(VoiceError) as err:
            student.encode(grid, frame_lengths=(LONG, 0))
        assert err.value.details.get("row") == 1 or err.value.details.get("frames") == 0, \
            (variant.label, err.value.details)


def test_a_row_too_short_for_the_named_stack_is_refused_naming_the_variant_and_the_count():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(TWO_STAGES_NO_PADDING)
    grid = torch.zeros((2, LONG, 80), dtype=torch.float64)
    with pytest.raises(ShapeContractError) as err:
        student.encode(grid, frame_lengths=(LONG, 6))
    assert err.value.details["variant"] == TWO_STAGES_NO_PADDING.label
    assert err.value.details["frames"] == 6


def test_no_default_subsampling_variant_appears_anywhere_in_this_contract():
    """R1 stays open in code, not only in prose: no derivation works without a named stack."""
    with pytest.raises(ConfigurationError):
        derive_position_lengths((LONG, SHORT), batch=2, total_frames=LONG, variant="two stages")
    for variant in SUBSAMPLE_VARIANTS:
        assert variant.padding in (0, 1), variant.label


# ------------------------------------------------- 2. lengths are derived, and the axis is a lie

def test_derived_row_lengths_are_the_authority_and_the_returned_count_is_the_padded_axis():
    """D-A in one node: the batch says 39, the row says 23, and only one of them is its ``T``."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    counts, lengths, padded = derive_position_lengths(
        (LONG, SHORT), batch=2, total_frames=LONG, variant=TWO_STAGES_NO_PADDING)
    assert counts == (LONG, SHORT)
    assert lengths == (stack_length(LONG, variant=TWO_STAGES_NO_PADDING),
                       stack_length(SHORT, variant=TWO_STAGES_NO_PADDING))
    assert lengths[0] == padded and lengths[1] < padded
    assert position_lengths((LONG, SHORT), variant=TWO_STAGES_NO_PADDING) == lengths
    student = build()
    grid = torch.zeros((2, LONG, 80), dtype=torch.float64)
    _, _, returned = student(grid, torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                          frame_lengths=(LONG, SHORT))
    assert returned == padded, "forward must keep reporting the axis it actually built"
    assert returned != lengths[1]


def test_a_batch_padded_past_its_longest_row_is_accepted_and_still_says_what_it_meant():
    """Bucket padding is legal: the axis may exceed every declaration, a declaration may not."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build()
    grid = torch.zeros((2, LONG + 40, 80), dtype=torch.float64)
    states = encode(student, grid, (LONG, SHORT))
    axis = stack_length(LONG + 40, variant=TWO_STAGES_NO_PADDING)
    assert int(states.shape[1]) == axis
    _, lengths, _ = derive_position_lengths((LONG, SHORT), batch=2, total_frames=LONG + 40,
                                            variant=TWO_STAGES_NO_PADDING)
    assert max(lengths) < axis


# ------------------------------------------------------------- 3. alone and batched, per variant

@pytest.mark.parametrize("variant", ALL_STEMS, ids=STEM_IDS)
@pytest.mark.parametrize("pad", PAD_VALUES, ids=["pad0", "pad3"])
def test_a_padded_row_reproduces_itself_at_every_valid_position(variant, pad):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(variant)
    grid, lengths, alone_grid = batch_of_two(torch, pad=pad)
    alone = encode(student, alone_grid, lengths[1:])
    states = encode(student, grid, lengths)
    valid = stack_length(SHORT, variant=variant)
    assert int(alone.shape[1]) == valid, (variant.label, "the alone axis is the row's own length")
    moved = per_position(alone[0], states[1][:valid])
    assert float(moved.max()) <= ATOL, (variant.label, pad, float(moved.max()))
    assert torch.allclose(alone[0], states[1][:valid], atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("pad", PAD_VALUES, ids=["pad0", "pad3"])
def test_the_pad_value_is_not_observable_at_a_valid_position(pad):
    """Two callers, one arithmetic: what the pad constant is must not reach a real position."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(TWO_STAGES_NO_PADDING)
    scored = []
    for value in PAD_VALUES:
        grid, lengths, _ = batch_of_two(torch, pad=value)
        scored.append(encode(student, grid, lengths))
    valid = stack_length(SHORT, variant=TWO_STAGES_NO_PADDING)
    moved = float(per_position(scored[0][1][:valid], scored[1][1][:valid]).max())
    assert moved <= ATOL, (pad, moved)


def test_the_first_and_last_valid_position_and_the_convolution_boundary_all_reproduce():
    """Named positions rather than a mean: a mean can hide a boundary by averaging it away."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = TWO_STAGES_NO_PADDING
    student = build(variant)
    grid, lengths, alone_grid = batch_of_two(torch, pad=3.0)
    alone = encode(student, alone_grid, lengths[1:])
    states = encode(student, grid, lengths)
    valid = stack_length(SHORT, variant=variant)
    radius = (student.config.conv_kernel_size - 1) // 2
    assert valid - 1 - 2 * radius >= 0, "the fixture is too short to reach past the boundary"
    moved = per_position(alone[0], states[1][:valid])
    for index in (0, 1, valid - 1 - radius, valid - 1 - 2 * radius, valid - 1):
        assert float(moved[index]) <= ATOL, (index, float(moved[index]))


@pytest.mark.parametrize("blocks", [1, 2, 4], ids=["1block", "2blocks", "4blocks"])
def test_exactness_does_not_collapse_as_blocks_accumulate(blocks):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = TWO_STAGES_NO_PADDING
    student = build(variant, block_count=blocks)
    grid, lengths, alone_grid = batch_of_two(torch)
    alone = encode(build(variant, block_count=blocks), alone_grid, lengths[1:])
    states = encode(student, grid, lengths)
    valid = stack_length(SHORT, variant=variant)
    assert float(per_position(alone[0], states[1][:valid]).max()) <= ATOL, blocks


def test_global_attention_is_as_exact_as_the_window_because_every_padded_key_is_masked():
    """``position_encoding="none"`` is the harder case: an unmasked pad reaches the whole row."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = TWO_STAGES_NO_PADDING
    student = build(variant, position_encoding="none", relative_window=0)
    grid, lengths, alone_grid = batch_of_two(torch)
    alone = encode(build(variant, position_encoding="none", relative_window=0),
                   alone_grid, lengths[1:])
    states = encode(student, grid, lengths)
    valid = stack_length(SHORT, variant=variant)
    moved = float(per_position(alone[0], states[1][:valid]).max())
    assert moved <= ATOL, moved
    assert moved <= 8.9e-16, "measured headroom; if it grows, look long before it reaches the atol"


def test_a_row_of_exactly_one_position_is_exact_inside_a_much_longer_batch():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    for variant in ALL_STEMS:
        floor = minimum_frames(variant)
        grid, lengths, alone_grid = batch_of_two(torch, short=floor)
        student = build(variant)
        alone = encode(student, alone_grid, lengths[1:])
        states = encode(student, grid, lengths)
        assert stack_length(floor, variant=variant) == 1, variant.label
        assert float(per_position(alone[0], states[1][:1]).max()) <= ATOL, variant.label
        assert bool(torch.isfinite(states).all()), variant.label


# ------------------------------------------------------------------------------------ 4. no NaN

@pytest.mark.parametrize("encoding", ["none", "relative_shaw_windowed"],
                         ids=["global-attn", "windowed-attn"])
def test_padded_query_positions_are_finite(encoding):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    window = 3 if encoding == "relative_shaw_windowed" else 0
    student = build(position_encoding=encoding, relative_window=window)
    grid, lengths, _ = batch_of_two(torch)
    states = encode(student, grid, lengths)
    valid = stack_length(SHORT, variant=TWO_STAGES_NO_PADDING)
    assert bool(torch.isfinite(states).all()), "a padded query produced a NaN inside the batch"
    assert valid < int(states.shape[1]), "the fixture must actually contain padded positions"


def test_a_query_outside_every_window_would_be_a_softmax_of_all_minus_infinity():
    """Why the fallback exists, demonstrated rather than asserted about the code.

    The intersection of the pad mask and the Shaw window is empty for a query far enough past the
    row's end, and a softmax over an all-refused row is NaN. The encoder keeps the padded keys out
    and falls back to the pad mask alone for such a row. This node recomputes the naive version to
    show the failure the fallback prevents, then ships the real path to show it is absent.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    valid = stack_length(SHORT, variant=TWO_STAGES_NO_PADDING)
    total = stack_length(LONG, variant=TWO_STAGES_NO_PADDING)
    window = fixture_config().relative_window
    assert total - valid > window, "the fixture must actually contain a stranded query row"
    index = torch.arange(total)
    delta = index.view(total, 1) - index.view(1, total)
    keys = torch.tensor([[True] * valid + [False] * (total - valid)])
    naive = keys.view(1, 1, 1, total) & (delta.abs() <= window).view(1, 1, total, total)
    scores = torch.zeros((1, 1, total, total), dtype=torch.float64).masked_fill(
        ~naive, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    stranded = slice(valid + window, total)
    assert bool(torch.isnan(weights[0, 0, stranded]).any()), (
        "the naive mask is supposed to be the NaN case; if it is not, this control has gone vacuous")
    values = torch.randn((1, 1, total, fixture_config().d_model // fixture_config().attention_heads),
                         dtype=torch.float64)
    values[:, :, valid:] = float("nan")            # what a NaN query row becomes in the next block
    merged = weights @ values
    assert bool(torch.isnan(merged[0, 0, :valid]).all()), (
        "a zero weight times a NaN value is NaN, so one NaN at a padded position reaches every "
        "valid query - which is why the fallback prevents it instead of cleaning it up afterwards")
    student = build()
    grid, lengths, _ = batch_of_two(torch)
    assert bool(torch.isfinite(encode(student, grid, lengths)).all())


def test_a_row_whose_last_position_straddles_the_stem_padding_boundary_is_still_inert():
    """The input-grid zeroing, isolated from the two block masks.

    With ``padding=1`` the stem's own last position reaches one frame past the row, and there the
    convolution substitutes a zero. If the caller's padding is not zero the two are different
    numbers: measured on the raw stem at 101 frames, position 50 moves by ``7.0e-1`` for a pad of
    ``3.0`` and stays exactly put for a pad of ``0.0``. Zeroing the declared pads first is what
    makes the caller's padding and the convolution's padding one and the same thing.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = ONE_STAGE_PAD_ONE
    short = 101
    valid = stack_length(short, variant=variant)
    student = build(variant)
    with torch.no_grad():
        _, lengths, alone_grid = batch_of_two(torch, short=short, pad=3.0)
        stem_alone = student.subsample(alone_grid, (short,))[0]
        raw_padded = student.subsample(batch_of_two(torch, short=short, pad=3.0)[0][1:2],
                                       (short,))[0]
        raw_zeroed = student.subsample(batch_of_two(torch, short=short, pad=0.0)[0][1:2],
                                       (short,))[0]
    assert float((stem_alone[:valid - 1] - raw_padded[:valid - 1]).abs().max()) == 0.0
    assert float((stem_alone[valid - 1] - raw_padded[valid - 1]).abs().max()) > 1e-2, (
        "the boundary this node exists to show has moved")
    assert float((stem_alone - raw_zeroed[:valid]).abs().max()) == 0.0
    for pad in PAD_VALUES:
        grid, lengths, alone_grid = batch_of_two(torch, short=short, pad=pad)
        after = per_position(encode(build(variant), alone_grid, lengths[1:])[0],
                             encode(build(variant), grid, lengths)[1][:valid])
        assert float(after.max()) <= ATOL, (pad, float(after.max()))


def test_attention_reads_only_the_keys_the_declaration_calls_valid():
    """Block level, so the stem cannot be blamed for the block's own arithmetic."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build()
    attention = student.blocks[0].attn
    valid = stack_length(SHORT, variant=TWO_STAGES_NO_PADDING)
    total = stack_length(LONG, variant=TWO_STAGES_NO_PADDING)
    torch.manual_seed(5)
    wide = torch.randn((1, total, student.config.d_model), dtype=torch.float64)
    mask = torch.tensor([[True] * valid + [False] * (total - valid)])
    with torch.no_grad():
        batched = attention(wide, mask)
        alone = attention(wide[:, :valid], mask[:, :valid])
        inert = attention(wide, torch.ones_like(mask))
    assert float(per_position(alone[0], batched[0][:valid]).max()) <= ATOL
    assert float(per_position(alone[0], inert[0][:valid]).max()) > ATOL, (
        "the pad mask is not reaching the attention scores")


def test_the_convolution_module_is_exact_at_the_boundary_it_could_not_help_but_read():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build()
    conv = student.blocks[0].conv
    radius = (student.config.conv_kernel_size - 1) // 2
    valid = stack_length(SHORT, variant=TWO_STAGES_NO_PADDING)
    total = stack_length(LONG, variant=TWO_STAGES_NO_PADDING)
    torch.manual_seed(5)
    wide = torch.randn((1, total, student.config.d_model), dtype=torch.float64)
    mask = torch.tensor([[True] * valid + [False] * (total - valid)])
    with torch.no_grad():
        batched = conv(wide, mask)
        alone = conv(wide[:, :valid], mask[:, :valid])
        inert = conv(wide, torch.ones_like(mask))
    for index in (0, valid - 1 - 2 * radius, valid - 1 - radius, valid - 1):
        assert float(per_position(alone[0][index], batched[0][index]).max()) <= ATOL, index
    assert float(per_position(alone[0], inert[0][:valid]).max()) > ATOL, (
        "masking the pointwise output changed nothing, so it is not where it works")


# ---------------------------------------------------------------- 5. controls: every mask, every block

def _inert_mask(student, part):
    """Make one mask in every block inert by declaring its positions all-valid.

    No production flag exists for this, and none should: the control walks the same forward
    signatures the blocks use, so what it measures is the shipped code without that one mask.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    for block in student.blocks:
        module = getattr(block, part)
        original = module.forward
        if part == "attn":
            module.forward = lambda x, key_valid, _o=original: _o(x, torch.ones_like(key_valid))
        else:
            module.forward = lambda x, pos_valid, _o=original: _o(x, torch.ones_like(pos_valid))
    return student


def _batch_and_alone(torch, student, reference_student):
    grid, lengths, alone_grid = batch_of_two(torch)
    return per_position(encode(reference_student, alone_grid, lengths[1:])[0],
                        encode(student, grid, lengths)[1][:stack_length(
                            SHORT, variant=TWO_STAGES_NO_PADDING)])


@pytest.mark.parametrize("parts", [("attn",), ("conv",), ("attn", "conv")],
                         ids=["attention-off", "conv-off", "both-off"])
def test_either_mask_made_inert_is_observable_at_a_valid_position(parts):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build()
    for part in parts:
        _inert_mask(student, part)
    moved = float(_batch_and_alone(torch, student, build()).max())
    assert moved > 1e-2, (parts, moved)


def test_removing_both_is_at_least_as_visible_as_removing_either():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    seen = {}
    for parts in (("attn",), ("conv",), ("attn", "conv")):
        student = build()
        for part in parts:
            _inert_mask(student, part)
        seen[parts] = float(_batch_and_alone(torch, student, build()).max())
    assert seen[("attn",)] > ATOL and seen[("conv",)] > ATOL, seen
    assert seen[("attn", "conv")] >= max(seen[("attn",)], seen[("conv",)]), seen


@pytest.mark.parametrize("index", [0, 1], ids=["block0-inert", "block1-inert"])
def test_the_mask_belongs_to_every_block_not_only_to_the_first(index):
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build()
    block = student.blocks[index]
    original = block.conv.forward
    block.conv.forward = lambda x, pos_valid, _o=original: _o(x, torch.ones_like(pos_valid))
    assert float(_batch_and_alone(torch, student, build()).max()) > 1e-2, index


# --------------------------------------------------------------- 6. loss: the row, batched or not

def test_the_same_row_losses_the_same_number_whether_scored_alone_or_inside_the_batch():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    cfg = fixture_config()
    spec = TdtSpec(durations=cfg.durations)
    student = build(TWO_STAGES_NO_PADDING)
    grid, lengths, alone_grid = batch_of_two(torch)
    ids = torch.tensor([[1, 2]], dtype=torch.int64)
    valid = stack_length(SHORT, variant=TWO_STAGES_NO_PADDING)
    logits_alone, _, _ = student(alone_grid, ids, frame_lengths=lengths[1:])
    logits_batch, _, _ = student(grid, torch.tensor([[1, 2], [1, 2]], dtype=torch.int64),
                              frame_lengths=lengths)
    rows_alone = logits_alone[0][:valid].tolist()
    rows_batch = logits_batch[1][:valid].tolist()
    reference_alone = loss([rows_alone], [[1, 2]], [[valid, 2]], spec)[0]
    reference_batch = loss([rows_batch], [[1, 2]], [[valid, 2]], spec)[0]
    assert abs(reference_alone - reference_batch) <= ATOL, (reference_alone, reference_batch)

    from nativevoice.training.tdt_loss import tdt_loss as torch_loss
    value_alone = torch_loss(logits_alone[:, :valid].contiguous(), ids,
                            torch.tensor([[valid, 2]], dtype=torch.int64), spec)
    value_batch = torch_loss(logits_batch[1:2, :valid].contiguous(), ids,
                            torch.tensor([[valid, 2]], dtype=torch.int64), spec)
    assert abs(float(value_alone[0].detach()) - float(value_batch[0].detach())) <= ATOL, (
        float(value_alone[0].detach()), float(value_batch[0].detach()))


def test_a_batch_is_scored_with_per_row_lengths_not_with_the_axis_it_was_padded_to():
    """The refusal that catches the mistake, from the model side.

    Handing the loss the padded axis for a short row is not a shape error - extra rows are inert -
    so what proves the contract is that the row's own length is a different, smaller number and
    that the axis length is accepted where the row length is refused.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    spec = TdtSpec(durations=fixture_config().durations)
    student = build(TWO_STAGES_NO_PADDING)
    grid, lengths, _ = batch_of_two(torch)
    axis = stack_length(LONG, variant=TWO_STAGES_NO_PADDING)
    valid = stack_length(SHORT, variant=TWO_STAGES_NO_PADDING)
    logits, _, returned = student(grid, torch.tensor([[1, 2], [1, 2]], dtype=torch.int64),
                              frame_lengths=lengths)
    assert returned == axis > valid
    rows = logits[1][:valid].tolist()
    scored = loss([rows], [[1, 2]], [[valid, 2]], spec)[0]
    assert scored == scored and scored > 0.0
    from nativevoice.training.tdt_loss import tdt_loss as torch_loss
    narrow = logits[1:2, :valid].contiguous()
    with pytest.raises(Exception) as err:                              # declared axis > available rows
        torch_loss(narrow, torch.tensor([[1, 2]], dtype=torch.int64),
                   torch.tensor([[axis, 2]], dtype=torch.int64), spec)
    assert "at least one encoder position" not in str(err.value)


def test_float32_is_not_held_to_the_float64_tolerance_it_cannot_reach():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    for variant in ALL_STEMS:
        student = build(variant, dtype="float")
        grid, lengths, alone_grid = batch_of_two(torch, dtype="float")
        alone = encode(student, alone_grid, lengths[1:])
        states = encode(student, grid, lengths)
        valid = stack_length(lengths[1], variant=variant)
        moved = float(per_position(alone[0], states[1][:valid]).max())
        assert moved <= FLOAT32_ATOL, (variant.label, moved)
        assert moved > ATOL, (variant.label, moved, "float32 would fake a float64 pass")


# ---------------------------------------------------- 7. what is still not inert, kept assertable

def test_batch_normalisation_in_train_mode_still_reads_the_batch_it_was_handed():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    for variant in SUBSAMPLE_VARIANTS:
        student = build(variant, normalization="batch_norm")
        student.train()
        grid, lengths, alone_grid = batch_of_two(torch)
        with torch.no_grad():
            alone = student.encode(alone_grid, frame_lengths=lengths[1:])
            states = student.encode(grid, frame_lengths=lengths)
            alone_in_its_own_shape = student.encode(grid[1:2], frame_lengths=(SHORT,))
        valid = stack_length(SHORT, variant=variant)
        pad_moved = float(per_position(alone[0], states[1][:valid]).max())
        composed = float(per_position(alone_in_its_own_shape[0][:valid], states[1][:valid]).max())
        assert pad_moved > 1e-2, (variant.label, pad_moved)
        assert composed > 1e-2, (variant.label, composed)


def test_a_padded_batch_can_move_a_clean_row_through_the_statistics_it_leaves_behind():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = TWO_STAGES_NO_PADDING
    student = build(variant, normalization="batch_norm")
    grid, lengths, alone_grid = batch_of_two(torch)
    student.train()
    encode(student, grid, lengths)                                    # updates the running stats
    student.eval()
    after = encode(student, alone_grid, lengths[1:])
    before = encode(build(variant, normalization="batch_norm"), alone_grid, lengths[1:])
    valid = stack_length(SHORT, variant=variant)
    assert float(per_position(before[0], after[0][:valid]).max()) > 1e-2


def test_layer_norm_is_the_case_the_masks_were_written_for():
    """Stated as a positive control, because the batch-norm nodes only prove the negative."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = TWO_STAGES_NO_PADDING
    student = build(variant)
    grid, lengths, alone_grid = batch_of_two(torch)
    moved = per_position(encode(build(variant), alone_grid, lengths[1:])[0],
                         encode(student, grid, lengths)[1][:stack_length(SHORT, variant=variant)])
    assert float(moved.max()) <= ATOL


SWEEP_AXIS = 200
# The lengths that used to fail, kept named because they are the reason the sweep exists.
PREVIOUSLY_LEAKY = (98, 101, 102)


def unmasked_subsample(subsample, grid, frame_counts, variant):
    """The shipped stem convolutions run *without* the between-stage zeroing.

    This is the mutation control for criterion 5: it uses the same modules and the same weights, so
    any difference it shows is the zeroing and nothing else.
    """
    x = grid.unsqueeze(1)
    for conv in subsample.convs:
        x = conv(x)
    return x.reshape(x.shape[0], x.shape[2], x.shape[3])


@pytest.mark.parametrize("variant", ALL_STEMS, ids=STEM_IDS)
def test_the_stem_is_padding_inert_at_every_valid_position_for_every_supported_length(variant):
    """Criterion 1: a sweep, not a sample - the defect was parity-conditioned and samples miss it.

    Every length from this variant's own floor to the padded axis is compared at the stem against the
    same row scored alone, with the pads zeroed so the measurement isolates the stem's own intermediate
    positions from the caller's pad values (the input-grid zeroing has its own node). The assertion is on
    the *set* of diverging positions being empty, so no tolerance is doing any work. The full encoder is
    then re-checked at every fifth length plus the lengths that used to fail, with a nonzero pad, because
    that path is what a caller actually runs.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(variant)
    failures = []
    for frames in range(minimum_frames(variant), SWEEP_AXIS + 1):
        row, padded = row_and_padded(torch, frames, SWEEP_AXIS, pad=0.0)
        with torch.no_grad():
            alone = student.subsample(row.unsqueeze(0), (frames,))[0]
            batched = student.subsample(padded.unsqueeze(0), (frames,))[0]
        t = alone.shape[0]
        diverging = [i for i in range(t) if not torch.equal(alone[i], batched[i])]
        if diverging:
            failures.append((frames, diverging))
        if frames % 5 == 0 or frames in PREVIOUSLY_LEAKY:
            grid = batch_with_long_row(torch, row, frames, SWEEP_AXIS, pad=3.0)
            whole = per_position(encode(student, row.unsqueeze(0), (frames,))[0],
                                 encode(student, grid, (SWEEP_AXIS, frames))[1][:t])
            assert float(whole.max()) <= ATOL, (variant.label, frames, float(whole.max()))
    assert failures == [], (variant.label, failures[:6])


@pytest.mark.parametrize("variant", ALL_STEMS, ids=STEM_IDS)
def test_every_candidate_is_exact_at_the_row_that_has_only_one_position(variant):
    """Criterion 2: the floor row. For two-stage padding=1 the defect used to eat its only position."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    frames = minimum_frames(variant)
    student = build(variant)
    row, _ = row_and_padded(torch, frames, SWEEP_AXIS, pad=3.0)
    grid = batch_with_long_row(torch, row, frames, SWEEP_AXIS, pad=3.0)
    alone = encode(student, row.unsqueeze(0), (frames,))
    states = encode(student, grid, (SWEEP_AXIS, frames))
    assert int(alone.shape[1]) == stack_length(frames, variant=variant) == 1, variant.label
    assert float((alone[0][0] - states[1][0]).abs().max()) <= ATOL, variant.label
    assert bool(torch.isfinite(states).all()), variant.label


@pytest.mark.parametrize("variant", ALL_STEMS, ids=STEM_IDS)
@pytest.mark.parametrize("pad", PAD_VALUES, ids=["pad0", "pad3"])
def test_the_final_valid_position_is_named_and_checked(variant, pad):
    """Criterion 3: the last position explicitly, because it is where every boundary defect lands."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(variant)
    grid, lengths, alone_grid = batch_of_two(torch, pad=pad)
    alone = encode(student, alone_grid, lengths[1:])
    states = encode(student, grid, lengths)
    valid = stack_length(SHORT, variant=variant)
    moved = per_position(alone[0], states[1][:valid])
    assert float(moved[valid - 1]) <= ATOL, (variant.label, pad, float(moved[valid - 1]))
    assert float(moved.max()) <= ATOL, (variant.label, pad, float(moved.max()))


def test_a_tight_batch_and_a_single_row_agree_and_the_stem_stays_bit_identical():
    """Criterion 4: the zeroing must be a no-op when nothing is padded, and its own claim is bit-level.

    An all-valid declaration yields an all-True mask, so the stem must produce the *same bits*. The
    encoder underneath it cannot be promised bit-identical, and this node says so rather than tuning
    the claim away: a batched matrix product reduces its sums in a different order than a single-row
    one, so the states differ in the last bits - measured ``4.4e-16`` to ``8.9e-16``, and even two
    identical rows inside one batch differ by ``6.7e-16`` for two of the four candidates. That is
    reduction order, not contamination, and it is why the invariance nodes assert a tolerance while
    this one asserts bits where bits are actually available.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    for variant in ALL_STEMS:
        student = build(variant)
        torch.manual_seed(11)
        row = torch.randn((SWEEP_AXIS, 80), dtype=torch.float64)
        with torch.no_grad():
            single = student.subsample(row.unsqueeze(0), (SWEEP_AXIS,))
            tight = student.subsample(torch.stack([row, row], dim=0), (SWEEP_AXIS, SWEEP_AXIS))
            alone = student.encode(row.unsqueeze(0), frame_lengths=(SWEEP_AXIS,))
            paired = student.encode(torch.stack([row, row], dim=0),
                                    frame_lengths=(SWEEP_AXIS, SWEEP_AXIS))
        assert torch.equal(single[0], tight[0]), variant.label
        assert torch.equal(tight[0], tight[1]), variant.label
        valid = stack_length(SWEEP_AXIS, variant=variant)
        moved = float((alone[0][:valid] - paired[0][:valid]).abs().max())
        rows_apart = float((paired[0][:valid] - paired[1][:valid]).abs().max())
        assert moved <= ATOL, (variant.label, moved)
        assert rows_apart <= ATOL, (variant.label, rows_apart)


def test_removing_the_intermediate_mask_reopens_the_defect():
    """Criterion 5: the zeroing is load-bearing, proved against the same weights and modules.

    Without it, two-stage padding=1 diverges at exactly the lengths the arithmetic predicted
    (``frames mod 4`` of 1 or 2) and the unpadded candidates do not diverge at all - which is the
    signature the investigation recorded before the fix, asserted here so a future silent removal of
    the mask fails this node rather than a training run.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    variant = TWO_STAGES_PAD_ONE
    student = build(variant)
    for frames in PREVIOUSLY_LEAKY:
        row, padded = row_and_padded(torch, frames, SWEEP_AXIS, pad=0.0)
        with torch.no_grad():
            alone = unmasked_subsample(student.subsample, row.unsqueeze(0), (frames,), variant)[0]
            batched = unmasked_subsample(student.subsample, padded.unsqueeze(0), (frames,),
                                         variant)[0]
            fixed_alone = student.subsample(row.unsqueeze(0), (frames,))[0]
            fixed_batched = student.subsample(padded.unsqueeze(0), (frames,))[0]
        t = alone.shape[0]
        leaked = float((alone[t - 1] - batched[t - 1]).abs().max())
        repaired = float((fixed_alone[t - 1] - fixed_batched[t - 1]).abs().max())
        assert leaked > 1e-2, (frames, leaked, "the control no longer leaks: is the arithmetic gone?")
        assert repaired <= ATOL, (frames, repaired)
    inert = []
    for variant in (ONE_STAGE_NO_PADDING, ONE_STAGE_PAD_ONE, TWO_STAGES_NO_PADDING):
        student = build(variant)
        for frames in range(minimum_frames(variant), SWEEP_AXIS + 1, 3):
            row, padded = row_and_padded(torch, frames, SWEEP_AXIS, pad=0.0)
            with torch.no_grad():
                a = unmasked_subsample(student.subsample, row.unsqueeze(0), (frames,), variant)[0]
                b = unmasked_subsample(student.subsample, padded.unsqueeze(0), (frames,), variant)[0]
            if any(not torch.equal(a[i], b[i]) for i in range(a.shape[0])):
                inert.append((variant.label, frames))
    assert inert == [], inert[:4]


def test_padded_input_frames_take_no_gradient_and_the_row_still_takes_all_of_it():
    """Criterion 6: gradient discipline through the masks, and no dead parameter."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    student = build(TWO_STAGES_PAD_ONE)
    row, _ = row_and_padded(torch, SHORT, SWEEP_AXIS, pad=3.0)
    grid = batch_with_long_row(torch, row, SHORT, SWEEP_AXIS, pad=3.0).requires_grad_(True)
    ids = torch.tensor([[1, 2], [1, 2]], dtype=torch.int64)
    logits, _, _ = student(grid, ids, frame_lengths=(SWEEP_AXIS, SHORT))
    logits.sum().backward()
    grad = grid.grad[1]
    assert float(grad[SHORT:].abs().sum()) == 0.0, "a padded frame still received gradient"
    assert float(grad[:SHORT].abs().sum()) > 0.0, "the row's own frames received no gradient"
    unreached = [name for name, p in student.named_parameters() if p.grad is None]
    assert unreached == [], unreached
