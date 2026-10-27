"""Focused tests for the pure length and mask-pattern contract. Standard library only.

These assert arithmetic and refusals. They assert nothing about the encoder: a pattern being
well-defined says no more about batched encoding than a ruler says about a building. The masks
built from them are measured in ``tests/test_modeling_encoder_masks.py``, which keeps the same
refusal that batching is not claimed safe.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from nativevoice.audio.features import FBankConfig
from nativevoice.core.errors import ConfigurationError
from nativevoice.modeling.lengths import (
    ONE_STAGE_NO_PADDING,
    ONE_STAGE_PAD_ONE,
    SubFloorInputError,
    TWO_STAGES_NO_PADDING,
    TWO_STAGES_PAD_ONE,
    UndefinedAxisError,
    bridge,
    stack_length,
)
from nativevoice.modeling.masking import (position_lengths, position_lengths_from_audio,
                                         valid_frame_mask, valid_position_mask)

VARIANTS = (ONE_STAGE_NO_PADDING, ONE_STAGE_PAD_ONE, TWO_STAGES_NO_PADDING, TWO_STAGES_PAD_ONE)
VARIANT_IDS = ["1stage-pad0", "1stage-pad1", "2stage-pad0", "2stage-pad1"]
CFG = FBankConfig()

#: Frames where all four candidate stacks are defined, with the positions each produced in the
#: design pass. Restated here so a change to the arithmetic becomes a decision, not a drift.
LENGTH_TABLE = {
    7: (3, 4, 1, 2),
    98: (48, 49, 23, 25),
    2998: (1498, 1499, 748, 750),
}


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_position_lengths_delegate_to_stack_length_for_every_variant(variant):
    frames = sorted(LENGTH_TABLE)
    assert position_lengths(frames, variant=variant) == tuple(
        stack_length(f, variant=variant) for f in frames)


@pytest.mark.parametrize("frames", sorted(LENGTH_TABLE))
def test_position_lengths_table_matches_the_measured_bridge_values(frames):
    for variant, expect in zip(VARIANTS, LENGTH_TABLE[frames]):
        assert position_lengths([frames], variant=variant) == (expect,), variant.label


def test_a_stacked_axis_below_the_floor_raises_instead_of_reporting_zero():
    """No silent zero: a stack without padding cannot run over a short axis, and says so."""
    with pytest.raises(UndefinedAxisError):
        position_lengths([2], variant=TWO_STAGES_NO_PADDING)
    with pytest.raises(UndefinedAxisError):
        position_lengths([1], variant=ONE_STAGE_NO_PADDING)
    assert position_lengths([1], variant=ONE_STAGE_PAD_ONE) == (1,)
    assert position_lengths([1], variant=TWO_STAGES_PAD_ONE) == (1,)
    assert position_lengths([3], variant=ONE_STAGE_NO_PADDING) == (1,)


def test_position_lengths_from_audio_agrees_with_the_bridge_row_by_row():
    for n in (1360, 16000, 480000, 2000, 720, 719, 399):
        for variant in VARIANTS:
            try:
                expect = bridge(n, frontend=CFG, variant=variant).positions
            except SubFloorInputError:
                with pytest.raises(SubFloorInputError):
                    position_lengths_from_audio([n], frontend=CFG, variant=variant)
                continue
            assert position_lengths_from_audio([n], frontend=CFG, variant=variant) == (expect,)


def test_position_lengths_from_audio_preserves_row_order():
    """A non-monotone input order must come back non-monotone: no silent sorting of rows."""
    audio = (16000, 480000, 64000)
    for variant in VARIANTS:
        rows = position_lengths_from_audio(audio, frontend=CFG, variant=variant)
        assert rows == tuple(bridge(n, frontend=CFG, variant=variant).positions for n in audio)
        assert rows[0] < rows[1] and rows[2] < rows[1], (variant.label, rows)


def test_sub_floor_audio_is_refused_with_the_sample_count_named():
    with pytest.raises(SubFloorInputError) as err:
        position_lengths_from_audio([1359, 16000], frontend=CFG, variant=TWO_STAGES_NO_PADDING)
    assert err.value.details["n_samples"] == 1359
    assert err.value.details["variant"] == TWO_STAGES_NO_PADDING.label


def test_the_padded_batch_case_seven_hundred_forty_eight_positions_and_twenty_three_real():
    """The case that motivated the contract: one second of audio in a batch padded to thirty."""
    lengths = position_lengths_from_audio([16000, 480000], frontend=CFG,
                                          variant=TWO_STAGES_NO_PADDING)
    assert lengths == (23, 748)
    total = max(lengths)
    mask = valid_position_mask(total, lengths)
    assert len(mask) == 2 and all(len(row) == 748 for row in mask)
    assert sum(mask[0]) == 23 and sum(mask[1]) == 748
    assert mask[0][:23] == (True,) * 23 and mask[0][23:] == (False,) * 725
    assert sum(mask[0]) != total, "the padded axis must never be read as a row's own length"


@pytest.mark.parametrize("total,rows,expected", [
    (4, [4], ((True, True, True, True),)),
    (4, [1], ((True, False, False, False),)),
    (5, [5, 2, 3], ((True,) * 5, (True, True, False, False, False),
                     (True, True, True, False, False))),
    (748, [23], ((True,) * 23 + (False,) * 725,)),
])
def test_valid_position_mask_patterns(total, rows, expected):
    assert valid_position_mask(total, rows) == expected


def test_valid_frame_mask_uses_the_frame_axis_granularity():
    frames = (CFG.frame_count(16000), CFG.frame_count(480000))
    assert frames == (98, 2998)
    mask = valid_frame_mask(max(frames), frames)
    assert sum(mask[0]) == 98 and sum(mask[1]) == 2998 and len(mask[0]) == 2998
    positions = position_lengths(frames, variant=TWO_STAGES_NO_PADDING)
    pmask = valid_position_mask(max(positions), positions)
    assert (sum(pmask[0]), sum(mask[0])) == (23, 98), "two axes, one authority"


@pytest.mark.parametrize("variant", VARIANTS, ids=VARIANT_IDS)
def test_masks_are_contiguous_prefixes_with_a_tail_equal_to_the_padding(variant):
    """The pure half of the propagation contract: the shape of the mask, not the encoder."""
    audio = (16000, 64000, 480000, 32000)
    rows = position_lengths_from_audio(audio, frontend=CFG, variant=variant)
    total = max(rows)
    for row in valid_position_mask(total, rows):
        assert row == (True,) * sum(row) + (False,) * (total - sum(row))
    assert total == stack_length(max(CFG.frame_count(n) for n in audio), variant=variant)


@pytest.mark.parametrize("bad_total,rows,needle", [
    (0, [1], "at least 1"),
    (-3, [1], "at least 1"),
    (3, [4], "exceeds the padded dimension"),
    (3, [0], "at least one valid position"),
    (3, [-1], "at least one valid position"),
    (3, [1.5], "must be an integer"),
    (3, [True], "must be an integer"),
    (3, [], "must not be empty"),
])
def test_invalid_mask_dimensions_are_refused(bad_total, rows, needle):
    with pytest.raises(ConfigurationError) as err:
        valid_position_mask(bad_total, rows)
    assert needle in str(err.value), str(err.value)


@pytest.mark.parametrize("bad_total", ["33", None, 4.0])
def test_a_non_integer_padded_dimension_is_refused_as_such(bad_total):
    with pytest.raises(ConfigurationError) as err:
        valid_position_mask(bad_total, [1])
    assert "must be an integer" in str(err.value)


def test_a_refusal_names_the_offending_row_index():
    with pytest.raises(ConfigurationError) as err:
        valid_position_mask(5, [5, 9, 2])
    assert err.value.details["row"] == "1" and err.value.details["total_positions"] == 5


@pytest.mark.parametrize("frames,needle", [
    ([-1], "non-negative"),
    ([1.5], "must be an integer"),
    ([True], "must be an integer"),
    ([], "must not be empty"),
    ("44", "sequence of integers"),
])
def test_position_lengths_refuses_bad_frame_counts(frames, needle):
    with pytest.raises(ConfigurationError) as err:
        position_lengths(frames, variant=TWO_STAGES_NO_PADDING)
    assert needle in str(err.value), str(err.value)


def test_position_lengths_refuses_a_missing_or_unnamed_variant():
    with pytest.raises(TypeError):
        position_lengths([98])                                   # type: ignore[call-arg]
    for bad in (None, "two stages", 2):
        with pytest.raises(ConfigurationError) as err:
            position_lengths([98], variant=bad)                  # type: ignore[arg-type]
        assert "no default" in str(err.value)


def test_position_lengths_from_audio_requires_both_authorities():
    with pytest.raises(TypeError):
        position_lengths_from_audio([16000], variant=TWO_STAGES_NO_PADDING)   # type: ignore[call-arg]
    with pytest.raises(TypeError):
        position_lengths_from_audio([16000], frontend=CFG)                    # type: ignore[call-arg]


def test_an_unpadded_batch_masks_to_all_true():
    mask = valid_position_mask(748, [748, 748])
    assert all(row == (True,) * 748 for row in mask)


def test_masks_are_immutable_and_reproducible():
    first = valid_position_mask(4, [2, 4])
    second = valid_position_mask(4, [2, 4])
    assert first == second
    assert isinstance(first, tuple) and isinstance(first[0], tuple)
    with pytest.raises(TypeError):
        first[0][0] = True                                       # type: ignore[index]
    assert hash(first) == hash(second)


def test_masking_module_imports_no_tensor_library_or_product_runtime():
    """AST-level proof: this file cannot quietly become the encoder's dependency."""
    path = (pathlib.Path(__file__).resolve().parents[1] / "src" / "nativevoice"
            / "modeling" / "masking.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [ast.unparse(node) for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert imports, "the module was expected to import its contract from lengths"
    for line in imports:
        for banned in ("torch", "numpy", "audio", "stt", "models", "training", "tts"):
            assert banned not in line, line
