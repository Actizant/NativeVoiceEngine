"""Focused tests for the provisional frontend-to-lattice length bridge.

Everything here is integer arithmetic over the frozen frontend's own `frame_count`, so the
expectations are re-derived rather than copied: the four variants are measured against the
shipped `FBankConfig`, and the boundary values are cross-checked by exhaustive search.

These tests do not touch the loss, the lattice, the frontend or any gate, and they do not
revise the values pinned in `tests/test_tdt_lattice.py`. The two-stage no-padding pair
98 -> 23 and 2,998 -> 748 is asserted *in addition to* that existing pin, so a re-rule of the
subsampling convention shows up here as a deliberate edit rather than a silent drift.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from nativevoice.audio.features import FBankConfig
from nativevoice.core.errors import ConfigurationError
from nativevoice.modeling.lengths import (
    KNOWN_OPEN_DEFECTS,
    ONE_STAGE_NO_PADDING,
    ONE_STAGE_PAD_ONE,
    PROVISIONAL_CONTRACT,
    RESOLVED_DEFECTS,
    SUBSAMPLE_VARIANTS,
    TWO_STAGES_NO_PADDING,
    TWO_STAGES_PAD_ONE,
    LengthSeamError,
    StackSpec,
    SubFloorInputError,
    UndefinedAxisError,
    bridge,
    frequency_bins,
    minimum_frames,
    minimum_samples,
    stage_length,
    stack_length,
)
from nativevoice.training.tdt_lattice import TdtSpec, loss

CFG = FBankConfig()
PARITY = FBankConfig.whisper_parity()

VARIANT_IDS = ["1stage-pad0", "1stage-pad1", "2stage-pad0", "2stage-pad1"]

#: Measured in Step 1A against the shipped frontend and a real convolution, restated here as
#: the expectation so a change in either is a decision and not an accident.
TABLE = {
    ONE_STAGE_NO_PADDING: {"t_1s": 48, "t_30s": 1498, "t_parity_1s": 50, "mel": 39,
                           "floor": 720, "step": 320, "ms": 20.0},
    ONE_STAGE_PAD_ONE: {"t_1s": 49, "t_30s": 1499, "t_parity_1s": 51, "mel": 40,
                        "floor": 400, "step": 320, "ms": 20.0},
    TWO_STAGES_NO_PADDING: {"t_1s": 23, "t_30s": 748, "t_parity_1s": 24, "mel": 19,
                            "floor": 1360, "step": 640, "ms": 40.0},
    TWO_STAGES_PAD_ONE: {"t_1s": 25, "t_30s": 750, "t_parity_1s": 26, "mel": 20,
                         "floor": 400, "step": 640, "ms": 40.0},
}


def axis(variant, n):
    """Positions for n samples, computed from the frontend rather than through the refusal path."""
    return stack_length(CFG.frame_count(n), variant=variant)


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_matches_the_measured_table_for_every_variant(variant):
    expect = TABLE[variant]
    got = bridge(16000, frontend=CFG, variant=variant)
    assert got.positions == expect["t_1s"], (variant.label, got.positions)
    assert got.frames == CFG.frame_count(16000) == 98
    assert got.n_samples == 16000
    assert axis(variant, 480000) == expect["t_30s"]
    assert stack_length(PARITY.frame_count(16000), variant=variant) == expect["t_parity_1s"]
    assert frequency_bins(80, variant=variant) == expect["mel"]
    assert minimum_samples(CFG, variant) == expect["floor"]
    assert got.samples_per_position == expect["step"]
    assert got.ms_per_position == pytest.approx(expect["ms"], abs=1e-12)


def test_bridge_uses_the_frozen_frontend_frame_count_verbatim():
    """The bridge must not re-implement the frontend: it delegates the frame count.

    Compared against the shipped method for every input length up to 12.5 s, under all four
    variants, including the refusal boundary.
    """
    for variant in SUBSAMPLE_VARIANTS:
        need = minimum_frames(variant)
        floor = minimum_samples(CFG, variant)
        checked = refused = 0
        for n in range(1, 20001):
            frames = CFG.frame_count(n)
            if frames < need:
                with pytest.raises(SubFloorInputError):
                    bridge(n, frontend=CFG, variant=variant)
                refused += 1
                continue
            got = bridge(n, frontend=CFG, variant=variant)
            assert got.frames == frames and got.positions == stack_length(frames, variant=variant)
            checked += 1
        # The refusal boundary is exactly "below the floor", with no overlap and no gap.
        assert (checked, refused) == (20001 - floor, floor - 1), (variant.label, checked, refused)


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_floor_is_exact_and_one_sample_below_refuses(variant):
    floor = minimum_samples(CFG, variant)
    assert floor == TABLE[variant]["floor"]
    at = bridge(floor, frontend=CFG, variant=variant)
    assert at.positions == 1 and at.frames == minimum_frames(variant)
    with pytest.raises(SubFloorInputError):
        bridge(floor - 1, frontend=CFG, variant=variant)


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_refusal_names_the_sample_count_frames_and_variant(variant):
    floor = minimum_samples(CFG, variant)
    with pytest.raises(SubFloorInputError) as err:
        bridge(floor - 1, frontend=CFG, variant=variant)
    d = err.value.details
    assert d["n_samples"] == floor - 1
    assert d["variant"] == variant.label
    assert d["minimum_samples"] == floor
    assert d["frames"] == CFG.frame_count(floor - 1) < d["minimum_frames"]
    assert "no encoder position" in str(err.value)


def test_bridge_undefined_axis_is_a_refusal_not_a_silent_zero():
    """What a real convolution does with an axis shorter than its kernel and no padding.

    Returning 0 here would let a caller read "no positions" as a legitimate answer when the
    operation is not defined at all, which is the confusion the T=0 divergence already shows.
    """
    with pytest.raises(UndefinedAxisError):
        stack_length(1, variant=TWO_STAGES_NO_PADDING)          # 1 -> 1 stage would need a real axis
    with pytest.raises(UndefinedAxisError):
        stage_length(2, variant=ONE_STAGE_NO_PADDING)
    assert stage_length(3, variant=ONE_STAGE_NO_PADDING) == 1
    assert stage_length(1, variant=ONE_STAGE_PAD_ONE) == 1      # padding makes short axes legal
    assert stage_length(0, variant=ONE_STAGE_PAD_ONE) == 0
    with pytest.raises(UndefinedAxisError):
        stage_length(0, variant=ONE_STAGE_NO_PADDING)


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_positions_are_monotone_in_audio_length(variant):
    previous = 0
    for n in range(minimum_samples(CFG, variant), 20001, 7):
        now = bridge(n, frontend=CFG, variant=variant).positions
        assert now >= previous, (n, previous, now)
        previous = now


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_stair_step_matches_the_declared_rate(variant):
    floor = minimum_samples(CFG, variant)
    step = TABLE[variant]["step"]
    second = min(n for n in range(floor, floor + 4 * step) if axis(variant, n) >= 2)
    assert second - floor == step, (variant.label, floor, second)
    assert bridge(second, frontend=CFG, variant=variant).positions == 2
    assert bridge(second - 1, frontend=CFG, variant=variant).positions == 1


def test_bridge_pinned_examples_survive_beside_the_existing_test():
    """98 -> 23 and 2,998 -> 748 for the two-stage no-padding stack.

    The same pair is pinned by `tests/test_tdt_lattice.py:398-407`. Asserting it again here is
    not duplication of purpose: that file treats it as an assumption about a component that does
    not exist, while this file treats it as the contract of a module that does.
    """
    assert CFG.frame_count(16000) == 98 and CFG.frame_count(480000) == 2998
    assert bridge(16000, frontend=CFG, variant=TWO_STAGES_NO_PADDING).positions == 23
    assert axis(TWO_STAGES_NO_PADDING, 480000) == 748
    assert PARITY.frame_count(16000) == 101
    assert stack_length(101, variant=TWO_STAGES_NO_PADDING) == 24


def test_bridge_four_variants_are_four_different_answers():
    """The variant is load-bearing and non-defaulted: collapse would be a silent bug."""
    one_second = {v.label: axis(v, 16000) for v in SUBSAMPLE_VARIANTS}
    assert len(set(one_second.values())) == 4, one_second
    mel = {v.label: frequency_bins(80, variant=v) for v in SUBSAMPLE_VARIANTS}
    assert len(set(mel.values())) == 4, mel
    assert one_second[TWO_STAGES_NO_PADDING.label] == 23 and one_second[TWO_STAGES_PAD_ONE.label] == 25


def test_bridge_padded_and_unpadded_cannot_be_reconciled_by_a_shift():
    gaps = {axis(TWO_STAGES_PAD_ONE, n) - axis(TWO_STAGES_NO_PADDING, n) for n in range(1360, 20000)}
    assert len(gaps) > 1, gaps
    assert min(gaps) >= 1 and max(gaps) <= 2, gaps


def _first_defined_frame(variant):
    """Smallest frame axis, 1..400, for which the whole stack is defined and yields a position."""
    for f in range(1, 401):
        try:
            if stack_length(f, variant=variant) >= 1:
                return f
        except UndefinedAxisError:
            continue
    raise AssertionError(f"no defined axis below 401 for {variant.label}")


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_minimum_frames_is_derived_not_searched(variant):
    """The backwards recurrence must equal a forward search over every frame count."""
    derived = minimum_frames(variant)
    assert derived == _first_defined_frame(variant), (variant.label, derived)
    assert stack_length(derived, variant=variant) == 1


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_one_frame_below_the_floor_is_not_a_position(variant):
    """Whatever the stack does one frame short, it never reports a usable position."""
    derived = minimum_frames(variant)
    if derived == 1:
        assert stack_length(0, variant=variant) == 0        # padding cannot conjure an axis
        return
    try:
        below = stack_length(derived - 1, variant=variant)
    except UndefinedAxisError:
        return                                             # a refusal is the honest answer
    assert below == 0 or below < 1, (variant.label, derived, below)


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_bridge_frequency_axis_follows_the_same_law(variant):
    assert frequency_bins(80, variant=variant) == stack_length(80, variant=variant)
    assert frequency_bins(80, variant=variant) == TABLE[variant]["mel"]


def test_bridge_carries_no_default_variant_or_default_frontend():
    """RR-1: the variant is required, so no length can be produced by an implicit convention."""
    with pytest.raises(TypeError):
        bridge(16000, frontend=CFG)                                # type: ignore[call-arg]
    with pytest.raises(TypeError):
        stack_length(98)                                           # type: ignore[call-arg]
    with pytest.raises(TypeError):
        frequency_bins(80)                                         # type: ignore[call-arg]
    with pytest.raises(TypeError):
        bridge(16000, variant=TWO_STAGES_NO_PADDING)               # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [
    dict(label="x", stages=0, kernel=3, stride=2, padding=0),
    dict(label="x", stages=1, kernel=0, stride=2, padding=0),
    dict(label="x", stages=1, kernel=3, stride=0, padding=0),
    dict(label="x", stages=1, kernel=3, stride=2, padding=-1),
    dict(label="x", stages=True, kernel=3, stride=2, padding=0),
    dict(label="x", stages=1, kernel=2.5, stride=2, padding=0),
    dict(label="", stages=1, kernel=3, stride=2, padding=0),
])
def test_bridge_rejects_geometry_it_could_not_interpret(bad):
    with pytest.raises(ConfigurationError):
        StackSpec(**bad)


def test_bridge_rejects_audio_lengths_it_cannot_count():
    for bad in (-1, 3.5, "16000", True, None):
        with pytest.raises(ConfigurationError):
            bridge(bad, frontend=CFG, variant=TWO_STAGES_NO_PADDING)   # type: ignore[arg-type]


class MinimalFrontend:
    """Only what the protocol asks for: a frame count and the geometry to express it."""

    sample_rate = 16000
    hop_samples = 160
    window_samples = 400

    def frame_count(self, n_samples):
        return 0 if n_samples < 400 else 1 + (n_samples - 400) // 160


def test_bridge_never_looks_at_feature_values():
    """R4, enforced structurally rather than by promise.

    A frontend stub that exposes no values and no dtype still yields identical lengths, so the
    bridge cannot be choosing a float32 cast, a clamp or a normalisation - it has no access to
    anything to normalise.
    """
    stub = MinimalFrontend()
    for variant in SUBSAMPLE_VARIANTS:
        a = bridge(480000, frontend=stub, variant=variant)
        b = bridge(480000, frontend=CFG, variant=variant)
        assert (a.frames, a.positions, a.samples_per_position) == (b.frames, b.positions,
                                                                   b.samples_per_position)
        assert a == b


def test_bridge_module_imports_nothing_heavy_or_product_side():
    """No module-level heavy import, and no dependency on the product or training side.

    Amended when the encoder skeleton landed: a function-level ``import torch`` is legitimate and
    is in fact what ``tools/audit.py`` requires, so this scan now mirrors that rule - module-level
    imports must be clean, and any heavy import must be nested rather than at module scope. The
    previous version rejected both, which would have made the lazy-import rule impossible to
    satisfy inside this package.
    """
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "nativevoice" / "modeling"
    heavy = ("torch", "numpy", "sherpa_onnx", "onnxruntime", "faster_whisper", "ctranslate2")
    banned_sources = ("audio.features", "stt.", "models.", "training.", "tts")
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {id(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))}
        inside_functions = {id(n) for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
                            for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom))}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            line = ast.unparse(node)
            for token in banned_sources:
                assert token not in line, f"{path.name}: {token} in {line}"
            if any(token in line for token in heavy):
                assert id(node) not in top, f"{path.name}: module-level heavy import {line}"
                assert id(node) in inside_functions, f"{path.name}: heavy import outside a function"


def test_bridge_seam_tripwire_detects_a_mis_declared_length_in_the_real_lattice():
    """The provisional contract, enforced against the shipped loss rather than asserted in prose.

    A one-position error is invisible *inside* the loss - extra trailing rows are inert - so this
    test does two things: it shows the loss cannot see the difference, and it shows the seam can.
    """
    n, variant, U, D = 16000, TWO_STAGES_NO_PADDING, 2, (0, 1, 2)
    spec = bridge(n, frontend=CFG, variant=variant)
    assert spec.positions == 23

    def lattice(rows):
        targets = [[j + 1 for j in range(U)]]
        width = (U + 1) + len(D)
        table = [[[-0.5] * width for _ in range(U + 1)] for _ in range(rows)]
        return table

    def value(declared_T, rows):
        return loss([lattice(rows)], targets=[ [j + 1 for j in range(U)] ],
                    lengths=[[declared_T, U]], spec=TdtSpec(durations=D))[0]

    honest = value(spec.positions, spec.positions)
    bigger = value(spec.positions + 1, spec.positions + 1)
    smaller = value(spec.positions - 1, spec.positions + 1)
    assert honest != bigger and honest != smaller, "the loss stopped caring about T"

    # The failure the seam exists to catch: 24 rows of logits, honestly declared as T=23,
    # produce the same number as 23 rows. Nothing downstream of here can tell they differ.
    inert = value(spec.positions, spec.positions + 1)
    assert inert == honest
    with pytest.raises(LengthSeamError) as err:
        spec.check_seam(spec.positions + 1)
    d = err.value.details
    assert d["declared_T"] == 24 and d["encoder_positions"] == 23 and d["n_samples"] == n
    assert spec.check_seam(spec.positions) == 23


def test_bridge_keeps_open_decisions_open_and_records_the_closed_one():
    """The record is machine-checkable: nobody may quietly promote these to settled facts."""
    assert "A1 remains unresolved" in PROVISIONAL_CONTRACT
    assert "provisional" in PROVISIONAL_CONTRACT
    joined = " ".join(KNOWN_OPEN_DEFECTS)
    assert "stage count and padding not ruled" in joined
    assert "normalisation" in joined
    assert "T=0 divergence" not in joined
    resolved = " ".join(RESOLVED_DEFECTS)
    assert "T=0 divergence" in resolved
    assert "2026-10-26" in resolved
