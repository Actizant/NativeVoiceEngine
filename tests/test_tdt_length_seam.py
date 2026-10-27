"""C4: the frontend-to-encoder length seam at the TDT scoring boundary.

The seam is one route and it already exists: sample counts, through the frozen
``FBankConfig`` frontend and the declared ``StackSpec``, become per-row encoder position counts by
``nativevoice.modeling.masking.position_lengths_from_audio``. No second producer and no alias was
added. What this file pins is that a declared ``T_b`` which disagrees with that route is refused by
both implementations of the loss, with the same exception type and the same detail fields, before a
single lattice cell is computed.

Why a comparison rather than a smarter check: rows past ``T_b`` are inert, because the terminal
sentinel never reads a logit at index ``T``. An under-declared ``T_b`` therefore leaves nothing in
the tensor for a later inspection to find - one utterance scored at ``T_b`` of 49, 47, 43 and 1
returns four different finite numbers and refuses none of them, which is a node in this file
(:func:`test_four_under_declared_lengths_score_four_finite_losses_and_only_the_seam_refuses_them`)
rather than a claim in prose. The declared length has to be measured against something outside the
logits, and that something is the frontend.

``expected_positions`` is optional at the function boundary for one reason only: the arithmetic and
oracle suites call these functions directly. Omission is a migration condition, not an accepted
steady state, and the last nodes here are what hold that line - an inventory of the public scoring
surface and an AST sweep of ``src/`` and ``tools/`` for a scoring call that skips the seam. Both are
vacuous-but-live rather than unconditional enforcement: there is currently no production scoring call
site at all, so the sweep has no traffic to protect. The inventory also names the residual hole rather
than pretending it away - :func:`nativevoice.training.tdt_lattice.validate` reads a ``lengths``
argument and does not consult the seam, so a caller that composes it with ``alpha_table`` reaches the
lattice around both scoring functions. Closing that means giving ``validate`` the same optional
parameter, which touches a guard the arithmetic suites depend on, so it is reported instead of
silently widened here.

``torch`` is absent from the project environment; nodes that build tensors report NOT RUN there and
run in the isolated CPU-torch verification environment. Reference-only nodes run everywhere.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib

import pytest

from nativevoice.audio.features import FBankConfig
from nativevoice.modeling.encoder import derive_position_lengths
from nativevoice.modeling.lengths import (
    ONE_STAGE_NO_PADDING,
    ONE_STAGE_PAD_ONE,
    SUBSAMPLE_VARIANTS,
    TWO_STAGES_NO_PADDING,
    TWO_STAGES_PAD_ONE,
    bridge,
    minimum_frames,
    minimum_samples,
    stack_length,
)
from nativevoice.modeling.masking import position_lengths, position_lengths_from_audio
from nativevoice.training import tdt_lattice, tdt_loss as tdt_loss_module
from nativevoice.training.errors import TdtAlignmentImpossible, TdtContractError
from nativevoice.training.tdt_lattice import TdtSpec, loss as ref_loss

_NO_TORCH = ("NOT RUN: torch is absent from this environment; the autograd transcription runs only "
             "in the isolated CPU-torch verification environment installed outside this repository")

D = (0, 1, 2)
ND = len(D)
VOCAB = 4                                   # ids 0..3, 0 is the blank slot
WIDTH = VOCAB + ND
U = 2
TARGET = [1, 2]
SPEC = TdtSpec(durations=D, blank_index=0)
CFG = FBankConfig()
PARITY = FBankConfig.whisper_parity()
VARIANT_IDS = ["1stage-pad0", "1stage-pad1", "2stage-pad0", "2stage-pad1"]
SECONDS = (1360, 2000, 8000, 16000, 40000, 80000, 160000, 480000)   # all >= the largest floor

ROOT = pathlib.Path(__file__).resolve().parents[1]


def value(b, i, u, w):
    """Deterministic, bounded, non-degenerate logits - no zero variance, no infinities."""
    return math.sin((b + 1) * 3.1 + i * 0.7 + u * 1.3 + w * 0.41) * 0.9


def rows_for(batch_index, count):
    return [[[value(batch_index, i, u, w) for w in range(WIDTH)] for u in range(U + 1)]
            for i in range(count)]


def seam_positions(samples_per_row, variant):
    return position_lengths_from_audio(samples_per_row, frontend=CFG, variant=variant)


# ------------------------------------------------------- 1. the accepted case, and its neighbours

@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_the_seam_derived_length_is_accepted_and_changes_no_number(variant):
    samples = (16000, 40000)
    positions = seam_positions(samples, variant)
    logits = [rows_for(b, positions[b]) for b in range(len(samples))]
    lengths = [[positions[b], U] for b in range(len(samples))]
    plain = ref_loss(logits, [list(TARGET) for _ in samples], lengths, SPEC)
    seamed = ref_loss(logits, [list(TARGET) for _ in samples], lengths, SPEC,
                      expected_positions=positions)
    assert plain == seamed, (variant.label, plain, seamed)
    assert all(value_ == value_ and abs(value_) < 1e4 for value_ in plain), plain


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_an_under_declared_length_is_refused_only_by_the_seam(variant):
    """The case the logits cannot reveal: fewer declared positions than the audio carries."""
    positions = seam_positions((16000,), variant)
    declared = positions[0] - 2
    logits = [rows_for(0, positions[0])]
    lengths = [[declared, U]]
    targets = [list(TARGET)]
    without = ref_loss(logits, targets, lengths, SPEC)          # accepted today, silently different
    assert without == without and abs(without[0]) < 1e4
    with pytest.raises(TdtContractError) as err:
        ref_loss(logits, targets, lengths, SPEC, expected_positions=positions)
    assert err.value.details["declared_T"] == declared
    assert err.value.details["expected_T"] == positions[0]
    assert err.value.details["batch"] == 0


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_an_over_declared_length_inside_the_padded_axis_is_invisible_except_to_the_seam(variant):
    """Rows exist, so every existing guard passes. Only the frontend knows the row is too long."""
    positions = seam_positions((16000,), variant)
    axis = positions[0] + 4
    declared = axis
    logits = [rows_for(0, axis)]
    lengths = [[declared, U]]
    targets = [list(TARGET)]
    assert ref_loss(logits, targets, lengths, SPEC)             # no pre-existing guard objects
    with pytest.raises(TdtContractError) as err:
        ref_loss(logits, targets, lengths, SPEC, expected_positions=positions)
    assert err.value.details["expected_T"] == positions[0]
    assert err.value.details["declared_T"] == declared > positions[0]


def test_an_over_declared_length_beyond_the_rows_is_refused_twice_over():
    """The row budget refuses it, and so does the seam once the rows are supplied - neither guard
    can be dropped without the other noticing."""
    positions = seam_positions((16000,), TWO_STAGES_NO_PADDING)
    rows = rows_for(0, positions[0])
    lengths = [[positions[0] + 3, U]]
    with pytest.raises(TdtContractError) as err:
        ref_loss([rows], [list(TARGET)], lengths, SPEC)
    assert "fewer logit rows" in str(err.value)
    assert err.value.details == {"batch": 0, "T": positions[0] + 3, "rows": positions[0]}
    with pytest.raises(TdtContractError) as seam_err:
        ref_loss([rows + rows_for(0, 3)], [list(TARGET)], lengths, SPEC,
                 expected_positions=positions)
    assert seam_err.value.details == {"batch": 0, "declared_T": positions[0] + 3,
                                      "expected_T": positions[0]}


# --------------------------------------------------------------- 2. the expected sequence itself

@pytest.mark.parametrize("splice,needle,ident", [
    (lambda p: (), "every row", "empty"),
    (lambda p: p[:1], "every row", "short"),
    (lambda p: p + p, "every row", "long"),
    (lambda p: "4949", "not text", "text"),
    (lambda p: (p[0], "49"), "integer position counts", "string-element"),
    (lambda p: (p[0], float(p[1])), "integer position counts", "float-element"),
    (lambda p: (p[0], True), "integer position counts", "bool-element"),
    (lambda p: (p[0], 0), "no encoder position", "zero-element"),
    (lambda p: (p[0], -1), "no encoder position", "negative-element"),
], ids=["empty", "short", "long", "text", "string-element", "float-element",
         "bool-element", "zero-element", "negative-element"])
def test_a_malformed_expected_position_sequence_is_refused(splice, needle, ident):
    """The first element is always the true count, so the refusal is the splice's and not a mismatch.

    Without that, a wrong value in position 0 would trip the comparison at row 0 and the malformed
    entry in position 1 would never be reached - the node would pass for the wrong reason.
    """
    positions = seam_positions((16000, 16000), TWO_STAGES_NO_PADDING)
    logits = [rows_for(b, positions[b] + 1) for b in range(2)]
    lengths = [[positions[b], U] for b in range(2)]
    bad = splice(positions)
    with pytest.raises(TdtContractError) as err:
        ref_loss(logits, [list(TARGET)] * 2, lengths, SPEC, expected_positions=bad)
    assert needle in str(err.value), (ident, needle, str(err.value))


def test_a_false_position_count_is_not_a_number_of_positions():
    """``True`` is an ``int`` in Python; it must not be mistaken for one position."""
    positions = seam_positions((16000,), TWO_STAGES_NO_PADDING)
    with pytest.raises(TdtContractError) as err:
        ref_loss([rows_for(0, positions[0])], [list(TARGET)], [[positions[0], U]], SPEC,
                 expected_positions=(True,))
    assert "integer position counts" in str(err.value), str(err.value)
    assert err.value.details == {"batch": 0, "expected_positions": True}


def test_four_under_declared_lengths_score_four_finite_losses_and_only_the_seam_refuses_them():
    """The measurement behind the design decision, kept as a node rather than as prose.

    ``one stage, padding 1`` over one second of audio is 49 positions. Scored at 49, 47, 43 and 1
    the reference returns four different finite numbers and refuses none of them, because a row
    past the declared ``T`` is never read: there is nothing inside the logits for a later
    inspection to find. With the seam supplied, three of the four are refused and the fourth - the
    correct one - is accepted.
    """
    positions = seam_positions((16000,), ONE_STAGE_PAD_ONE)
    assert positions == (49,), positions
    logits = [rows_for(0, 49)]
    unrefused = [ref_loss(logits, [list(TARGET)], [[t, U]], SPEC) for t in (49, 47, 43, 1)]
    assert len({value_[0] for value_ in unrefused}) == 4, unrefused
    assert all(value_[0] == value_[0] and abs(value_[0]) < 1e4 for value_ in unrefused), unrefused
    for t in (47, 43, 1):
        with pytest.raises(TdtContractError) as err:
            ref_loss(logits, [list(TARGET)], [[t, U]], SPEC, expected_positions=positions)
        assert err.value.details == {"batch": 0, "declared_T": t, "expected_T": 49}
    assert ref_loss(logits, [list(TARGET)], [[49, U]], SPEC, expected_positions=positions) == (
        unrefused[0])


# ------------------------------------------------- 3. four-way identity across the candidates

@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_the_four_routes_from_audio_to_positions_agree_over_a_sweep(variant):
    """bridge == position_lengths_from_audio == position_lengths == derive_position_lengths.

    One arithmetic authority, four spellings, and a node that fails the day anyone adds a fifth.
    """
    for samples in SECONDS:
        spec = bridge(samples, frontend=CFG, variant=variant)
        producer = position_lengths_from_audio((samples,), frontend=CFG, variant=variant)[0]
        from_frames = position_lengths((spec.frames,), variant=variant)[0]
        counts, derived, padded = derive_position_lengths((spec.frames,), batch=1,
                                                          total_frames=spec.frames, variant=variant)
        assert producer == spec.positions == from_frames == derived[0], (variant.label, samples)
        assert counts == (spec.frames,) and padded == spec.positions
        assert stack_length(spec.frames, variant=variant) == spec.positions


def test_the_frames_to_positions_map_is_not_injective_so_positions_are_the_contract():
    """``T_b`` is what the lattice consumes; recovering frames from it is impossible."""
    for variant in SUBSAMPLE_VARIANTS:
        seen = {}
        for frames in range(minimum_frames(variant), 400):
            seen.setdefault(stack_length(frames, variant=variant), []).append(frames)
        collide = sorted(t for t, fs in seen.items() if len(fs) > 1)
        assert collide, variant.label
    two_stage_pad_one = {f: stack_length(f, variant=TWO_STAGES_PAD_ONE) for f in (98, 99)}
    assert two_stage_pad_one[98] == two_stage_pad_one[99], two_stage_pad_one


def test_the_frozen_frontend_and_the_whisper_parity_instrument_are_different_authorities():
    """If a caller passes the wrong frontend, the seam disagrees loudly rather than quietly."""
    assert CFG.frame_count(16000) == 98 and PARITY.frame_count(16000) == 101
    frozen = seam_positions((16000,), TWO_STAGES_NO_PADDING)[0]
    other = position_lengths_from_audio((16000,), frontend=PARITY,
                                        variant=TWO_STAGES_NO_PADDING)[0]
    assert frozen != other, (frozen, other)
    with pytest.raises(TdtContractError) as err:
        ref_loss([rows_for(0, other)], [list(TARGET)], [[other, U]], SPEC,
                 expected_positions=(frozen,))
    assert err.value.details == {"batch": 0, "declared_T": other, "expected_T": frozen}


def test_audio_below_a_variants_floor_is_refused_before_the_seam_can_be_consulted():
    for variant in SUBSAMPLE_VARIANTS:
        too_short = minimum_samples(CFG, variant) - 1
        with pytest.raises(Exception) as err:
            seam_positions((too_short,), variant)
        assert type(err.value).__name__ in ("SubFloorInputError", "ConfigurationError")


SPEC_NOZERO = TdtSpec(durations=(1, 2), blank_index=0)


def test_the_seam_is_consulted_before_the_recursion_runs():
    """Ordering is measured, not claimed: the same input answers with two different exception types.

    ``T = 1`` with ``U = 2`` and no zero-duration token has no admissible path, so without the seam
    this input is refused by the lattice as impossible. With the seam it is refused as a contract
    breach instead - which only happens if the comparison is consulted first. A check moved after
    the recursion reproduces the first error and fails this node.
    """
    positions = seam_positions((16000,), TWO_STAGES_NO_PADDING)
    with pytest.raises(TdtAlignmentImpossible):
        ref_loss([rows_for(0, 1)], [list(TARGET)], [[1, U]], SPEC_NOZERO)
    with pytest.raises(TdtContractError) as err:
        ref_loss([rows_for(0, 1)], [list(TARGET)], [[1, U]], SPEC_NOZERO,
                 expected_positions=positions)
    assert err.value.details == {"batch": 0, "declared_T": 1, "expected_T": positions[0]}


@pytest.mark.parametrize("variant", SUBSAMPLE_VARIANTS, ids=VARIANT_IDS)
def test_the_torch_transcription_also_refuses_before_it_recurses(variant):
    """The same ordering, in the autograd transcription: contract error, not impossibility."""
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    positions = seam_positions((16000,), variant)
    tensor = torch.tensor([[[[value(0, i, u, w) for w in range(WIDTH)] for u in range(U + 1)]
                            for i in range(1)]], dtype=torch.float64)
    targets = torch.tensor([TARGET], dtype=torch.int64)
    lengths = torch.tensor([[1, U]], dtype=torch.int64)
    with pytest.raises(TdtAlignmentImpossible):
        tdt_loss_module.tdt_loss(tensor, targets, lengths, SPEC_NOZERO)
    with pytest.raises(TdtContractError):
        tdt_loss_module.tdt_loss(tensor, targets, lengths, SPEC_NOZERO,
                                 expected_positions=positions)


# ------------------------------------------------------------ 4. both implementations, one refusal

@pytest.mark.parametrize("case", ["under", "over-in-axis", "wrong-frontend", "bool"])
def test_the_torch_transcription_refuses_the_same_seam_cases(case):
    """Same exception type, same detail fields, and the refusal is the seam's in both paths.

    Each case is built so that every pre-existing guard passes first: the rows are at least as
    many as the declared ``T``, and the declared ``T`` is at least one. What is left to refuse it
    is the frontend-derived length.
    """
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    frozen = seam_positions((16000,), TWO_STAGES_NO_PADDING)[0]
    parity = position_lengths_from_audio((16000,), frontend=PARITY, variant=TWO_STAGES_NO_PADDING)[0]
    table = {
        "under": (frozen, frozen - 2, (frozen,), {"batch": 0, "declared_T": frozen - 2,
                                                  "expected_T": frozen}),
        "over-in-axis": (frozen + 4, frozen + 4, (frozen,),
                         {"batch": 0, "declared_T": frozen + 4, "expected_T": frozen}),
        "wrong-frontend": (parity, parity, (frozen,),
                           {"batch": 0, "declared_T": parity, "expected_T": frozen}),
        "bool": (frozen, frozen, (True,), {"batch": 0, "expected_positions": True}),
    }
    rows, declared, expected, want_details = table[case]
    tensor = torch.tensor([[[[value(0, i, u, w) for w in range(WIDTH)] for u in range(U + 1)]
                            for i in range(rows)]], dtype=torch.float64)
    targets = torch.tensor([TARGET], dtype=torch.int64)
    lengths = torch.tensor([[declared, U]], dtype=torch.int64)
    with pytest.raises(TdtContractError) as torch_err:
        tdt_loss_module.tdt_loss(tensor, targets, lengths, SPEC, expected_positions=expected)
    with pytest.raises(TdtContractError) as ref_err:
        ref_loss([rows_for(0, rows)], [list(TARGET)], [[declared, U]], SPEC,
                 expected_positions=expected)
    assert type(torch_err.value) is type(ref_err.value)
    assert torch_err.value.details == want_details, torch_err.value.details
    assert ref_err.value.details == want_details, ref_err.value.details


def test_the_refusal_names_the_row_that_disagrees_and_not_just_the_first_one():
    """A padded two-row batch where row 0 is honest and row 1 is not.

    Both rows are supplied to the axis the batch actually pads to, so every pre-existing guard
    passes for both, and the loss scores both rows happily - which is why the ``batch`` index in the
    refusal is the thing worth pinning: a checker that reported row 0 would send the caller to fix
    the wrong utterance.
    """
    positions = seam_positions((16000, 16000), TWO_STAGES_NO_PADDING)
    axis = positions[0] + 5
    logits = [rows_for(0, axis), rows_for(1, axis)]
    lengths = [[positions[0], U], [axis, U]]
    scored = ref_loss(logits, [list(TARGET)] * 2, lengths, SPEC)
    assert len(scored) == 2 and scored[0] != scored[1], scored
    with pytest.raises(TdtContractError) as err:
        ref_loss(logits, [list(TARGET)] * 2, lengths, SPEC, expected_positions=positions)
    assert err.value.details == {"batch": 1, "declared_T": axis, "expected_T": positions[0]}


def test_the_torch_transcription_names_the_same_row_in_a_padded_batch():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    positions = seam_positions((16000, 16000), TWO_STAGES_NO_PADDING)
    axis = positions[0] + 5
    tensor = torch.tensor([[[[value(b, i, u, w) for w in range(WIDTH)] for u in range(U + 1)]
                            for i in range(axis)] for b in (0, 1)], dtype=torch.float64)
    targets = torch.tensor([TARGET, TARGET], dtype=torch.int64)
    lengths = torch.tensor([[positions[0], U], [axis, U]], dtype=torch.int64)
    with pytest.raises(TdtContractError) as torch_err:
        tdt_loss_module.tdt_loss(tensor, targets, lengths, SPEC, expected_positions=positions)
    with pytest.raises(TdtContractError) as ref_err:
        ref_loss([rows_for(0, axis), rows_for(1, axis)], [list(TARGET)] * 2,
                 [[positions[0], U], [axis, U]], SPEC, expected_positions=positions)
    assert torch_err.value.details == ref_err.value.details == {
        "batch": 1, "declared_T": axis, "expected_T": positions[0]}


def test_an_iterable_of_python_ints_is_the_accepted_input_type():
    """The documented contract, tested: any iterable of ints is accepted, nothing is coerced.

    A ``tuple`` is what ``position_lengths_from_audio`` returns; ``list``, ``range`` and a generator
    are the other spellings a caller can reach for without importing a tensor library. A float or a
    tensor element is refused rather than rounded, and text is refused rather than iterated.
    """
    positions = seam_positions((16000,), TWO_STAGES_NO_PADDING)
    logits = [rows_for(0, positions[0])]
    lengths = [[positions[0], U]]
    for container in (tuple(positions), list(positions), range(positions[0], positions[0] + 1),
                      (x for x in positions)):
        assert ref_loss(logits, [list(TARGET)], lengths, SPEC,
                        expected_positions=container) == ref_loss(
            logits, [list(TARGET)], lengths, SPEC), container
    with pytest.raises(TdtContractError):
        ref_loss(logits, [list(TARGET)], lengths, SPEC, expected_positions=positions[0])
    with pytest.raises(TdtContractError):
        ref_loss(logits, [list(TARGET)], lengths, SPEC,
                 expected_positions="2" * positions[0])


def test_the_torch_transcription_accepts_the_seam_derived_length_unchanged():
    torch = pytest.importorskip("torch", reason=_NO_TORCH)
    positions = seam_positions((16000,), TWO_STAGES_NO_PADDING)
    tensor = torch.tensor([[[[value(0, i, u, w) for w in range(WIDTH)] for u in range(U + 1)]
                            for i in range(positions[0])]], dtype=torch.float64)
    targets = torch.tensor([TARGET], dtype=torch.int64)
    lengths = torch.tensor([[positions[0], U]], dtype=torch.int64)
    plain = tdt_loss_module.tdt_loss(tensor, targets, lengths, SPEC)
    seamed = tdt_loss_module.tdt_loss(tensor, targets, lengths, SPEC,
                                      expected_positions=positions)
    reference = ref_loss([rows_for(0, positions[0])], [list(TARGET)], [[positions[0], U]], SPEC)
    assert float(torch.abs(plain - seamed).max()) == 0.0
    assert abs(float(seamed[0]) - reference[0]) <= 1e-9, (float(seamed[0]), reference[0])


def test_omitting_the_seam_is_backwards_compatible_by_design():
    """Documented as a migration state: the parameter is keyword-only and defaults to off."""
    positions = seam_positions((16000,), TWO_STAGES_PAD_ONE)
    logits = [rows_for(0, positions[0])]
    lengths = [[positions[0], U]]
    assert ref_loss(logits, [list(TARGET)], lengths, SPEC) == ref_loss(
        logits, [list(TARGET)], lengths, SPEC, expected_positions=None)
    for fn in (tdt_lattice.loss, tdt_loss_module.tdt_loss):
        param = inspect.signature(fn).parameters["expected_positions"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, fn.__name__
        assert param.default is None, fn.__name__


# ----------------------------------------------------------------- 5. guards against a bypass

#: The two, and only two, entry points that score a lattice.
_SEAM_GUARDED = {
    "nativevoice.training.tdt_lattice.loss",
    "nativevoice.training.tdt_loss.tdt_loss",
}
#: Every public callable in the two scoring modules that reads a ``lengths`` argument. Two of them
#: score and both carry the seam; ``validate`` is the one seam-free member, and it is named here
#: rather than hidden, because a caller who composes ``validate`` with ``alpha_table`` by hand
#: reaches the lattice around this guard. That composition is what the AST sweep of ``src/`` and
#: ``tools/`` below is for - it covers call sites in this repository, not a hand-rolled recursion.
#: A third scorer, or a renamed one, breaks the first assertion instead of quietly becoming a way
#: to skip the seam.
_LENGTHS_SURFACE = {
    "nativevoice.training.tdt_lattice.check_expected_positions",
    "nativevoice.training.tdt_lattice.loss",
    "nativevoice.training.tdt_lattice.validate",
    "nativevoice.training.tdt_loss.check_expected_positions",
    "nativevoice.training.tdt_loss.tdt_loss",
}
_SEAM_FREE = {"nativevoice.training.tdt_lattice.validate"}


def test_the_public_scoring_surface_is_exactly_two_seam_guarded_callables():
    takes_lengths, without_the_seam = set(), set()
    for module in (tdt_lattice, tdt_loss_module):
        for name, obj in vars(module).items():
            if name.startswith("_") or not callable(obj):
                continue
            try:
                parameters = inspect.signature(obj).parameters
            except (TypeError, ValueError):
                continue
            if "lengths" not in parameters:
                continue
            full = f"{module.__name__}.{name}"
            takes_lengths.add(full)
            if "expected_positions" not in parameters:
                without_the_seam.add(full)
    assert _SEAM_GUARDED <= takes_lengths, takes_lengths
    assert takes_lengths == _LENGTHS_SURFACE, takes_lengths
    assert without_the_seam == _SEAM_FREE, without_the_seam


_SCORERS = ("loss", "tdt_loss")
_SRC = ROOT / "src"


def _dotted(node):
    """``a.b.c`` as a list, or ``None`` when the callee is not a plain dotted path."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return list(reversed(parts))


def _is_module(dotted):
    """Whether a dotted name under ``src/`` is a module or package rather than a function."""
    path = dotted.replace(".", "/")
    return (_SRC / f"{path}.py").exists() or (_SRC / path / "__init__.py").exists()


def _scoring_import_map(tree):
    """A file's scorer bindings and its module bindings, kept apart.

    ``nativevoice.training.tdt_loss`` is both a module and, inside it, a function of the same name,
    so ``from nativevoice.training import tdt_loss`` binds the module while
    ``from nativevoice.training.tdt_lattice import loss`` binds a scorer. Resolving each dotted
    candidate against the tree is what stops the guard from reading one spelling as the other, and
    from treating ``validate(...)``, ``TdtSpec(...)`` or any other imported name as a scorer.
    """
    direct, modules, starred = {}, {}, False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            parent = node.module or ""
            if not parent.startswith("nativevoice.training"):
                continue
            for alias in node.names:
                if alias.name == "*":
                    starred = True
                    continue
                dotted = f"{parent}.{alias.name}"
                handle = alias.asname or alias.name
                if _is_module(dotted):
                    modules[handle] = dotted
                elif alias.name in _SCORERS:
                    direct[handle] = dotted
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nativevoice.training") and _is_module(alias.name):
                    modules[alias.name] = alias.name
                    modules[alias.asname or alias.name.rsplit(".", 1)[-1]] = alias.name
    return direct, modules, starred


def unguarded_scoring_calls(paths):
    """Public scoring calls in ``paths`` that omit ``expected_positions``."""
    offenders = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        direct, modules, starred = _scoring_import_map(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            parts = _dotted(node.func)
            if parts is None:
                continue
            called = None
            if len(parts) == 1:
                called = direct.get(parts[0])
                if called is None and starred and parts[0] in _SCORERS:
                    called = f"nativevoice.training.<star-imported>.{parts[0]}"
            else:
                head = ".".join(parts[:-1])
                if parts[-1] in _SCORERS and head in modules:
                    called = f"{modules[head]}.{parts[-1]}"
            if called and not any(k.arg == "expected_positions" for k in node.keywords):
                offenders.append((str(path), node.lineno, called))
    return offenders


def _production_sources():
    return sorted((ROOT / "src" / "nativevoice").rglob("*.py")) + sorted(
        (ROOT / "tools").glob("*.py"))


def test_no_production_scoring_call_omits_the_seam():
    """Vacuous today by construction and stated as such: there is no production scoring site yet.

    It becomes a real guard the moment a training or inference path appears, which is exactly when
    the optional default has to be revisited.
    """
    assert unguarded_scoring_calls(_production_sources()) == []


def test_the_ast_guard_reports_a_synthetic_unguarded_call(tmp_path):
    """The verifier, verified: a call that skips the seam must be found by the guard."""
    bare = tmp_path / "bare.py"
    bare.write_text(
        "from nativevoice.training.tdt_lattice import loss\n"
        "loss(logits, targets, lengths)\n", encoding="utf-8")
    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        "from nativevoice.training.tdt_lattice import loss\n"
        "loss(logits, targets, lengths, expected_positions=positions)\n", encoding="utf-8")
    aliased = tmp_path / "aliased.py"
    aliased.write_text(
        "from nativevoice.training import tdt_loss as scoring\n"
        "scoring.tdt_loss(logits, targets, lengths, spec)\n", encoding="utf-8")
    module_import = tmp_path / "module_import.py"
    module_import.write_text(
        "from nativevoice.training import tdt_lattice\n"
        "tdt_lattice.loss(logits, targets, lengths)\n", encoding="utf-8")
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text(
        "def loss(a, b, c):\n"
        "    return a\n"
        "loss(logits, targets, lengths)\n", encoding="utf-8")
    starred = tmp_path / "starred.py"
    starred.write_text(
        "from nativevoice.training.tdt_lattice import *\n"
        "loss(logits, targets, lengths)\n", encoding="utf-8")
    helper = tmp_path / "helper.py"
    helper.write_text(
        "from nativevoice.training.tdt_lattice import validate\n"
        "validate(logits, targets, lengths, spec)\n", encoding="utf-8")
    module_guarded = tmp_path / "module_guarded.py"
    module_guarded.write_text(
        "from nativevoice.training import tdt_lattice\n"
        "tdt_lattice.loss(logits, targets, lengths, expected_positions=positions)\n",
        encoding="utf-8")
    assert len(unguarded_scoring_calls([bare])) == 1
    assert unguarded_scoring_calls([guarded]) == []
    assert len(unguarded_scoring_calls([aliased])) == 1, "an aliased scorer must not slip through"
    assert len(unguarded_scoring_calls([module_import])) == 1
    assert len(unguarded_scoring_calls([starred])) == 1
    assert unguarded_scoring_calls([unrelated]) == [], "a local function is not the scorer"
    assert unguarded_scoring_calls([helper]) == [], "a helper call is not a scorer call"
    assert unguarded_scoring_calls([module_guarded]) == []


def test_the_seam_checker_is_shared_not_duplicated():
    """One implementation of the comparison, reached from both scoring functions."""
    assert tdt_loss_module.check_expected_positions is tdt_lattice.check_expected_positions
    with pytest.raises(TdtContractError) as err:
        tdt_lattice.check_expected_positions([[9, 2]], (4,), batch=1)
    assert err.value.details == {"batch": 0, "declared_T": 9, "expected_T": 4}


def test_the_seam_touches_no_arithmetic():
    """The checker stands in front of the recursion; it adds no term and changes no cell.

    ``expected_positions`` appears in neither :func:`validate` nor :func:`alpha_table`, and every
    mutation flag that used to move the number still moves it by the same amount, which is the
    observable form of "the mathematics is untouched".
    """
    positions = seam_positions((16000,), ONE_STAGE_NO_PADDING)
    logits = [rows_for(0, positions[0])]
    lengths = [[positions[0], U]]
    baseline = ref_loss(logits, [list(TARGET)], lengths, SPEC, expected_positions=positions)
    for fn in (tdt_lattice.validate, tdt_lattice.alpha_table, tdt_lattice.is_feasible):
        assert "expected_positions" not in inspect.signature(fn).parameters, fn.__name__
    for flag, value in (("mutate_token_requires_positive", True),
                        ("mutate_forbid_closing_blank", True),
                        ("mutate_only_blank_closes", True),
                        ("mutate_duration_index_offset", 1)):
        moved = ref_loss(logits, [list(TARGET)], lengths, SPEC, **{flag: value})
        assert moved != baseline, (flag, moved, baseline)
    # `mutate_allow_blank_zero` is deliberately not probed here: a zero-duration blank is a self-loop
    # that the sweep cannot admit, so it changes no number and is not a control. That is asserted
    # where it belongs - in tests/test_tdt_lattice.py (exact rationals) and
    # tests/test_tdt_lattice_reference.py (path-count growth), not as a no-op next to live flags.
    assert tdt_lattice.is_feasible(0, 0, SPEC) is True
    with pytest.raises(TdtContractError):
        tdt_lattice.validate([[]], [[]], [[0, 0]], SPEC)
