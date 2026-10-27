"""Integer length arithmetic between the frozen log-mel frontend and the TDT lattice.

Standard library only. No tensor library, no feature values, no parameters: every function
here maps integers to integers, which is what makes it safe to write before the student's
architecture is ruled.

The convention under test is the *proposed* subsampling stack - repeated 3-tap, stride-2
convolutions over a (frames x mel) grid. Four candidates are named explicitly and none is a
default, because the stage count and the padding choice are open rulings:

======================  ================  ================  ==========  ==============
variant                 T for 1 s (98 fr) T for 30 s (2998) mel axis    floor (samples)
======================  ================  ================  ==========  ==============
one stage, no padding   48                1498              39          720  (45.0 ms)
one stage, padding 1    49                1499              40          400  (25.0 ms)
two stages, no padding  23                748               19          1360 (85.0 ms)
two stages, padding 1   25                750               20          400  (25.0 ms)
======================  ================  ================  ==========  ==============

Two measured facts shape this file. First, with no padding an axis shorter than the kernel is
not a zero-length output but an *undefined* operation - a real convolution refuses it - so the
floor of each variant is reported as a refusal naming the caller's sample count rather than as
a silent ``0``. Second, padding does not change the rate at which positions appear (40 ms per
position after two stages, 20 ms after one, in both paddings); it changes the offset and the
floor, and the padded/unpadded gap varies with the input, so the two conventions cannot be
converted into one another by a shift.

Provisional contract, and its limit: ``declared T == encoder output position count``. A1 is
**specified and provisionally ruled** for this implementation (2026-10-27: positions are the unit of
both ``T`` and the durations, the terminal node is virtual, ``sum D == T`` exactly, blanks exclude
duration 0 and tokens include it) and it **remains unresolved against the paper** - the source never
states what ``T`` counts, and a frame-indexed ``T`` with the durations scaled by the stack's stride is
provably isomorphic to this reading, so no lattice test can choose between them. Extra trailing logit
rows are inert, so nothing inside the loss can detect a mis-declared ``T``: the convention is now
compared inside both scorers by
:func:`nativevoice.training.tdt_lattice.check_expected_positions` when a caller supplies
``expected_positions`` - optional, so this is caller-conditional enforcement, not a guarantee - and
:meth:`LengthBridge.check_seam` stays the per-utterance tripwire for a caller that already holds a
bridge. Both enforce *our* convention; neither proves the paper's terminal semantics.

Resolved defect, recorded because it shaped this file: at ``T == 0`` the stdlib reference and the
torch transcription used to disagree. The reference returned ``-0.0`` for an empty target, raised
``TdtAlignmentImpossible`` for an unreachable one, and raised an uncaught ``IndexError`` exactly
when ``0 in D`` and ``U == 1`` - the sentinel rule makes every intermediate cell inadmissible at
``T == 0``, so ``(0, 1)`` was the only cell that could reach for a row ``validate`` had never
required to exist. The torch path raised ``TdtContractError`` for all of them. As of 2026-10-26 the
reference refuses ``T == 0`` outright, for every target length and duration set, so the two
implementations agree; :func:`nativevoice.training.tdt_lattice.is_feasible` still answers ``True``
for ``(0, 0)``, on purpose, because it is a geometry predicate that never sees a logits tensor.
This bridge independently refuses sub-floor audio, naming the caller's sample count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from nativevoice.core.errors import ConfigurationError, VoiceError

#: The interface convention this package enforces, and the assumption it cannot settle.
PROVISIONAL_CONTRACT = (
    "declared T == encoder output position count (provisional interface convention; A1 is specified "
    "and provisionally RULED for this implementation on 2026-10-27, and A1 remains unresolved "
    "against the paper)"
)

#: Decisions this package deliberately leaves open rather than absorbs.
KNOWN_OPEN_DEFECTS = (
    "R1 subsampling stage count and padding not ruled (four candidates represented, none default); "
    "A1 clause S4 - every admissible input needs complete real-frame coverage on the frontend time "
    "axis and at the frozen mel width, ruled 2026-10-27 - makes the two padding=0 stacks "
    "ineligible because their coverage is conditional (one stage: only odd frame counts; two "
    "stages: only counts that are 3 mod 4), while both padding=1 stacks cover every admissible "
    "frame count. Boundary zero-padding is a separate, permitted property, not coverage. "
    "Authoritative record: docs/asr-architecture.md section 7.2. One stage padding=1 and two "
    "stages padding=1 remain in competition and neither is preferred",
    "R4 feature dtype conversion and normalisation intentionally outside this bridge",
    "model capacity, vocabulary, blank id and duration configuration not chosen here",
    "A1 (what the source's T counts) is specified and provisionally ruled as this package's "
    "implementation contract; it remains unresolved against the paper, and the length seam stays "
    "provisional",
)

#: Behaviour that used to be wrong, is now fixed, and is pinned by tests rather than by prose.
RESOLVED_DEFECTS = (
    "T=0 divergence between nativevoice.training.tdt_lattice and nativevoice.training.tdt_loss "
    "resolved on 2026-10-26: validate now refuses T=0 for every U and every duration set, "
    "matching the torch guard. Crash cases were U=1 with 0 in D; silent success was U=0.",
)


class SubFloorInputError(VoiceError):
    """The supplied audio cannot produce a single encoder position for this variant."""


class UndefinedAxisError(VoiceError):
    """An axis shorter than the kernel with no padding: the operation is undefined, not empty."""


class LengthSeamError(VoiceError):
    """A caller declared a lattice length that the bridge does not agree the encoder produced."""


class FrameFrontend(Protocol):
    """The part of the frozen frontend this bridge may see.

    Deliberately structural, and deliberately narrow: a count of frames, plus the geometry
    needed to express that count in samples and milliseconds. `FeatureGrid` and its values are
    not reachable through this protocol, which is how "no dtype or normalisation decision" is a
    property of the code rather than a promise in a comment.
    """

    sample_rate: int
    hop_samples: int
    window_samples: int

    def frame_count(self, n_samples: int) -> int: ...


@dataclass(frozen=True, slots=True)
class StackSpec:
    """A named subsampling configuration. Required wherever a length is computed."""

    label: str
    stages: int
    kernel: int
    stride: int
    padding: int

    def __post_init__(self) -> None:
        for name in ("stages", "kernel", "stride", "padding"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(f"{name} must be an integer", **{name: value})
        if self.stages < 1:
            raise ConfigurationError("a subsampling stack needs at least one stage", stages=self.stages)
        if self.kernel < 1:
            raise ConfigurationError("kernel must be positive", kernel=self.kernel)
        if self.stride < 1:
            raise ConfigurationError("stride must be positive", stride=self.stride)
        if self.padding < 0:
            raise ConfigurationError("padding must be non-negative", padding=self.padding)
        if not self.label:
            raise ConfigurationError("a stack configuration must be named so refusals can name it")

    @property
    def frames_per_position(self) -> int:
        """Frontend frames consumed by one output position: `stride ** stages`, padding-independent."""
        return self.stride ** self.stages

    def __str__(self) -> str:
        return (f"{self.label} (stages={self.stages}, kernel={self.kernel}, "
                f"stride={self.stride}, padding={self.padding})")


ONE_STAGE_NO_PADDING = StackSpec("one stage, no padding", stages=1, kernel=3, stride=2, padding=0)
ONE_STAGE_PAD_ONE = StackSpec("one stage, padding 1", stages=1, kernel=3, stride=2, padding=1)
TWO_STAGES_NO_PADDING = StackSpec("two stages, no padding", stages=2, kernel=3, stride=2, padding=0)
TWO_STAGES_PAD_ONE = StackSpec("two stages, padding 1", stages=2, kernel=3, stride=2, padding=1)

#: Every candidate the student architecture may still choose between. No entry is a default.
SUBSAMPLE_VARIANTS: tuple[StackSpec, ...] = (
    ONE_STAGE_NO_PADDING, ONE_STAGE_PAD_ONE, TWO_STAGES_NO_PADDING, TWO_STAGES_PAD_ONE,
)


def stage_length(axis: int, *, variant: StackSpec) -> int:
    """One strided convolution over one axis.

    ``variant`` has no default: a length means nothing without saying which candidate stack
    produced it, and that stack is an unresolved ruling rather than a settled convention.
    """
    if axis + 2 * variant.padding < variant.kernel:
        if axis == 0 and variant.padding > 0:
            return 0                      # padding cannot conjure an axis out of nothing
        raise UndefinedAxisError(
            "axis is shorter than the kernel with no padding, so the operation is undefined",
            axis=axis, kernel=variant.kernel, padding=variant.padding, variant=variant.label,
        )
    return (axis + 2 * variant.padding - variant.kernel) // variant.stride + 1


def stack_length(axis: int, *, variant: StackSpec) -> int:
    """The length of one axis after the whole named stack."""
    out = axis
    for _ in range(variant.stages):
        out = stage_length(out, variant=variant)
    return out


def minimum_frames(variant: StackSpec) -> int:
    """Smallest frame axis that yields at least one position, derived rather than searched.

    Walking the stack backwards: an output of ``need`` requires an input of at least
    ``stride * (need - 1) + kernel - 2 * padding`` per stage, floored at one.
    """
    need = 1
    for _ in range(variant.stages):
        need = max(1, variant.stride * (need - 1) + variant.kernel - 2 * variant.padding)
    return need


def minimum_samples(frontend: FrameFrontend, variant: StackSpec) -> int:
    """Smallest audio length that yields at least one encoder position."""
    return (minimum_frames(variant) - 1) * frontend.hop_samples + frontend.window_samples


def frequency_bins(mel_bins: int, *, variant: StackSpec) -> int:
    """The frequency axis after the same stack: the input width of the projection per position."""
    return stack_length(mel_bins, variant=variant)


@dataclass(frozen=True, slots=True)
class LengthBridge:
    """One utterance's journey from samples to the length a lattice must be told."""

    n_samples: int
    frames: int
    positions: int
    variant: StackSpec
    floor_samples: int
    samples_per_position: int
    ms_per_position: float

    def check_seam(self, declared_T: int) -> int:
        """Assert the provisional contract for one utterance and return the agreed length.

        A mismatch is the failure mode that is silent inside the loss - extra rows are inert and
        a wrong ``T`` still yields a finite, plausible number - so it is refused here, with both
        lengths in the message.
        """
        if declared_T != self.positions:
            raise LengthSeamError(
                "declared lattice length does not match the encoder positions the frontend implies",
                declared_T=declared_T, encoder_positions=self.positions,
                frames=self.frames, n_samples=self.n_samples, variant=self.variant.label,
            )
        return self.positions


def bridge(n_samples: int, *, frontend: FrameFrontend, variant: StackSpec) -> LengthBridge:
    """Compute the encoder output length for ``n_samples`` under the named variant.

    Refuses audio that cannot produce a position, naming the caller's sample count, the frames
    that count produced, the variant in play and the floor it fell short of.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ConfigurationError("n_samples must be an integer count of samples", n_samples=n_samples)
    if n_samples < 0:
        raise ConfigurationError("n_samples cannot be negative", n_samples=n_samples)
    frames = frontend.frame_count(n_samples)
    need = minimum_frames(variant)
    floor = minimum_samples(frontend, variant)
    if frames < need:
        raise SubFloorInputError(
            "this audio produces no encoder position for the named subsampling variant",
            n_samples=n_samples, frames=frames, variant=variant.label,
            minimum_samples=floor, minimum_frames=need,
        )
    hop = frontend.hop_samples
    step = hop * variant.frames_per_position
    return LengthBridge(
        n_samples=n_samples, frames=frames, positions=stack_length(frames, variant=variant),
        variant=variant, floor_samples=floor, samples_per_position=step,
        ms_per_position=1000.0 * step / frontend.sample_rate,
    )
