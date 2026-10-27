"""Per-row length and mask-pattern contract for padded batches. Pure standard library.

What this module owns is the arithmetic that D-A asked for: valid lengths come from the frontend
and the named subsampling variant, never from a tensor's shape. That distinction is measured, not
rhetorical - a batch padded to the 30 s ceiling holds 748 positions for every row while the row
that carries one second of audio has 23, so ``states.shape[1]`` is the wrong authority by 725
positions. :class:`nativevoice.modeling.encoder.Student` calls these functions to build its masks
and returns the per-row ``T_b`` from ``forward`` alongside the padded axis, so neither length can be
mistaken for the other.

What this module deliberately is not:

* no application of masks to a network. The input-grid zeroing, the subsampling stem's between-stage
  zeroing, the attention key mask and the convolution-module mask live in
  :mod:`nativevoice.modeling.encoder` and are measured in ``tests/test_modeling_encoder_masks.py``;
  this file stays torch-free so the patterns can be tested in the project environment. Under the
  provisionally chosen ``layer_norm`` direction the geometry is inert for all four subsampling
  candidates (measured in this fixture: ``5.6e-16`` to ``1.4e-15`` in float64 for both a longer padding
  bucket and a different batch partner - the arithmetic floor, three orders inside the accepted
  tolerance - while two identical rows inside one batch are bit-identical).
  ``batch_norm``, which stays selectable, is not: it pools over the length axis at every normalisation
  site, so pad columns and other rows both enter its statistics, and an unmasked padded batch moves
  valid positions by up to ``5.3e-01`` under layer normalisation and ``3.3e+00`` under batch
  normalisation. Neither figure is a claim about speech, training or production readiness.
* no torch, no numpy, no tensors: masks are immutable boolean patterns, which is the only form
  that can be tested in the project environment, where torch is absent.
* no default subsampling variant, and no length may be omitted: ``variant`` is required by every
  function here, because the stage count and padding rule are still the owner's open decision
  (R1) and they change these numbers (the same second of audio yields 23, 25, 48 or 49 positions).
* no new error vocabulary: bad arguments raise ``ConfigurationError`` from
  ``nativevoice.core.errors``; a frame count too short for a stack without padding propagates
  ``UndefinedAxisError`` from :func:`nativevoice.modeling.lengths.stack_length`, and audio below a
  variant's floor propagates ``SubFloorInputError`` from :func:`nativevoice.modeling.lengths.bridge`.
  Where a length cannot produce a position, this module refuses; it never reports a silent zero.
"""

from __future__ import annotations

from typing import Sequence

from nativevoice.core.errors import ConfigurationError
from nativevoice.modeling.lengths import StackSpec, bridge, stack_length

__all__ = ["position_lengths", "position_lengths_from_audio", "valid_frame_mask",
           "valid_position_mask"]


def _as_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer", **{name: value})
    return value


def _sequence(values, name: str) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ConfigurationError(f"{name} must be a sequence of integers, not text", **{name: values})
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ConfigurationError(f"{name} must be a sequence of integers", **{name: str(values)}) from exc
    if not items:
        raise ConfigurationError(f"{name} must not be empty", name=name)
    return items


def _require_variant(variant) -> None:
    if not isinstance(variant, StackSpec):
        raise ConfigurationError(
            "a subsampling variant is required and there is no default: the stage count and "
            "padding rule are an open architectural decision",
            variant=str(variant))


def position_lengths(frame_counts: Sequence[int], *, variant: StackSpec) -> tuple[int, ...]:
    """Encoder output positions per row, from each row's own frame count.

    Delegates to :func:`nativevoice.modeling.lengths.stack_length`, so the arithmetic is shared
    with the frozen bridge rather than restated. A frame count too short for the named stack
    raises rather than yielding ``0``.
    """
    _require_variant(variant)
    out = []
    for index, frames in enumerate(_sequence(frame_counts, "frame_counts")):
        frames = _as_int(frames, f"frame_counts[{index}]")
        if frames < 0:
            raise ConfigurationError("frame counts must be non-negative",
                                     index=index, frame_counts=frames)
        out.append(stack_length(frames, variant=variant))
    return tuple(out)


def position_lengths_from_audio(n_samples: Sequence[int], *, frontend, variant: StackSpec) -> tuple[int, ...]:
    """The same lengths, taken from sample counts through the frozen frontend's own arithmetic.

    This is the single path that keeps the frontend as the authority for frame counts: the bridge
    supplies both the frames and the positions, and audio too short to produce a position is
    refused with the sample count in the message.
    """
    _require_variant(variant)
    out = []
    for index, n in enumerate(_sequence(n_samples, "n_samples")):
        n = _as_int(n, f"n_samples[{index}]")
        out.append(bridge(n, frontend=frontend, variant=variant).positions)
    return tuple(out)


def _mask(total: int, valid: int, *, total_name: str, label: str) -> tuple[bool, ...]:
    total = _as_int(total, total_name)
    valid = _as_int(valid, f"{total_name}[{label}]")
    if total < 1:
        raise ConfigurationError(f"{total_name} must be at least 1", **{total_name: total})
    if valid < 1:
        raise ConfigurationError(
            "every padded row must declare at least one valid position; a row of length zero is "
            "not a maskable row", **{total_name: total, "valid": valid, "row": label})
    if valid > total:
        raise ConfigurationError("a row's valid length exceeds the padded dimension it must fit in",
                                 **{total_name: total, "valid": valid, "row": label})
    return (True,) * valid + (False,) * (total - valid)


def valid_position_mask(total_positions: int, per_row_positions: Sequence[int]) -> tuple[tuple[bool, ...], ...]:
    """Boolean pattern per row over a padded position axis: ``True`` where the row is real.

    ``total_positions`` is the padded axis length the model actually produced, which for a
    right-padded batch equals the longest row's length; it is a dimension the caller states, never
    a value inferred from a tensor here.
    """
    rows = _sequence(per_row_positions, "per_row_positions")
    return tuple(_mask(total_positions, row, total_name="total_positions", label=str(i))
                 for i, row in enumerate(rows))


def valid_frame_mask(total_frames: int, per_row_frames: Sequence[int]) -> tuple[tuple[bool, ...], ...]:
    """The same pattern on the frame axis, for callers that pad before the subsampling stack."""
    rows = _sequence(per_row_frames, "per_row_frames")
    return tuple(_mask(total_frames, row, total_name="total_frames", label=str(i))
                 for i, row in enumerate(rows))
