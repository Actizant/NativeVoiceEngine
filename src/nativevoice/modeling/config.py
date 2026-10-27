"""Explicit configuration for the Conformer-TDT student shape skeleton.

Every field is required and there is no default configuration anywhere in this module: a value
that has a default is a decision nobody made. The label field exists for the same reason - it
forces whoever constructs a config to say what it is, and the test fixtures that use it say
"test fixture", not "proposal".

Still open, and therefore still required arguments rather than constants: model capacity
(``d_model``, ``block_count``, heads, feed-forward multiplier), the attention variant, the
normalisation placement (the normalisation *kind* carries a provisional ruling, recorded in
``OPEN_ARCHITECTURE_DECISIONS``, and it is still a required field - a direction is not a
default), the convolution-module activation, the prediction-network form, the
vocabulary and blank id, the duration set, and the subsampling stack (``StackSpec``, four
candidates, none selected). The joint TDT emission head, terminal-state semantics and the
``T == 0`` contract live in ``nativevoice.training`` and are not restated here.

Nothing in this package has been trained. A forward pass through a random initialisation proves
shape arithmetic, finiteness and gradient reachability; it says nothing about recognition.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Tuple

from nativevoice.core.errors import ConfigurationError
from nativevoice.modeling.lengths import StackSpec

#: Choices that are enumerable but NOT ruled. Each is required in the config, so the caller
#: states which candidate it is using instead of inheriting one from a library default.
NORMALIZATIONS = ("layer_norm", "batch_norm")
POSITION_ENCODINGS = ("none", "relative_shaw_windowed")
MACARON_MODES = ("macaron", "single")
CONV_ACTIVATIONS = ("swish", "glu")
PREDICTION_KINDS = ("embedding_only", "lstm")

#: Architecture questions this skeleton deliberately leaves open. Kept importable so tests can
#: assert the list has not quietly shrunk into a set of "decisions".
OPEN_ARCHITECTURE_DECISIONS = (
    "positional information: none vs relative (Shaw-windowed here) vs rotary/DeKERF - not ruled",
    "normalisation kind is PROVISIONALLY RULED to layer_norm - a statistic over the channel "
    "axis at each position, never pooled across the batch or across time; batch_norm stays "
    "enumerated and remains measurable, its padding and composition failures are pinned by "
    "tests; normalisation placement and the pre-norm residual order are NOT ruled",
    "convolution-module activation: swish vs glu, not ruled",
    "depthwise same-length padding (kernel-1)//2 is a shape requirement of the block, not a "
    "ruling on the subsampling stem's padding",
    "prediction network form: embedding only vs LSTM stack, not ruled",
    "macaron half-step scale 0.5 is provisional; the residual order is provisional",
    "capacity: d_model, block_count, heads, feed-forward multiplier, joint and prediction dims",
    "vocabulary size, blank id and start symbol: no vocabulary exists in this repository",
    "duration set D and Nd: the reference default (0,1,2) is a fixture, not an approval",
    "A1: specified and provisionally RULED as this implementation's contract (clauses S1-S3, S5-S8: "
    "T and T_b count encoder output positions, durations share that unit, the terminal node is "
    "virtual and row T reads no logit, sum D == T exactly, blanks exclude duration 0 and tokens "
    "include it); what the source's T counts remains unresolved against the paper, so the length "
    "seam stays provisional and expected_positions stays an optional, caller-conditional guard",
    "batching along the frame axis is inert for every subsampling candidate under the "
    "provisional layer_norm direction (measured 5.6e-16 to 1.4e-15, the float64 arithmetic "
    "floor, three orders inside the accepted tolerance) once the stem, attention, convolution "
    "and grid masks are applied; "
    "batch_norm would instead read the pad columns and the other rows at all ten sites and "
    "a padded batch would write running statistics a clean row later reads (measured 1.9e-1). "
    "R1 - stage count and padding - is untouched by any of this: the masks are variant-agnostic "
    "and no candidate is preferred. A1 clause S4 (ruled 2026-10-27; authoritative record "
    "docs/asr-architecture.md section 7.2) prunes that field on arithmetic rather than preference: "
    "complete real-frame coverage is required for every admissible frame count, so the padding=0 "
    "stacks are excluded - one stage covers only odd counts, two stages only counts that are "
    "3 mod 4 - while both padding=1 stacks cover every admissible length. Frequency-axis coverage "
    "at the frozen mel width, and the boundary zero-padding the padding=1 stacks are permitted to "
    "use, are separate properties defined there. The two padding=1 stacks stay eligible, and "
    "between them no candidate is preferred",
)


def _require_int(value, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer", **{name: value})
    if value < minimum:
        raise ConfigurationError(f"{name} must be at least {minimum}", **{name: value})
    return value


@dataclass(frozen=True, slots=True)
class StudentConfig:
    """One fully explicit description of a student shape. No field has a default."""

    label: str
    d_model: int
    attention_heads: int
    feed_forward_mult: int
    conv_kernel_size: int
    block_count: int
    dropout: float
    normalization: str
    position_encoding: str
    relative_window: int
    macaron: str
    conv_activation: str
    mel_bins: int
    subsample: StackSpec
    vocab_size: int
    blank_index: int
    durations: Tuple[int, ...]
    prediction_dim: int
    prediction_kind: str
    prediction_layers: int
    prediction_start_symbol: int
    joint_dim: int

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ConfigurationError("a config must be labelled so its status is never ambiguous")
        _require_int(self.d_model, "d_model", minimum=1)
        _require_int(self.attention_heads, "attention_heads", minimum=1)
        if self.d_model % self.attention_heads:
            raise ConfigurationError(
                "d_model must divide evenly into attention heads",
                d_model=self.d_model, attention_heads=self.attention_heads)
        _require_int(self.feed_forward_mult, "feed_forward_mult", minimum=1)
        _require_int(self.conv_kernel_size, "conv_kernel_size", minimum=1)
        if self.conv_kernel_size % 2 == 0:
            raise ConfigurationError(
                "the depthwise convolution uses (kernel-1)//2 padding to preserve length, "
                "which needs an odd kernel", conv_kernel_size=self.conv_kernel_size)
        _require_int(self.block_count, "block_count", minimum=1)
        _require_int(self.mel_bins, "mel_bins", minimum=1)
        _require_int(self.joint_dim, "joint_dim", minimum=1)
        _require_int(self.prediction_dim, "prediction_dim", minimum=1)
        _require_int(self.prediction_layers, "prediction_layers", minimum=0)
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise ConfigurationError("dropout must be a number", dropout=self.dropout)
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ConfigurationError("dropout must be in [0, 1)", dropout=self.dropout)

        if not isinstance(self.subsample, StackSpec):
            raise ConfigurationError("subsample must be an explicit StackSpec", subsample=self.subsample)
        for name, allowed in (("normalization", NORMALIZATIONS), ("position_encoding", POSITION_ENCODINGS),
                              ("macaron", MACARON_MODES), ("conv_activation", CONV_ACTIVATIONS),
                              ("prediction_kind", PREDICTION_KINDS)):
            value = getattr(self, name)
            if value not in allowed:
                raise ConfigurationError(
                    f"{name} must be one of the candidates this skeleton implements; anything else "
                    f"is an unimplemented architectural choice",
                    **{name: value, "allowed": list(allowed)})
        if self.position_encoding == "relative_shaw_windowed":
            _require_int(self.relative_window, "relative_window", minimum=1)
        elif self.relative_window < 0:
            raise ConfigurationError("relative_window must be non-negative", relative_window=self.relative_window)

        d = tuple(self.durations)
        object.__setattr__(self, "durations", d)
        if not d:
            raise ConfigurationError("durations must not be empty")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in d):
            raise ConfigurationError("durations must be integers", durations=list(d))
        if min(d) < 0:
            raise ConfigurationError("durations must be non-negative", durations=list(d))
        if list(d) != sorted(set(d)):
            raise ConfigurationError("durations must be strictly increasing and distinct", durations=list(d))
        if not any(x > 0 for x in d):
            raise ConfigurationError("durations must contain at least one positive value", durations=list(d))

        _require_int(self.vocab_size, "vocab_size", minimum=1)
        if isinstance(self.blank_index, bool) or not isinstance(self.blank_index, int) or self.blank_index < 0:
            raise ConfigurationError("blank_index must be a non-negative integer", blank_index=self.blank_index)
        if self.blank_index >= self.vocab_size:
            raise ConfigurationError("blank_index must sit inside the token slice",
                                     blank_index=self.blank_index, vocab_size=self.vocab_size)
        if isinstance(self.prediction_start_symbol, bool) or not isinstance(self.prediction_start_symbol, int) \
                or not 0 <= self.prediction_start_symbol < self.vocab_size:
            raise ConfigurationError("prediction_start_symbol must be a vocabulary id",
                                     prediction_start_symbol=self.prediction_start_symbol)
        if self.prediction_kind == "embedding_only" and self.prediction_layers:
            raise ConfigurationError("an embedding-only predictor takes no layers",
                                     prediction_layers=self.prediction_layers)

    @property
    def n_durations(self) -> int:
        return len(self.durations)

    @property
    def logit_width(self) -> int:
        """The emission head's slot count: token slice first (blank included), duration slice last."""
        return self.vocab_size + self.n_durations

    def field_names(self) -> tuple:
        return tuple(f.name for f in fields(self))

    def __str__(self) -> str:
        return (f"{self.label}: {self.block_count} x d{self.d_model}/{self.attention_heads}h, "
                f"subsample={self.subsample}, vocab={self.vocab_size}, Nd={self.n_durations}")


#: Not asserted by any test: this is only a count of the required fields declared above. What the
#: tests do pin is that no field carries a default at all, in
#: ``tests/test_modeling_encoder_shapes.py::test_config_refuses_every_attempt_to_omit_a_field``, so a
#: configuration cannot be produced by omission and no default architecture exists to fall back on.
REQUIRED_FIELD_COUNT = len(tuple(fields(StudentConfig)))
