"""Shape-side components of the Conformer-TDT student. Deliberately not a model that runs.

Status while the architecture is under review: this package holds two things that need no
unset decision to be useful - an integer length bridge between the frozen frontend and the TDT
lattice (``lengths``), and a fully explicit configuration record for the student shape
(``config``). ``encoder`` builds real ``torch.nn`` layers from that record but is imported
explicitly, because ``torch`` is absent from the project environment and importing it at module
level is refused by ``tools/audit.py``. Nothing here reads a dataset, opens a socket, loads a
checkpoint, updates a weight or recognises anything, and ``GATED_TRAINING_CALLERS`` in
``tests/test_corpus_reader_fence.py`` remains empty.

The subsampling convention is NOT chosen: four candidates are named in ``lengths`` and every
function requires one. A1 clause S4 (ruled 2026-10-27; authoritative record
``docs/asr-architecture.md`` section 7.2) prunes that field on arithmetic rather than preference:
complete real-frame coverage is required for every admissible frame count, so the two ``padding=0``
stacks are ineligible - one stage covers only odd counts, two stages only counts that are 3 mod 4 -
while both ``padding=1`` stacks cover every admissible length. Frequency-axis coverage at the frozen
mel width, and the boundary zero-padding the padding=1 stacks are permitted to use, are separate
properties defined there. The ``padding=1`` pair stays in competition and neither is preferred.
``config.OPEN_ARCHITECTURE_DECISIONS`` lists what stays open, including A1 - specified and
provisionally ruled as this package's implementation contract, yet unresolved against the paper -
which the length seam treats as a provisional interface contract rather than a settled fact. Model
capacity, vocabulary, blank id and duration configuration are required arguments for the same reason.
"""

from __future__ import annotations

from nativevoice.modeling.config import (
    CONV_ACTIVATIONS,
    MACARON_MODES,
    NORMALIZATIONS,
    OPEN_ARCHITECTURE_DECISIONS,
    PREDICTION_KINDS,
    POSITION_ENCODINGS,
    REQUIRED_FIELD_COUNT,
    StudentConfig,
)
from nativevoice.modeling.lengths import (
    KNOWN_OPEN_DEFECTS,
    ONE_STAGE_NO_PADDING,
    ONE_STAGE_PAD_ONE,
    PROVISIONAL_CONTRACT,
    RESOLVED_DEFECTS,
    SUBSAMPLE_VARIANTS,
    TWO_STAGES_NO_PADDING,
    TWO_STAGES_PAD_ONE,
    LengthBridge,
    LengthSeamError,
    StackSpec,
    SubFloorInputError,
    UndefinedAxisError,
    bridge,
    frequency_bins,
    minimum_frames,
    minimum_samples,
    stack_length,
)

__all__ = [
    "CONV_ACTIVATIONS",
    "KNOWN_OPEN_DEFECTS",
    "MACARON_MODES",
    "NORMALIZATIONS",
    "ONE_STAGE_NO_PADDING",
    "ONE_STAGE_PAD_ONE",
    "OPEN_ARCHITECTURE_DECISIONS",
    "PREDICTION_KINDS",
    "POSITION_ENCODINGS",
    "PROVISIONAL_CONTRACT",
    "RESOLVED_DEFECTS",
    "REQUIRED_FIELD_COUNT",
    "SUBSAMPLE_VARIANTS",
    "TWO_STAGES_NO_PADDING",
    "TWO_STAGES_PAD_ONE",
    "LengthBridge",
    "LengthSeamError",
    "StackSpec",
    "StudentConfig",
    "SubFloorInputError",
    "UndefinedAxisError",
    "bridge",
    "frequency_bins",
    "minimum_frames",
    "minimum_samples",
    "stack_length",
]
