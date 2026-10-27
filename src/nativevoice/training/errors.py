"""Error types for the training package.

Both derive from :class:`~nativevoice.core.errors.VoiceError`, which is what
``tools/audit.py``\\'s error-taxonomy check admits - it collects every ``VoiceError``
subclass "however locally it is declared", so a subsystem may raise its own category
without editing the product taxonomy. No builtin exception is raised anywhere in this
package, so the single admitted ``TypeError`` guard in the audit is untouched.
"""

from __future__ import annotations

from nativevoice.core.errors import ConfigurationError, VoiceError


class TdtContractError(ConfigurationError):
    """The caller's tensors, ids, lengths or duration set cannot describe a lattice.

    Derives from :class:`ConfigurationError`, hence from ``ValueError`` as well, so a host
    that catches the builtin still catches the category - while the audit sees our type.
    """


class TdtAlignmentImpossible(VoiceError):
    """No admissible path exists, or the lattice produced a non-finite likelihood.

    Raised instead of returning ``inf`` or ``NaN``: an impossible alignment is a contract
    violation between the encoder length and the target length, and a silent ``inf`` trains
    nothing while looking like a number.
    """
