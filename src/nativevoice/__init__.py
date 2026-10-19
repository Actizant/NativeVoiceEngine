"""NativeVoice - provider-independent, local-first voice infrastructure.

The `core`, `audio`, `language`, `models`, `hardware` and `session` layers are
standard-library only by design: the control path of a local-first system must
not acquire third-party code, network access, or optional dependencies as a
condition of working. Neural engines are the only place where heavy runtimes
appear, and they enter behind `nativevoice.models.registry` approval.
"""

__version__ = "0.1.0"
