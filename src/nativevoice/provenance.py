"""Curated licence and rights findings per upstream source, with what was actually read.

`gap` records evidence that is missing. `prohibition` records a term that was read and forbids
something. They are different keys on purpose: "nobody checked yet" and "checking found no" lead to
different work, and a record that merges them will eventually be read as the latter.

Dates are the dates the source was inspected from this machine, not the dates the source was written.
"""

from __future__ import annotations

#: keyed by model_id in the installed registry, or by upstream repository id
CANDIDATE_EVIDENCE: dict[str, dict] = {
    "faster-whisper-base": {
        "upstream": "Systran/faster-whisper-base (Hugging Face)",
        "read_on": "2026-10-12",
        "evidence": "repository tag license:mit read from the Hugging Face API",
        "gap": (
            "weights are a third-party conversion of OpenAI Whisper; the licence text itself was not "
            "archived locally, and no statement was read about Whisper's training-data provenance"
        ),
        "prohibition": "",
    },
    "faster-whisper-small": {
        "upstream": "Systran/faster-whisper-small (Hugging Face)",
        "read_on": "2026-10-12",
        "evidence": "repository tag license:mit read from the Hugging Face API",
        "gap": "as faster-whisper-base: conversion provenance and training-data disclosure unread",
        "prohibition": "",
    },
    "kokoro-int8-multi-lang-v1_1": {
        "upstream": "hexgrad/Kokoro-82M, packaged by k2-fsa",
        "read_on": "2026-10-12",
        "evidence": "Apache-2.0 licence text shipped inside the installed package (LICENSE, read directly)",
        "gap": (
            "the package bundles an espeak-ng data tree under GPL-3.0, and the per-voice consent and "
            "dataset terms are referenced on an upstream voices page that was not read or archived"
        ),
        "prohibition": "",
    },
    "kitten-nano-en-v0_8-int8": {
        "upstream": "KittenML/kitten-tts-nano-0.8-int8, packaged by k2-fsa",
        "read_on": "2026-10-13",
        "evidence": "model-card front matter license: apache-2.0; the sherpa tarball publishes no licence of its own",
        "gap": (
            "no training-data disclosure and no voice-actor consent for the eight named voices; "
            "espeak-ng data under GPL-3.0 is required by the runtime"
        ),
        "prohibition": "",
    },
}
