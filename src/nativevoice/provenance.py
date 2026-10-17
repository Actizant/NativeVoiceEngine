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
    # The selected recognition teacher. It has no manifest and is in neither registry,
    # so this entry records what is known about it, not a permission to serve it. The
    # questions are kept apart on purpose - the project's failure mode so far has been
    # one flag standing in for all of them.
    "faster-whisper-large-v3": {
        "upstream": (
            "Systran/faster-whisper-large-v3 (Hugging Face), a CTranslate2 conversion of "
            "openai/whisper-large-v3"
        ),
        "read_on": "2026-10-14",
        "source_identity": (
            "revision edaa852ec7e145841d8ffdb056a99866b5f0a478, model.bin 3,087,284,237 bytes, cached "
            "locally and complete; the same revision the repository's own "
            "artifacts/evaluation/multilingual-large-v3-int8.json recorded, and the revision "
            "tools/provision.py pins this asset to"
        ),
        "evidence": (
            "SHA-256 69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1 recomputed over "
            "model.bin on this machine on 2026-10-14, matching the digest published by the source for "
            "that revision; all seven files of the snapshot are recorded, with their sizes and both the "
            "digest computed here and the digest the source publishes, in "
            "artifacts/provenance/faster-whisper-large-v3.json"
        ),
        "digest_verification": (
            "this pins the bytes to what the source publishes for the revision, which is more than "
            "drift detection - it does not reach OpenAI's own release, because the conversion is a "
            "third party's and that link is unanchored. The evidence record states the same limits as "
            "non-claims: a matching digest says nothing about licensing, original-model authenticity, "
            "commercial clearance, accuracy, or whether the asset fits the target accelerator. The "
            "publisher's cached file table also carries a xet_hash for model.bin; nothing in this "
            "repository computes or compares it, its semantics were not established on this machine, "
            "and it is recorded as uncompared metadata rather than a second verification"
        ),
        "licence_spdx": "NOASSERTION",
        "gap": (
            "the conversion is tagged license:mit by its publisher while openai/whisper-large-v3 is "
            "tagged license:apache-2.0; neither states which terms govern converted weights derived "
            "from the other, no licence text ships inside the snapshot, and no statement was read about "
            "the training data. Recorded as NOASSERTION: asserting either tag would be a guess "
            "presented as a fact"
        ),
        "prohibition": "",
        "redistribution": (
            "not established either way, so nothing here clears this asset for bundling or "
            "distribution. A model trained from it inherits no rights its teacher does not have"
        ),
        "local_measurement_authorization": (
            "permitted for measurement use on the machine that holds it, as a measurement-scope asset. "
            "That is not product-path eligibility, and an approval override cannot widen scope"
        ),
        "accuracy": (
            "unverified. The only pass on record is a 13-clip cue-recall check that its own report "
            "labels a gross-error and script detector rather than a quality measure, run on 13 "
            "user-supplied machine-made samples that report calls neither human speech nor a licensed "
            "corpus and for which it records no reference transcripts; no WER or CER is reported for any "
            "language anywhere in the repository's evidence"
        ),
        "gpu_compatibility": (
            "unverified. No accelerated run of this asset is recorded anywhere: the pass on record is CPU "
            "(int8, 4 threads, 3,246 MiB resident) and its own report reads its timings as that machine "
            "under that memory pressure, partly served from the pagefile. Whether the asset fits the "
            "project's target accelerator at all, and how fast it would run there, has never been measured"
        ),
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
