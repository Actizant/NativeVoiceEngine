"""One-shot local provisioning of approved speech assets.

This is the seam the specification asks to keep separate: initial asset
acquisition may use the network, ordinary runtime may not. Nothing under
`nativevoice/` imports this file, and nothing here is required to speak or hear
once the assets exist.

Why the manifest is written *here* rather than by hand: an approval that is not
accompanied by the evidence it rested on is a guess with a signature under it.
Every manifest produced records the licence as published by the source, where
that was read from, on what date, the digest computed at provisioning time, and
whose authorisation was relied on.

Integrity caveat recorded deliberately: the sherpa-onnx model zoo does not publish
digests for its tarballs, so the SHA-256 of those files is **computed at provisioning**,
which detects later corruption or drift but cannot prove the first fetch was authentic.
Where an upstream digest *does* exist it is compared against, and a mismatch is fatal
rather than a warning: a Hugging Face snapshot carries, in the hub cache's own
``trees/<revision>.json``, the size and git blob hash of every file and the LFS SHA-256
of every large one, so a pinned asset is checked against what its publisher claims
before any manifest is written. A match identifies bytes; it says nothing about licence.

Three things this tool does *not* do, each because a manifest that claimed it would be
a fabrication:

* **clear a licence.** It records the licence the source publishes, the evidence
  string that was read, and the name of whoever authorised the recording. Whether those
  facts amount to permission is the approval gate's question and a human's decision, so
  ``review`` is written as ``approved`` only when ``--grant-product-approval`` is
  supplied deliberately - a name in ``--approved-by`` is evidence about a person, not a
  decision by one.
* **decide scope from the asset.** `--scope` is required and has no default, so
  provisioning cannot stamp a research-only download onto the shipped voice path by
  silence. `product` means this registry may serve it; it says nothing about
  redistribution rights, and `measurement` is refused by the product path even under
  an approval override. An asset the catalogue locks to one scope - the large-v3
  teacher is measurement-only - is refused for the other, by the tool rather than by
  convention.
* **approve a measurement asset.** `--scope measurement` can never write `approved`:
  what authorizes running such an asset locally is its scope, a named authoriser and a
  complete provenance record, which is a different question from clearing it for the
  shipped path, and keeping the two apart is the only reason the boundary means anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nativevoice.core.errors import ConfigurationError  # noqa: E402
from nativevoice.config.paths import hf_hub_cache_root  # noqa: E402
from nativevoice.models.api import (  # noqa: E402
    Asset,
    AssetKind,
    AssetScope,
    ModelManifest,
    Provenance,
    ResourceRequirement,
    ReviewStatus,
)
from nativevoice.core.language import Task  # noqa: E402
from nativevoice.models.manifest_io import save_manifest  # noqa: E402
from nativevoice.models.roots import (  # noqa: E402
    default_registry_root,
    resolve_registry_root,
    scope_tree,
)

USER_AGENT = "ultron-native-voice/0.1 (asset provisioning)"

#: Catalogue of provisionable assets. `licence_evidence` is the string that was
#: actually read from the source, not a summary of it.
CATALOGUE: dict[str, dict] = {
    "kokoro-int8-multi-lang-v1_1": {
        "task": Task.TTS,
        "engine": "sherpa-onnx-kokoro",
        "display_name": "Kokoro 82M (int8, multi-language)",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-multi-lang-v1_1.tar.bz2",
        "upstream_project": "hexgrad/Kokoro-82M v1.1, packaged by k2-fsa sherpa-onnx",
        "licence_spdx": "Apache-2.0",
        "licence_evidence": "Hugging Face tag license:apache-2.0 on hexgrad/Kokoro-82M, read 2026-10-12",
        "bundled_notice": (
            "Contains an espeak-ng data directory. espeak-ng and its dictionaries are GPL-3.0 "
            "(COPYING read from the upstream repository on 2026-10-12). Acceptable for local use; "
            "redistributing this asset carries GPL-3.0 obligations that must be met separately."
        ),
        "sample_rate": 24000,
        "quantization": "int8",
        "ram_mb_estimate": 400,
        "declared_languages": ("en", "zh"),
        "language_basis": "front-end files shipped in the package: lexicon-us-en, lexicon-gb-en, lexicon-zh and the zh date/number FSTs. Languages without a lexicon here are not supported by this asset, whatever espeak-ng could romanise.",
        "notes": "Multi-language offline TTS. Language coverage is verified by inspecting the extracted files, not assumed.",
    },
    "kitten-nano-en-v0_8-int8": {
        "task": Task.TTS,
        "engine": "sherpa-onnx-kitten",
        "display_name": "KittenTTS nano 0.8 (int8, English)",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_8-int8.tar.bz2",
        "upstream_project": "KittenML/kitten-tts-nano-0.8-int8, packaged by k2-fsa sherpa-onnx",
        "licence_spdx": "Apache-2.0",
        "licence_evidence": (
            "Hugging Face front matter license: apache-2.0 on KittenML/kitten-tts-nano-0.8-int8, "
            "read 2026-10-13; the sherpa release tarball publishes no licence text of its own, so the "
            "upstream repository is the licence source"
        ),
        "bundled_notice": (
            "Redistribution rights are NOT established for this asset: `redistributable` is recorded as "
            "False because the runtime needs an espeak-ng data directory, which is GPL-3.0, and the model card "
            "discloses neither training data nor voice-actor consent. The eight named voices are the only "
            "identity evidence available. The model card names eight "
            "voices (Bella, Jasper, Luna, Bruno, Rosie, Hugo, Kiki, Leo) but discloses no training data and "
            "no voice-actor consent basis, so redistribution rights are unestablished even though the "
            "checkpoint itself is Apache-2.0"
        ),
        "sample_rate": 24000,
        "quantization": "int8",
        "ram_mb_estimate": 250,
        "declared_languages": ("en",),
        "language_basis": "English-only per the upstream model card and the packaged tokens/lexicon",
        "notes": "Isolated candidate under evaluation for latency; not a default and not promoted. Acquired into a trial root so the installed Kokoro asset and the shared catalogue are untouched.",
        "trial": True,
    },
    "faster-whisper-base": {
        "task": Task.STT,
        "engine": "faster-whisper",
        "display_name": "Whisper base (CT2)",
        "url": "",
        "hf_repo": "Systran/faster-whisper-base",
        "upstream_project": "OpenAI Whisper, converted by Systran for CTranslate2",
        "licence_spdx": "MIT",
        "licence_evidence": "Hugging Face tag license:mit on Systran/faster-whisper-base, read 2026-10-12",
        "sample_rate": 16000,
        "quantization": "ct2-default",
        "ram_mb_estimate": 200,
        "declared_languages": ("en", "hi", "bn"),
        "language_basis": "OpenAI Whisper's published 97-language list includes English, Hindi and Bengali; vendor claim, measured here only for the languages the round-trip evaluation covers",
        "notes": "Already resident in the local Hugging Face cache; provisioning verifies digests, it does not download.",
    },
    "faster-whisper-small": {
        "task": Task.STT,
        "engine": "faster-whisper",
        "display_name": "Whisper small (CT2)",
        "url": "",
        "hf_repo": "Systran/faster-whisper-small",
        "upstream_project": "OpenAI Whisper, converted by Systran for CTranslate2",
        "licence_spdx": "MIT",
        "licence_evidence": "Hugging Face tag license:mit on Systran/faster-whisper-small, read 2026-10-12",
        "sample_rate": 16000,
        "quantization": "ct2-default",
        "ram_mb_estimate": 500,
        "declared_languages": ("en", "hi", "bn"),
        "language_basis": "OpenAI Whisper's published 97-language list includes English, Hindi and Bengali; vendor claim, measured here only for the languages the round-trip evaluation covers",
        "notes": "Already resident in the local Hugging Face cache.",
    },
    # The selected recognition teacher. Recording it here authorises provisioning *for
    # measurement only*: it does not resolve the licence, does not claim the asset fits
    # this machine's accelerator, and does not make it servable by the shipped path.
    "faster-whisper-large-v3": {
        "task": Task.STT,
        "engine": "faster-whisper",
        "display_name": "Whisper large-v3 (CT2)",
        "url": "",
        "hf_repo": "Systran/faster-whisper-large-v3",
        # The commit the cached snapshot must be, and the one the repository's own
        # evaluation record and `provenance.py` name. A pin nobody checks is a comment.
        "revision": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
        "allowed_scope": AssetScope.MEASUREMENT,
        "upstream_project": "OpenAI Whisper large-v3, converted by Systran for CTranslate2",
        "licence_spdx": "NOASSERTION",
        "licence_evidence": (
            "no licence text ships inside the snapshot; the publisher tags the conversion "
            "license:mit while openai/whisper-large-v3 is tagged license:apache-2.0 and neither "
            "states which terms govern converted weights, and no statement was read about the "
            "training data. Recorded as NOASSERTION: asserting either tag would be a guess "
            "presented as a fact"
        ),
        "redistributable": False,
        "sample_rate": 16000,
        # A name for how the publisher says it converted the weights, nothing more: it is
        # not a measured property of these bytes and no check here confirms it.
        "quantization": "ct2-default",
        # Not a guess: 3,246.4 MiB peak RSS for the whole recorded int8 CPU pass, in
        # artifacts/evaluation/multilingual-large-v3-int8.json, rounded up. That pass is
        # marked `production_configuration: false` and ran partly from the pagefile, so it
        # sizes the asset without describing any configuration this project would ship.
        # `requirements.measured` stays False for exactly that reason.
        "ram_mb_estimate": 3300,
        "declared_languages": (),
        "notes": (
            "Recognition teacher candidate, local measurement only. Its 97-language set is a "
            "vendor claim verified nowhere in this repository, so no language is recorded until "
            "one is measured. RAM is the recorded whole-pass peak resident size on a different "
            "machine class under memory pressure, not a fitted budget."
        ),
    },
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        got = 0
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
            got += len(block)
            if total:
                print(f"  {got / 2**20:.0f}/{total / 2**20:.0f} MB", end="\r", flush=True)
    temporary.replace(destination)
    if total:
        print(f"  {total / 2**20:.1f} MB downloaded")
    return destination


def extract(archive: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    # filter="data" refuses absolute paths and ../ traversal in the tar member names;
    # a model archive is not a place to accept the filesystem's defaults.
    with tarfile.open(archive, "r:*") as handle:
        handle.extractall(into, filter="data")
    return into


def hf_snapshot_dir(repo: str) -> Path:
    """The repository directory a Hugging Face snapshot lives in.

    Resolved through :func:`~nativevoice.config.paths.hf_hub_cache_root`, which is the
    same function the loading engine uses, so an ``HF_HOME`` or ``HF_HUB_CACHE`` set on
    this machine moves both callers together. A provisioner that digested
    ``~/.cache/huggingface`` while the engine reads ``D:/models`` would be recording
    evidence about bytes nothing ever loads.
    """
    return hf_hub_cache_root() / ("models--" + repo.replace("/", "--"))


def snapshot_files(repo: str, revision: str = "") -> list[Path]:
    """The files of one cached snapshot, chosen by revision when one is pinned.

    With ``revision`` set, the snapshot directory *is* the answer: ``huggingface_hub``
    names it after the commit it fetched, so a pinned asset is recorded only if the
    bytes for exactly that commit are present. Falling back to "the newest directory"
    would silently digest a different revision than the one the provenance record
    names, and the manifest would then be evidence for the wrong file.

    Without a pin, the newest snapshot is used, and which one that was is recorded in
    every asset's relative path.
    """
    root = hf_snapshot_dir(repo)
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    if revision:
        pinned = snapshots / revision
        return sorted(path for path in pinned.rglob("*") if path.is_file() or path.is_symlink()) if pinned.is_dir() else []
    latest = max(snapshots.iterdir(), key=lambda item: item.stat().st_mtime)
    return sorted(path for path in latest.rglob("*") if path.is_file() or path.is_symlink())


def published_digests(repository: Path, revision: str) -> dict[str, dict]:
    """What the source published about one snapshot, read from the local hub cache.

    ``trees/<revision>.json`` is written by ``huggingface_hub`` when the snapshot is
    fetched: every file carries the size and the git blob SHA-1 of its content, and an
    LFS file additionally carries ``lfs_sha256`` - the digest the source's own storage
    claims for it. That is upstream *metadata*, read offline, so a mismatch is fatal
    rather than a warning here. What it cannot do is reach past the publisher: an LFS
    digest match proves these are the bytes the source serves for this revision, not
    that they are original, licensed, or usable.

    For a pinned asset this record is required rather than optional. Absent, unreadable
    or empty used to be reported as "no published digest", which turned a missing local
    file into a positive claim about upstream and let the comparison pass without
    comparing anything; each of those now refuses, naming the file. An unpinned asset
    asks for nothing, because there is no revision whose record could be looked up.
    """
    if not revision:
        return {}
    tree = repository / "trees" / f"{revision}.json"
    if not tree.is_file():
        raise ConfigurationError(
            "no published-digest record is cached for the revision this asset is pinned to",
            expected=str(tree),
            revision=revision,
            remedy=(
                "fetch this snapshot with huggingface_hub so the cache holds trees/<revision>.json; this "
                "tool does not download, and it will not record a pinned asset it cannot compare - an "
                "absent record is not evidence that no digest is published"
            ),
        )
    try:
        payload = json.loads(tree.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfigurationError(
            "the published-digest record for the pinned revision cannot be read",
            path=str(tree),
            detail=f"{type(error).__name__}: {error}",
            remedy="re-fetch the snapshot; a corrupt metadata file is refused rather than treated as silence",
        ) from error
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict) or not files:
        raise ConfigurationError(
            "the published-digest record for the pinned revision lists no files",
            path=str(tree),
            revision=revision,
            remedy="re-fetch the snapshot; an empty file table compares nothing, so nothing is recorded",
        )
    return dict(files)


def git_blob_sha1(path: Path) -> str:
    """The git blob hash of a file's content: ``sha1("blob <n>\\0" + bytes)``.

    This is what a non-LFS entry's ``blob_id`` is, so it can be checked against the
    file on disk rather than only against its size.
    """
    digest = hashlib.sha1()
    data = path.read_bytes()
    digest.update(b"blob %d\x00" % len(data))
    digest.update(data)
    return digest.hexdigest()


def check_published_digest(path: Path, relative: str, digest: str, size: int, meta: dict) -> str:
    """Compare one file with the digest published for it; return the basis. Fatal on mismatch."""
    if not meta:
        return "no digest published by the source for this file"
    published_size = int(meta.get("size") or 0)
    if published_size and published_size != size:
        raise ConfigurationError(
            "the size published by the source does not match the file on disk",
            path=relative,
            published_size=published_size,
            found_size=size,
            remedy="re-fetch the snapshot; this asset is not recorded until the bytes agree",
        )
    oid = str(meta.get("lfs_sha256") or "")
    if oid:
        if digest.lower() != oid.lower():
            raise ConfigurationError(
                "the digest published by the source does not match the file on disk",
                path=relative,
                published=oid,
                computed=digest,
                remedy="re-fetch the snapshot; a digest mismatch is fatal here, not a warning",
            )
        return "computed SHA-256 matches the LFS digest the source publishes for this revision"
    blob = str(meta.get("blob_id") or "")
    if blob and git_blob_sha1(path) != blob.lower():
        raise ConfigurationError(
            "the git blob published by the source does not match the file on disk",
            path=relative,
            published=blob,
            remedy="re-fetch the snapshot; a digest mismatch is fatal here, not a warning",
        )
    return "computed content matches the git blob the source publishes for this revision" if blob else (
        "no digest published by the source for this file"
    )


def flatten(root: Path) -> Path:
    """A sherpa archive unpacks to a single same-named directory; lift its contents.

    The registry resolves every asset path as `root/model_id/<relative_path>`, so a
    nested second level would make each digest point at a file that is not there.
    """
    inner = [path for path in root.iterdir() if path.is_dir()]
    if len(inner) == 1 and (inner[0] / "model.int8.onnx").exists() or len(inner) == 1 and any(
        suffix in {".onnx", ".bin"} for suffix in [x.suffix for x in inner[0].iterdir()]
    ):
        for item in sorted(inner[0].iterdir()):
            item.replace(root / item.name)
        inner[0].rmdir()
    return root


# What one file inside an asset package *is*, decided by name. The previous map asked
# only "is this a weight?" and called everything else a tokenizer asset, so a CT2
# `config.json` was recorded as vocabulary data and a licence file or a stray script
# inherited a label nobody chose. Anything outside this table is refused: an explicit
# stop is a finding a human can act on, a silent wrong kind is not.
#
# Refusing a real package proved to be the right diagnosis and the wrong end state: the
# teacher snapshot and the installed Kokoro package both carry a README, a licence text,
# VCS attributes and one helper script, and none of those is a tokenizer asset. They are
# now named by exact filename, so a package's documentation is digested and described as
# documentation instead of being dropped from the integrity record or filed under a kind
# that implies the runtime reads it.
_WEIGHT_SUFFIXES = {".onnx", ".bin", ".model"}
_MODEL_CONFIG_NAMES = {"config.json", "preprocessor_config.json"}
_TOKENIZER_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "vocabulary.json",
    "vocabulary.txt",
    "tokens.txt",
}
# sherpa's text front-end: pinyin/number FSTs and the jieba dictionaries.
_FRONT_END_SUFFIXES = {".fst", ".utf8"}
_FRONT_END_NAME_PREFIXES = ("lexicon-",)
# The espeak-ng data tree is front-end data end to end, and its voice dicts are
# extension-less names like `af_dict` under `espeak-ng-data/lang/<family>/`.
_FRONT_END_ROOT_DIRS = ("espeak-ng-data",)
_FRONT_END_NAME_SUFFIXES = ("_dict",)
# The terms the asset ships under, as text - not a summary of them, and not prose.
_LICENCE_NAMES = {"LICENSE", "LICENCE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt", "NOTICE"}
# Prose about the package: a model card or a directory note. Read by people, never loaded.
_DOCUMENTATION_NAMES = {"README", "README.md", "README.txt", "README.rst"}
# Packaging and VCS residue that travels with the files and describes none of them.
_PACKAGE_METADATA_NAMES = {".gitattributes", ".gitignore", ".mailmap", ".huggingface.yaml", "version.txt"}
# Code shipped inside a data package. Its licence is its own question, so it is never
# filed as documentation - and a `.py` in a model directory is not a skip-list entry, it
# is a file with a digest that somebody must be able to see.
_SOURCE_SUFFIXES = {".py"}


def classify_asset(path: Path, directory: Path) -> AssetKind:
    """The kind of one asset file, or ``ConfigurationError`` naming it.

    Never defaults. An unrecognised file is a question about what the package
    contains, and guessing it would put a digest of the wrong thing into the
    provenance record that later review relies on.
    """
    relative = path.relative_to(directory).as_posix()
    parts = relative.split("/")
    name = parts[-1]
    suffix = path.suffix.lower()
    if name in _MODEL_CONFIG_NAMES:
        return AssetKind.MODEL_CONFIG
    if suffix in _WEIGHT_SUFFIXES:
        return AssetKind.MODEL
    if name in _LICENCE_NAMES:
        return AssetKind.LICENCE
    if name in _DOCUMENTATION_NAMES:
        return AssetKind.DOCUMENTATION
    if name in _PACKAGE_METADATA_NAMES:
        return AssetKind.PACKAGE_METADATA
    if name in _TOKENIZER_NAMES or suffix in _FRONT_END_SUFFIXES:
        return AssetKind.TOKENIZER
    if name.startswith(_FRONT_END_NAME_PREFIXES) or name.endswith(_FRONT_END_NAME_SUFFIXES):
        return AssetKind.TOKENIZER
    if parts[0] in _FRONT_END_ROOT_DIRS:
        return AssetKind.TOKENIZER
    if suffix in _SOURCE_SUFFIXES:
        return AssetKind.SOURCE
    raise ConfigurationError(
        "cannot classify an asset file, so this asset is not recorded",
        path=relative,
        remedy=(
            "extend the provisioning asset-kind table deliberately, or keep the file out of the "
            "package; recording it as a tokenizer asset is not an option"
        ),
    )


# The registry-root rules are imported above from `nativevoice.models.roots`, not
# reimplemented here: the measurement loader has to refuse the same roots this tool
# does, and two copies of that rule drift apart until the guard that survives is the
# weaker one. The imported names are also what `tests/test_provision.py` imports, so
# provisioning and the loader are provably checking one rule against one tree.


def check_scope_contract(name: str, spec: dict, scope: AssetScope) -> None:
    """Refuse a scope the catalogue never authorised this asset for.

    The teacher is the case that matters: it is on disk to measure something, its
    licence is unresolved, and a later ``--scope product`` would otherwise be answerable
    by a tool that had never been told that is not permission. The catalogue, not the
    caller, owns which scope an acquisition is authorised for; ``--scope`` still has no
    default, because a required argument that is *also* checked stops both a silence and
    a guess.
    """
    allowed = spec.get("allowed_scope")
    if allowed is not None and scope is not allowed:
        raise ConfigurationError(
            "this asset is not provisionable for that scope",
            asset=name,
            scope=scope.value,
            allowed_scope=allowed.value,
            remedy=f"provision {name} with --scope {allowed.value}, or have its rights position reviewed first",
        )


def review_state(args) -> ReviewStatus:
    """What this run is allowed to record about approval - usually nothing.

    ``--approved-by`` names the person who authorised the recording, which is evidence,
    not the decision. Before, supplying a name was sufficient to write ``approved`` into
    the review field of every manifest this tool produced, so the licence gate's central
    fact was produced by the tool rather than granted by a human - and a measurement
    download could carry product approval as a side effect of the flag that meant "this
    is research only". Approval is now a separate, deliberate act, and one a measurement
    asset cannot take.
    """
    scope = AssetScope(args.scope)
    if args.grant_product_approval and scope is not AssetScope.PRODUCT:
        raise ConfigurationError(
            "product approval cannot be granted for a measurement asset",
            scope=scope.value,
            remedy=(
                "drop --grant-product-approval: authorising a local measurement is not clearing "
                "the asset for the shipped path, and the two records must not be written by one flag"
            ),
        )
    return ReviewStatus.APPROVED if args.grant_product_approval else ReviewStatus.UNREVIEWED


def tree_key(relative: str) -> str:
    """A cached snapshot path as the hub's own file index names it.

    Provisioning records ``snapshots/<revision>/model.bin``; the published digest table
    is keyed on ``model.bin``. Reconstructing one from the other by string surgery would
    break on a nested file, so the prefix that is not part of the file's identity is
    removed by position.
    """
    parts = relative.split("/")
    if len(parts) >= 3 and parts[0] == "snapshots":
        return "/".join(parts[2:])
    return relative


def digest_notice(total: int, checked: int, stamp: str, metadata: str = "none") -> str:
    """What the recorded digests actually prove, worded differently when they do not.

    The sentence used to be unconditional - "upstream publishes no digest for this
    artefact" - while the module header promised a fatal comparison. Both cannot be
    true of the same asset, and a manifest that contradicts its own tool is the kind of
    record that gets believed selectively.

    ``metadata`` is what this run actually had to compare against, because the three
    ways a comparison does not happen mean different things: ``none`` for an artefact
    whose source publishes nothing (a sherpa tarball), ``unpinned`` for a Hugging Face
    asset with no revision to look a record up by, and ``read`` for a pinned asset whose
    record was cached and parsed yet matched none of these files. Only the first is a
    statement about upstream; the other two are statements about this cache.
    """
    if checked:
        return (
            f"{checked}/{total} files matched a digest the source publishes for this revision; the "
            f"remainder carry none, so their SHA-256 was computed at provisioning {stamp} and pins "
            "drift-detection, not authenticity. A digest agreement identifies the bytes the publisher "
            "serves for a revision: it is not a licence, and it does not prove that a converted model "
            "came from the model it claims to convert."
        )
    if metadata == "unpinned":
        return (
            f"SHA-256 computed at provisioning {stamp}; this asset carries no revision pin, so no "
            "per-revision record was looked up and no comparison happened. That is a statement about "
            "this run, not about what the publisher does or does not release: this pins "
            "drift-detection, not authenticity."
        )
    if metadata == "read":
        return (
            f"SHA-256 computed at provisioning {stamp}; the published-digest record for the pinned "
            "revision was read and names none of these files, so nothing was compared. The pin is "
            "recorded, the bytes are not corroborated by it: this pins drift-detection, not "
            "authenticity."
        )
    return (
        f"Digest computed at provisioning {stamp}; upstream publishes no digest for this artefact, "
        "so this pins drift-detection not authenticity."
    )


def build_manifest(name: str, directory: Path, files: list[Path], args) -> ModelManifest:
    spec = CATALOGUE[name]
    # Required on the CLI, never inferred here: an absent --scope cannot quietly
    # resolve to the shipped path for an asset acquired to measure something.
    scope = AssetScope(args.scope)
    check_scope_contract(name, spec, scope)
    revision = str(spec.get("revision") or "")
    # Published digests exist only for a cache snapshot at a known revision. For a pinned
    # Hugging Face asset the record is required, so an absent or unreadable one refuses the
    # asset here instead of becoming a claim about what upstream publishes.
    is_hf = bool(spec.get("hf_repo"))
    published = published_digests(directory, revision) if is_hf else {}
    if published:
        metadata_state = "read"
    elif is_hf and not revision:
        metadata_state = "unpinned"
    else:
        metadata_state = "none"
    assets: list[Asset] = []
    checked = 0
    for path in files:
        kind = classify_asset(path, directory)
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size if path.is_file() else 0
        digest = sha256(path) if path.is_file() else ""
        meta = published.get(tree_key(relative)) or {}
        if digest:
            check_published_digest(path, relative, digest, size, meta)
            checked += 1 if meta else 0
        assets.append(Asset(kind=kind, relative_path=relative, sha256=digest, size_bytes=size))
    recorded = tuple(assets)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    review = review_state(args)
    granted = review is ReviewStatus.APPROVED
    notice = " ".join(
        part.strip()
        for part in (
            spec["licence_evidence"] + ".",
            spec.get("bundled_notice", ""),
            digest_notice(len(recorded), checked, stamp, metadata_state),
        )
        if part.strip()
    )
    provenance = Provenance(
        source=spec["url"] or f"https://huggingface.co/{spec.get('hf_repo','')}",
        upstream_project=spec["upstream_project"],
        licence_spdx=spec["licence_spdx"],
        licence_notice=notice,
        consent_basis="not applicable: pretrained weights carry no personal data; voice recordings and profiles are governed separately",
        redistributable=spec.get("redistributable", False),
        notes=spec.get("bundled_notice", ""),
    )
    sentences = [
        spec["notes"],
        f"Authorisation relied on: {args.authority}.",
        f"Recorded for the {scope.value} path; that is a routing permission, not a rights clearance.",
    ]
    if not granted:
        sentences.append(
            "Recorded without product approval: the review status stays unreviewed and the licence "
            "gate refuses this asset until a human grants approval deliberately."
        )
    # Said in the record rather than only in the catalogue comment, because a manifest is
    # what a later reader has in hand: an unmeasured RAM figure and a declared quantization
    # are both easy to mistake for measurements once they sit alone in a JSON field.
    sentences.append(
        f"Declared, not measured: the {spec['ram_mb_estimate']} MiB RAM figure is carried as an "
        f"estimate with requirements.measured false, and quantization {spec['quantization']!r} names "
        "the conversion the catalogue declares, not a property verified of the bytes on disk."
    )
    if spec.get("revision"):
        sentences.append(f"Pinned to revision {revision}.")
    if spec.get("language_basis"):
        sentences.append(spec["language_basis"])
    return ModelManifest(
        model_id=name,
        engine=spec["engine"],
        task=spec["task"],
        display_name=spec["display_name"],
        # The revision is the asset's identity for a pinned snapshot, so it goes where a
        # reader of the manifest looks for the version rather than only into prose.
        version=revision or str(spec.get("version", "1")),
        languages=tuple(sorted(args.language or spec.get("declared_languages") or [])),
        minimum_profile="balanced" if spec["ram_mb_estimate"] >= 400 else "lite",
        sample_rate=spec["sample_rate"],
        quantization=spec["quantization"],
        provenance=provenance,
        review=review,
        approved_by=args.approved_by,
        # An approval date on an asset that was not approved would be the second
        # fabrication this function had to stop.
        approved_on=stamp[:10] if granted else "",
        requirements=ResourceRequirement(
            ram_mb=spec["ram_mb_estimate"],
            disk_mb=max(1, sum(asset.size_bytes for asset in recorded) // 2**20),
            threads=2,
            measured=False,
        ),
        assets=recorded,
        asset_root=str(Path(directory).resolve()),
        use_scope=scope,
        notes=" ".join(sentences),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asset", nargs="+", required=True, choices=sorted(CATALOGUE))
    parser.add_argument(
        "--scope",
        required=True,
        choices=sorted(scope.value for scope in AssetScope),
        help=(
            "which registry may serve what is provisioned: 'product' is the shipped voice path, "
            "'measurement' is local research only. There is no default, because an unanswered "
            "question would otherwise be answered 'product' for every asset this tool is later "
            "pointed at. Scope is not a licence decision: a measurement asset stays refused by the "
            "product path, and a product asset is not therefore cleared for redistribution."
        ),
    )
    parser.add_argument(
        "--approved-by",
        required=True,
        help=(
            "the person who authorised this recording, named: a manifest without a name is not an "
            "approval, and supplying one is evidence rather than a decision - the review status "
            "still comes from --grant-product-approval"
        ),
    )
    parser.add_argument(
        "--grant-product-approval",
        action="store_true",
        help=(
            "write review=approved, the licence gate's central fact, deliberately and only for "
            "--scope product. Without it the asset is recorded unreviewed and the gate refuses it, "
            "which is the correct state for anything acquired to measure: an approval a tool "
            "produced as a side effect of another flag is not a human's decision"
        ),
    )
    parser.add_argument("--authority", required=True, help="the instruction relied on, quoted")
    parser.add_argument("--language", action="append", help="language code actually verified for this asset; repeat per language")
    parser.add_argument(
        "--languages-unverified",
        action="store_true",
        help="record the asset with no languages, for the first pass before the archive has been inspected",
    )
    parser.add_argument(
        "--root",
        default="",
        help="model root for this scope (defaults to the configured one; a root inside the other scope's tree is refused)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan without touching the network or the disk")
    parser.add_argument("--force", action="store_true", help="refetch even if the asset is already on disk")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.language and not args.languages_unverified:
        # each asset in the catalogue carries the languages its own files support,
        # so an explicit list is only required when provisioning a mix of them
        if len(args.asset) > 1:
            parser.error("one --language per asset is required when provisioning several, so a language is never stamped onto a model that cannot do it")

    try:
        root = resolve_registry_root(AssetScope(args.scope), args.root)
    except ConfigurationError as error:
        print(f"REFUSED: {error}")
        return 1
    for name in args.asset:
        spec = CATALOGUE[name]
        target = root / name
        try:
            check_scope_contract(name, spec, AssetScope(args.scope))
        except ConfigurationError as error:
            # Before the dry run, and before anything else: a plan this tool is not
            # allowed to carry out should be refused while it is still only a plan.
            # Refused per asset, like every other refusal here, so one locked asset
            # cannot turn a batch into a silent nothing-happened success either.
            print(f"  REFUSED: {error}")
            continue
        if args.dry_run:
            print(f"{name}: [{args.scope}] would fetch {spec['url'] or 'local HF snapshot ' + spec.get('hf_repo','')} -> {target}")
            continue
        print(f"provisioning {name} for the {args.scope} path")
        if spec.get("url"):
            if any(target.rglob("*.onnx")) and not args.force:
                print("  already provisioned; re-verifying digests (use --force to refetch)")
            else:
                archive = download(spec["url"], root / (name + ".tar.bz2"))
                extract(archive, target)
                archive.unlink(missing_ok=True)
            directory = flatten(target)
            files = sorted(path for path in directory.rglob("*") if path.is_file())
        else:
            revision = str(spec.get("revision") or "")
            directory = hf_snapshot_dir(spec["hf_repo"])
            files = snapshot_files(spec["hf_repo"], revision)
            if not files:
                where = f" at revision {revision}" if revision else ""
                print(f"  MISSING: no local snapshot for {spec['hf_repo']}{where}; refusing to record it")
                continue
        try:
            manifest = build_manifest(name, directory, files, args)
        except ConfigurationError as error:
            # Nothing is written for this asset: a manifest whose kinds were guessed
            # is worse evidence than no manifest, and the message names the file.
            print(f"  REFUSED: {error}")
            continue
        path = save_manifest(manifest, str(root / "manifests" / f"{name}.json"))
        print(f"  {len(files)} files, {manifest.requirements.disk_mb} MB, licence {manifest.provenance.licence_spdx}")
        # The review status is printed because it is no longer a foregone conclusion:
        # whoever runs this needs to see which of the two things they just recorded.
        print(f"  scope {manifest.use_scope.value}, review {manifest.review.value}, {manifest.engine} v{manifest.version}")
        print(f"  digest recorded for {sum(1 for a in manifest.assets if a.sha256)}/{len(manifest.assets)} files")
        print(f"  manifest -> {path}")
    print(f"registry root for {args.scope}: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
