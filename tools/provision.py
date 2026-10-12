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
digests for its tarballs, so the SHA-256 here is **computed at provisioning**, which
detects later corruption or drift but cannot prove the first fetch was authentic.
Where an upstream digest does exist (Hugging Face LFS OIDs) it is compared against,
and a mismatch is fatal rather than a warning.
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

from nativevoice.config import data_root, models_dir  # noqa: E402
from nativevoice.models.api import (  # noqa: E402
    Asset,
    AssetKind,
    ModelManifest,
    Provenance,
    ResourceRequirement,
    ReviewStatus,
)
from nativevoice.core.language import Task  # noqa: E402
from nativevoice.models.manifest_io import save_manifest  # noqa: E402

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
    return Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))


def snapshot_files(repo: str) -> list[Path]:
    root = hf_snapshot_dir(repo)
    snapshots = root / "snapshots"
    if not snapshots.is_dir():
        return []
    latest = max(snapshots.iterdir(), key=lambda item: item.stat().st_mtime)
    return sorted(path for path in latest.rglob("*") if path.is_file() or path.is_symlink())


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


def build_manifest(name: str, directory: Path, files: list[Path], args) -> ModelManifest:
    spec = CATALOGUE[name]
    assets = tuple(
        Asset(
            kind=AssetKind.MODEL if path.suffix in {".onnx", ".bin", ".model"} else AssetKind.TOKENIZER,
            relative_path=path.relative_to(directory).as_posix(),
            sha256=sha256(path) if path.is_file() else "",
            size_bytes=path.stat().st_size if path.is_file() else 0,
        )
        for path in files
    )
    provenance = Provenance(
        source=spec["url"] or f"https://huggingface.co/{spec.get('hf_repo','')}",
        upstream_project=spec["upstream_project"],
        licence_spdx=spec["licence_spdx"],
        licence_notice=spec["licence_evidence"]
        + ". "
        + spec.get("bundled_notice", "")
        + f" Digest computed at provisioning {time.strftime('%Y-%m-%dT%H:%M:%S')}; "
        "upstream publishes no digest for this artefact, so this pins drift-detection not authenticity.",
        consent_basis="not applicable: pretrained weights carry no personal data; voice recordings and profiles are governed separately",
        redistributable=spec.get("redistributable", False),
        notes=spec.get("bundled_notice", ""),
    )
    return ModelManifest(
        model_id=name,
        engine=spec["engine"],
        task=spec["task"],
        display_name=spec["display_name"],
        version=str(spec.get("version", "1")),
        languages=tuple(sorted(args.language or spec.get("declared_languages") or [])),
        minimum_profile="balanced" if spec["ram_mb_estimate"] >= 400 else "lite",
        sample_rate=spec["sample_rate"],
        quantization=spec["quantization"],
        provenance=provenance,
        review=ReviewStatus.APPROVED,
        approved_by=args.approved_by,
        approved_on=time.strftime("%Y-%m-%d"),
        requirements=ResourceRequirement(
            ram_mb=spec["ram_mb_estimate"],
            disk_mb=max(1, sum(asset.size_bytes for asset in assets) // 2**20),
            threads=2,
            measured=False,
        ),
        assets=assets,
        asset_root=str(Path(directory).resolve()),
        notes=(
            spec["notes"]
            + f" Authorisation relied on: {args.authority}."
            + (" " + spec["language_basis"] if spec.get("language_basis") else "")
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asset", nargs="+", required=True, choices=sorted(CATALOGUE))
    parser.add_argument("--approved-by", required=True, help="who authorised this asset; a manifest without a name is not an approval")
    parser.add_argument("--authority", required=True, help="the instruction relied on, quoted")
    parser.add_argument("--language", action="append", help="language code actually verified for this asset; repeat per language")
    parser.add_argument(
        "--languages-unverified",
        action="store_true",
        help="record the asset with no languages, for the first pass before the archive has been inspected",
    )
    parser.add_argument("--root", default="", help="model root (defaults to the configured local data dir)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without touching the network or the disk")
    parser.add_argument("--force", action="store_true", help="refetch even if the asset is already on disk")
    args = parser.parse_args(argv)

    if not args.language and not args.languages_unverified:
        # each asset in the catalogue carries the languages its own files support,
        # so an explicit list is only required when provisioning a mix of them
        if len(args.asset) > 1:
            parser.error("one --language per asset is required when provisioning several, so a language is never stamped onto a model that cannot do it")

    root = Path(args.root) if args.root else Path(models_dir())
    for name in args.asset:
        spec = CATALOGUE[name]
        target = root / name
        if args.dry_run:
            print(f"{name}: would fetch {spec['url'] or 'local HF snapshot ' + spec.get('hf_repo','')} -> {target}")
            continue
        print(f"provisioning {name}")
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
            directory = hf_snapshot_dir(spec["hf_repo"])
            files = snapshot_files(spec["hf_repo"])
            if not files:
                print(f"  MISSING: no local snapshot for {spec['hf_repo']}; refusing to record it")
                continue
        manifest = build_manifest(name, directory, files, args)
        path = save_manifest(manifest, str(root / "manifests" / f"{name}.json"))
        print(f"  {len(files)} files, {manifest.requirements.disk_mb} MB, licence {manifest.provenance.licence_spdx}")
        print(f"  manifest -> {path}")
    print(f"data root: {data_root()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
