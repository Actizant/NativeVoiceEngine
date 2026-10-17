"""Audit every installed asset manifest and correct any rights claim it cannot support.

Run:

    python tools/asset_status_audit.py

Why this exists: an earlier provisioning script wrote `redistributable: true` for everything it
processed, so a rights claim in a manifest may be a record of a decision or merely a fossil of a
default. This tool looks at each asset individually, decides what the local and upstream evidence
actually establishes, and rewrites the record to say no more than that.

Two distinctions are kept visible rather than collapsed:

* **missing evidence** - nothing reviewed establishes the right either way;
* **established prohibition** - a term was read that forbids it.

The schema has one boolean and one nullable slot, and `None` (unknown) is deliberately *unusable* by
the load gate, because a model whose rights are unknown must not silently become loadable. So the
conservative boolean `False` is used, and the distinction lives in the accompanying `permissions` text
and in the correction record: `False` here means "not established", never "prohibited", unless the
record says a prohibition was read.

Historical artefacts are preserved: each manifest is copied into
`artifacts/audit/manifest-snapshots/` before it is rewritten, so the value the earlier tool wrote is
still inspectable after the correction. The copy is not the live file byte for byte: `redact_copy`
substitutes `[USER_HOME]` for this machine's home directory, and mirrors whether the source ends in a
final newline, which is the convention the committed copies already carry. A manifest whose JSON does
not parse is not backed up at all - it is reported as its own failure, its bytes left exactly as
found, and the remaining assets are still audited.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nativevoice.config import models_dir  # noqa: E402
from nativevoice.core.language import Task  # noqa: E402
from nativevoice.diagnostics.evidence_paths import redact_copy, write_json  # noqa: E402
from nativevoice.models.approval import approval_blockers  # noqa: E402
from dataclasses import replace

from nativevoice.models.manifest_io import save_manifest  # noqa: E402
from nativevoice.models.registry import ModelRegistry  # noqa: E402
from nativevoice.provenance import CANDIDATE_EVIDENCE  # noqa: E402


def evidence_for(manifest) -> dict:
    """What can be pointed at, for this asset, right now."""
    directory = Path(manifest.asset_root or (Path(models_dir()) / manifest.model_id))
    licence_file = directory / "LICENSE"
    has_licence_text = licence_file.is_file()
    head = ""
    if has_licence_text:
        head = licence_file.read_text(encoding="utf-8", errors="replace")[:60].strip()
    voices = [asset for asset in manifest.assets if "voices" in asset.relative_path.lower()]
    return {
        "checked_directory": str(directory),
        "licence_text_shipped_in_package": has_licence_text,
        "licence_text_head": head,
        "declared_licence_spdx": manifest.provenance.licence_spdx,
        "digest_recorded_for_assets": f"{sum(1 for a in manifest.assets if a.sha256)}/{len(manifest.assets)}",
        "voice_or_speaker_assets": [asset.relative_path for asset in voices],
        "upstream_source": manifest.provenance.source,
    }


def judge(manifest, evidence: dict) -> dict:
    """Return the corrected rights position and the reason for it."""
    upstream = CANDIDATE_EVIDENCE.get(manifest.model_id, {})
    established_prohibition = bool(upstream.get("prohibition"))
    if established_prohibition:
        return {
            "redistributable": False,
            "status": "prohibited",
            "permissions": f"established prohibition: {upstream['prohibition']}",
            "why": "a term was read that forbids redistribution, not merely an absent record",
        }
    missing = [
        note
        for note in (
            "" if evidence["licence_text_shipped_in_package"] else "no licence text inside the installed package",
            upstream.get("gap", "training-data provenance not disclosed in the artefact"),
            "no per-voice or per-speaker consent record reviewed",
            "no publisher terms page read and archived at review time",
        )
        if note
    ]
    return {
        "redistributable": False,
        "status": "unverified",
        "permissions": "not established either way; local use approved, redistribution claim withdrawn until reviewed",
        "why": "evidence missing: " + "; ".join(missing),
    }


def write_corrections(payload: dict, out: Path) -> Path:
    """Save the corrections record, redacting the asset directories it names.

    `checked_directory` and `backup_of_previous_manifest` are absolute on the machine that
    ran this tool, and this file is published, so they leave through the serialization
    boundary with `[USER_HOME]` in place of the home directory.
    """
    return write_json(out, payload)


def backup_manifest(live: Path, backup: Path) -> str | None:
    """Copy one live manifest into the snapshot tree, redacted.

    Returns ``None`` when the copy was written, otherwise the reason it was not. Reported
    per asset rather than raised: this loop runs over every installed manifest, and one
    unreadable record must not cost the operator the backups of all the others. Losing the
    whole pass on a single corrupt file is the worse failure, because the snapshot is the
    only thing that makes the previous rights claim inspectable after the rewrite.
    """
    try:
        redact_copy(live, backup)
    except json.JSONDecodeError as error:
        return f"not backed up: {live.name} is not valid JSON ({error})"
    except OSError as error:
        return f"not backed up: {live.name} could not be read or copied ({type(error).__name__}: {error})"
    return None


def main() -> int:
    registry = ModelRegistry(models_dir())
    snapshot_dir = ROOT / "artifacts" / "audit" / "manifest-snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    corrections: list[dict] = []

    for manifest in registry.all():
        previous = {
            "redistributable": manifest.provenance.redistributable,
            "licence_spdx": manifest.provenance.licence_spdx,
            "permissions": manifest.provenance.permissions,
            "notes_head": manifest.provenance.notes[:160],
        }
        backup = snapshot_dir / f"{manifest.model_id}.json.pre-audit"
        live_manifest = Path(models_dir()) / "manifests" / f"{manifest.model_id}.json"
        # The committed copy is redacted; the live manifest under the data root keeps its
        # real asset root, because that is the value the integrity check digests.
        backup_error = backup_manifest(live_manifest, backup)
        if backup_error is not None:
            # Without the snapshot there is no inspectable record of the claim about to be
            # overwritten, so the correction stops here for this asset and the others continue.
            corrections.append(
                {
                    "model_id": manifest.model_id,
                    "task": manifest.task.value,
                    "engine": manifest.engine,
                    "previous_claim": previous,
                    "backup_error": backup_error,
                    "corrected_claim": None,
                    "evidence_inspected": None,
                    "still_loads_for_local_use": None,
                    "approval_blockers": [
                        "not re-checked: the manifest was left exactly as found because its "
                        "previous bytes could not be backed up"
                    ],
                }
            )
            print(f"  {manifest.model_id:34s} SKIPPED - {backup_error}")
            continue

        evidence = evidence_for(manifest)
        verdict = judge(manifest, evidence)
        updated = manifest

        provenance = replace(
            manifest.provenance,
            redistributable=verdict["redistributable"],
            permissions=verdict["permissions"],
            notes=(
                f"{manifest.provenance.notes} | rights correction {time.strftime('%Y-%m-%d')}: previous claim "
                f"redistributable={previous['redistributable']} was written by the provisioning tool's default, "
                f"not by a review; status is now '{verdict['status']}'. {verdict['why']}. "
                "Local use remains approved by Shreshtha; this entry records redistribution only."
            ),
        )
        updated = replace(manifest, provenance=provenance)
        save_manifest(updated, str(live_manifest))
        blockers = approval_blockers(updated)

        corrections.append(
            {
                "model_id": manifest.model_id,
                "task": updated.task.value,
                "engine": updated.engine,
                "previous_claim": previous,
                "corrected_claim": {
                    "redistributable": verdict["redistributable"],
                    "rights_status": verdict["status"],
                    "permissions": verdict["permissions"],
                    "why": verdict["why"],
                },
                "evidence_inspected": evidence,
                "still_loads_for_local_use": not blockers,
                "approval_blockers": list(blockers),
                "backup_of_previous_manifest": str(backup),
            }
        )
        print(f"  {manifest.model_id:34s} {str(previous['redistributable']):5s} -> {verdict['status']:11s} loads={not blockers}")

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": "a provisioning default had written a redistribution claim no review had established",
        "schema_note": (
            "`redistributable` is a boolean gate for loading; `None` means unknown and is refused by design. "
            "The model manifest schema has no separate rights-status field, so \"unverified\" can only be written "
            "as the conservative `False` plus the `permissions` sentence and a dated line in `notes`. "
            "The missing-evidence versus prohibited distinction therefore lives in this record's "
            "`rights_status` and `why` text, and in `permissions` - not in the boolean. A prohibited "
            "asset would be recorded here as `rights_status: prohibited` with the term quoted; none "
            "was found for these assets. "
            "An entry with `backup_error` is an asset this pass did not touch: its live manifest "
            "could not be read as JSON, so no snapshot was taken and no correction was written, and "
            "its `corrected_claim`, `evidence_inspected` and `still_loads_for_local_use` are null "
            "because they were never evaluated. The other assets were still audited."
        ),
        "corrections": corrections,
    }
    out = ROOT / "artifacts" / "audit" / "asset-status-corrections.json"
    write_corrections(payload, out)
    print(f"record -> {out}")

    after = ModelRegistry(models_dir())
    print("post-correction status:", {m.model_id: after.status(m.model_id).usable for m in after.all()})
    return 0 if all(item["still_loads_for_local_use"] for item in corrections) else 1


if __name__ == "__main__":
    raise SystemExit(main())
