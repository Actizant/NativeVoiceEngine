"""`python -m nativevoice.cli.diagnose` - print what is actually known.

Three sections, because they answer three different questions: which machine this
answer came from, what this build can really do per language and task, and whether
the code is provably offline. Exit status is 0 only when no criterion is FAIL; a run
with no capability data exits 0 too but prints NOT_RUN, because "we have no data" is
a reported state here, not an implied pass - and `--strict` upgrades any NOT_RUN to
FAIL for callers that want absence of evidence to be evidence of absence.

Hardware probing is intentionally absent: this command reads core data and a
capability JSON file only, so it runs on a machine with no audio devices at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from ..core.errors import VoiceError
from ..core.language import Capability, CapabilityStatus, Task
from ..diagnostics.offline import package_source_root, static_scan
from ..eval.capability_matrix import build_matrix
from ..eval.report import (
    AcceptanceReport,
    CriterionResult,
    CriterionStatus,
    Evidence,
    host_summary,
    judge,
    not_run,
)

PROGRAM = "python -m nativevoice.cli.diagnose"

STATUS_ORDER = (
    CapabilityStatus.MEASURED,
    CapabilityStatus.PARTIAL,
    CapabilityStatus.DECLARED,
    CapabilityStatus.NO_MODEL,
    CapabilityStatus.UNSUPPORTED,
)


class CapabilityFileError(VoiceError):
    """The capability JSON is absent, unreadable, or not shaped like capabilities."""


def build_parser(argv: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Report host summary, real per-language capability status, and offline scan findings.",
    )
    parser.add_argument(
        "--capabilities",
        default="",
        metavar="PATH",
        help="JSON file holding the Capability list (a list, or {'capabilities': [...]}).",
    )
    parser.add_argument(
        "--scan-path",
        action="append",
        default=[],
        metavar="PATH",
        help="Extra directory tree to scan for network imports (repeatable).",
    )
    parser.add_argument("--max-wer", type=float, default=None, metavar="RATE", help="Fail any measured WER above this.")
    parser.add_argument("--max-first-audio-ms", type=float, default=None, metavar="MS", help="Fail any measured first-audio latency above this.")
    parser.add_argument("--strict", action="store_true", help="Treat NOT_RUN criteria as FAIL for the exit code.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON instead of text.")
    parser.add_argument("--version", action="store_true", help="Print the package version and exit.")
    return parser


def _entry_from_dict(raw: dict) -> tuple[str, Capability]:
    missing = {"language", "task", "status"} - set(raw)
    if missing:
        raise CapabilityFileError("capability entry missing keys", keys=sorted(missing), entry=str(raw)[:120])
    try:
        task = Task(str(raw["task"]))
        status = CapabilityStatus(str(raw["status"]).lower())
    except ValueError as exc:
        raise CapabilityFileError("capability entry has an unknown task or status", error=str(exc)) from exc
    value = raw.get("metric_value")
    limitations = raw.get("limitations") or ()
    if isinstance(limitations, str):
        limitations = (limitations,)
    return (
        str(raw["language"]),
        Capability(
            task=task,
            status=status,
            language=str(raw["language"]),
            engine=str(raw.get("engine", "")),
            metric_name=str(raw.get("metric_name", "")),
            metric_value=None if value is None else float(value),
            measured_on=str(raw.get("measured_on", "")),
            model_id=str(raw.get("model_id", "")),
            limitations=tuple(str(x) for x in limitations),
            evidence=str(raw.get("evidence", "")),
        ),
    )


def load_capabilities(path: str) -> list[tuple[str, Capability]]:
    """Read a capability JSON file into (language, Capability) pairs."""
    if not path:
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise CapabilityFileError("capability file unreadable", path=path, error=str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise CapabilityFileError("capability file is not valid JSON", path=path, error=str(exc)) from exc
    rows = payload.get("capabilities", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise CapabilityFileError("capability file must contain a list of capabilities", path=path)
    return [_entry_from_dict(raw) for raw in rows]


def capability_criteria(entries: Sequence[tuple[str, Capability]], *, max_wer: float | None, max_first_audio_ms: float | None) -> list:
    """One criterion per capability: PASS only where a measurement exists and clears a limit.

    A DECLARED capability is never turned into a PASS. Without a threshold flag the
    verdict is NOT_RUN, i.e. "measured, but nothing was acceptance-gated on it", which
    keeps the exit code tied to thresholds the caller actually set.
    """
    rows = []
    for language, capability in entries:
        name = f"{language}/{capability.task.value}"
        label = capability.metric_name or "metric"
        if capability.status is CapabilityStatus.MEASURED and capability.metric_value is not None:
            if label.upper().startswith("WER") and max_wer is not None:
                rows.append(judge(f"{name} {label}", capability.metric_value, max_wer, unit="", direction="max", language=language))
            elif label == "first_audio_ms" and max_first_audio_ms is not None:
                rows.append(judge(f"{name} {label}", capability.metric_value, max_first_audio_ms, unit=" ms", direction="max", language=language))
            else:
                rows.append(
                    not_run(
                        f"{name} {label}",
                        f"measured value {capability.metric_value} is recorded but no threshold was supplied for {label}",
                        language=language,
                    )
                )
            continue
        rows.append(
            not_run(
                f"{name} {capability.status.value}",
                capability_reason(capability),
                language=language,
            )
        )
    return rows


def capability_reason(capability: Capability) -> str:
    if capability.status is CapabilityStatus.DECLARED:
        return "declared by upstream only: not evaluated on this machine, so it cannot pass acceptance"
    if capability.status is CapabilityStatus.NO_MODEL:
        return "no approved model installed" + (f" (engine {capability.engine})" if capability.engine else "")
    if capability.status is CapabilityStatus.UNSUPPORTED:
        return "installed engine does not do this task for this language"
    if capability.status is CapabilityStatus.PARTIAL:
        return "partial: " + ("; ".join(capability.limitations) if capability.limitations else "limitations unstated")
    return "no metric recorded for this measured capability"


def offline_criteria(paths: Sequence[str], command_line: str) -> list:
    """Static-scan criterion per scanned tree: any finding is a FAIL, not a warning."""
    rows = []
    for path in paths:
        findings = static_scan(path)
        rendered = "; ".join(str(f) for f in findings[:12])
        rows.append(
            judge(
                f"offline static scan: {path}",
                float(len(findings)),
                0.0,
                unit=" findings",
                direction="max",
                reason=f"network-capable imports: {rendered}" if findings else "no network-capable imports found (diagnostics/offline.py is allowlisted as the guard itself)",
                evidence=Evidence.capture(command_line),
            )
        )
    return rows


def summarise(entries: Sequence[tuple[str, Capability]]) -> str:
    """Installed vs missing, per status, in the order a reader triages in."""
    if not entries:
        return "  no capability data supplied: nothing is installed *and* nothing is missing as far as this run knows"
    matrix = build_matrix(entries)
    lines = [matrix.render_text()]
    counts: dict[str, list[str]] = {}
    for language, capability in entries:
        counts.setdefault(capability.status.value, []).append(f"{language}/{capability.task.value}")
    for status in STATUS_ORDER:
        rows = counts.get(status.value, [])
        if rows:
            lines.append(f"  {status.value} ({len(rows)}): {', '.join(sorted(rows))}")
    return "\n".join(lines)


def build_report(entries, paths, *, max_wer, max_first_audio_ms, capability_error: str, command_line: str) -> AcceptanceReport:
    report = AcceptanceReport(
        title="ULTRON Native Voice - diagnose",
        generated_at=Evidence.capture(command_line).captured_at,
        host=host_summary(),
    )
    if capability_error:
        # A supplied-but-broken capability file is a FAIL: the operator asked for a
        # verdict and the data needed to give one is unusable.
        report.add(
            CriterionResult(
                name="capability data",
                status=CriterionStatus.FAIL,
                measured=None,
                threshold=None,
                unit="",
                direction="max",
                reason=capability_error,
            )
        )
    elif not entries:
        report.add(not_run("capability data", "no --capabilities file was given, so installed/missing capabilities are unknown"))
    else:
        report.extend(capability_criteria(entries, max_wer=max_wer, max_first_audio_ms=max_first_audio_ms))
    report.extend(offline_criteria(paths, command_line))
    return report


def main(argv: Sequence[str] | None = None, stream=None) -> int:
    """Run the diagnose command. Returns the process exit code."""
    out = stream or sys.stdout
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(argv_list)
    args = parser.parse_args(argv_list)
    if args.version:
        from .. import __version__

        print(f"ultron-native-voice {__version__}", file=out)
        return 0

    paths = [package_source_root(), *args.scan_path]
    entries: list[tuple[str, Capability]] = []
    capability_error = ""
    try:
        entries = load_capabilities(args.capabilities)
    except CapabilityFileError as exc:
        capability_error = str(exc)

    command_line = " ".join([PROGRAM, *argv_list])
    report = build_report(
        entries,
        paths,
        max_wer=args.max_wer,
        max_first_audio_ms=args.max_first_audio_ms,
        capability_error=capability_error,
        command_line=command_line,
    )
    summary = report.summarise()

    if args.json:
        print(json.dumps({"report": report.to_dict(), "capabilities": summarise(entries), "exit_code": _exit_code(summary, args.strict)}, indent=2, sort_keys=True), file=out)
    else:
        print("== host ==", file=out)
        print(f"  {report.host}", file=out)
        print("== capabilities (installed vs missing) ==", file=out)
        print(summarise(entries), file=out)
        print("== offline static scan ==", file=out)
        for path in paths:
            findings = static_scan(path)
            print(f"  {path}: {len(findings)} finding(s)", file=out)
            for finding in findings[:20]:
                print(f"    {finding}", file=out)
        print("== acceptance view ==", file=out)
        print(report.render(), file=out)

    code = _exit_code(summary, args.strict)
    if code:
        blamed = "FAIL rows" if summary.failed else "NOT_RUN rows, escalated by --strict"
        print(f"exit {code}: see {blamed} above", file=out)
    return code


def _exit_code(summary, strict: bool) -> int:
    """0 only with no FAIL. `--strict` additionally fails on any NOT_RUN."""
    if summary.failed:
        return 1
    if strict and summary.not_run:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
