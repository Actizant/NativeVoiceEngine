from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "tests"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

from fixtures import dc_offset_capture  # noqa: E402,F401  (re-exported as a pytest fixture)


def pytest_addoption(parser) -> None:
    """Opt-in switch for tests that load an approved model asset and run inference.

    Off by default: those tests load ~hundreds of MB of weights and take seconds,
    so the green suite must not depend on them. Pass ``--run-approved-assets`` (or
    set ``ULTRON_RUN_APPROVED_ASSET=1``) to actually execute them.
    """
    parser.addoption(
        "--run-approved-assets",
        action="store_true",
        default=False,
        help="run tests that perform real inference on an approved model asset",
    )


def pytest_configure(config) -> None:
    # Surface the CLI flag through the same env var the existing skip guards read,
    # so a single mechanism drives both the new and pre-existing asset tests.
    if config.getoption("--run-approved-assets", default=False):
        os.environ["ULTRON_RUN_APPROVED_ASSET"] = "1"
