"""Command surface for the measurement layer.

Deliberately thin: argument parsing and printing only. Every judgement lives in
`nativevoice.eval` and `nativevoice.diagnostics`, so a claim printed here can be
reproduced by calling the same functions from Python. Standard library `argparse`,
no formatting dependency, and no network import anywhere in this package.
"""

from __future__ import annotations

__all__: list[str] = []
