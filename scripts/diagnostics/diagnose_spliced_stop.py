#!/usr/bin/env python3
"""Wrapper that runs Tiberius with the NSS spliced-stop debug hook enabled.

Invoke exactly like `tiberius.py`, but add `--debug_out <dir>`. The
debug dir will receive `spans.jsonl` with one record per detected
spliced-stop span (see `spliced_stop_hook.py` for the schema). Everything
else — model/config loading, GTF/GFF3 output, logging — is the normal
Tiberius pipeline.

Example:
    python scripts/diagnostics/diagnose_spliced_stop.py \\
        --genome tests/data/small.fa --model_cfg chlorophyta \\
        --out small.gtf --debug_out debug_small/
    python scripts/diagnostics/analyze_spliced_stop_debug.py debug_small/

Afterwards you have:
    debug_small/spans.jsonl  — per-span JSONL records
    small.gtf                — same output you'd get from tiberius.py

Run on a small genome (e.g. Bathycoccus, ~15 Mb). Do NOT run on brain /
HPC — this is strictly a local diagnostic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _split_debug_arg(argv: list[str]) -> tuple[list[str], str | None]:
    """Pull `--debug_out <dir>` or `--debug_out=<dir>` out of argv before
    handing the remainder to Tiberius' own argument parser (which doesn't
    know this flag)."""
    rest: list[str] = []
    debug_dir: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--debug_out":
            if i + 1 >= len(argv):
                print("--debug_out requires a path argument", file=sys.stderr)
                sys.exit(2)
            debug_dir = argv[i + 1]
            i += 2
            continue
        if a.startswith("--debug_out="):
            debug_dir = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    return rest, debug_dir


def main() -> int:
    rest, debug_dir = _split_debug_arg(sys.argv[1:])
    if debug_dir is None:
        print(
            "ERROR: --debug_out <dir> is required (that's the whole point of this wrapper)",
            file=sys.stderr,
        )
        return 2

    debug_path = Path(debug_dir).expanduser().resolve()
    debug_path.mkdir(parents=True, exist_ok=True)
    os.environ["TIBERIUS_SPLICED_STOP_DEBUG_DIR"] = str(debug_path)
    # Also enable the stderr timing probes in PredictionGTF._patch_spliced_stops
    # so a single invocation of this wrapper gives both the per-span jsonl and
    # the per-call timing breakdown.
    os.environ.setdefault("TIBERIUS_SPLICED_STOP_TIMING", "1")

    # Replace argv so Tiberius' arg parser sees only its known flags.
    sys.argv = [sys.argv[0]] + rest

    # Importing the hook triggers the monkey-patch of
    # PredictionGTF._patch_spliced_stops. It MUST happen before
    # `tiberius.main.run_tiberius` runs so the patched method is bound.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import spliced_stop_hook  # noqa: F401

    from tiberius import parseCmd
    from tiberius.main import run_tiberius

    args = parseCmd()
    run_tiberius(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
