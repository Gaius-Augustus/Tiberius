"""Shim to the Paludamentum evidence pipeline.

The Nextflow pipeline and its launcher live in the git submodule
``<repo>/paludamentum`` (https://github.com/Gaius-Augustus/Paludamentum).
The import is lazy, so that direct Tiberius runs never need the submodule.
"""
from __future__ import annotations

import sys
from pathlib import Path

PALUDAMENTUM_ROOT = Path(__file__).resolve().parent.parent / "paludamentum"


def _launcher():
    if not (PALUDAMENTUM_ROOT / "main.nf").exists():
        raise SystemExit(
            f"The Paludamentum pipeline was not found at {PALUDAMENTUM_ROOT}.\n"
            "It is a git submodule of Tiberius. Fetch it with:\n"
            "    git submodule update --init --recursive"
        )
    # The submodule root must come first on sys.path. Otherwise the directory
    # <repo>/paludamentum itself is imported as an empty namespace package.
    root = str(PALUDAMENTUM_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from paludamentum import launcher
    return launcher


def pipeline_paths(root_override: str | None = None):
    return _launcher().pipeline_paths(root_override or PALUDAMENTUM_ROOT)


def resolve_nf_config(value: str) -> Path:
    return _launcher().resolve_nf_config(value, PALUDAMENTUM_ROOT)


def run_nextflow_pipeline(args) -> None:
    _launcher().run_nextflow_pipeline(args, genefinder="tiberius", pipeline_root=PALUDAMENTUM_ROOT)
