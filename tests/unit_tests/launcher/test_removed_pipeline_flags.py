"""The Nextflow evidence pipeline moved to Paludamentum; its options must point there."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("argv", [
    ["--nf_config", "conf/slurm_generic.config", "--genome", "g.fa", "--model_cfg", "diatoms"],
    ["--params_yaml", "params.yaml", "--nf_config=local"],
    ["--genome", "g.fa", "--model_cfg", "diatoms", "--proteins", "p.faa"],
])
def test_removed_pipeline_flags_point_to_paludamentum(argv):
    proc = subprocess.run(
        [sys.executable, str(REPO / "tiberius.py"), *argv],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 2
    assert "Paludamentum" in proc.stdout + proc.stderr
    assert "no longer part of Tiberius" in proc.stdout + proc.stderr


def test_no_pipeline_files_left():
    assert not (REPO / "paludamentum").exists()
    assert not (REPO / "conf").exists()
    assert not (REPO / "tiberius" / "evidence_pipeline_wrapper.py").exists()
    assert not (REPO / ".gitmodules").exists()
