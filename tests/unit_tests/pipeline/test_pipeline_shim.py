"""The evidence pipeline lives in the Paludamentum submodule; test the shim to it."""
import sys
from pathlib import Path

import pytest

from tiberius import evidence_pipeline_wrapper as shim

REPO = Path(__file__).resolve().parents[3]
HAS_SUBMODULE = (REPO / "paludamentum" / "main.nf").exists()
needs_submodule = pytest.mark.skipif(not HAS_SUBMODULE, reason="paludamentum submodule not checked out")


def test_root_is_the_submodule_directory():
    assert shim.PALUDAMENTUM_ROOT == REPO / "paludamentum"


def test_missing_submodule_gives_a_clean_hint(tmp_path, monkeypatch):
    monkeypatch.setattr(shim, "PALUDAMENTUM_ROOT", tmp_path / "paludamentum")
    with pytest.raises(SystemExit, match="git submodule update --init --recursive"):
        shim.pipeline_paths()


@needs_submodule
def test_pipeline_paths_exist():
    root, main_nf, base_config = shim.pipeline_paths()
    assert root == (REPO / "paludamentum").resolve()
    assert main_nf.is_file() and base_config.is_file()
    assert (root / "conf" / "blosum62.csv").is_file()


@needs_submodule
def test_launcher_is_imported_from_the_submodule():
    shim.pipeline_paths()
    module = sys.modules["paludamentum.launcher"]
    assert Path(module.__file__).resolve() == (REPO / "paludamentum" / "paludamentum" / "launcher.py").resolve()


@needs_submodule
@pytest.mark.parametrize("name", ["base", "local", "slurm_generic", "greifswald_hpc", "user_hpc_template"])
def test_conf_shims_include_existing_configs(name):
    shim_file = REPO / "conf" / f"{name}.config"
    assert f"includeConfig '../paludamentum/conf/{name}.config'" in shim_file.read_text()
    assert (REPO / "paludamentum" / "conf" / f"{name}.config").is_file()


@needs_submodule
def test_nf_config_resolution(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO)
    assert shim.resolve_nf_config("conf/slurm_generic.config") == REPO / "conf" / "slurm_generic.config"
    monkeypatch.chdir(tmp_path)
    assert shim.resolve_nf_config("slurm_generic") == (REPO / "paludamentum" / "conf" / "slurm_generic.config").resolve()
