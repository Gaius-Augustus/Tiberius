#!/usr/bin/env python3
import os
import sys
import json
import yaml
import subprocess
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from tiberius.tiberius_args import parseCmd
import importlib.metadata


from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()

SCRIPT_ROOT = Path(__file__).resolve().parent
SINGULARITY_IMAGE_REPO = "gaiusaugustus/tiberius"
# Pinned, tested image tag. Bump when a new image is published.
SINGULARITY_IMAGE_VERSION = importlib.metadata.version("tiberius")
SINGULARITY_IMAGE_URI = f"docker://{SINGULARITY_IMAGE_REPO}:{SINGULARITY_IMAGE_VERSION}"
SINGULARITY_IMAGE_PATH = SCRIPT_ROOT / "singularity" / f"tiberius_{SINGULARITY_IMAGE_VERSION}.sif"
DOCKER_HUB_TAGS_URL = (
    f"https://hub.docker.com/v2/repositories/{SINGULARITY_IMAGE_REPO}/tags"
    "?page_size=100"
)
# The Nextflow evidence pipeline moved to Paludamentum
# (https://github.com/Gaius-Augustus/Paludamentum), which runs Tiberius as one
# of its gene finders. These options of tiberius.py were removed with it.
REMOVED_PIPELINE_FLAGS = (
    "-c", "--nf_config", "--profile", "--nextflow_bin", "--resume", "--work_dir",
    "--check_tools", "--skip_singularity_check", "--dry_run",
    "--outdir", "--threads", "--proteins", "--odb12Partitions",
    "--rnaseq_single", "--rnaseq_paired", "--rnaseq_sra_single", "--rnaseq_sra_paired",
    "--isoseq", "--isoseq_sra", "--mode", "--scoring_matrix",
    "--prothint_conflict_filter", "--tiberius_result",
)
PALUDAMENTUM_URL = "https://github.com/Gaius-Augustus/Paludamentum"


def reject_removed_pipeline_flags(argv) -> None:
    """Exit with a pointer to Paludamentum if an option of the removed pipeline is used."""
    used = sorted({arg.split("=", 1)[0] for arg in argv if arg.split("=", 1)[0] in REMOVED_PIPELINE_FLAGS})
    if not used:
        return
    console.print(
        f"[bold red]{', '.join(used)}: the Nextflow evidence pipeline is no longer part of Tiberius.[/bold red]"
    )
    console.print(f"It moved to Paludamentum ({PALUDAMENTUM_URL}), which runs Tiberius as one of its gene finders:")
    console.print(f"    git clone --recursive {PALUDAMENTUM_URL}")
    console.print("    paludamentum --nf_config slurm_generic --genome genome.fa --model_cfg diatoms")
    console.print("    paludamentum --params_yaml params.yaml --nf_config slurm_generic")
    sys.exit(2)


def has_nvidia_container_cli() -> bool:
    if not shutil.which("nvidia-smi"):
        return False

    if not shutil.which("nvidia-container-cli"):
        return False

    try:
        subprocess.run(
            ["nvidia-container-cli", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        return False

    return True


def singularity_supports_nvccli() -> bool:
    # --nvccli requires Singularity >= 3.9 / Apptainer >= 1.0 built with nvccli support.
    try:
        result = subprocess.run(
            ["singularity", "run", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return "--nvccli" in (result.stdout + result.stderr)


def load_params_yaml(params_path: str) -> tuple[Path, dict]:
    """Load a params YAML file and return (path, data)."""
    params_file = Path(params_path).expanduser().resolve()
    if not params_file.exists():
        console.print(f"[bold red]Params YAML not found:[/bold red] {params_path}")
        sys.exit(1)
    with params_file.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        console.print(f"[bold red]Expected a mapping at top-level of params file:[/bold red] {params_file}")
        sys.exit(1)
    return params_file, data

def hydrate_args_from_params(args):
    """
    If --params_yaml is provided, populate missing Tiberius args (genome,
    model_cfg) from that file, e.g. from the params file of a Paludamentum run.
    """
    if not args.params_yaml:
        return args

    params_path, params = load_params_yaml(args.params_yaml)
    base_dir = params_path.parent

    if not args.genome and params.get("genome"):
        genome_path = Path(params["genome"])
        if not genome_path.is_absolute():
            genome_path = (base_dir / genome_path).resolve()
        args.genome = str(genome_path)

    if not args.model_cfg:
        tiberius_cfg = params.get("tiberius") or {}
        if isinstance(tiberius_cfg, dict) and tiberius_cfg.get("model_cfg"):
            cfg_path = Path(tiberius_cfg["model_cfg"])
            if not cfg_path.is_absolute():
                cfg_path = (base_dir / cfg_path).resolve()
            args.model_cfg = str(cfg_path)

    return args

def resolve_model_cfg(cfg_value: str) -> Path:
    """
    Resolve a model config path. Accepts bare names like 'diatoms' or 'diatoms.yaml'
    and searches the local model_cfg directory if a direct path is not found.
    """

    def resolve_candidate(parent_dir):
        alt = parent_dir / cfg_value
        if alt.exists():
            return alt.resolve()

        stem = candidate.name
        if stem.endswith(".yaml") or stem.endswith(".yml"):
            stem = stem.rsplit(".", 1)[0]

        for ext in (".yaml", ".yml"):
            alt = parent_dir / f"{stem}{ext}"
            if alt.exists():
                return alt.resolve()
        return None

    candidate = Path(cfg_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    project_root = Path(__file__).resolve().parent
    cfg_dir = project_root / "model_cfg"
    cfg_file = resolve_candidate(cfg_dir)
    if cfg_file is not None:
        return cfg_file

    project_root = Path(__file__).resolve().parent
    cfg_dir = project_root / "model_cfg" / "superseded"
    cfg_file = resolve_candidate(cfg_dir)
    if cfg_file is not None:
        console.print(f"WARNING: The chosen model {cfg_value} is superseded, there may be a newer model available")
        return cfg_file

    console.print(f"[bold red]Model config not found:[/bold red] {cfg_value}")
    console.print(f"Searched: {candidate}, {cfg_dir}/{cfg_value}, {cfg_dir}/{candidate.name}.yaml/.yml")
    sys.exit(1)

def _parse_semver(tag: str):
    """Parse '1.2.3' or 'v1.2.3' into a tuple of ints; return None if not semver-shaped."""
    stripped = tag.lstrip("vV")
    parts = stripped.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def _fetch_latest_image_tag(timeout: float = 3.0):
    """Query Docker Hub for the highest semver-shaped tag. Returns None on any failure."""
    try:
        req = urllib.request.Request(
            DOCKER_HUB_TAGS_URL,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None

    versions = []
    for tag_obj in payload.get("results", []):
        name = tag_obj.get("name", "")
        parsed = _parse_semver(name)
        if parsed is not None:
            versions.append((parsed, name))
    if not versions:
        return None
    versions.sort(key=lambda x: x[0])
    return versions[-1][1]


def _warn_if_newer_image_available():
    """Print a warning if Docker Hub advertises a newer semver tag than the pinned one."""
    latest_tag = _fetch_latest_image_tag()
    if latest_tag is None:
        return
    current_parsed = _parse_semver(SINGULARITY_IMAGE_VERSION)
    latest_parsed = _parse_semver(latest_tag)
    if current_parsed is None or latest_parsed is None:
        return
    if latest_parsed > current_parsed:
        console.print(
            f"[yellow][WARNING] A newer Singularity image is available: "
            f"{latest_tag} (current pinned: {SINGULARITY_IMAGE_VERSION}). "
            f"Update Tiberius (e.g. 'git pull') to use it.[/yellow]"
        )


def _list_old_image_files(image_dir: Path, current_path: Path):
    """Return cached tiberius_*.sif files that don't match the pinned version."""
    if not image_dir.exists():
        return []
    current_resolved = current_path.resolve()
    return sorted(
        p for p in image_dir.glob("tiberius_*.sif")
        if p.resolve() != current_resolved
    )


def _cleanup_old_image_files(old_images):
    for p in old_images:
        try:
            p.unlink()
            console.print(f"[INFO] Removed old Singularity image {p}")
        except OSError as exc:
            console.print(f"[yellow][WARNING] Could not remove {p}: {exc}[/yellow]")


def run_tiberius_in_singularity(args):
    if os.environ.get("TIBERIUS_IN_SINGULARITY") == "1":
        return False

    _warn_if_newer_image_available()

    image_path = SINGULARITY_IMAGE_PATH
    pulled_now = False
    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[INFO] Pulling Singularity image to {image_path}")
        subprocess.run(
            ["singularity", "pull", str(image_path), SINGULARITY_IMAGE_URI],
            check=True,
        )
        pulled_now = True

    old_images = _list_old_image_files(image_path.parent, image_path)
    if args.cleanup_old_singularity_images:
        if old_images:
            _cleanup_old_image_files(old_images)
    elif pulled_now and old_images:
        listing = "\n  ".join(str(p) for p in old_images)
        console.print(
            "[yellow][WARNING] Old Singularity image(s) still on disk:\n"
            f"  {listing}\n"
            "Pass --cleanup_old_singularity_images to remove them.[/yellow]"
        )
    cmd = ["singularity", "run"]
    if has_nvidia_container_cli() and singularity_supports_nvccli():
        cmd += ["--nvccli"]
    cmd += ["--nv"]
    # Only forward CUDA_VISIBLE_DEVICES when explicitly set; passing an empty
    # value hides all GPUs from the container (issue #105).
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        cmd += ["--env", f"CUDA_VISIBLE_DEVICES={cuda_visible}"]
    cmd += [
        "--cleanenv",
        str(image_path), "/usr/bin/python3",
        "/opt/Tiberius/tiberius.py"
        ]

    passthrough = [arg for arg in sys.argv[1:] if arg != "--singularity"]
    cmd.extend(passthrough)
    env = os.environ.copy()
    env["TIBERIUS_IN_SINGULARITY"] = "1"
    console.print("[INFO] Launching Tiberius inside Singularity.")
    completed = subprocess.run(cmd, env=env)
    raise SystemExit(completed.returncode)

def validate_mode(args) -> str:
    """
    Determine which mode to run and enforce required argument combinations.
    Returns one of: show_cfg, list_cfg, tiberius.
    Exits with a helpful message if required args are missing.
    """
    if args.show_cfg:
        if not args.model_cfg:
            console.print("[bold red]--show_cfg requires --model_cfg[/bold red]")
            sys.exit(1)
        return "show_cfg"

    if args.list_cfg:
        return "list_cfg"

    missing = []
    if not args.genome:
        missing.append("--genome")
    if not args.model_cfg and not args.model_lstm_old and not args.model_old and not args.model:
        missing.append("--model_cfg")
    if missing:
        console.print(f"[bold red]Missing required argument(s): {', '.join(missing)}[/bold red]")
        sys.exit(1)
    return "tiberius"

def load_yaml(cfg_path: Path) -> dict:
    """
    Reads the config file and returns a Python dict.
    If the file is not valid YAML an error is raised early.
    """
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        console.print(f"[bold red]YAML syntax error:[/bold red]\n{exc}")
        sys.exit(1)

def pretty_dump(data: dict) -> str:
    """
    Returns a canonical YAML string with 2-space indentation, preserving order.
    Comments inside values (like the long 'comment' field) are kept as-is
    because they’re part of the string – no special handling needed.
    """
    return yaml.dump(
        data,
        sort_keys=False,
        indent=2,
        width=88,
        default_flow_style=False
    )

def print_config(cfg_path: Path) -> None:
    raw_cfg = pretty_dump(load_yaml(cfg_path))
    syntax = Syntax(raw_cfg, "yaml", line_numbers=True, word_wrap=False)
    console.print(syntax)

def list_available_configs(cfg_dir: Path) -> None:
    """
    Scan cfg_dir for YAML files and print 'file_stem: target_species'.
    Raises early if no configs are found or a file lacks target_species.
    """
    cfg_paths = sorted(cfg_dir.glob("*.yml")) + sorted(cfg_dir.glob("*.yaml"))

    if not cfg_paths:
        console.print(f"[yellow]No *.yaml files found in {cfg_dir}[/yellow]")
        sys.exit(1)

    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Config", style="green")
    table.add_column("Target species")

    for cfg in cfg_paths:
        data = load_yaml(cfg)
        species = data.get("target_species", "<missing key>")
        table.add_row(cfg.stem, str(species))

    console.print(table)

def main():
    reject_removed_pipeline_flags(sys.argv[1:])
    args = parseCmd()
    args = hydrate_args_from_params(args)
    if args.model_cfg:
        args.model_cfg = str(resolve_model_cfg(args.model_cfg))

    mode = validate_mode(args)
    if mode == "show_cfg":
        print_config(Path(args.model_cfg))
    elif mode == "list_cfg":
        project_root = Path(__file__).resolve().parent
        cfg_dir = project_root / "model_cfg"
        list_available_configs(cfg_dir)
    else:
        if args.singularity:
            run_tiberius_in_singularity(args)
        from tiberius.main import run_tiberius
        run_tiberius(args)

if __name__ == '__main__':
    main()
