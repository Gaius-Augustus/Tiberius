#!/usr/bin/env python3
import os
import sys
import yaml
import subprocess
import shutil
from copy import deepcopy
from pathlib import Path
from tiberius.tiberius_args import parseCmd
from tiberius.evidence_pipeline_wrapper import run_nextflow_pipeline

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()
SCRIPT_ROOT = Path(__file__).resolve().parent
SINGULARITY_IMAGE_URI = "docker://larsgabriel23/tiberius:latest"
SINGULARITY_IMAGE_PATH = SCRIPT_ROOT / "singularity" / "tiberius.sif"
DEFAULT_PARAMS = {
    "threads": 48,
    "outdir": "tiberius_results",
    "genome": None,
    "proteins": None,
    "rnaseq_sra_single": [],
    "rnaseq_sra_paired": [],
    "isoseq_sra": [],
    "rnaseq_single": [],
    "rnaseq_paired": [],
    "isoseq": [],
    "tiberius": {
        "run": True,
        "result": None,
        "model_cfg": None,
    },
    "mode": None,
    "scoring_matrix": str((SCRIPT_ROOT / "conf" / "blosum62.csv").resolve()),
    "prothint_conflict_filter": False,
}


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
    If --params_yaml is provided (and not using --run_nextflow), populate
    missing Tiberius args (genome, model_cfg) from that file.
    """
    if not args.params_yaml or args.nf_config:
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

def merge_dicts(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base (in-place) when overlay values are not None."""
    for key, value in overlay.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def collect_cli_params(args) -> dict:
    """Extract pipeline-relevant CLI args into a params override dict."""
    overrides = {}
    def maybe_set(key, val):
        if val not in [None, [], ""]:
            overrides[key] = val

    maybe_set("threads", args.threads)
    maybe_set("outdir", args.outdir)
    maybe_set("genome", args.genome)
    maybe_set("proteins", args.proteins if args.proteins else None)
    maybe_set("rnaseq_single", args.rnaseq_single if args.rnaseq_single else None)
    maybe_set("rnaseq_paired", args.rnaseq_paired if args.rnaseq_paired else None)
    maybe_set("rnaseq_sra_single", args.rnaseq_sra_single if args.rnaseq_sra_single else None)
    maybe_set("rnaseq_sra_paired", args.rnaseq_sra_paired if args.rnaseq_sra_paired else None)
    maybe_set("isoseq", args.isoseq if args.isoseq else None)
    maybe_set("isoseq_sra", args.isoseq_sra if args.isoseq_sra else None)
    maybe_set("mode", args.mode)
    maybe_set("scoring_matrix", args.scoring_matrix)
    maybe_set("prothint_conflict_filter", args.prothint_conflict_filter if args.prothint_conflict_filter else None)

    tib = {}
    if args.model_cfg:
        tib["model_cfg"] = args.model_cfg
    if args.tiberius_result:
        tib["result"] = args.tiberius_result
    if tib:
        tib["run"] = True
        overrides["tiberius"] = tib
    return overrides

def ensure_params_yaml(args):
    """
    Ensure args.params_yaml points to a file. If not supplied, build one from
    defaults + optional params file + CLI overrides, then write to outdir/params.yaml.
    """
    # Start with defaults
    params = deepcopy(DEFAULT_PARAMS)
    base_dir = Path.cwd()

    if args.params_yaml:
        params_path, loaded = load_params_yaml(args.params_yaml)
        base_dir = params_path.parent
        merge_dicts(params, loaded)

    cli_overrides = collect_cli_params(args)
    merge_dicts(params, cli_overrides)

    if not params.get("genome"):
        console.print("[bold red]A genome file must be specified (via --genome or params).[/bold red]")
        sys.exit(1)
    if params.get("tiberius", {}).get("run") and not params.get("tiberius", {}).get("model_cfg"):
        console.print("[bold red]A model config file must be specified (params.tiberius.model_cfg or --model_cfg).[/bold red]")
        sys.exit(1)

    outdir = params.get("outdir") or DEFAULT_PARAMS["outdir"]
    outdir_path = Path(outdir)
    if not outdir_path.is_absolute():
        outdir_path = (Path.cwd() / outdir_path).resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    params_path = outdir_path / "params.yaml"
    with params_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(params, fh, sort_keys=False)

    args.params_yaml = str(params_path)
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

def run_tiberius_in_singularity(args):
    if os.environ.get("TIBERIUS_IN_SINGULARITY") == "1":
        return False

    image_path = SINGULARITY_IMAGE_PATH
    if not image_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"[INFO] Pulling Singularity image to {image_path}")
        subprocess.run(
            ["singularity", "pull", str(image_path), SINGULARITY_IMAGE_URI],
            check=True,
        )
    cmd = ["singularity", "exec"]
    if has_nvidia_container_cli():
        cmd += ["--nvccli"]
    cmd += [
        "--nv",
        str(image_path), "python3",
        str(Path(__file__).resolve())
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
    Returns one of: show_cfg, list_cfg, nextflow, tiberius.
    Exits with a helpful message if required args are missing.
    """
    if args.show_cfg:
        if not args.model_cfg:
            console.print("[bold red]--show_cfg requires --model_cfg[/bold red]")
            sys.exit(1)
        return "show_cfg"

    if args.list_cfg:
        return "list_cfg"

    if args.nf_config or args.params_yaml:
        return "nextflow"

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
    elif mode == "nextflow":
        if not args.nf_config:
            args.nf_config = str((SCRIPT_ROOT / "conf" / "base.config").resolve())
        args = ensure_params_yaml(args)
        run_nextflow_pipeline(args)
    else:
        if args.singularity:
            run_tiberius_in_singularity(args)
        from tiberius.main import run_tiberius
        run_tiberius(args)

if __name__ == '__main__':
    main()
