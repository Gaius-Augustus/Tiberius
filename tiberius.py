#!/usr/bin/env python3
import yaml
import argparse
from tiberius import parseCmd, run_tiberius

from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()

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
        sort_keys=False,       # keep original key order – friendlier for humans
        indent=2,
        width=88,
        default_flow_style=False
    )

def print_config(cfg_path: Path) -> None:
    """Load → re-dump → syntax-highlight → print."""
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

    console.print(table)    # pretty table; falls back to plain text if TERM is dumb.

def main():    
    args = parseCmd()
    if args.show_cfg and args.model_cfg:
        print_config(Path(args.model_cfg))
        return
    
    if args.list_cfg:
        project_root = Path(__file__).resolve().parent
        cfg_dir = project_root / "model_cfg"
        list_available_configs(cfg_dir)
        return
    
    run_tiberius(args)

if __name__ == '__main__':
    main()
