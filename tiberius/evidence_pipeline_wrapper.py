from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import yaml


GLOB_CHARS = set("*?[]{}")

DEFAULT_TOOL_BINARIES: Dict[str, str] = {
    "hisat2": "hisat2",
    "hisat2_build": "hisat2-build",
    "minimap2": "minimap2",
    "stringtie": "stringtie",
    "samtools": "samtools",
    "transdecoder_longorfs": "TransDecoder.LongOrfs",
    "transdecoder_predict": "TransDecoder.Predict",
    "transdecoder_util_gtf2fa": "gtf_genome_to_cdna_fasta.pl",
    "transdecoder_util_orf2genome": "cdna_alignment_orf_to_genome_orf.pl",
    "transdecoder_gtf2gff": "gtf_to_alignment_gff3.pl",
    "diamond": "diamond",
    "bedtools": "bedtools",
    "miniprot": "miniprot",
    "miniprot_boundary_scorer": "miniprot_boundary_scorer",
    "miniprothint": "miniprothint.py",
    "bam2hints": "bam2hints",
}

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "hisat2": "HISAT2 aligner (params.tools.hisat2)",
    "hisat2_build": "HISAT2 indexer (params.tools.hisat2_build)",
    "minimap2": "Minimap2 aligner (params.tools.minimap2)",
    "stringtie": "StringTie assembler (params.tools.stringtie)",
    "samtools": "Samtools (params.tools.samtools)",
    "transdecoder_longorfs": "TransDecoder.LongOrfs (params.tools.transdecoder_longorfs)",
    "transdecoder_predict": "TransDecoder.Predict (params.tools.transdecoder_predict)",
    "transdecoder_util_gtf2fa": "gtf_genome_to_cdna_fasta.pl (params.tools.transdecoder_util_gtf2fa)",
    "transdecoder_util_orf2genome": "cdna_alignment_orf_to_genome_orf.pl (params.tools.transdecoder_util_orf2genome)",
    "transdecoder_gtf2gff": "gtf_to_alignment_gff3.pl (params.tools.transdecoder_gtf2gff)",
    "diamond": "DIAMOND aligner (params.tools.diamond)",
    "bedtools": "BEDTools (params.tools.bedtools)",
    "miniprot": "MiniProt aligner (params.tools.miniprot)",
    "miniprot_boundary_scorer": "MiniProt boundary scorer (params.tools.miniprot_boundary_scorer)",
    "miniprothint": "MiniProtHint (params.tools.miniprothint)",
    "bam2hints": "BAM2HINTS (params.tools.bam2hints)",
}

GENERAL_COMMANDS = {
    "nextflow": "Nextflow executable used to launch the pipeline",
    "java": "Java runtime (version 11 or newer)",
    "singularity": "Singularity/Apptainer runtime",
    "python3": "System Python 3 interpreter",
    "perl": "Perl interpreter (needed for aln2hints.pl)",
    "fasterq-dump": "SRA Toolkit fasterq-dump utility",
}

OPTIONAL_COMMANDS = {
    "prefetch": "SRA Toolkit prefetch utility (optional but recommended for SRA downloads)",
}


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent 


def pipeline_paths(root_override: str | None = None) -> Tuple[Path, Path, Path]:
    repo_root = Path(root_override).expanduser().resolve() if root_override else _default_repo_root()
    pipeline_main = repo_root / "tiberius" / "main.nf"
    base_config = repo_root / "conf" / "base.config"
    return repo_root, pipeline_main, base_config


def load_params(params_path: Path) -> Dict:
    with params_path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Expected a mapping at the top-level of {params_path}, got {type(data).__name__}.")
    return data


def has_glob_char(value: str) -> bool:
    return any(char in value for char in GLOB_CHARS)


def expand_braces(pattern: str) -> List[str]:
    start = pattern.find("{")
    if start == -1:
        return [pattern]
    end = pattern.find("}", start)
    if end == -1:
        return [pattern]
    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    choices = pattern[start + 1 : end].split(",")
    expanded = []
    for choice in choices:
        expanded.extend(expand_braces(f"{prefix}{choice}{suffix}"))
    return expanded


def iter_strings(value) -> Iterable[str]:
    if isinstance(value, (list, tuple, set)):
        for sub in value:
            yield from iter_strings(sub)
    elif isinstance(value, (str, Path)):
        yield str(value)
    elif value is None:
        return
    else:
        raise TypeError(f"Unsupported value type in params file: {type(value)}")


def resolve_data_entries(value, base_dir: Path) -> Tuple[List[Path], List[str]]:
    resolved: List[Path] = []
    errors: List[str] = []
    for raw in iter_strings(value):
        cleaned = raw.strip()
        if not cleaned:
            continue
        expanded = os.path.expandvars(os.path.expanduser(cleaned))
        brace_patterns = expand_braces(expanded) if "{" in cleaned else [expanded]
        for pattern in brace_patterns:
            candidate = Path(pattern)
            candidates = []
            if candidate.is_absolute():
                candidates = [candidate]
            else:
                candidates = [(base_dir / candidate).resolve(), (Path.cwd() / candidate).resolve()]

            if has_glob_char(pattern):
                matches: List[str] = []
                for cand in candidates:
                    matches.extend(glob.glob(str(cand), recursive=True))
                if matches:
                    for match in matches:
                        resolved.append(Path(match))
                else:
                    errors.append(f"No files matched pattern '{cleaned}'.")
            else:
                resolved.extend(candidates)
    return resolved, errors


def validate_input_data(params: Dict, params_path: Path) -> List[str]:
    errors: List[str] = []
    base_dir = params_path.parent

    def ensure_required(key: str, label: str) -> None:
        value = params.get(key)
        if not value:
            errors.append(f"{label} ('{key}') is not set in {params_path}")
            return
        check_entries(value, label)

    def check_entries(value, label: str) -> None:
        files, errs = resolve_data_entries(value, base_dir)
        errors.extend(errs)
        if not files:
            return
        for file_path in files:
            if not file_path.exists():
                errors.append(f"{label} missing: {file_path}")
            elif not file_path.is_file():
                errors.append(f"{label} is not a file: {file_path}")

    ensure_required("genome", "Genome FASTA")
    # ensure_required("proteins", "Protein FASTA")

    optional_fields = {
        "rnaseq_single": "RNA-Seq single-end FASTQ",
        "rnaseq_paired": "RNA-Seq paired-end FASTQ",
        "isoseq": "Iso-Seq FASTQ",
        "scoring_matrix": "Scoring matrix",
        "proteins": "Protein FASTA",
    }
    for key, label in optional_fields.items():
        if key in params and params.get(key):
            check_entries(params[key], label)

    # tiberius_cfg = params.get("tiberius") or {}
    # if isinstance(tiberius_cfg, dict) and tiberius_cfg.get("run"):
    #     model_cfg = tiberius_cfg.get("model_cfg")
    #     if not model_cfg:
    #         errors.append("Tiberius is enabled but params.tiberius.model_cfg is missing.")
    #     else:
    #         check_entries(model_cfg, "Tiberius model_cfg")

    return errors


def resolve_executable(command: str, relative_to: Path) -> Path | None:
    expanded = os.path.expandvars(os.path.expanduser(command))
    has_sep = os.sep in expanded or (os.altsep and os.altsep in expanded)
    if has_sep:
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = (relative_to / candidate).resolve()
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
        return None
    found = shutil.which(expanded)
    return Path(found).resolve() if found else None


def check_java_version(java_path: Path) -> Tuple[bool, str | None]:
    try:
        proc = subprocess.run(
            [str(java_path), "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return False, str(exc)
    output = proc.stderr or proc.stdout
    version_line = output.splitlines()[0] if output else ""
    marker = '"'
    if marker in version_line:
        version = version_line.split(marker)[1]
        major = version.split(".")[0]
        try:
            if int(major) >= 11:
                return True, None
        except ValueError:
            pass
        return False, f"Java version {version} detected, but 11+ is required."
    return False, "Unable to parse Java version output."


def build_tool_command_map(params: Dict) -> Dict[str, str]:
    overrides = params.get("tools") or {}
    command_map = DEFAULT_TOOL_BINARIES.copy()
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key in command_map and value:
                command_map[key] = str(value)
    return command_map


def validate_executables(
    params: Dict,
    nextflow_bin: str,
    check_tool_binaries: bool,
    skip_singularity_check: bool,
    repo_root: Path,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    def check_command(command: str, description: str, mandatory: bool = True, java: bool = False) -> None:
        path = resolve_executable(command, repo_root)
        if not path:
            msg = f"{description} not found on PATH (looked for '{command}')."
            if mandatory:
                errors.append(msg)
            else:
                warnings.append(msg)
            return
        if java:
            ok, problem = check_java_version(path)
            if not ok:
                errors.append(problem or f"Unable to validate Java executable at {path}.")

    for cmd, desc in GENERAL_COMMANDS.items():
        if cmd == "nextflow":
            check_command(nextflow_bin, desc)
        elif cmd == "singularity" and skip_singularity_check:
            continue
        elif cmd == "java":
            check_command(cmd, desc, java=True)
        else:
            check_command(cmd, desc)

    if check_tool_binaries:        
        for cmd, desc in OPTIONAL_COMMANDS.items():
            check_command(cmd, desc, mandatory=False)
        tools = build_tool_command_map(params)
        for key, cmd in tools.items():
            label = TOOL_DESCRIPTIONS.get(key, f"Tool '{key}'")
            check_command(cmd, label)

        tiberius_cfg = params.get("tiberius") or {}
        if isinstance(tiberius_cfg, dict) and tiberius_cfg.get("run"):
            check_command("tiberius.py", "Tiberius CLI (tiberius.py)")

    return errors, warnings


def run_nextflow(
    params_path: Path,
    config_path: Path,
    base_config_path: Path,
    pipeline_main_path: Path,
    profile: str | None,
    nextflow_bin: str,
    resume: bool,
    work_dir: str | None,
    extra_args: Sequence[str],
) -> int:
    launch_cwd = Path.cwd()
    cmd = [
        nextflow_bin,
        "run",
        str(pipeline_main_path),
        "-params-file",
        str(params_path),
        "-c",
        str(base_config_path),
        "-c",
        str(config_path),
    ]
    if profile:
        cmd.extend(["-profile", profile])
    if resume:
        cmd.append("-resume")
    if work_dir:
        cmd.extend(["-work_dir", work_dir])
    if extra_args:
        cmd.extend(extra_args)

    print("[INFO] Launching Nextflow with command:")
    print("       " + " ".join(shlex.quote(part) for part in cmd))

    completed = subprocess.run(cmd, cwd=launch_cwd)
    return completed.returncode


def run_nextflow_pipeline(args) -> None:
    params_yaml = args.params_yaml
    config = args.nf_config
    if not params_yaml or not config:
        raise SystemExit("Launching Nextflow requires --params_yaml and --config.")

    extra_args = list(args.nextflow_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    params_path = Path(params_yaml).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()
    repo_root, pipeline_main, base_config = pipeline_paths()
    work_dir: Path | None = None
    if getattr(args, "work_dir", None):
        work_dir = Path(args.work_dir).expanduser().resolve()

    if not params_path.exists():
        raise SystemExit(f"Params YAML not found: {params_path}")
    if not config_path.exists():
        raise SystemExit(f"Nextflow config not found: {config_path}")
    if not pipeline_main.exists():
        raise SystemExit(f"Pipeline entry point missing: {pipeline_main}")
    if not base_config.exists():
        raise SystemExit(f"Base config not found: {base_config}")

    params = load_params(params_path)

    print("[INFO] Validating input files...")
    data_errors = validate_input_data(params, params_path)
    if data_errors:
        for error in data_errors:
            print(f"[ERROR] {error}")
        raise SystemExit("Input validation failed.")

    print("[INFO] Validating required executables...")
    exec_errors, exec_warnings = validate_executables(
        params=params,
        nextflow_bin=args.nextflow_bin,
        check_tool_binaries=args.check_tools,
        skip_singularity_check=args.skip_singularity_check,
        repo_root=repo_root,
    )
    if exec_warnings:
        for warning in exec_warnings:
            print(f"[WARN] {warning}")
    if exec_errors:
        for error in exec_errors:
            print(f"[ERROR] {error}")
        raise SystemExit("Executable validation failed.")

    if args.dry_run:
        print("[INFO] Dry run requested; skipping Nextflow execution.")
        return

    returncode = run_nextflow(
        params_path=params_path,
        config_path=config_path,
        base_config_path=base_config,
        pipeline_main_path=pipeline_main,
        profile=args.profile,
        nextflow_bin=args.nextflow_bin,
        resume=args.resume,
        work_dir=str(work_dir) if work_dir else None,
        extra_args=extra_args,
    )
    if returncode != 0:
        raise SystemExit(returncode)
