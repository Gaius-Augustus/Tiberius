#!/usr/bin/env python3

# ==============================================================
# Authors: Lars Gabriel
#
# Running Tiberius for single genome prediction
# ==============================================================

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict
import subprocess as sp
from packaging.version import Version
import bricks2marble as b2m
import requests
import yaml


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log_config = []
MIN_TF_VERSION = "2.13"
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
SCRIPT_ROOT = SCRIPT_DIR


class MissingConfigFieldError(RuntimeError):
    """Raised when the model-config file lacks one or more required fields."""


def load_model_config(
    filepath,
    required = ("weights_url", "softmasking", "clamsa"),
):
    """
    Read a YAML model-config file and return its contents as a dict.

    Parameters
    ----------
    filepath : str
        Path to the YAML file.
    required : Sequence[str], optional
        Keys that *must* be present (and non-null) in the YAML.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    repo_config = f"{SCRIPT_ROOT}/../model_cfg/{filepath.split('/')[-1]}"

    if not repo_config.endswith(".yaml"):
        repo_config += ".yaml"
    if os.path.exists(repo_config):
        filepath = repo_config
    else:
        raise FileNotFoundError(f"File not found: {filepath}")


    logging.info(f'Model Config File: {os.path.abspath(filepath)}')
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Treat absent keys *or* keys explicitly set to null/None as missing
    missing = [k for k in required if data.get(k) is None]
    if missing:
        raise MissingConfigFieldError(
            f"{filepath} is missing required field(s): {', '.join(missing)}"
        )

    return data


def check_tf_version(tf_version: str) -> bool:
    if Version(tf_version) < Version(MIN_TF_VERSION):
        logging.warning(
            "You are using TensorFlow version %s, which is older than the "
            "recommended minimum version %s. It may fail during prediction.",
            tf_version,
            MIN_TF_VERSION,
        )
        return False
    return True


def check_seq_len(seq_len: int) -> bool:
    """Ensure seq_len is divisible by 9 and 2."""
    if not (seq_len % 9 == 0 and seq_len % 2 == 0):
        logging.error(
            f'ERROR: The argument "seq_len" has to be  divisible by 9 and by 2 for the model to work! Please change the value {seq_len} to a different value!'
        )
        sys.exit(1)
    return True


def compute_parallel_factor(seq_len: int) -> int:
    sqrt_n = int(math.sqrt(seq_len))

    for delta in range(seq_len):
        lower = sqrt_n - delta
        if lower >= 1 and seq_len % lower == 0:
            return lower

        upper = sqrt_n + delta
        if seq_len % upper == 0:
            return upper

    raise RuntimeError(f"Could not compute a valid parallel factor for seq_len={seq_len}")


def check_file_exists(file_path: str | Path) -> None:
    """Check if a file exists at the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

def get_gpu_memory_gb(device_index: int = 0) -> float:
    """
    Return total memory of the selected NVIDIA GPU in GB.
    """
    try:
        result = sp.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        logging.warning(
            "nvidia-smi not found. Cannot auto-detect GPU memory.\n"
            "Batch size is set to the default value of 16."
        )
        return None
    except sp.CalledProcessError as exc:
        logging.warning(
            f"Failed to query GPU memory with nvidia-smi: {exc}\n"
            "Batch size is set to the default value of 16."
        )
        return None

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if device_index >= len(lines):
        logging.warning(
            f"GPU index {device_index} out of range. Found {len(lines)} GPU(s).\n"
            "Batch size is set to the default value of 16.\n"
            "Please set the batch size manually with the --batch_size option if this is not correct."
        )
        return None

    memory_mb = float(lines[device_index])
    return memory_mb / 1024.0

def compute_auto_batch_size(
    seq_len: int,
    gpu_memory_gb: float,
    safety_factor: float = 0.9,
    min_batch_size: int = 1,
    max_batch_size: int | None = None,
) -> int:
    """
    Estimate batch size from available GPU memory and seq_len using
    approximately linear scaling in batch_size * seq_len.

    Calibrated by default with:
        seq_len=500400, batch_size=16 on 80 GB GPU
    """
    preferred_batch_sizes = [1, 2, 4, 8, 16, 24,
                            32, 64, 96, 128, 160, 192]
    max_deviation = 3
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if gpu_memory_gb <= 0:
        raise ValueError(f"gpu_memory_gb must be > 0, got {gpu_memory_gb}")
    if safety_factor <= 0:
        raise ValueError(f"safety_factor must be > 0, got {safety_factor}")

    if gpu_memory_gb >= 70:
        ref_mem, ref_bs = 80.0, 18
    elif gpu_memory_gb >= 16:
        ref_mem, ref_bs = 25.0, 8
    else:
        ref_mem, ref_bs = 8.0, 2
    estimated = (gpu_memory_gb / ref_mem) \
        * ref_bs \
        * (500_004 / seq_len) \
        * safety_factor

    batch_size = max(min_batch_size, estimated)
    candidates = [b for b in preferred_batch_sizes if abs(b-batch_size) < max_deviation]
    if candidates:
        batch_size = min(candidates, key=lambda x: abs(x - batch_size))

    if max_batch_size is not None:
        batch_size = min(batch_size, max_batch_size)
    batch_size = int(batch_size+0.5)
    logging.info(
        "Auto-computed batch_size=%s from seq_len=%s and gpu_memory_gb=%.2f "
        "If GPU runs out of memory, please set the batch size manually with the --batch_size option.",
        batch_size,
        seq_len,
        gpu_memory_gb,
    )
    return batch_size

def download_weights(url: str, file_path: str | Path) -> str | Path:
    """Download model weights unless a usable local file already exists."""

    print
    if (
        file_path.endswith(".tar.gz")
        and os.path.isdir(file_path[:-7])
        and  os.listdir(file_path[:-7])
    ):
        file_path = file_path[:-7]
        logging.info(
            f"Using existing model weights file at {file_path} ."
        )
    elif os.path.exists(file_path) and not os.path.getsize(file_path) == 0:
        logging.info(
            f"Using existing model weights file at {file_path} ."
        )
    else:
        logging.info(
            f"Model weights will be downloaded to {file_path} ."
        )
        logging.info(f"Weights for Tiberius model will be downloaded from {url}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=10000):
                    f.write(chunk)
    return file_path

def extract_tar_gz(file_path: Path, dest_dir: Path) -> None:
    sp.run(
        ["tar", "-xzf", str(file_path), "-C", str(dest_dir)],
        check=True,
    )

def is_writable(file_path: str | Path) -> bool:
    """Check whether a path is writable."""
    return os.access(file_path, os.W_OK)


def resolve_model_paths(args):
    """Resolve config and model paths based on command-line arguments."""
    config = None
    model_path = None
    model_path_hmm = None

    if not args.model_old and not args.model_lstm_old:
        if args.model_cfg:
            config = load_model_config(args.model_cfg)
        elif args.model:
            model_path = os.path.abspath(args.model)
        else:
            raise FileNotFoundError(
                "A model config file has to be specified with --model_cfg!"
            )

        if model_path:
            check_file_exists(model_path)
            log_config.append(f"Model path: {model_path}")

        model_path_hmm = os.path.abspath(args.model_hmm) if args.model_hmm else None
        if model_path_hmm:
            check_file_exists(model_path_hmm)
            log_config.append(f"Model HMM path: {model_path_hmm}")

    return config, model_path, model_path_hmm


def resolve_weight_download(config: Dict[str, Any]) -> str:
    """Resolve or download model weights from config."""


    model_weights_dir = f"{SCRIPT_DIR}/../model_weights"
    model_file_name = config["weights_url"].split("/")[-1]

    model_path_exist = f"{model_weights_dir}/{model_file_name.split('.')[0]}"
    if os.path.exists(model_path_exist):
        return model_path_exist

    if not os.path.exists(model_weights_dir):
        if is_writable(os.path.dirname(model_weights_dir)):
            os.makedirs(model_weights_dir)

    if not os.path.exists(model_weights_dir) or not is_writable(model_weights_dir):
        model_weights_dir = os.getcwd()

    if not is_writable(model_weights_dir):
        logging.error(
            "No model weights provided, and candidate directories for download are not writable. Please download the model weights manually (see README.md) and specify them with --model!"
        )
        sys.exit(1)

    model_path = download_weights(
        config["weights_url"], f"{model_weights_dir}/{model_file_name}"
    )

    if model_path and model_path[-3:] in ["tgz", ".gz"]:
        logging.info(f"Extracting weights to {model_weights_dir}")
        extract_tar_gz(model_path, model_weights_dir)
        model_path = model_path[:-4] if model_path.endswith(".tgz") else model_path[:-7]

    return model_path


def import_tensorflow():
    """Import TensorFlow lazily after argument validation."""
    if "tensorflow" in sys.modules:
        return sys.modules["tensorflow"]

    import tensorflow as tf

    return tf


def run_tiberius(args):
    config, model_path, model_path_hmm = resolve_model_paths(args)

    gtf_out = os.path.abspath(args.out)

    if args.seq_len is not None:
        seq_len = args.seq_len
    elif config is not None:
        seq_len = config["default_seq_len"]
    else:
        seq_len = 500_040
    log_config.append(f"chunk length: {seq_len}")


    if args.batch_size is None:
        gpu_mem = get_gpu_memory_gb()
        batch_size = 16 if gpu_mem is None else compute_auto_batch_size(seq_len, gpu_mem)
    else:
        batch_size = args.batch_size
    log_config.append(f"batch size: {batch_size}")


    min_seq_len = args.min_genome_seqlen
    log_config.append(f"minimum sequence length: {min_seq_len}")

    check_seq_len(seq_len)

    softmasking = not args.no_softmasking if not config else config["softmasking"]
    log_config.append(f"Softmasking: {softmasking}")

    genome_path = os.path.abspath(args.genome)
    check_file_exists(genome_path)


    if config:
        model_path = resolve_weight_download(config)

    if model_path and not os.path.exists(model_path):
        logging.error(
            "Error: The model weights could not be downloaded. Please download the model weights manually (see README.md) and specify them with --model!"
        )
        sys.exit(1)

    clamsa_prefix = args.clamsa

    if config:
        if clamsa_prefix is not None and not config["clamsa"]:
            logging.error("Error: ClaMSA input data was provided but the model provided does not support ClaMSA input.")
        if clamsa_prefix is not None and not config["clamsa"]:
            logging.error("Error: A model that requires ClaMSA input is used but no ClaMSA input was provided using --clamsa .")


    tf = import_tensorflow()
    check_tf_version(tf.__version__)

    if seq_len > 500_400:
        logging.error(
            f"\nWARNING: The sequence length {args.seq_len} can be too long for TensorFlow version {tf.__version__}. "
            "If it fails, please use a sequence length <= 500400 (--seq_len).\n"
        )

    from tiberius.eval_model_class import PredictionGTF

    parallel_factor = (
        compute_parallel_factor(seq_len)
        if args.parallel_factor == 0
        else args.parallel_factor
    )
    log_config.append(f"HMM parallel factor: {parallel_factor}")
    logging.info("")
    start_time = time.time()

    pred_gtf = PredictionGTF(
        model_path_lstm_old=args.model_lstm_old,
        model_path_old=args.model_old,
        model_path=model_path,
        model_path_hmm=model_path_hmm,
        seq_len=seq_len,
        batch_size=batch_size,
        hmm=True,
        hmm_emitter_epsilon=args.hmm_eps,
        hmm_initial_exon_len=args.hmm_initial_exon_len,
        hmm_initial_intron_len=args.hmm_initial_intron_len,
        hmm_initial_ir_len=args.hmm_initial_ir_len,
        temp_dir=None,
        num_hmm=1,
        hmm_factor=1,
        genome=None,
        softmask=softmasking,
        parallel_factor=parallel_factor,
    )

    pred_gtf.load_model(summary=0)

    predict_fun = pred_gtf.predict_function
    repred_fun = pred_gtf.repredict_function

    if args.codingseq:
        open(args.codingseq, "w").close()
    if args.protseq:
        open(args.protseq, "w").close()

    def postprocess(fasta: b2m.struct.Fasta, \
                annot: b2m.struct.Annotation) -> b2m.struct.Annotation:

        b2m.tools.check_min_coding_length(
            annot, 200, remove=True
        )
        b2m.tools.check_inframe_stop_codons(
            annot, fasta, remove=True
        )
        if args.codingseq:
            annot.sequence_to_file(target="coding", fasta=fasta, path=args.codingseq, mode="a")
        if args.protseq:
            annot.sequence_to_file(target="protein", fasta=fasta, path=args.protseq, mode="a")
        return annot

    clamsa=None
    if clamsa_prefix:
        clamsa = pred_gtf.load_clamsa_data(clamsa_prefix=clamsa_prefix, seq_names=seq,
                            strand="+", chunk_len=seq_len, pad=True)

    b2m.tools.annotate.annotate_genome(
        fasta = Path(genome_path).expanduser(),
        predict_func = predict_fun,
        output = gtf_out,
        allow_extract_gz = True,
        T_max = seq_len,
        T_delta = 0.1,
        T_factors = [2, 9, parallel_factor],
        model_name = "Tiberius",
        min_sequence_size = min_seq_len,
        reprediction_factor= 0.5,
        postprocess = postprocess,
        repredict_func = repred_fun,
        concat_strand_to_reprediction=True,
        log_config=log_config,
        group_size_limit=args.group_size_limit
    )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Tiberius took {duration/60:.4f} minutes to execute.")


def main():
    from tiberius import parseCmd

    try:
        args = parseCmd()
        run_tiberius(args)
    except Exception as exc:
        logging.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()