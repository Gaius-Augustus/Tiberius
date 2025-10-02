import json
import logging
from pathlib import Path
from typing import Any

from .data import DatasetConfig
from .model import TiberiusConfig
from .train import TrainerConfig


def load_json(path: Path | str | None) -> dict[str, Any]:
    """Loads a given json file as a dictionary."""
    if path is None: return {}
    p = Path(path).expanduser()
    with p.open("r") as f:
        return json.load(f)


def deep_update(
    base: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    """Replace entries of the possibly nested base dictionary that are
    inside the override dictionary. Inner dictionaries are also updated
    and not overwritten.
    """
    out = dict(base)
    for k, v in update.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def log_diff(
    defaults: dict[str, Any],
    final: dict[str, Any],
    prefix: str,
) -> None:
    """Log keys of the `final` dictionary that are different compared to
    the `defaults` dictionary. The log info starts with `prefix`.
    """
    logger = logging.getLogger("config")
    changed = {
        k: final[k]
        for k in final.keys() if defaults.get(k) != final[k]
    }
    if len(changed) > 0:
        logger.info(f"[{prefix}] overrides: {changed}")
    else:
        logger.info(f"[{prefix}] no overrides; using pure defaults")


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "run.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def split_config(
    file: Path | str,
) -> tuple[TiberiusConfig, DatasetConfig, TrainerConfig]:
    with open(file, "r") as f:
        total_dict = json.load(f)
    if (
        "model" not in total_dict
        or "dataset" not in total_dict
        or "trainer" not in total_dict
    ):
        raise ValueError(
            "A total config has to consist of model, dataset and trainer."
        )

    return (
        TiberiusConfig(**total_dict["model"]),
        DatasetConfig(**total_dict["dataset"]),
        TrainerConfig(**total_dict["trainer"]),
    )
