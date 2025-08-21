# config_utils.py
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

def load_json(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    with p.open("r") as f:
        return json.load(f)

def deep_update(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """
    override default parameter.
    - Dicts are merged deep
    - Lists / scalars are replaced
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def log_diff(defaults: Dict[str, Any], final: Dict[str, Any], prefix: str) -> None:
    """Log only the keys that changed vs defaults (shallow compare for brevity)."""
    logger = logging.getLogger("config")
    changed = {k: final[k] for k in final.keys() if defaults.get(k) != final[k]}
    if changed:
        logger.info("[%s] overrides: %s", prefix, changed)
    else:
        logger.info("[%s] no overrides; using pure defaults", prefix)
