#!/usr/bin/env python3
"""Download and extract all non-superseded Tiberius model weights.

Reads every *.yaml in model_cfg/ (ignoring model_cfg/superseded/), collects
unique weights_url values, downloads the archives to model_weights/, extracts
them, then removes the .tar.gz files.
"""

import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_CFG_DIR = REPO_ROOT / "model_cfg"
MODEL_WEIGHTS_DIR = REPO_ROOT / "model_weights"


def _normalise_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(path=parsed.path.replace("//", "/")).geturl()


def _stem(filename: str) -> str:
    if filename.endswith(".tar.gz"):
        return filename[:-7]
    if filename.endswith(".tgz"):
        return filename[:-4]
    return filename


def _download(url: str, dest: Path) -> None:
    logging.info("Downloading %s", url)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)


def _extract(archive: Path, dest_dir: Path) -> None:
    import subprocess
    logging.info("Extracting %s", archive.name)
    subprocess.run(
        ["tar", "-xzf", str(archive), "-C", str(dest_dir)],
        check=True,
    )


def main() -> None:
    MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg_files = sorted(MODEL_CFG_DIR.glob("*.yaml"))
    if not cfg_files:
        logging.error("No YAML configs found in %s", MODEL_CFG_DIR)
        sys.exit(1)

    seen: set[str] = set()

    for cfg_path in cfg_files:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        url = cfg.get("weights_url")
        if not url:
            logging.warning("%s: no weights_url, skipping", cfg_path.name)
            continue

        url = _normalise_url(url)

        if url in seen:
            logging.info("%s: URL already queued, skipping", cfg_path.name)
            continue
        seen.add(url)

        filename = url.rsplit("/", 1)[-1]
        archive = MODEL_WEIGHTS_DIR / filename
        extracted = MODEL_WEIGHTS_DIR / _stem(filename)

        if extracted.is_dir() and any(extracted.iterdir()):
            logging.info("%s: already extracted at %s, skipping", cfg_path.name, extracted.name)
            continue

        if not archive.exists():
            _download(url, archive)

        _extract(archive, MODEL_WEIGHTS_DIR)
        archive.unlink()
        logging.info("Deleted %s", archive.name)

    logging.info("Done — %d unique archive(s) processed.", len(seen))


if __name__ == "__main__":
    main()
