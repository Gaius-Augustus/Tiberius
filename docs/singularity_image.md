### Singularity image lifecycle

This document describes how Tiberius pins, fetches, validates, and cleans up
its Singularity image. It is aimed at developers maintaining the launcher in
[`tiberius.py`](../tiberius.py) and at maintainers publishing new images.

All logic lives in [`tiberius.py`](../tiberius.py); no extra dependencies are
introduced (`urllib`, `json` from the stdlib).


#### Architecture overview

The image used by `--singularity` is identified by a single source-of-truth
constant in [`tiberius.py`](../tiberius.py):

```python
SINGULARITY_IMAGE_REPO    = "larsgabriel23/tiberius"
SINGULARITY_IMAGE_VERSION = "2.0.0"   # pinned, tested image tag
SINGULARITY_IMAGE_URI     = f"docker://{SINGULARITY_IMAGE_REPO}:{SINGULARITY_IMAGE_VERSION}"
SINGULARITY_IMAGE_PATH    = SCRIPT_ROOT / "singularity" / f"tiberius_{SINGULARITY_IMAGE_VERSION}.sif"
DOCKER_HUB_TAGS_URL       = f"https://hub.docker.com/v2/repositories/{SINGULARITY_IMAGE_REPO}/tags?page_size=100"
```

Two consequences worth noting:

- The pinned tag is part of the source code, not a runtime variable. Updating
  the image is therefore a code change — reviewable, revertable, and tied to
  whatever Tiberius release ships with that image.
- The local `.sif` filename embeds the version (`tiberius_2.0.0.sif`). A
  version bump produces a different cache path, so the new image is fetched
  cleanly without overwriting the old one.


#### End-to-end flow

When a user runs `python tiberius.py --singularity ...`, the launcher (in
[`run_tiberius_in_singularity`](../tiberius.py)) performs the following steps:

1. Skip if already running inside the container (`TIBERIUS_IN_SINGULARITY=1`).
2. **Update check** — `_warn_if_newer_image_available()` queries Docker Hub
   and prints a yellow `[WARNING]` if a newer semver tag exists.
3. **Local cache check** — if `SINGULARITY_IMAGE_PATH` does not exist:
   - create the parent directory,
   - run `singularity pull <path> <uri>`,
   - record `pulled_now = True`.
4. **Old-image handling** — list cached `tiberius_*.sif` files that don't
   match the pinned version. Two branches:
   - `--cleanup_old_singularity_images` set: delete each old file, log per
     deletion. OSErrors are caught and logged as a `[WARNING]`.
   - flag not set, but a fresh pull just happened and old files exist:
     print a yellow `[WARNING]` listing them and pointing to the flag.
   - flag not set and no fresh pull: stay silent (avoids nag spam every run).
5. Build and exec the `singularity run` command with `--nv`, `--cleanenv`,
   `CUDA_VISIBLE_DEVICES` passthrough, and `--nvccli` if
   `nvidia-container-cli` is available.


#### Update detection

`_fetch_latest_image_tag()` issues a single GET to the Docker Hub API:

```
https://hub.docker.com/v2/repositories/larsgabriel23/tiberius/tags?page_size=100
```

It filters tags through `_parse_semver()` (accepts `1.2.3` or `v1.2.3`,
rejects `latest`, `dev`, `devel_train`, etc.), sorts by tuple comparison,
and returns the highest. `_warn_if_newer_image_available()` then compares
against `SINGULARITY_IMAGE_VERSION` and warns only on strict-greater.

Failure modes are deliberately silent — offline runs, DNS failures,
timeouts, malformed JSON, all return `None` and skip the warning. Network
timeout is 3 s.

If you change the tag scheme (e.g. add a `-rc1` suffix), update
`_parse_semver` accordingly. Currently any non-numeric component causes the
tag to be skipped, which is the safe default.


#### Cache layout

```
<repo>/singularity/
    tiberius_1.1.8.sif   # left from a previous version
    tiberius_2.0.0.sif   # currently pinned
```

- The `singularity/` directory is created lazily on first pull.
- Old `.sif` files are kept by default to allow rollback by pinning the
  previous version in code.
- `--cleanup_old_singularity_images` scans for `tiberius_*.sif` files whose
  resolved path differs from `SINGULARITY_IMAGE_PATH.resolve()` and deletes
  them. Files unrelated to this naming convention are never touched.


#### Behaviour matrix

| Pinned image cached? | Old `.sif` present? | `--cleanup_old_singularity_images` | Behaviour                                         |
| -------------------- | ------------------- | ---------------------------------- | ------------------------------------------------- |
| yes                  | no                  | off                                | nothing                                           |
| yes                  | yes                 | off                                | nothing (avoids nag spam)                         |
| yes                  | yes                 | on                                 | old `.sif` deleted                                |
| no (fresh pull)      | no                  | off                                | new image pulled                                  |
| no (fresh pull)      | yes                 | off                                | new image pulled, warning lists old `.sif` files  |
| no (fresh pull)      | yes                 | on                                 | new image pulled, old `.sif` files deleted        |

`_warn_if_newer_image_available()` runs unconditionally at the start of
every `--singularity` invocation, independent of the cache state.


#### Release checklist (publishing a new image)

1. Build and tag the image with a semver tag (no leading `v`):
   ```shell
   docker build -t larsgabriel23/tiberius:2.1.0 .
   docker push larsgabriel23/tiberius:2.1.0
   ```
   Optionally also retag and push `:latest` for Docker users — the
   launcher does not consume `:latest`, but the README's Docker example
   does.
2. Bump `SINGULARITY_IMAGE_VERSION` in [`tiberius.py`](../tiberius.py).
3. Commit the bump in the same PR as any code changes that depend on the
   new image. The image and the launcher version that uses it should land
   together.
4. Cut the Tiberius release.

After step 1, users still on the old launcher will see the
`[WARNING] A newer Singularity image is available: 2.1.0 ...` message and
can `git pull` to fetch the bumped constant. Their next `--singularity`
run then auto-pulls the new `.sif` (different cache path) without any
manual cleanup.


#### Testing locally

Quick check that the registry query works with the current network:

```shell
python -c "
import json, urllib.request
with urllib.request.urlopen(
    'https://hub.docker.com/v2/repositories/larsgabriel23/tiberius/tags?page_size=100',
    timeout=5,
) as r:
    print([t['name'] for t in json.load(r).get('results', [])])
"
```

To exercise the warning branch without publishing a new image, edit
`SINGULARITY_IMAGE_VERSION` to a value strictly lower than any tag on Hub
(e.g. `0.0.1`) and run with `--singularity`.

To exercise cleanup, drop a stub file into `singularity/` matching the
pattern:

```shell
touch singularity/tiberius_0.0.1.sif
python tiberius.py --singularity --cleanup_old_singularity_images ...
```

The stub will be removed; only the pinned `.sif` survives.


#### Code references

- [`tiberius.py`](../tiberius.py) — constants, helpers, and
  `run_tiberius_in_singularity`.
- [`tiberius/tiberius_args.py`](../tiberius/tiberius_args.py) —
  `--singularity` and `--cleanup_old_singularity_images` flag definitions.
- [`Dockerfile`](../Dockerfile) — image build recipe; tag and push with the
  semver constant defined above.
