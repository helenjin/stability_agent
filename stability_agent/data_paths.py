"""Resolve dataset locations through the repo-root ``data/`` folder.

Code should reach datasets via these helpers (or the registry) instead of
hard-coding ``../mythbusters/...`` paths. Resolution is by convention:
``data/raw/<name>``, ``data/stability_runs/<name>``, etc. The registry
(``data/registry.yaml``) is the authoritative description; reading it is
optional and only used when PyYAML is available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
REGISTRY_PATH = DATA_DIR / "registry.yaml"


def raw_corpus(name: str) -> Path:
    """Path to a local assessment corpus, e.g. ``raw_corpus('claimspy_v1')``."""
    return DATA_DIR / "raw" / name


def stability_run(name: str) -> Path:
    """Path to a soft-stability run dir, e.g. ``stability_run('parametric')``."""
    return DATA_DIR / "stability_runs" / name


def processed(name: str) -> Path:
    return DATA_DIR / "processed" / name


def fixture(name: str) -> Path:
    return DATA_DIR / "fixtures" / name


def load_registry() -> dict[str, Any]:
    """Parse ``data/registry.yaml`` (requires PyYAML). Raises if unavailable."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Reading the registry requires PyYAML. Install pyyaml, or resolve "
            "paths by convention with raw_corpus()/stability_run()."
        ) from exc
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
