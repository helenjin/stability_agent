"""Input/output helpers for soft-stability experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class StabilityRun:
    """One example's stability trace for one source directory."""

    example_id: str
    source_name: str
    path: Path
    radii: dict[int, dict[str, Any]]


def parse_int_keyed_json(path: Path) -> dict[int, Any]:
    """Load a JSON object whose keys are expected to be integer-like."""

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[int, Any] = {}
    for key, value in raw.items():
        try:
            out[int(key)] = value
        except (TypeError, ValueError):
            continue
    return out


def iter_stability_runs(paths: Iterable[Path]) -> list[StabilityRun]:
    """Load all numeric JSON files from one or more stability output dirs."""

    runs: list[StabilityRun] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.suffix == ".json" and path.stem.isdigit():
            runs.append(
                StabilityRun(
                    example_id=path.stem,
                    source_name=path.parent.name,
                    path=path,
                    radii=parse_int_keyed_json(path),
                )
            )
            continue
        for json_path in sorted(path.glob("*.json"), key=_numeric_path_key):
            if not json_path.stem.isdigit():
                continue
            runs.append(
                StabilityRun(
                    example_id=json_path.stem,
                    source_name=path.name,
                    path=json_path,
                    radii=parse_int_keyed_json(json_path),
                )
            )
    return runs


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _numeric_path_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.name)
    except ValueError:
        return (10**12, path.name)
