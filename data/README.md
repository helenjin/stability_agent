# data/

Single entry point for every dataset `stability_agent` uses. See `registry.yaml`
for the authoritative list (paths, sizes, structure, provenance).

```
data/
  raw/             symlinks -> local SciFy/ClaimSpy assessment corpora
  stability_runs/  symlinks -> soft-stability rate outputs (one per source condition)
  processed/       normalized canonical-schema exports (generated)
  fixtures/        tiny samples for tests (e.g. popqa_fixture*)
  registry.yaml    name -> path / hf_id + provenance
```

## How datasets enter `data/`

- **Local corpora** (`raw/`, `stability_runs/`) are **per-corpus symlinks** with clean
  names pointing at their source location. Targets are absolute and host-specific —
  recreate them with `scripts/setup_data.sh` on a new machine.
- **Hugging Face datasets** are **register-only**: not copied here. They are recorded
  in `registry.yaml` under `hf_datasets` and loaded on demand via
  `datasets.load_dataset(hf_id, ...)`.

## Conventions

- Code should resolve dataset locations through `data/` (or the registry), never via
  hard-coded `../mythbusters/...` paths.
- `raw/` symlinks are read-only sources — never write into them.
- Generated artifacts go in `processed/`; keep `raw/` pristine.
