#!/usr/bin/env bash
# Recreate data/ symlinks to local source corpora.
# Edit SRC if the mythbusters checkout lives elsewhere on this machine.
set -euo pipefail

SRC="${MYTHBUSTERS_DIR:-/mnt/md0/helenjin/mythbusters}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p data/raw data/stability_runs data/processed data/fixtures

# local assessment corpora -> data/raw/<clean name>
ln -sfn "$SRC/data/v2/claimspy_v1_ICL_Agent_2"          data/raw/claimspy_v1
ln -sfn "$SRC/data/v2/claimspy_v2"                      data/raw/claimspy_v2
ln -sfn "$SRC/data/v2/scholar_qa"                       data/raw/scholar_qa
ln -sfn "$SRC/data/v2/scholarqa_assessments_claude_4.5" data/raw/scholarqa_claude45
ln -sfn "$SRC/data/v2/codescientist"                    data/raw/codescientist
ln -sfn "$SRC/data/v2/evaluation-data"                  data/raw/evaluation_data

# soft-stability runs -> data/stability_runs/<source condition>
SS="$SRC/notebooks/scify_soft_stability"
ln -sfn "$SS/ss_rate_dicts_all"               data/stability_runs/all
ln -sfn "$SS/ss_rate_dicts_all_nonweb"        data/stability_runs/nonweb
ln -sfn "$SS/ss_rate_dicts_all_parametric_v0" data/stability_runs/parametric
ln -sfn "$SS/ss_rate_dicts_all_parametric"    data/stability_runs/parametric_legacy
ln -sfn "$SS/ss_rate_dicts_all_v0"            data/stability_runs/all_v0
ln -sfn "$SS/ss_rate_dicts_all_v1"            data/stability_runs/all_v1

echo "data/ symlinks created (SRC=$SRC)."
