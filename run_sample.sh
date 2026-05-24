#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR="$SCRIPT_DIR"
DEFAULT_ENV_FILE="$REPO_DIR/environment.lock.yml"
if [ ! -f "$DEFAULT_ENV_FILE" ]; then
  DEFAULT_ENV_FILE="$REPO_DIR/environment.yml"
fi
ENV_FILE="${PHILOSOPHER_ENV_FILE:-$DEFAULT_ENV_FILE}"
ENV_NAME="${PHILOSOPHER_CONDA_ENV:-philosopher}"
OUTDIR="${PHILOSOPHER_OUTDIR:-$REPO_DIR/phi_out_sample}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Install Miniconda or Anaconda first." >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_TOOL="conda"
if command -v mamba >/dev/null 2>&1; then
  ENV_TOOL="mamba"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Creating conda environment '$ENV_NAME' from $ENV_FILE using $ENV_TOOL"
  "$ENV_TOOL" env create -n "$ENV_NAME" -f "$ENV_FILE"
else
  echo "Using existing conda environment '$ENV_NAME'"
fi

conda activate "$ENV_NAME"

if ! python - <<'PY' >/dev/null 2>&1
import pkg_resources
PY
then
  echo "Installing compatible setuptools into '$ENV_NAME' to provide pkg_resources"
  python -m pip install "setuptools==75.1.0"
fi

python -m pip install --no-deps -e "$REPO_DIR"

mkdir -p "$OUTDIR"

echo "Running bundled sample manifest with environment '$ENV_NAME'"
philosophers-stone \
  --manifest_csv "$REPO_DIR/phi_manifest.csv" \
  --outdir "$OUTDIR" \
  "$@"
