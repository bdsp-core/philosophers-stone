#!/bin/bash
# Execute the Philosophers Stone batch pipeline against the local phi_manifest.csv

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/phi_manifest.csv"
OUTDIR="$SCRIPT_DIR/phi_out"
RESULTS_CSV="$OUTDIR/phi_results.csv"
DEVICE=1 # Change this to select a different GPU, or remove to use CPU/GPU auto-detect.

mkdir -p "$OUTDIR"

python "$SCRIPT_DIR/philosopher.py" \
  --manifest_csv "$MANIFEST" \
  --outdir "$OUTDIR" \
  --device-id "$DEVICE" \
  # --collect-heads \
  # --save-json \
  # --save-plots
