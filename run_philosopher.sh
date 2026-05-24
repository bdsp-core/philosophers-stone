#!/bin/bash
# Execute the Philosopher's Stone batch pipeline against the local phi_manifest.csv

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/phi_manifest.csv"
OUTDIR="$SCRIPT_DIR/phi_out"
DEVICE_ARG=()
if [ -n "${PHILOSOPHER_DEVICE_ID:-}" ]; then
  DEVICE_ARG=(--device-id "$PHILOSOPHER_DEVICE_ID")
fi

mkdir -p "$OUTDIR"

if command -v philosophers-stone >/dev/null 2>&1; then
  philosophers-stone \
    --manifest_csv "$MANIFEST" \
    --outdir "$OUTDIR" \
    "${DEVICE_ARG[@]}"
else
  python "$SCRIPT_DIR/philosopher.py" \
    --manifest_csv "$MANIFEST" \
    --outdir "$OUTDIR" \
    "${DEVICE_ARG[@]}"
fi
