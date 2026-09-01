#!/usr/bin/env bash
# Export Ultralytics YOLO detection weights to ONNX for the examples.
#
# The weights are Ultralytics YOLO, licensed AGPL-3.0 (the EdgeFirst Model Zoo
# distributes the same weights under the same license). This repository never
# redistributes them; each user exports their own copy.
#
# Usage: scripts/export-yolo.sh [model] [imgsz]
#   model  Ultralytics model name (default: yolo11n)
#   imgsz  square input size (default: 640)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-yolo11n}"
IMGSZ="${2:-640}"

if [ ! -d venv ]; then
    python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --quiet --upgrade ultralytics

OUT="examples/ultralytics/src/model"
mkdir -p "$OUT"
yolo export model="${MODEL}.pt" format=onnx imgsz="${IMGSZ}"
mv "${MODEL}.onnx" "$OUT/"
echo "Exported $OUT/${MODEL}.onnx"
echo "NOTE: the committed example modules expect EdgeFirst Model Zoo file names"
echo "(e.g. yolo11n-t-b8b.fp32.onnx); rename accordingly or adjust build.rs."
