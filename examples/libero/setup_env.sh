#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."

echo "=== Installing libero requirements ==="
uv pip sync "$REPO_ROOT/examples/libero/requirements.txt" "$REPO_ROOT/third_party/libero/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match

echo "=== Installing openpi-client ==="
uv pip install -e "$REPO_ROOT/packages/openpi-client"

echo "=== Installing libero ==="
uv pip install -e "$REPO_ROOT/third_party/libero"

echo "=== Installing wandb (>=0.18 for long API keys) ==="
uv pip install "wandb>=0.18.0"

echo "=== Installing opencv ==="
uv pip install opencv-python-headless

echo "=== Setting PYTHONPATH ==="
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT/third_party/libero

echo "=== Setting MuJoCo rendering ==="
export MUJOCO_GL=osmesa

echo "=== Done! ==="
