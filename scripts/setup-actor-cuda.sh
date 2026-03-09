#!/usr/bin/env bash
set -euo pipefail

# Install CUDA-enabled PyTorch stack into projects/actor/.venv
# Usage:
#   ./scripts/setup-actor-cuda.sh
#   CUDA_TAG=cu121 TORCH_VERSION=2.5.1 ./scripts/setup-actor-cuda.sh
#   SKIP_ACTOR_SYNC=1 CUDA_TAG=cu121 TORCH_VERSION=2.5.1 ./scripts/setup-actor-cuda.sh

CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
SKIP_ACTOR_SYNC="${SKIP_ACTOR_SYNC:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTOR_PROJECT="$REPO_ROOT/projects/actor"
PY="$ACTOR_PROJECT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Actor virtualenv Python not found: $PY"
  echo "Run: uv sync --project projects/actor"
  exit 1
fi

INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"

if [[ -z "${CUDA_HOME:-}" && -n "${CUDA_PATH:-}" ]]; then
  export CUDA_HOME="$CUDA_PATH"
fi
if [[ -z "${CUDA_PATH:-}" && -n "${CUDA_HOME:-}" ]]; then
  export CUDA_PATH="$CUDA_HOME"
fi

echo "Installing CUDA PyTorch packages into actor environment..."
echo "  Python : $PY"
echo "  Index  : $INDEX_URL"
echo "  Torch  : $TORCH_VERSION"

uv pip install --python "$PY" --index-url "$INDEX_URL" --upgrade --force-reinstall \
  "torch==${TORCH_VERSION}" \
  torchvision \
  torchaudio

echo "CUDA package install completed."
echo "Run CUDA preflight:"
echo "  $PY ./scripts/check_gpu.py"

echo "Running CUDA preflight now..."
"$PY" "$REPO_ROOT/scripts/check_gpu.py"

if [[ "$SKIP_ACTOR_SYNC" != "1" ]]; then
  echo "Syncing actor project dependencies (canonical mambapy runtime)..."
  uv sync --project "$ACTOR_PROJECT" --python 3.12
fi
