#!/usr/bin/env bash
# Build the project environment at ../../envs/env_sr.
#
# torch is pinned to the cu126 build deliberately: CUDA 13 dropped Volta, and
# this project runs on both V100 (PSC `GPU-shared`) and H100 nodes. The cu130
# default wheel would silently exclude every V100 node from the queue.
#
# Idempotent -- safe to re-run; conda/pip skip what is already satisfied.
set -euo pipefail

# Site-specific locations come from the environment, so nothing here is tied
# to one machine or allocation. Override either as needed:
#   CONDA=/path/to/conda ENV_DIR=/path/to/env bash env/create_env.sh
REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
CONDA="${CONDA:-$(command -v conda || echo conda)}"
ENV_DIR="${ENV_DIR:-$REPO_DIR/../envs/env_sr}"
PY_VERSION=3.13

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo ">> creating conda env at $ENV_DIR (python $PY_VERSION)"
    "$CONDA" create -y -p "$ENV_DIR" -c conda-forge "python=$PY_VERSION"
else
    echo ">> reusing existing env at $ENV_DIR"
fi

PIP="$ENV_DIR/bin/pip"
"$PIP" install --upgrade pip

echo ">> installing torch (cu126)"
"$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo ">> installing the rest"
"$PIP" install \
    timm \
    numpy \
    rasterio \
    geopandas \
    shapely \
    pyproj \
    scikit-image \
    scipy \
    pandas \
    pyarrow \
    einops \
    pyyaml \
    tqdm \
    matplotlib \
    pillow

echo ">> installing this project (editable)"
"$PIP" install -e "$REPO_DIR"

echo
echo ">> done. activate with:"
echo "     export PATH=$ENV_DIR/bin:\$PATH"
echo "   or run directly: $ENV_DIR/bin/python ..."
"$ENV_DIR/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.version.cuda}")
PY
