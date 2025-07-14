#!/usr/bin/env bash
# setup_qokit_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# Creates a conda env, installs GPU/CPU deps, builds QOKit in editable mode
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="qokit-env"
PYTHON_VER="3.10"
CUDA_VER="12.8"

echo "✔️  Creating conda env '$ENV_NAME' with Python $PYTHON_VER"
conda create -y -n "$ENV_NAME" python="$PYTHON_VER"

echo "✔️  Activating '$ENV_NAME'"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "✔️  Installing CPU packages"
conda install -y -c conda-forge \
    numpy scipy pandas psutil scikit-learn matplotlib \
    numba notebook

echo "✔️  Installing Qiskit & Aer"
conda install -y -c conda-forge qiskit qiskit-aer

echo "✔️  Installing CuPy (CUDA $CUDA_VER)"
conda install -y -c conda-forge cupy cudatoolkit="$CUDA_VER"

echo "✔️  Cloning & installing QOKit (editable)"
# assume you've already unpacked QOKit-main.zip into ./QOKit
cd QOKit
pip install -e .

echo "✔️  Compiling native C simulator for FUR backends"
pushd qokit/fur/c
make
popd

echo "✔️  Environment setup complete!"
echo "   To activate:    conda activate $ENV_NAME"
echo "   Verify with:    python - <<<'import qokit; print(qokit.__version__)'"
