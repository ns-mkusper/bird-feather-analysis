#!/bin/bash
set -e

echo "=== 🪶 Feather Molt Pipeline: Environment Setup ==="

if ! command -v conda &> /dev/null; then
    if [ ! -d "$HOME/miniforge3" ]; then
        echo "[1/3] Installing Miniforge3..."
        curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh
        bash miniforge.sh -b -p "$HOME/miniforge3"
        rm miniforge.sh
    fi
    export PATH="$HOME/miniforge3/bin:$PATH"
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[2/3] Creating 'feather_env'..."
conda create -y -n feather_env -c conda-forge python=3.10 || echo "feather_env already exists"
conda activate feather_env

echo "[3/3] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision ultralytics pandas open_clip_torch einops kornia timm mlx_vlm grad-cam jupyterlab ipykernel ipywidgets onnxruntime jupyter-resource-usage opencv-python matplotlib python-dotenv 'celery[redis]' flower redis

python -m ipykernel install --user --name feather_env --display-name "Feather Env"

echo "=== Setup Complete ==="
