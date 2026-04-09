#!/bin/bash
set -e

echo "=== 🪶 Feather Molt Pipeline: Environment Setup ==="
echo "This script will install all dependencies, ML models, and configure R + Python to work together."

# 1. Install Miniforge if conda is not available
if ! command -v conda &> /dev/null; then
    if [ ! -d "$HOME/miniforge3" ]; then
        echo "[1/4] Installing Miniforge3 for environment management..."
        curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh
        bash miniforge.sh -b -p "$HOME/miniforge3"
        rm miniforge.sh
    fi
    export PATH="$HOME/miniforge3/bin:$PATH"
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[2/4] Creating 'feather_env' (Python Core)..."
conda create -y -n feather_env -c conda-forge python=3.10 || echo "feather_env already exists"
conda activate feather_env

echo "[3/4] Installing Native Apple R 4.4 and XQuartz (for 'pavo' UI requirements)..."
# We bypass Conda's R here to ensure precompiled CRAN binary compatibility for heavy geo/spatial packages
if [ ! -d "/Library/Frameworks/R.framework/Versions/4.4-arm64" ]; then
    echo "Requires sudo to install Apple Native R and X11..."
    curl -O https://cloud.r-project.org/bin/macosx/big-sur-arm64/base/R-4.4.2-arm64.pkg
    sudo installer -pkg R-4.4.2-arm64.pkg -target /
    rm R-4.4.2-arm64.pkg
fi

if [ ! -d "/opt/X11" ]; then
    echo "Installing XQuartz..."
    curl -L -O https://github.com/XQuartz/XQuartz/releases/download/XQuartz-2.8.5/XQuartz-2.8.5.pkg
    sudo installer -pkg XQuartz-2.8.5.pkg -target /
    rm XQuartz-2.8.5.pkg
fi

export R_HOME="/Library/Frameworks/R.framework/Resources"
export PATH="$R_HOME/bin:$PATH"

echo "Installing R packages (pavo, jpeg) natively..."
Rscript -e "install.packages(c('pavo', 'jpeg'), repos='https://cloud.r-project.org/', type='mac.binary.big-sur-arm64')"

echo "[4/4] Installing Python ML dependencies..."
pip install --upgrade pip
pip install torch torchvision
pip install ultralytics pandas open_clip_torch einops kornia timm mlx_vlm grad-cam ray jupyterlab ipykernel ipywidgets  onnxruntime jupyter-resource-usage opencv-python matplotlib python-dotenv

# Force rpy2 to build against the newly installed Native R
pip install --force-reinstall --no-cache-dir rpy2

python -m ipykernel install --user --name feather_env --display-name "Feather Env (Python + R)"

echo "=== Setup Complete! ==="
echo "You can now run ./start_jupyter.sh"
