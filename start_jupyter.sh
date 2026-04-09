#!/bin/bash
export PATH=$HOME/miniforge3/bin:$PATH
source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate feather_env

# Explicitly tell Python/rpy2 to use Apple Native R, bypassing conda R completely
export R_HOME="/Library/Frameworks/R.framework/Resources"
export PATH="$R_HOME/bin:$PATH"

# Crucial fix for MLX/Torch and R running in the same memory space on macOS
export KMP_DUPLICATE_LIB_OK=TRUE

echo "Starting Jupyter Lab in 'feather_env'..."
# FIX: Start Jupyter from the Project Root, not the notebooks folder, so all folders are visible in the UX!
cd "$(dirname "$0")"
nohup jupyter lab --ip=0.0.0.0 --port=8889 --no-browser --NotebookApp.token='' --NotebookApp.password='' > jupyter.log 2>&1 &
echo "Jupyter Lab started on port 8889!"
echo "View logs at: cat jupyter.log"
