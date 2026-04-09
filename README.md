# Feather Molt Analysis Pipeline

A distributed, GPU-accelerated pipeline for extracting, segmenting, and analyzing longitudinal molt patterns in bird feathers using Apple Silicon (mps).

## Project Structure
- `data/raw/`: Drop the uncompressed feather `.jpg` files here.
- `data/processed/`: The pipeline will output isolated, normalized feather crops here.
- `notebooks/`: Contains the `minimal_slice_native.ipynb` PoC for interactive development (includes R `pavo` integration).
- `src/`: Contains the distributed Ray script (`full_run_ray.py`) for processing the entire dataset across the cluster.
- `models/`: Store custom weights (like `yolov8-feather.pt`) here.

## How to Use (Interactive Jupyter)
1. Execute `./start_jupyter.sh`
2. Connect via browser (e.g. `http://<compute-cluster-ip>:8889/lab`).

## How to Use (Distributed Ray Cluster)
1. Ensure Ray is started on the head node and all worker nodes with OSX clustering enabled:
   ```bash
   export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
   ray start --head --port=6379  # (On head node)
   ray start --address='<head-ip>:6379' # (On worker nodes)
   ```
2. From the project root on the head node, run:
   ```bash
   export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1
   python3 src/full_run_ray.py
   ```
