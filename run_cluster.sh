#!/bin/bash
set -e

echo "=========================================="
echo "🚀 STARTING MULTI-NODE RAY CLUSTER 🚀"
echo "=========================================="

KEY="~/.ssh/ubuntu-mac-openteams-admin"
USER="openteams"
HEAD_IP="10.0.0.148"
WORKER_IPS=("10.0.0.63" "10.0.0.19" "10.0.0.118")
REPO_DIR="/Users/openteams/Feather_Molt_Project"
PYTHON_BIN="/Users/openteams/miniforge3/envs/feather_env/bin/python"
PIP_BIN="/Users/openteams/miniforge3/envs/feather_env/bin/pip"
RAY_BIN="/Users/openteams/miniforge3/envs/feather_env/bin/ray"

# Stop existing processes on HEAD and prepare
echo "1. Preparing HEAD Node ($HEAD_IP)..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
    $PIP_BIN install -q 'ray[default]' python-dotenv mlx_vlm
    $RAY_BIN stop -f > /dev/null 2>&1 || true
            if [ ! -d "$REPO_DIR/.git" ]; then
            echo "      Cloning repository to $ip..."
            git clone git@github.com:ns-mkusper/birth-feather-thesis.git $REPO_DIR > /dev/null 2>&1
        fi
        cd $REPO_DIR && git fetch origin && git reset --hard origin/main > /dev/null 2>&1
"

# Export HF_TOKEN if available in the repo to copy to workers
echo "2. Syncing Environment & Bootstrapping WORKER Nodes..."
for ip in "${WORKER_IPS[@]}"; do
    echo "   -> Syncing to Worker: $ip"
    ssh -i $KEY -o StrictHostKeyChecking=no $USER@$ip "
        if [ ! -d \"/Users/openteams/miniforge3\" ]; then
            echo \"      [+] Installing Miniforge3 on \$ip...\"
            curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh
            bash miniforge.sh -b -p /Users/openteams/miniforge3 > /dev/null 2>&1
            rm miniforge.sh
        fi
        
        source /Users/openteams/miniforge3/etc/profile.d/conda.sh
        if ! conda env list | grep -q \"feather_env\"; then
            conda remove -y --name feather_env --all > /dev/null 2>&1 || true
            echo \"      [+] Creating feather_env and installing dependencies on \$ip (this may take 5+ minutes)...\"
            conda create -y -n feather_env -c conda-forge python=3.10 > /dev/null 2>&1
            conda activate feather_env
            pip install -q torch torchvision ultralytics pandas open_clip_torch einops kornia timm mlx_vlm grad-cam 'ray[default]' opencv-python python-dotenv
        else
            conda activate feather_env
            pip install -q 'ray[default]' python-dotenv mlx_vlm
        fi
        
                if [ ! -d "$REPO_DIR/.git" ]; then
            echo "      Cloning repository to $ip..."
            git clone git@github.com:ns-mkusper/birth-feather-thesis.git $REPO_DIR > /dev/null 2>&1
        fi
        cd $REPO_DIR && git fetch origin && git reset --hard origin/main > /dev/null 2>&1
        $RAY_BIN stop -f > /dev/null 2>&1 || true
    " &
    
    # Sync the .env file so the workers have the HuggingFace token
    scp -i $KEY -o StrictHostKeyChecking=no .env $USER@$ip:$REPO_DIR/.env > /dev/null 2>&1 || true
done
wait # Wait for all worker nodes to finish bootstrapping

echo "3. Starting HEAD Node ($HEAD_IP)..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
    # Use a tiny Python script to proxy the traffic from localhost to 10.0.0.x (Fixes MacOS aiohttp IPv6 bug)
    pkill -f "tiny_proxy.py" || true
    cat << 'PROX' > tiny_proxy.py
import socket, select, sys, _thread

def forward(source, destination):
    while True:
        try:
            string = source.recv(1024)
            if string: destination.sendall(string)
            else: break
        except Exception: break
    source.close(); destination.close()

def main(local_port, remote_host, remote_port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', local_port))
    server.listen(10)
    while True:
        local_socket, _ = server.accept()
        remote_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        remote_socket.connect((remote_host, remote_port))
        _thread.start_new_thread(forward, (local_socket, remote_socket))
        _thread.start_new_thread(forward, (remote_socket, local_socket))

if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
PROX
    nohup $PYTHON_BIN tiny_proxy.py 9090 10.0.0.246 9090 > /dev/null 2>&1 &
    nohup $PYTHON_BIN tiny_proxy.py 3000 10.0.0.246 3000 > /dev/null 2>&1 &
    sleep 2
    RAY_PROMETHEUS_HOST="http://127.0.0.1:9090" RAY_GRAFANA_HOST="http://127.0.0.1:3000" RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 $RAY_BIN start --head --node-ip-address=$HEAD_IP --port=6379 --dashboard-host=0.0.0.0 --metrics-export-port=8080 > /dev/null 2>&1
"
echo "   ✅ Head Node active! Dashboard available at http://$HEAD_IP:8265"

echo "4. Joining WORKER Nodes to the cluster..."
for ip in "${WORKER_IPS[@]}"; do
    echo "   -> Joining Worker: $ip"
    ssh -i $KEY -o StrictHostKeyChecking=no $USER@$ip "
        RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 $RAY_BIN start --address=$HEAD_IP:6379 --metrics-export-port=8080 > /dev/null 2>&1
    " &
done
wait
echo "   ✅ All Workers connected to Head Node!"

echo "5. Launching Distributed AI Pipeline!"
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
    cd $REPO_DIR
    nohup $PYTHON_BIN src/full_run_ray.py > ray_cluster_distributed.log 2>&1 &
"
echo "=========================================="
echo "🎉 MULTI-NODE CLUSTER DEPLOYED 🎉"
echo "=========================================="
echo "To monitor progress:"
echo "1. Ray Dashboard: Open http://$HEAD_IP:8265 in your browser."
echo "2. Process Gallery: Open http://$HEAD_IP:8889/lab and run Live_Processing_Gallery.ipynb"
echo "=========================================="
