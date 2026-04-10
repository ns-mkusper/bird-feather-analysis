#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "🚀 STARTING MULTI-NODE CELERY CLUSTER 🚀"
echo "=========================================="

KEY="~/.ssh/ubuntu-mac-openteams-admin"
USER="openteams"
HEAD_IP="10.0.0.148"
WORKER_IPS=("10.0.0.63" "10.0.0.19" "10.0.0.118")
REPO_DIR="/Users/openteams/Feather_Molt_Project"
PYTHON_BIN="/Users/openteams/miniforge3/envs/feather_env/bin/python"
PIP_BIN="/Users/openteams/miniforge3/envs/feather_env/bin/pip"

BROKER_URL="redis://$HEAD_IP:6379/0"
RESULT_BACKEND="redis://$HEAD_IP:6379/1"
WORKER_CONCURRENCY="${WORKER_CONCURRENCY:-1}"

sync_env_file() {
  local ip="$1"
  scp -i $KEY -o StrictHostKeyChecking=no .env $USER@$ip:$REPO_DIR/.env >/dev/null 2>&1 || true
}

echo "1. Preparing HEAD node ($HEAD_IP)..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
  source /Users/openteams/miniforge3/etc/profile.d/conda.sh
  conda activate feather_env >/dev/null 2>&1 || true
  $PIP_BIN install -q --upgrade pip
  $PIP_BIN install -q torch torchvision ultralytics pandas open_clip_torch einops kornia timm mlx_vlm grad-cam opencv-python python-dotenv 'celery[redis]' flower redis pi-heif

  REDIS_BIN=\"\"
  if [ -x /opt/homebrew/bin/redis-server ]; then
    REDIS_BIN=/opt/homebrew/bin/redis-server
  elif [ -x /usr/local/bin/redis-server ]; then
    REDIS_BIN=/usr/local/bin/redis-server
  elif command -v redis-server >/dev/null 2>&1; then
    REDIS_BIN=\$(command -v redis-server)
  else
    if [ -x /opt/homebrew/bin/brew ]; then
      /opt/homebrew/bin/brew list redis >/dev/null 2>&1 || /opt/homebrew/bin/brew install redis
    fi
    if [ -x /opt/homebrew/bin/redis-server ]; then
      REDIS_BIN=/opt/homebrew/bin/redis-server
    else
      conda install -y -n feather_env -c conda-forge redis >/dev/null 2>&1 || true
      if [ -x /Users/openteams/miniforge3/envs/feather_env/bin/redis-server ]; then
        REDIS_BIN=/Users/openteams/miniforge3/envs/feather_env/bin/redis-server
      fi
    fi
  fi
  if [ -z \"\$REDIS_BIN\" ]; then
    echo 'redis-server not found on head node after install attempts'
    exit 1
  fi
  echo \"\$REDIS_BIN\" > /tmp/feather_redis_bin_path

  if [ ! -d '$REPO_DIR/.git' ]; then
    git clone git@github.com:ns-mkusper/birth-feather-thesis.git '$REPO_DIR' >/dev/null 2>&1
  fi
  cd '$REPO_DIR' && git fetch origin && git reset --hard origin/main >/dev/null 2>&1

  pkill -f 'src/full_run_distributed.py' >/dev/null 2>&1 || true
  pkill -f 'celery -A src.celery_app worker' >/dev/null 2>&1 || true
  pkill -f 'celery -A src.celery_app flower' >/dev/null 2>&1 || true
  pkill -f 'redis-server .*feather_redis.conf' >/dev/null 2>&1 || true
"

sync_env_file "$HEAD_IP"

echo "2. Preparing WORKER nodes..."
for ip in "${WORKER_IPS[@]}"; do
  echo "   -> $ip"
  ssh -i $KEY -o StrictHostKeyChecking=no $USER@$ip "
    if [ ! -d '/Users/openteams/miniforge3' ]; then
      curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh
      bash miniforge.sh -b -p /Users/openteams/miniforge3 >/dev/null 2>&1
      rm miniforge.sh
    fi

    source /Users/openteams/miniforge3/etc/profile.d/conda.sh
    if ! conda env list | grep -q 'feather_env'; then
      conda create -y -n feather_env -c conda-forge python=3.10 >/dev/null 2>&1
    fi
    conda activate feather_env

    /Users/openteams/miniforge3/envs/feather_env/bin/pip install -q --upgrade pip
    /Users/openteams/miniforge3/envs/feather_env/bin/pip install -q torch torchvision ultralytics pandas open_clip_torch einops kornia timm mlx_vlm grad-cam opencv-python python-dotenv 'celery[redis]' flower redis pi-heif

    if [ ! -d '$REPO_DIR/.git' ]; then
      git clone git@github.com:ns-mkusper/birth-feather-thesis.git '$REPO_DIR' >/dev/null 2>&1
    fi
    cd '$REPO_DIR' && git fetch origin && git reset --hard origin/main >/dev/null 2>&1

    pkill -f 'celery -A src.celery_app worker' >/dev/null 2>&1 || true
    pkill -f 'src/full_run_distributed.py' >/dev/null 2>&1 || true
  " &
  sync_env_file "$ip"
done
wait

echo "3. Starting Redis on HEAD..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
  REDIS_BIN=\$(cat /tmp/feather_redis_bin_path)
  cat > /tmp/feather_redis.conf <<'CONF'
port 6379
bind 0.0.0.0
protected-mode no
dbfilename feather_dump.rdb
dir /tmp
appendonly no
CONF
  nohup \$REDIS_BIN /tmp/feather_redis.conf >/tmp/feather_redis.log 2>&1 &
"

sleep 2

echo "4. Starting Celery workers..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
  cd '$REPO_DIR'
  export PYTHONPATH='$REPO_DIR'
  export BROKER_URL='$BROKER_URL'
  export RESULT_BACKEND='$RESULT_BACKEND'
  nohup script -q /tmp/celery_head_pty.log $PYTHON_BIN -m celery -A src.celery_app worker --pool=threads --without-mingle --loglevel=INFO --concurrency=$WORKER_CONCURRENCY --hostname=head@%h > celery_worker.log 2>&1 < /dev/null & disown
"

for ip in "${WORKER_IPS[@]}"; do
  ssh -i $KEY -o StrictHostKeyChecking=no $USER@$ip "
    cd '$REPO_DIR'
    export PYTHONPATH='$REPO_DIR'
    export BROKER_URL='$BROKER_URL'
    export RESULT_BACKEND='$RESULT_BACKEND'
    nohup script -q /tmp/celery_worker_pty.log $PYTHON_BIN -m celery -A src.celery_app worker --pool=threads --without-mingle --loglevel=INFO --concurrency=$WORKER_CONCURRENCY --hostname=worker@%h > celery_worker.log 2>&1 < /dev/null & disown
  " &
done
wait

echo "5. Starting Flower dashboard on HEAD..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
  cd '$REPO_DIR'
  export PYTHONPATH='$REPO_DIR'
  export BROKER_URL='$BROKER_URL'
  export RESULT_BACKEND='$RESULT_BACKEND'
  export FLOWER_UNAUTHENTICATED_API=true
  nohup $PYTHON_BIN -m celery -A src.celery_app flower --address=0.0.0.0 --port=5555 > flower.log 2>&1 &
"

echo "6. Launching distributed feather pipeline..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$HEAD_IP "
  cd '$REPO_DIR'
  export PYTHONPATH='$REPO_DIR'
  export BROKER_URL='$BROKER_URL'
  export RESULT_BACKEND='$RESULT_BACKEND'
  nohup $PYTHON_BIN src/full_run_distributed.py > distributed_pipeline.log 2>&1 &
"

echo "=========================================="
echo "🎉 CELERY CLUSTER DEPLOYED 🎉"
echo "=========================================="
echo "Flower dashboard: http://$HEAD_IP:5555"
echo "Pipeline log (head): $REPO_DIR/distributed_pipeline.log"
echo "Worker logs: $REPO_DIR/celery_worker.log"
echo "=========================================="
