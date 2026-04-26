#!/bin/bash
set -euo pipefail

KEY="$HOME/.ssh/ubuntu-mac-cluster_user-admin"
USER="cluster_user"
DEFAULT_NODES_CSV="10.0.0.148,10.0.0.63,10.0.0.19,10.0.0.118"
NODES_CSV="${NODES_CSV:-$DEFAULT_NODES_CSV}"
IFS=',' read -r -a NODES <<< "$NODES_CSV"
REPO_DIR="~/Feather_Molt_Project"
PYTHON_BIN="~/miniforge3/envs/feather_env/bin/python"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"
MAX_IMAGES="${MAX_IMAGES:-}"
ENABLE_VLM="${ENABLE_VLM:-0}"
ENABLE_VLM_SCORING="${ENABLE_VLM_SCORING:-0}"
ENABLE_VLM_METADATA="${ENABLE_VLM_METADATA:-1}"
VLM_MODEL="${VLM_MODEL:-mlx-community/Qwen3-VL-8B-Instruct-4bit}"

SHARD_COUNT="${#NODES[@]}"
RUN_ID="${FEATHER_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="data/runs/${RUN_ID}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-${RUN_ROOT}/processed}"
METRICS_SUBDIR="${METRICS_SUBDIR:-${RUN_ROOT}/pipeline_metrics}"
LOG_SUBDIR="${LOG_SUBDIR:-${RUN_ROOT}/logs}"
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse --verify HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null || true)" ]; then
  GIT_DIRTY=1
else
  GIT_DIRTY=0
fi
SSH_OPTS=(-o StrictHostKeyChecking=no)
if [ -f "$KEY" ]; then
  SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no)
fi

echo "Starting sharded run across ${SHARD_COUNT} nodes..."
echo "Run ID: ${RUN_ID}"
echo "Output: ${OUTPUT_SUBDIR}"
echo "Metrics: ${METRICS_SUBDIR}"
echo "Logs: ${LOG_SUBDIR}"
echo "Git: commit=${GIT_COMMIT} branch=${GIT_BRANCH} dirty=${GIT_DIRTY}"
echo "VLM model: ${VLM_MODEL} (enable=${ENABLE_VLM}, scoring=${ENABLE_VLM_SCORING}, metadata=${ENABLE_VLM_METADATA})"

for i in "${!NODES[@]}"; do
  ip="${NODES[$i]}"
  echo "-> launching shard $i on $ip"
  ssh ${SSH_OPTS[@]} $USER@$ip "
    cd '$REPO_DIR'
    export PYTHONPATH='$REPO_DIR'
    export FEATHER_ENABLE_VLM='$ENABLE_VLM'
    export FEATHER_ENABLE_VLM_SCORING='$ENABLE_VLM_SCORING'
    export FEATHER_ENABLE_VLM_METADATA='$ENABLE_VLM_METADATA'
    export FEATHER_VLM_MODEL='$VLM_MODEL'
    export PYTHONUNBUFFERED=1
    export FEATHER_RUN_ID='${RUN_ID}'
    export FEATHER_NODE_ID='${ip}'
    export FEATHER_METRICS_REDIS_HOST='10.0.0.148'
    export FEATHER_METRICS_REDIS_PORT='6379'
    export FEATHER_METRICS_REDIS_DB='2'
    export FEATHER_METRICS_DIR='$REPO_DIR/$METRICS_SUBDIR'
    mkdir -p '$OUTPUT_SUBDIR' '$METRICS_SUBDIR' '$LOG_SUBDIR'
    extra_args=''
    if [ -n '$MAX_IMAGES' ]; then
      extra_args=\"--max-images '$MAX_IMAGES'\"
    fi
    nohup $PYTHON_BIN src/full_run_sharded.py \
      --input-dir '$REPO_DIR/data/raw' \
      --output-dir '$REPO_DIR/$OUTPUT_SUBDIR' \
      --shard-index $i \
      --shard-count $SHARD_COUNT \
      \$extra_args \
      > '$REPO_DIR/$LOG_SUBDIR/shard_${i}_${ip}.log' 2>&1 &
  " &
done

wait
mkdir -p "$REPO_DIR/data/runs"
{
  echo "run_id,start_utc,output_subdir,metrics_subdir,log_subdir,enable_vlm,enable_vlm_scoring,enable_vlm_metadata,vlm_model,max_images,git_commit,git_branch,git_dirty"
  echo "${RUN_ID},$(date -u +%Y-%m-%dT%H:%M:%SZ),${OUTPUT_SUBDIR},${METRICS_SUBDIR},${LOG_SUBDIR},${ENABLE_VLM},${ENABLE_VLM_SCORING},${ENABLE_VLM_METADATA},${VLM_MODEL},${MAX_IMAGES:-all},${GIT_COMMIT},${GIT_BRANCH},${GIT_DIRTY}"
} > "$REPO_DIR/data/runs/${RUN_ID}.meta.csv"
if [ ! -f "$REPO_DIR/data/runs/run_history.csv" ] || ! head -n 1 "$REPO_DIR/data/runs/run_history.csv" | grep -q "git_commit"; then
  echo "run_id,start_utc,output_subdir,enable_vlm,enable_vlm_scoring,enable_vlm_metadata,vlm_model,max_images,git_commit,git_branch,git_dirty" > "$REPO_DIR/data/runs/run_history.csv"
fi
echo "${RUN_ID},$(date -u +%Y-%m-%dT%H:%M:%SZ),${OUTPUT_SUBDIR},${ENABLE_VLM},${ENABLE_VLM_SCORING},${ENABLE_VLM_METADATA},${VLM_MODEL},${MAX_IMAGES:-all},${GIT_COMMIT},${GIT_BRANCH},${GIT_DIRTY}" >> "$REPO_DIR/data/runs/run_history.csv"
echo "All shard processes launched."
echo "Logs: $REPO_DIR/$LOG_SUBDIR on each node"
