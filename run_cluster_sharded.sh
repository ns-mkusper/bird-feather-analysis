#!/bin/bash
# Split data/raw into one shard per node and process each shard on its node
# (no broker needed). Nodes, SSH and VLM settings come from .env.
#
#   ./run_cluster_sharded.sh
#   MAX_IMAGES=20 FEATHER_RUN_ID=test-$(date +%s) ./run_cluster_sharded.sh
set -euo pipefail
# shellcheck source=cluster_env.sh
. "$(dirname "$0")/cluster_env.sh"

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"
MAX_IMAGES="${MAX_IMAGES:-}"
ENABLE_VLM="${FEATHER_ENABLE_VLM:-1}"
VLM_MODEL="${FEATHER_VLM_MODEL:-mlx-community/Qwen3.8-27B-4bit}"

SHARD_COUNT="${#NODES[@]}"
RUN_ID="${FEATHER_RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ROOT="data/runs/${RUN_ID}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-${RUN_ROOT}/processed}"
METRICS_SUBDIR="${METRICS_SUBDIR:-${RUN_ROOT}/pipeline_metrics}"
LOG_SUBDIR="${LOG_SUBDIR:-${RUN_ROOT}/logs}"
GIT_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null || true)" ]; then GIT_DIRTY=1; else GIT_DIRTY=0; fi

echo "Starting sharded run across ${SHARD_COUNT} nodes..."
echo "Run ID: ${RUN_ID}"
echo "Output: ${OUTPUT_SUBDIR}"
echo "Git: commit=${GIT_COMMIT} branch=${GIT_BRANCH} dirty=${GIT_DIRTY}"
echo "VLM model: ${VLM_MODEL} (enable=${ENABLE_VLM})"

for i in "${!NODES[@]}"; do
  node="${NODES[$i]}"
  echo "-> launching shard $i on $node"
  node_ssh "$node" "
    cd $REPO_DIR
    export FEATHER_ENABLE_VLM='$ENABLE_VLM'
    export FEATHER_VLM_MODEL='$VLM_MODEL'
    export PYTHONUNBUFFERED=1
    export FEATHER_RUN_ID='${RUN_ID}'
    export FEATHER_NODE_ID='${node}'
    export FEATHER_METRICS_REDIS_HOST='${FEATHER_METRICS_REDIS_HOST:-$HEAD}'
    export FEATHER_METRICS_REDIS_PORT='${FEATHER_METRICS_REDIS_PORT:-6379}'
    export FEATHER_METRICS_REDIS_DB='${FEATHER_METRICS_REDIS_DB:-2}'
    export FEATHER_METRICS_DIR=\"\$PWD/$METRICS_SUBDIR\"
    mkdir -p '$OUTPUT_SUBDIR' '$METRICS_SUBDIR' '$LOG_SUBDIR'
    extra_args=''
    if [ -n '$MAX_IMAGES' ]; then extra_args=\"--max-images '$MAX_IMAGES'\"; fi
    nohup $PIXI run python src/full_run_sharded.py \
      --input-dir \"\$PWD/data/raw\" \
      --output-dir \"\$PWD/$OUTPUT_SUBDIR\" \
      --shard-index $i \
      --shard-count $SHARD_COUNT \
      \$extra_args \
      > '$LOG_SUBDIR/shard_${i}_${node}.log' 2>&1 < /dev/null &
  " &
done
wait

# Run metadata lives with the runs on the head node.
node_ssh "$HEAD" "
  cd $REPO_DIR && mkdir -p data/runs
  {
    echo 'run_id,start_utc,output_subdir,metrics_subdir,log_subdir,enable_vlm,enable_vlm_scoring,enable_vlm_metadata,vlm_model,max_images,git_commit,git_branch,git_dirty'
    echo '${RUN_ID},$(date -u +%Y-%m-%dT%H:%M:%SZ),${OUTPUT_SUBDIR},${METRICS_SUBDIR},${LOG_SUBDIR},${ENABLE_VLM},${ENABLE_VLM_SCORING},${ENABLE_VLM_METADATA},${VLM_MODEL},${MAX_IMAGES:-all},${GIT_COMMIT},${GIT_BRANCH},${GIT_DIRTY}'
  } > data/runs/${RUN_ID}.meta.csv
  if [ ! -f data/runs/run_history.csv ] || ! head -n 1 data/runs/run_history.csv | grep -q git_commit; then
    echo 'run_id,start_utc,output_subdir,enable_vlm,enable_vlm_scoring,enable_vlm_metadata,vlm_model,max_images,git_commit,git_branch,git_dirty' > data/runs/run_history.csv
  fi
  echo '${RUN_ID},$(date -u +%Y-%m-%dT%H:%M:%SZ),${OUTPUT_SUBDIR},${ENABLE_VLM},${ENABLE_VLM_SCORING},${ENABLE_VLM_METADATA},${VLM_MODEL},${MAX_IMAGES:-all},${GIT_COMMIT},${GIT_BRANCH},${GIT_DIRTY}' >> data/runs/run_history.csv
"
echo "All shard processes launched."
echo "Logs: $REPO_DIR/$LOG_SUBDIR on each node"
