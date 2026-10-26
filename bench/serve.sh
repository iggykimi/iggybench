#!/usr/bin/env bash
# serve.sh - start the model server described by a config in Docker and wait until it answers.
# Usage: ./serve.sh configs/<tag>.env          start (replaces a previous container of the same tag)
#        ./serve.sh configs/<tag>.env stop     save the container log to $LOG_DIR and remove the container
# Optional variables:
#   SERVE_EXTRA="--flag value ..."   appended to SERVE_ARGS (variants: KV offload, MTP; see the config comments)
#   SAFE=1      0.05 less --gpu-memory-utilization and half --max-num-batched-tokens (min 4096; 8192 if unset).
#               run.sh uses it to restart a server that crashed mid-battery.
#   TIMEOUT=s   how long to wait for /health (default from the config, else 1800)
# Other servers are left running, so two configs with different PORTs can share the GPU.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"
CFG=${1:?"Usage: ./serve.sh configs/<tag>.env [stop]"}
# shellcheck disable=SC1090
source "$CFG"
: "${TAG:?TAG missing in $CFG}" "${PORT:?PORT missing in $CFG}"
ENGINE=${ENGINE:-vllm}
CT=$(container_name "$TAG")
BASE="http://${HOST:-127.0.0.1}:$PORT"
mkdir -p "$LOG_DIR"

if [ "${2:-}" = stop ]; then
  if $DOCKER ps -aq -f "name=^${CT}$" | grep -q .; then
    $DOCKER logs "$CT" > "$LOG_DIR/server_${CT}_$(date +%Y%m%d-%H%M%S).log" 2>&1
    $DOCKER rm -f "$CT" >/dev/null
    echo "stopped $CT · GPU memory now: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
  fi
  exit 0
fi

if [ -n "${BUILD:-}" ] && ! $DOCKER image inspect "$IMG" >/dev/null 2>&1; then   # derived images
  echo "building $IMG from $BUILD"; $DOCKER build -t "$IMG" "$BUILD" || exit 1
fi
case $ENGINE in
  vllm|sglang) ;;
  *) echo "ENGINE=$ENGINE: nothing to serve (the config's client() runs the model itself)"; exit 0 ;;
esac
[ -d "${MODEL_DIR:-}" ] || { echo "Model folder not found: ${MODEL_DIR:-<unset>} (run download_models.sh $CFG)"; exit 1; }

if [ "${SAFE:-0}" = 1 ]; then    # more memory headroom and smaller prefill chunks
  safe=(); mbt=0; i=0
  while (( i < ${#SERVE_ARGS[@]} )); do
    x=${SERVE_ARGS[i]}
    case $x in
      --gpu-memory-utilization)
        safe+=("$x" "$(awk -v u="${SERVE_ARGS[i+1]}" 'BEGIN { v = u - 0.05; if (v < 0.70) v = 0.70; printf "%.2f", v }')"); i=$((i+2)); continue ;;
      --max-num-batched-tokens)
        v=$(( SERVE_ARGS[i+1] / 2 )); (( v < 4096 )) && v=4096; safe+=("$x" "$v"); mbt=1; i=$((i+2)); continue ;;
    esac
    safe+=("$x"); i=$((i+1))
  done
  (( mbt )) || safe+=(--max-num-batched-tokens 8192)
  SERVE_ARGS=("${safe[@]}")
fi

mkdir -p "$CACHE_DIR"/{vllm,flashinfer,triton}
RUN=(-d --name "$CT" --gpus all --ipc=host --network host -v "$MODEL_DIR:/model:ro")
[ -n "${DRAFT_DIR:-}" ] && RUN+=(-v "$DRAFT_DIR:/draft:ro")      # speculative-decoding draft model
RUN+=("${DOCKER_ARGS[@]}")
if [ "$ENGINE" = vllm ]; then
  # keep vLLM's compile cache and the FlashInfer/Triton JIT kernels between starts
  RUN+=(-v "$CACHE_DIR/vllm:/root/.cache/vllm" -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer" -v "$CACHE_DIR/triton:/root/.triton")
  # shellcheck disable=SC2206
  RUN+=(--entrypoint vllm "$IMG" serve /model --served-model-name "$MODEL" --host "${HOST:-127.0.0.1}" --port "$PORT"
        "${SERVE_ARGS[@]}" ${SERVE_EXTRA:-})
else
  # SGLang: the launch script shipped with the model reads MODEL/HOST/PORT from the environment
  RUN+=(-e MODEL=/model -e HOST="${HOST:-127.0.0.1}" -e PORT="$PORT" "$IMG" "${SERVE_CMD[@]}")
fi

$DOCKER rm -f "$CT" >/dev/null 2>&1
echo "starting $CT ($ENGINE, $IMG)"
$DOCKER run "${RUN[@]}" >/dev/null || { echo "!! docker run failed"; exit 1; }

fail() {
  echo "!! $CT: $1 (last lines of the log below; full log in $LOG_DIR)"
  $DOCKER logs "$CT" > "$LOG_DIR/failed_${CT}_$(date +%Y%m%d-%H%M%S).log" 2>&1
  $DOCKER logs --tail 30 "$CT" 2>&1 | sed 's/^/   /'
  $DOCKER rm -f "$CT" >/dev/null 2>&1; exit 1
}
TIMEOUT=${TIMEOUT:-1800}; t=0
until curl -sf --max-time 5 "$BASE/health" >/dev/null && curl -sf --max-time 5 "$BASE/v1/models" >/dev/null; do
  $DOCKER ps -q -f "name=^${CT}$" | grep -q . || fail "the container exited"
  (( t >= TIMEOUT )) && fail "not ready after $TIMEOUT s"
  sleep 20; t=$((t+20))
done
echo "== $CT ready after ~$t s on $BASE"
$DOCKER logs "$CT" 2>&1 | grep -E "KV cache size|Maximum concurrency" | tail -2 | sed 's/^/   /'
echo "   GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"

if [ -z "${NOGEN:-}" ]; then      # embeddings, rerankers and ASR do not generate text
  resp=$(curl -s --max-time 300 "$BASE/v1/completions" -H 'Content-Type: application/json' \
         -d "{\"model\":\"$MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}" \
         | python3 -c 'import sys,json;print(repr(json.load(sys.stdin)["choices"][0]["text"]))' 2>&1 | head -c 200)
  echo "   generation check: $resp"
fi

# What exactly is running, for the results folder (suite.sh copies it as serve_info.txt)
{
  echo "tag=$TAG"; echo "engine=$ENGINE"; echo "image=$IMG"
  echo "image_digest=$($DOCKER image inspect -f '{{join .RepoDigests " "}} {{.Id}}' "$IMG" 2>/dev/null)"
  echo "hf_repo=${HF_REPO:-}"; echo "started=$(date -Iseconds)"
  printf 'docker_run='; printf '%q ' "${RUN[@]}"; echo
  echo "v1_models=$(curl -s --max-time 10 "$BASE/v1/models")"
} > "$LOG_DIR/serve_${CT}.txt"
