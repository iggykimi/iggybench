#!/usr/bin/env bash
# selfcheck.sh - pre-flight checks that need no GPU time (a few minutes; safe while someone else uses the GPU).
# Usage: ./selfcheck.sh [configs/<tag>.env ...]     (default: every config)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"
CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=("$HERE"/configs/*.env)
ok() { echo "   OK  $*"; }; ko() { echo "   !!  $*"; }

# one line per config: tag, engine, image, port, model dir, draft dir, serve flags
mapfile -t ROWS < <(for c in "${CFGS[@]}"; do
  # shellcheck disable=SC1090
  (source "$c"; printf '%s|%s|%s|%s|%s|%s|%s\n' "$(basename "$c")" "${ENGINE:-vllm}" "${IMG:-}" "${PORT:-}" \
     "${MODEL_DIR:-}" "${DRAFT_DIR:-}" "$(printf '%s\n' "${SERVE_ARGS[@]}" | grep -E '^--' | tr '\n' ' ')")
done)

echo "== 1. Docker and the vLLM image"
$DOCKER ps >/dev/null 2>&1 && ok "docker access ($DOCKER)" || ko "no docker access: join the docker group or set DOCKER to a command that has it"
HELP=$($DOCKER run --rm --entrypoint vllm "$VLLM_IMG" serve --help=all 2>&1)
FLAGS=$(for r in "${ROWS[@]}"; do IFS='|' read -r _ e im _ _ _ f <<< "$r"; [ "$e" = vllm ] && [ "$im" = "$VLLM_IMG" ] && echo "$f"; done \
        | tr ' ' '\n' | grep -E '^--' | sort -u)
for f in $FLAGS; do
  g=${f%%.*}; g=${g//_/-}         # --attention_config.indexer_kv_dtype -> --attention-config
  grep -q -- "$g" <<< "$HELP" && ok "$f" || ko "$f is not an option of vllm serve in $VLLM_IMG"
done
$DOCKER run --rm --entrypoint vllm "$VLLM_IMG" bench serve --help=all 2>&1 | grep -q -- "--save-detailed" \
  && ok "bench serve --save-detailed (every request is saved)" || ko "bench serve without --save-detailed: only summaries are saved"

echo "== 2. Models: present and complete"
for d in $(for r in "${ROWS[@]}"; do IFS='|' read -r _ _ _ _ m dr _ <<< "$r"; echo "$m"; echo "$dr"; done | sort -u); do
  [ -z "$d" ] && continue
  python3 - "$d" <<'PY'
import glob, json, os, subprocess, sys
p = sys.argv[1]; name = os.path.basename(p)
if not os.path.isdir(p):
    print(f"   !!  {name:<28} missing ({p})"); sys.exit()
size = subprocess.run(["du", "-sh", p], capture_output=True, text=True).stdout.split("\t")[0]
missing, partial = [], glob.glob(os.path.join(p, ".cache", "**", "*.incomplete"), recursive=True)
for idx in glob.glob(os.path.join(p, "**", "*.safetensors.index.json"), recursive=True):
    need = set(json.load(open(idx))["weight_map"].values())
    missing += [f for f in need if not os.path.exists(os.path.join(os.path.dirname(idx), f))]
weights = glob.glob(os.path.join(p, "**", "*.safetensors"), recursive=True)
state = "OK " if weights and not missing and not partial else "!! "
extra = (f" · {len(missing)} shards missing" if missing else "") + (f" · {len(partial)} partial" if partial else "") \
        + ("" if weights else " · no weights")
print(f"   {state} {name:<28} {size:>6}{extra}")
PY
done

echo "== 3. Datasets ($DATASETS_DIR)"
for f in gsm8k_test.jsonl sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json real8k.jsonl reasoning/math500.jsonl bfcl/BFCL_v3_simple.json; do
  [ -s "$DATASETS_DIR/$f" ] && ok "$f" || ko "missing $f (prepare_data.sh)"
done
ls "$DATASETS_DIR"/speed-bench/*.jsonl >/dev/null 2>&1 && ok "speed-bench" || ko "missing SPEED-Bench (T15 falls back to real8k)"
n=$(ls -d "$DATASETS_DIR"/asr/*/ 2>/dev/null | wc -l); (( n == 7 )) && ok "FLEURS: 7 languages" || ko "FLEURS: $n of 7 languages"

echo "== 4. Invoices and client code"
for s in invoices_clean invoices_degraded; do
  n=$(ls "$HERE/data/$s"/*.jpg 2>/dev/null | wc -l)
  [ -s "$HERE/data/$s/verdad.json" ] && (( n > 0 )) && ok "data/$s: $n images + verdad.json" || ko "data/$s incomplete ($n images)"
done
[ -d "$PYLIB_DIR/diffusers/src/diffusers" ] && ok "diffusers source" || ko "missing $PYLIB_DIR/diffusers (download_models.sh)"
[ -d "$PYLIB_DIR/laya" ] && ok "laya source" || ko "missing $PYLIB_DIR/laya (download_models.sh)"
ls -d "$HF_CACHE"/hub/models--convaiinnovations--laya >/dev/null 2>&1 && ok "Laya weights in the HF cache" || ko "missing Laya weights"
[ -n "${HF_TOKEN:-}" ] && ok "HF_TOKEN set" || ko "HF_TOKEN not set (needed for gated downloads only)"

echo "== 5. Ports used by the configs (must be free before serving)"
for p in $(for r in "${ROWS[@]}"; do IFS='|' read -r _ _ _ port _ <<< "$r"; echo "$port"; done | sort -u); do
  ss -ltn 2>/dev/null | grep -q ":$p " && ko "port $p is in use now" || ok "port $p free"
done

echo "== 6. GPU right now"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/   /' || ko "nvidia-smi not available"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/   /'
df -h "$MODELS_DIR" 2>/dev/null | tail -1 | awk '{print "   disk free for models: "$4}'
