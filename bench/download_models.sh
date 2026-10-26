#!/usr/bin/env bash
# download_models.sh - download the weights named by the configs (HF_REPO -> MODEL_DIR, DRAFT_REPO -> DRAFT_DIR),
# plus the Python sources and the Laya weights used by the e-mail and image clients. No GPU needed.
# Usage: ./download_models.sh [configs/<tag>.env ...]     (default: every config)
# Uses `hf` if installed on the host, otherwise the huggingface_hub inside the vLLM image. Gated or private repos
# read HF_TOKEN from the environment. Configs whose repo id was not recorded are listed and skipped.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"
CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=("$HERE"/configs/*.env)
mkdir -p "$MODELS_DIR" "$PYLIB_DIR" "$HF_CACHE"

# fetch <repo> <folder> [include patterns...]
fetch() {
  local repo=$1 dir=$2; shift 2
  local inc=(); local p; for p in "$@"; do inc+=(--include "$p"); done
  echo "== $(date +%T) $repo -> $dir"
  if command -v hf >/dev/null; then
    hf download "$repo" --local-dir "$dir" "${inc[@]}" 2>&1 | tail -1
  else
    $DOCKER run --rm -v "$MODELS_DIR:/models" -e HF_TOKEN --entrypoint hf "$VLLM_IMG" \
        download "$repo" --local-dir "/models/${dir#"$MODELS_DIR"/}" "${inc[@]}" 2>&1 | tail -1
  fi
  du -sh "$dir" 2>/dev/null
}

declare -A DONE=()
for c in "${CFGS[@]}"; do
  # each config in a subshell so its variables do not leak into the next one
  # shellcheck disable=SC1090
  mapfile -t jobs < <(source "$c"
    [ -n "${MODEL_DIR:-}" ] && printf '%s|%s|%s\n' "${HF_REPO:-}" "$MODEL_DIR" "${HF_INCLUDE[*]:-}"
    [ -n "${DRAFT_DIR:-}" ] && printf '%s|%s|\n' "${DRAFT_REPO:-}" "$DRAFT_DIR")
  for j in "${jobs[@]}"; do
    IFS='|' read -r repo dir inc <<< "$j"
    [ -n "${DONE[$dir]:-}" ] && continue; DONE[$dir]=1
    if [ -z "$repo" ]; then echo "!! $(basename "$c"): repo id not recorded; put the weights in $dir yourself"; continue; fi
    read -ra incs <<< "$inc"          # include patterns, split without glob expansion
    fetch "$repo" "$dir" "${incs[@]}"
  done
done

echo "== $(date +%T) Laya weights (HF cache) and source trees of diffusers and Laya (for image_gen.py and email_classify.py)"
$DOCKER run --rm -v "$HF_CACHE:/root/.cache/huggingface" -e HF_TOKEN --entrypoint python3 "$VLLM_IMG" -c \
  "from huggingface_hub import snapshot_download as s; print(s('convaiinnovations/laya'))" 2>&1 | tail -1
for r in huggingface/diffusers NandhaKishorM/laya; do
  n=${r#*/}
  if curl -LsSf --max-time 300 "https://github.com/$r/archive/refs/heads/main.zip" -o "$PYLIB_DIR/$n.zip" \
     && rm -rf "${PYLIB_DIR:?}/$n" "$PYLIB_DIR/$n-main" && python3 -m zipfile -e "$PYLIB_DIR/$n.zip" "$PYLIB_DIR/" \
     && mv "$PYLIB_DIR/$n-main" "$PYLIB_DIR/$n" && rm -f "$PYLIB_DIR/$n.zip"; then
    echo "   $n OK"
  else
    echo "   !! $n: download failed"
  fi
done
df -h "$MODELS_DIR" | tail -1
