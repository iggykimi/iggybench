#!/usr/bin/env bash
# run.sh - serve a config, run its tests, stop the server.
# Usage: ./run.sh configs/<tag>.env [T1 T2 ...]
#   With tests (or for a config without client()): suite.sh with those tests (default T1 T2 T3 T4).
#   Without tests, if the config defines client(): runs it with the results folder as $1 and logs GPU usage.
# Crash guard: suite.sh stops at the first dead-server check (exit 3, test name in caido.txt). run.sh then
# restarts the server once with SAFE=1 (less memory, smaller prefill chunks) and resumes the battery at the
# test that crashed. RESCUE=0 disables the restart.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"
CFG=${1:?"Usage: ./run.sh configs/<tag>.env [T1 T2 ...]"}; shift
# shellcheck disable=SC1090
source "$CFG"
SERVE_INFO="$LOG_DIR/serve_$(container_name "$TAG").txt"; export SERVE_INFO

"$HERE/serve.sh" "$CFG" || exit 1
trap '"$HERE/serve.sh" "$CFG" stop' EXIT

if [ $# -eq 0 ] && declare -F client >/dev/null; then
  OUT="$RESULTS_DIR/$TAG/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$OUT"
  BASE="http://${HOST:-127.0.0.1}:$PORT"; export BASE
  nvidia-smi --query-gpu="$GPU_FIELDS" --format=csv,noheader,nounits -l 1 > "$OUT/gpu.csv" 2>/dev/null & GPUPID=$!
  client "$OUT"; rc=$?
  kill "$GPUPID" 2>/dev/null
  echo "client finished (rc=$rc) · results in $OUT"
  exit "$rc"
fi

TESTS=("$@"); [ ${#TESTS[@]} -eq 0 ] && TESTS=(T1 T2 T3 T4)
"$HERE/suite.sh" "$CFG" "${TESTS[@]}"; rc=$?
if (( rc == 3 )) && [ "${RESCUE:-1}" = 1 ]; then
  last=$(ls -td "$RESULTS_DIR/$TAG"/*/ 2>/dev/null | head -1)
  crashed=$(cat "$last/caido.txt" 2>/dev/null)
  rest=(); on=0
  for x in "${TESTS[@]}"; do [ "$x" = "$crashed" ] && on=1; (( on )) && rest+=("$x"); done
  if (( ${#rest[@]} )); then
    echo "!! server died during $crashed: restarting with SAFE=1 and resuming with ${rest[*]}"
    "$HERE/serve.sh" "$CFG" stop
    SAFE=1 "$HERE/serve.sh" "$CFG" || exit 3
    "$HERE/suite.sh" "$CFG" "${rest[@]}"; rc=$?
  fi
fi
exit "$rc"
