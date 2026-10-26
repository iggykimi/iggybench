#!/usr/bin/env bash
# =============================================================================
#  suite.sh - LLM inference benchmark battery (vLLM bench serve + Python clients)
# -----------------------------------------------------------------------------
#  Usage: ./suite.sh configs/<tag>.env [T1 T2 T3 ...]      (no tests = T1 T2 T3 T4)
#  The server must already be up (serve.sh, or run.sh which wraps serve -> suite -> stop).
#
#  Optional variables:
#    QUICK=1    short version (few requests) to check that everything works
#    DRYRUN=1   launch nothing, only print what would run
#    ONLY=chat  only that profile in T3/T4 (chat | rag | agentes | razonamiento)
#    MAXC=64    skip points above this concurrency (slow models)
#    STOP_X=3   early stop: skip the harder points of a sweep when TTFT/TPOT > 3x the SLO
#    BENCH_IMG  image that runs `vllm bench serve` (default $VLLM_IMG)
#
#  Tests:
#    T1   Raw speed            1 user, 512 in / 1024 out
#    T2   Prefill curve        1 user, 1k..128k in / 16 out
#    T3   Closed-loop load     4 profiles, concurrency 1..200
#    T4   Realistic traffic    Poisson arrivals, increasing rate
#    T4F  Refined capacity     rates between the last T4 rate that met the SLO and the first that did not
#    T5   Extreme context      128k / 256k / 512k / 1M (if the server allows it)
#    T6   MTP                  T1 + chat at 1/16/64/150 (run with the server in MTP mode)
#    T7   Prefix cache         16k input with and without an 8k common prompt (conc 1/16/64)
#    T8   KV offloading        NPREF 32k documents that repeat; run with T8_LABEL=base and, after restarting
#                              the server with offload to CPU RAM, with T8_LABEL=offload
#    T9   Department           chat + RAG + agents at once for 25/50/100/200/400 people
#    (T10, a batch-1 quick-quality check replaced by T16 and T20-T22, is not included)
#    T11  Invoice vision       scanned invoices -> JSON;  T11D: degraded invoices
#    T12  Community standard   8192 in / 1024 out, 1 -> 128 users
#    T13  Tool-using agents    8 multi-step back-office tasks
#    T14  Coding sessions      long sessions 48k -> ~100k context;  T14F: refined from the previous T14
#    T15  Real text            SPEED-Bench / ShareGPT instead of random tokens
#    T16  GSM8K                full test set (1319)
#    T17  Repeatability        3 repetitions of two key points
#    T18  Bursty traffic       chat with burstiness 0.5
#    T19  Steady state         10 min at the highest chat rate that met the SLO in T4
#    T20  Hard reasoning       AIME 2025/2026 and MATH-500
#    T21  Function calling     BFCL v3
#    T22  Hard email triage
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"
CFG=${1:?"Usage: ./suite.sh configs/<tag>.env [T1 T2 T3 ...]"}; shift
# shellcheck disable=SC1090
source "$CFG"
TESTS=("$@"); [ ${#TESTS[@]} -eq 0 ] && TESTS=(T1 T2 T3 T4)

: "${TAG:?TAG missing in $CFG}" "${PORT:?PORT missing in $CFG}"
TOKENIZER=${TOKENIZER:-${MODEL_DIR:-${HF_REPO:-}}}
: "${TOKENIZER:?TOKENIZER (or MODEL_DIR / HF_REPO) missing in $CFG}"
QUICK=${QUICK:-0}
DRYRUN=${DRYRUN:-0}
ONLY=${ONLY:-}
RAG_IN=${RAG_IN:-8192}
BURSTINESS=${BURSTINESS:-1.0}     # 1.0 = Poisson; <1 = burstier
T4_DUR=${T4_DUR:-90}             # seconds of traffic per T4 point
BASE="http://${HOST:-127.0.0.1}:${PORT}"

# ---- SLO per profile (ms), passed to --goodput -------------------------------
#  chat:          TTFT <= 2 s,  TPOT <= 50 ms (>= 20 tok/s per user)
#  rag:           TTFT <= 5 s,  TPOT <= 50 ms
#  agentes:       TTFT <= 5 s,  TPOT <= 40 ms (>= 25 tok/s)
#  razonamiento:                TPOT <= 40 ms
declare -A SLO=(
  [chat]="ttft:2000 tpot:50"
  [rag]="ttft:5000 tpot:50"
  [agentes]="ttft:5000 tpot:40"
  [razonamiento]="tpot:40"
  [velocidad]="tpot:50"
  [prefill]="ttft:120000"
  [extremo]="ttft:600000"
)

# ---- Benchmark client image -------------------------------------------------
BENCH_IMG=${BENCH_IMG:-$VLLM_IMG}

# ---- Also save every request (TTFT, ITL, lengths) so any statistic can be recomputed ----
DETAIL=()
if [ "$DRYRUN" = 0 ] && [ "${SAVE_DETAILED:-1}" = 1 ] && \
   $DOCKER run --rm --entrypoint vllm "$BENCH_IMG" bench serve --help=all 2>&1 | grep -q -- "--save-detailed"; then
  DETAIL=(--save-detailed)
fi

# ---- Server info ------------------------------------------------------------
MODELS_JSON=$(curl -s --max-time 10 "$BASE/v1/models" || true)
if [ -z "${MODEL:-}" ]; then
  MODEL=$(python3 -c 'import sys,json;print(json.loads(sys.argv[1])["data"][0]["id"])' "$MODELS_JSON" 2>/dev/null || true)
fi
if [ -z "${MAXLEN:-}" ]; then
  MAXLEN=$(python3 -c 'import sys,json;print(json.loads(sys.argv[1])["data"][0].get("max_model_len",0))' "$MODELS_JSON" 2>/dev/null || echo 0)
fi
if [ -z "${MODEL:-}" ] || [ "${MAXLEN:-0}" = 0 ]; then
  if [ "$DRYRUN" = 1 ]; then MODEL=${MODEL:-model}; [ "${MAXLEN:-0}" = 0 ] && MAXLEN=262144
  else echo "No answer from $BASE/v1/models (or no max_model_len). Check PORT or set MODEL/MAXLEN in $CFG"; exit 1; fi
fi

# Tokenizer: local folder (mounted) or Hugging Face repo id
if [ -d "$TOKENIZER" ]; then TOKMNT=(-v "$TOKENIZER:/tok:ro"); TOKARG=/tok
elif [ -n "${HF_REPO:-}" ]; then TOKENIZER=$HF_REPO; TOKMNT=(); TOKARG=$HF_REPO
elif [[ "$TOKENIZER" == /* ]] && [ "$DRYRUN" = 0 ]; then echo "Tokenizer folder not found: $TOKENIZER"; exit 1
else TOKMNT=(); TOKARG=$TOKENIZER; fi

RUNID=$(date +%Y%m%d-%H%M%S)
OUT="$RESULTS_DIR/$TAG/$RUNID"
mkdir -p "$OUT"/{json,logs,metrics} "$HF_CACHE"
echo "tag,runid,test,profile,in,out,prefix,conc,rate,n,start,end,rc,file" > "$OUT/runs.csv"

{
  echo "tag=$TAG"; echo "model=$MODEL"; echo "max_model_len=$MAXLEN"; echo "port=$PORT"
  echo "img=$BENCH_IMG"; echo "tokenizer=$TOKENIZER"; echo "tests=${TESTS[*]}"; echo "quick=$QUICK"
  echo "slo_chat=${SLO[chat]}"; echo "slo_rag=${SLO[rag]}"; echo "slo_agentes=${SLO[agentes]}"
  echo "slo_razonamiento=${SLO[razonamiento]}"; echo "rag_in=$RAG_IN"; echo "burstiness=$BURSTINESS"
  echo "detalle_por_peticion=${#DETAIL[@]}"; echo "fecha=$(date -Iseconds)"
} > "$OUT/suite_info.txt"
echo "$MODELS_JSON" > "$OUT/server_models.json"
ps -eo args 2>/dev/null | grep '[v]llm serve' > "$OUT/server_cmd.txt" || true
[ -s "${SERVE_INFO:-}" ] && cp "$SERVE_INFO" "$OUT/serve_info.txt"   # written by serve.sh: image, digest, docker run line
nvidia-smi > "$OUT/nvidia-smi_inicio.txt" 2>/dev/null || true

# ---- GPU monitor in the background (1 sample/s) -----------------------------
GPUPID=""
if [ "$DRYRUN" = 0 ]; then
  nvidia-smi --query-gpu="$GPU_FIELDS" --format=csv,noheader,nounits -l 1 > "$OUT/gpu.csv" 2>/dev/null &
  GPUPID=$!
fi

# ---- vLLM monitor in the background (every 2 s): queue, KV, prefix cache, speculative decoding ----
metrics_poller() {
  while true; do
    local t; t=$(date +%s.%N)
    curl -s --max-time 2 "$BASE/metrics" 2>/dev/null | awk -v t="$t" '
      /^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc|prefix_cache_hits_total|prefix_cache_queries_total|num_preemptions_total|prompt_tokens_total|generation_tokens_total|spec_decode_num_accepted_tokens_total|spec_decode_num_draft_tokens_total)[{ ]/ {
        split($0, a, "{"); n = a[1]; sub(/ .*/, "", n); sub(/^vllm:/, "", n); s[n] += $NF; k = 1 }
      END { if (k) { printf "%s", t; for (x in s) printf ";%s=%s", x, s[x]; printf "\n" } }'
    sleep 2
  done
}
MPID=""
if [ "$DRYRUN" = 0 ]; then metrics_poller > "$OUT/vllm_metrics.csv" 2>/dev/null & MPID=$!; fi
trap '[ -n "$GPUPID" ] && kill $GPUPID 2>/dev/null; [ -n "$MPID" ] && kill $MPID 2>/dev/null' EXIT

# ---- Helpers ----------------------------------------------------------------
# n_req <conc> <minimum> [multiplier]  -> number of requests for the point
n_req() {
  local c=$1 min=$2 mult=${3:-4} x
  x=$(( c * mult )); (( x < min )) && x=$min
  if (( QUICK )); then x=$(( x / 4 )); (( x < 4 )) && x=4; fi
  echo "$x"
}
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
snap() { [ "$DRYRUN" = 1 ] && return; curl -s --max-time 10 "$BASE/metrics" > "$OUT/metrics/$1.txt" 2>/dev/null || true; }
NRUNS=0

# ---- Guard: if the server dies, stop sending tests to it ---------------------
# Exits with code 3 and writes the test name to caido.txt: run.sh restarts the server with more
# memory headroom and resumes from that test.
CUR_T=start
alive() { (( DRYRUN )) && return 0; local i; for i in 1 2 3; do curl -s --max-time 5 "$BASE/v1/models" >/dev/null && return 0; sleep 5; done; return 1; }
server_down() {
  echo "!! SERVER DOWN during $CUR_T"; echo "$CUR_T" > "$OUT/caido.txt"
  nvidia-smi > "$OUT/nvidia-smi_fin.txt" 2>/dev/null; exit 3
}

# bench <test> <profile> <in> <out> <prefix> <conc|-> <rate|inf> <n>
bench() {
  local T=$1 P=$2 IN=$3 OL=$4 PRE=$5 C=$6 R=$7 N=$8
  LAST_JSON=""; LAST_RC=0; LAST_P=$P
  if [ -n "${MAXC:-}" ] && [ "$C" != "-" ] && (( C > MAXC )); then
    echo "   · skip $T/$P conc=$C: above MAXC=$MAXC"; return
  fi
  if (( IN + PRE + OL > MAXLEN )); then
    echo "   · skip $T/$P in=$((IN+PRE)): above max_model_len=$MAXLEN"; return
  fi
  alive || server_down
  local NAME="${T}_${P}_in${IN}_pre${PRE}_out${OL}_c${C}_r${R}"
  local DSARGS=(--dataset-name random --random-input-len "$IN" --random-output-len "$OL")
  if [ "${DSMODE:-}" = speed ]; then
    # T15: real text from SPEED-Bench (NVIDIA) with a fixed input length (throughput_8k subset, etc.)
    DSARGS=(--dataset-name speed_bench --dataset-path "${SPEED_PATH:-/datasets/speed-bench}"
            --speed-bench-dataset-subset "${SPEED_SUBSET:-throughput_8k}" --speed-bench-output-len "$OL")
  elif [ "${DSMODE:-}" = real8k ]; then
    # T15 without SPEED-Bench: ~8k tokens of real text (concatenated ShareGPT conversations), 1024 out
    DSARGS=(--dataset-name custom --dataset-path /datasets/real8k.jsonl --custom-output-len "$OL")
  elif [ "${DSMODE:-}" = sharegpt ]; then
    # T15: real ShareGPT conversations with their real input and output lengths
    DSARGS=(--dataset-name sharegpt --dataset-path /datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json)
  fi
  if [ "${DSMODE:-}" = prefix_rep ]; then
    # T8: NPREF different long documents (PRE tokens) repeated in random order
    DSARGS=(--dataset-name prefix_repetition --prefix-repetition-prefix-len "$PRE"
            --prefix-repetition-suffix-len "$IN" --prefix-repetition-num-prefixes "$NPREF"
            --prefix-repetition-output-len "$OL")
  fi
  local ARGS=(bench serve --backend vllm --base-url "$BASE" --model "$MODEL"
              --tokenizer "$TOKARG" --trust-remote-code
              "${DSARGS[@]}"
              --ignore-eos --num-prompts "$N" --seed "$(( $(od -An -N4 -tu4 /dev/urandom) % 1000000 ))"
              --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
              --save-result --result-dir /out --result-filename "$NAME.json" "${DETAIL[@]}")
  (( PRE > 0 )) && [ -z "${DSMODE:-}" ] && ARGS+=(--random-prefix-len "$PRE")
  [ -n "${TOKMODE:-}" ] && ARGS+=(--tokenizer-mode "$TOKMODE")
  [ "$C" != "-" ] && ARGS+=(--max-concurrency "$C")
  [ "$R" != "inf" ] && ARGS+=(--request-rate "$R" --burstiness "$BURSTINESS")
  local G; read -ra G <<< "${SLO[${P%%@*}]:-}"; (( ${#G[@]} )) && ARGS+=(--goodput "${G[@]}")

  NRUNS=$((NRUNS+1))
  echo ">>> $(date +%T)  $T/$P  in=$IN pre=$PRE out=$OL conc=$C rate=$R n=$N"
  if (( DRYRUN )); then echo "    vllm ${ARGS[*]}"; return; fi

  local S E RC
  S=$(date +%s.%N)
  local PDMNT=()   # T15 with our own text: `--dataset-name custom` needs pandas, which the container lacks
  [ "${DSMODE:-}" = real8k ] && PDMNT=(-v "$HERE/minipandas:/minipandas:ro" -e PYTHONPATH=/minipandas)
  $DOCKER run --rm --network host "${TOKMNT[@]}" "${PDMNT[@]}" \
      -v "$OUT/json:/out" -v "$HF_CACHE:/root/.cache/huggingface" -v "$DATASETS_DIR:/datasets:ro" \
      --entrypoint vllm "$BENCH_IMG" "${ARGS[@]}" > "$OUT/logs/$NAME.log" 2>&1
  RC=$?
  E=$(date +%s.%N)
  LAST_JSON="$OUT/json/$NAME.json"; LAST_RC=$RC
  echo "$TAG,$RUNID,$T,$P,$IN,$OL,$PRE,$C,$R,$N,$S,$E,$RC,$NAME.json" >> "$OUT/runs.csv"
  if (( RC != 0 )); then
    echo "    !! FAILED (rc=$RC) -> $OUT/logs/$NAME.log"; tail -n 4 "$OUT/logs/$NAME.log" | sed 's/^/    /'
  else
    grep -E 'Successful requests|Failed requests|Request goodput|Output token throughput|P95 TTFT|Median TPOT' \
        "$OUT/logs/$NAME.log" | sed 's/^/    /'
  fi
}

# ---- Early stop ----------------------------------------------------------------
# After each point of a sweep: if it is clearly saturated (TTFT p95 or TPOT > STOP_X times the SLO,
# >5 % failed requests or, in T4, goodput < 50 %), the harder points are skipped.
STOP_X=${STOP_X:-3}
# check_json <json> <profile> [t4] -> prints the saturation reasons (empty if fine)
check_json() {
  [ -f "$1" ] || { echo "no result"; return; }
  python3 - "$1" "${SLO[${2%%@*}]:-}" "$STOP_X" "${3:-}" <<'PY2'
import json, sys
d = json.load(open(sys.argv[1]))
slo = dict(x.split(":") for x in sys.argv[2].split() if ":" in x)
X = float(sys.argv[3]); mode = sys.argv[4]
n, ok = d.get("num_prompts") or 0, d.get("completed") or 0
r = []
if n and ok < 0.95 * n: r.append(f"{n-ok} failed requests")
tt = d.get("p95_ttft_ms"); tp = d.get("p50_tpot_ms") or d.get("median_tpot_ms")
if "ttft" in slo and tt and tt > X * float(slo["ttft"]): r.append(f"TTFT p95 {tt/1000:.1f} s > {X:g}x SLO")
if "tpot" in slo and tp and tp > X * float(slo["tpot"]): r.append(f"TPOT {tp:.0f} ms > {X:g}x SLO")
rt, g = d.get("request_throughput"), d.get("request_goodput")
if mode == "t4" and rt and g is not None and g / rt < 0.5: r.append(f"goodput {100*g/rt:.0f} %")
print("; ".join(r))
PY2
}

saturated() {
  (( DRYRUN )) && return 1
  if [ "${LAST_RC:-0}" != 0 ]; then echo "    [stop] the point failed: skipping the rest of this profile"; return 0; fi
  [ -z "${LAST_JSON:-}" ] && return 1
  local why; why=$(check_json "$LAST_JSON" "$LAST_P" "${1:-}")
  if [ -n "$why" ]; then echo "    [stop] saturated ($why): skipping the harder points of this profile"; return 0; fi
  return 1
}

# ---- Tests ------------------------------------------------------------------
# Profile names and metrics/ snapshot names (antes/despues = before/after) are part of the output schema.
t1() {
  snap T1_antes
  bench T1 velocidad 512 1024 0 1 inf "$( (( QUICK )) && echo 2 || echo 8 )"
  snap T1_despues
}

t2() {
  local L=(1024 4096 16384 32768 65536 131072)
  (( QUICK )) && L=(1024 16384)
  snap T2_antes
  for IN in "${L[@]}"; do bench T2 prefill "$IN" 16 0 1 inf "$( (( QUICK )) && echo 2 || echo 5 )"; done
  snap T2_despues
}

t3() {
  local CC
  if want chat; then
    read -ra CC <<< "${T3_CC:-1 4 16 32 64 100 150 200}"; (( QUICK )) && CC=(1 16 64)
    snap T3_chat_antes
    for C in "${CC[@]}"; do bench T3 chat 1024 512 0 "$C" inf "$(n_req "$C" 20)"; saturated && break; done
    snap T3_chat_despues
  fi
  if want rag; then
    CC=(1 16 64 150); (( QUICK )) && CC=(1 16 64)
    snap T3_rag_antes
    for C in "${CC[@]}"; do bench T3 rag "$RAG_IN" 256 0 "$C" inf "$(n_req "$C" 20)"; saturated && break; done
    snap T3_rag_despues
  fi
  if want agentes; then
    CC=(1 16 64 100); (( QUICK )) && CC=(1 16 64)
    snap T3_agentes_antes
    for C in "${CC[@]}"; do bench T3 agentes 8192 1024 8192 "$C" inf "$(n_req "$C" 16)"; saturated && break; done
    snap T3_agentes_despues
  fi
  if want razonamiento; then
    CC=(1 16 64 150); (( QUICK )) && CC=(1 16)
    snap T3_razonamiento_antes
    for C in "${CC[@]}"; do bench T3 razonamiento 512 4096 0 "$C" inf "$(n_req "$C" 6 3)"; saturated && break; done
    snap T3_razonamiento_despues
  fi
}

t4() {
  local RR DUR=$T4_DUR N
  (( QUICK )) && DUR=30
  if want chat; then
    RR=(0.5 1 2 4 8 12 16); (( QUICK )) && RR=(1 4)
    snap T4_chat_antes
    for R in "${RR[@]}"; do
      N=$(python3 -c "print(max(30,int($R*$DUR)))")
      bench T4 chat 1024 512 0 - "$R" "$N"; saturated t4 && break
    done
    snap T4_chat_despues
  fi
  if want agentes; then
    RR=(0.5 1 2 3 4); (( QUICK )) && RR=(0.5 2)
    snap T4_agentes_antes
    for R in "${RR[@]}"; do
      N=$(python3 -c "print(max(20,int($R*$DUR)))")
      bench T4 agentes 8192 1024 8192 - "$R" "$N"; saturated t4 && break
    done
    snap T4_agentes_despues
  fi
}

t5() {
  snap T5_antes
  for L in 131072 262144 524288 1048576; do
    bench T5 extremo $((L - 1024)) 256 0 1 inf 1
  done
  bench T5 extremo $((131072 - 1024)) 256 0 4 inf 4      # 4 users at 128k at once
  snap T5_despues
}

t9() {   # Department: chat + RAG + coding agents AT THE SAME TIME, for N people
  # Per-person assumptions: 60 % use chat (1 question/min), 25 % RAG (1 query/2 min with 8k of
  # documents), 15 % code with agents (half of them with an agent active at any time).
  local NN=(25 50 100 200 400); (( QUICK )) && NN=(25 100)
  local DUR=${T9_DUR:-120} N RCH RRG AG NC NR NA why f
  snap T9_antes
  for N in "${NN[@]}"; do
    RCH=$(python3 -c "print(round($N*0.60/60,3))"); RRG=$(python3 -c "print(round($N*0.25/120,3))")
    AG=$(python3 -c "print(max(1,round($N*0.15/2)))")
    NC=$(python3 -c "print(max(10,int($RCH*$DUR)))"); NR=$(python3 -c "print(max(5,int($RRG*$DUR)))")
    NA=$(( AG * 6 )); (( NA < 6 )) && NA=6
    echo ">>> $(date +%T)  T9 · department of $N people: chat $RCH req/s · RAG $RRG req/s · $AG active agents"
    local p1 p2 p3
    bench T9 "chat@$N" 1024 512 0 - "$RCH" "$NC" > "$OUT/logs/T9_${N}_chat.txt" 2>&1 & p1=$!
    bench T9 "rag@$N" "$RAG_IN" 256 0 - "$RRG" "$NR" > "$OUT/logs/T9_${N}_rag.txt" 2>&1 & p2=$!
    bench T9 "agentes@$N" 8192 1024 8192 "$AG" inf "$NA" > "$OUT/logs/T9_${N}_agentes.txt" 2>&1 & p3=$!
    wait "$p1" "$p2" "$p3"   # only these three (the background monitors never finish)
    grep -hE "Successful|P95 TTFT|Median TPOT|skip" "$OUT"/logs/T9_"${N}"_*.txt | sed 's/^/  /'
    (( DRYRUN )) && continue
    why=""
    for f in "$OUT"/json/T9_*@"${N}"_*.json; do
      local pf; pf=$(basename "$f" | cut -d_ -f2)
      local w; w=$(check_json "$f" "$pf"); [ -n "$w" ] && why+="${pf}: $w; "
    done
    if [ -n "$why" ]; then echo "    [stop] department of $N saturated (${why%; }): not trying larger sizes"; break; fi
  done
  snap T9_despues
}

t15() {  # Control with REAL TEXT: how much do the numbers change versus random tokens? (bias in MoE models)
  # a) SPEED-Bench throughput_8k (real text, 8k input) at the same levels as T12 (random 8k/1k)
  local CC=(1 16 64 128); (( QUICK )) && CC=(1 16)
  snap T15_antes
  DSMODE=speed
  if ! ls "$DATASETS_DIR"/speed-bench/*.jsonl >/dev/null 2>&1 && [ -s "$DATASETS_DIR/real8k.jsonl" ]; then
    DSMODE=real8k; SPEED_PATH=x       # no SPEED-Bench: ~8k tokens of real ShareGPT text
    echo "t15_fuente=sharegpt-8k" >> "$OUT/suite_info.txt"
  fi
  local first=1
  for C in "${CC[@]}"; do
    bench T15 "estandar@real" 8192 1024 0 "$C" inf "$(n_req "$C" 8 4)"
    if (( first )) && [ "${LAST_RC:-0}" != 0 ] && [ -z "${SPEED_PATH:-}" ]; then
      # depending on the version, vLLM wants the folder or the subset's .jsonl file
      SPEED_PATH=/datasets/speed-bench/${SPEED_SUBSET:-throughput_8k}.jsonl
      echo "   (retrying SPEED-Bench with $SPEED_PATH)"
      bench T15 "estandar@real" 8192 1024 0 "$C" inf "$(n_req "$C" 8 4)"
    fi
    first=0; saturated && break
  done
  # b) chat with real ShareGPT conversations (real lengths) at the T3 chat levels
  [ -n "${T15_REAL_ONLY:-}" ] && { DSMODE=; snap T15_despues; return; }   # repeat run: part b is already measured
  DSMODE=sharegpt
  CC=(16 64 150); (( QUICK )) && CC=(16)
  for C in "${CC[@]}"; do bench T15 "chat@sharegpt" 1024 512 0 "$C" inf "$(n_req "$C" 20)"; saturated && break; done
  DSMODE=
  snap T15_despues
}

t19() {  # Steady-state confirmation (MLPerf requires >= 600 s): 10 min in a row at the highest chat rate that
         # met the SLO in this battery's T4. Does it hold with a hot GPU and a full cache?
  (( DRYRUN )) && { echo ">>> T19 long confirmation (dry-run)"; return; }
  local R
  R=$(python3 - "$OUT/json" "${SLO[chat]}" <<'PY'
import glob, json, os, sys
slo = dict(x.split(":") for x in sys.argv[2].split())
best = None
for f in glob.glob(os.path.join(sys.argv[1], "T4_chat_*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    r = d.get("request_rate")
    if r in (None, "inf") or (d.get("failed") or 0) > 0: continue
    if (d.get("p95_ttft_ms") or 1e9) <= float(slo["ttft"]) and (d.get("p95_tpot_ms") or 1e9) <= float(slo["tpot"]):
        best = max(best or 0, float(r))
print(best if best else "")
PY
)
  if [ -z "$R" ]; then echo "   · T19: no T4 chat point in this battery meets the SLO, skipped"; return; fi
  echo ">>> $(date +%T)  T19 · 10 min of chat in a row at $R req/s (capacity measured in T4)"
  snap T19_antes
  bench T19 "chat@estacionario" 1024 512 0 - "$R" "$(python3 -c "print(int($R*${T19_DUR:-600}))")"
  snap T19_despues
}

t4f() {  # Refined capacity (chat and agents) from this model's previous T4:
         #  · if one rate meets the SLO and a higher one does not: 3 rates in between (e.g. 4 and 8 -> 5, 6 and 7 req/s);
         #  · if it met the SLO up to the last rate tried: keep going up (x1.5, x2, x3) until it fails.
  (( DRYRUN )) && { echo ">>> T4F refined capacity (dry-run)"; return; }
  local PLAN P R N RR IN PRE OL
  PLAN=$(python3 - "$(dirname "$OUT")" "${SLO[chat]}" "${SLO[agentes]}" <<'PYF'
import glob, json, sys
d = sys.argv[1]
for prof, slo_txt, pattern in (("chat", sys.argv[2], "T4_chat_in1024_pre0_out512_c-_r"),
                               ("agentes", sys.argv[3], "T4_agentes_in8192_pre8192_out1024_c-_r")):
    slo = dict(x.split(":") for x in slo_txt.split())
    pts = {}
    for f in glob.glob(d + "/*/json/" + pattern + "*.json") + glob.glob(d + "/*/json/T4F_" + pattern[3:] + "*.json"):
        try:
            r = float(f.rsplit("_r", 1)[1][:-5]); j = json.load(open(f))
        except Exception:
            continue
        c = j.get("completed") or 0
        ok = c > 0 and (j.get("failed") or 0) <= 0.01 * c and (j.get("p95_ttft_ms") or 1e9) <= float(slo.get("ttft", 1e9)) \
             and (j.get("p95_tpot_ms") or 1e9) <= float(slo.get("tpot", 1e9))
        pts[r] = pts.get(r, True) and ok
    lo = hi = None
    for r in sorted(pts):
        if pts[r] and hi is None: lo = r
        elif not pts[r] and lo is not None and hi is None: hi = r
    if lo and hi and hi - lo > 0.2:
        print(prof, " ".join(f"{lo + (hi - lo) * k / 4:g}" for k in (1, 2, 3)))
    elif lo and hi is None and lo == max(pts):
        print(prof, " ".join(f"{lo * k:g}" for k in (1.5, 2, 3)))
PYF
)
  if [ -z "$PLAN" ]; then echo ">>> T4F: no useful previous T4 for this model; skipped"; return; fi
  snap T4F_antes
  while read -r P RR; do
    [ -z "$P" ] && continue
    echo ">>> $(date +%T)  T4F · refined capacity for $P: rates $RR"
    if [ "$P" = chat ]; then IN=1024; PRE=0; OL=512; else IN=8192; PRE=8192; OL=1024; fi
    for R in $RR; do
      N=$(python3 -c "print(max(30,int($R*$T4_DUR)))")
      bench T4F "$P" "$IN" "$OL" "$PRE" - "$R" "$N"; saturated t4 && break
    done
  done <<< "$PLAN"
  snap T4F_despues
}

t14f() {  # Refined coding sessions from the previous T14: midpoint between the last level that passes (<= 8 s and
          # >= 30 tok/s) and the first that fails, or more sessions (x1.5 and x2) if it passed up to the last level.
  (( DRYRUN )) && { echo ">>> T14F (dry-run)"; return; }
  local LEVELS
  LEVELS=$(python3 - "$(dirname "$OUT")" <<'PYS'
import glob, json, sys
ok, no = set(), set()
for f in glob.glob(sys.argv[1] + "/*/sesiones_base.json") + glob.glob(sys.argv[1] + "/*/sesiones_fino.json"):
    try: d = json.load(open(f))
    except Exception: continue
    for n in d.get("niveles", []):
        (ok if (n.get("ttft_p95") or 99) <= 8 and (n.get("dec_tok_s_p50") or 0) >= 30 and not n.get("errores") else no).add(n["sesiones"])
lo = max(ok) if ok else None
hi = min([x for x in no if lo and x > lo], default=None)
if lo and hi and hi - lo >= 8: print((lo + hi) // 2)
elif lo and hi is None: print(f"{int(lo * 1.5)},{lo * 2}")
PYS
)
  if [ -z "$LEVELS" ]; then echo ">>> T14F: no useful previous T14 for this model, or the interval is already narrow; skipped"; return; fi
  T14_LABEL=fino T14_SESSIONS=$LEVELS t14
}

t21() {  # Function calling: Berkeley Function Calling Leaderboard v3 (simple, parallel and irrelevance AST categories)
  (( DRYRUN )) && { echo ">>> T21 BFCL (dry-run)"; return; }
  if [ ! -s "$DATASETS_DIR/bfcl/BFCL_v3_simple.json" ]; then echo "!! T21: BFCL data missing (prepare_data.sh); skipped"; return; fi
  echo ">>> $(date +%T)  T21 · function calling (BFCL v3)"
  snap T21_antes
  python3 "$HERE/bfcl.py" "$BASE" "$MODEL" "$OUT/bfcl.json" "$DATASETS_DIR/bfcl" \
      --conc "${T21_CONC:-64}" --limite-min "${T21_MIN:-20}" 2>&1 | tee "$OUT/logs/T21_bfcl.log"
  snap T21_despues
}

t11d() {  # DEGRADED invoices with vision (bad scans, stamps, distractors, discounts)
  (( DRYRUN )) && { echo ">>> T11D (dry-run)"; return; }
  [ -s "$HERE/data/invoices_degraded/verdad.json" ] || { echo "!! T11D: data/invoices_degraded/ missing; skipped"; return; }
  echo ">>> $(date +%T)  T11D · degraded invoices with vision"
  python3 "$HERE/invoices_vision.py" "$BASE" "$MODEL" "$OUT/vision_dificil.json" --docs "$HERE/data/invoices_degraded" \
      --conc "${T11D_CONC:-1,8,32}" 2>&1 | tee "$OUT/logs/T11D_vision_dificil.log"
}

t22() {  # Hard email triage (several topics, phishing, quoted threads, languages, irony)
  (( DRYRUN )) && { echo ">>> T22 (dry-run)"; return; }
  echo ">>> $(date +%T)  T22 · hard email triage"
  python3 "$HERE/email_hard.py" "$BASE" "$MODEL" "$OUT/clasif_dificil.json" --conc 32 2>&1 | tee "$OUT/logs/T22_clasif.log"
}

t20() {  # Hard quality with reasoning: AIME 2025 + AIME 2026 (2 attempts) and MATH-500
  (( DRYRUN )) && { echo ">>> T20 hard quality (dry-run)"; return; }
  if [ ! -s "$DATASETS_DIR/reasoning/math500.jsonl" ] && [ ! -s "$DATASETS_DIR/reasoning/aime2025-I.jsonl" ]; then
    echo "!! T20: data missing (run prepare_data.sh); skipped"; return
  fi
  echo ">>> $(date +%T)  T20 · hard quality: AIME 2025/2026 and MATH-500 with reasoning"
  snap T20_antes
  python3 "$HERE/reasoning.py" "$BASE" "$MODEL" "$OUT/dificil${T20_SUFFIX:-}.json" "$DATASETS_DIR/reasoning" \
      --conc "${T20_CONC:-128}" --limite-min "${T20_MIN:-50}" --max-tokens "${T20_MAXTOK:-32768}" ${T20_AIME_ONLY:+--solo-aime} \
      2>&1 | tee "$OUT/logs/T20_dificil${T20_SUFFIX:-}.log"
  snap T20_despues
}

t16() {  # Quality comparable with the community: full GSM8K test (1319), no extended reasoning, 95 % CI
  (( DRYRUN )) && { echo ">>> T16 GSM8K (dry-run)"; return; }
  echo ">>> $(date +%T)  T16 · GSM8K (full test set, 1319 problems)"
  python3 "$HERE/gsm8k.py" "$BASE" "$MODEL" "$OUT/gsm8k.json" "$DATASETS_DIR/gsm8k_test.jsonl" 2>&1 | tee "$OUT/logs/T16_gsm8k.log"
}

t17() {  # Repeatability: 3 repetitions of two key points -> measurement error margin
  snap T17_antes
  local k
  for k in 1 2 3; do
    bench T17 "estandar@rep$k" 8192 1024 0 64 inf 256
    bench T17 "chat@rep$k" 1024 512 0 64 inf 256
  done
  snap T17_despues
}

t18() {  # BURSTY traffic (BurstGPT: real traffic is more irregular than Poisson) -> chat with burstiness 0.5
  local RR=(2 4 6 8 12) N; (( QUICK )) && RR=(4)
  snap T18_antes
  local B0=$BURSTINESS; BURSTINESS=0.5
  for R in "${RR[@]}"; do
    N=$(python3 -c "print(max(30,int($R*$T4_DUR)))")
    bench T18 "chat@rafagas" 1024 512 0 - "$R" "$N"; saturated t4 && break
  done
  BURSTINESS=$B0
  snap T18_despues
}

t14() {  # Long coding sessions: context 48k -> ~100k, ~1k new tokens per turn, prefix cache
  (( DRYRUN )) && { echo ">>> T14 sessions (dry-run)"; return; }
  local LBL=${T14_LABEL:-base}
  echo ">>> $(date +%T)  T14 · long coding sessions ($LBL)"
  snap "T14_${LBL}_antes"
  local S; S=$(date +%s)
  python3 "$HERE/coding_sessions.py" "$BASE" "$MODEL" "$OUT/sesiones_${LBL}.json" --etiqueta "$LBL" \
      --sesiones "${T14_SESSIONS:-4,8,16,32,64}" --inicio "${T14_START:-48000}" 2>&1 | tee "$OUT/logs/T14_${LBL}.log"
  echo "T14_${LBL}_window,$S,$(date +%s)" >> "$OUT/suite_info.txt"
  snap "T14_${LBL}_despues"
}

t13() {  # Tool-using agents: 8 multi-step back-office tasks, checked automatically
  (( DRYRUN )) && { echo ">>> T13 agents (dry-run)"; return; }
  echo ">>> $(date +%T)  T13 · tool-using agents (8 multi-step tasks)"
  snap T13_antes
  local S; S=$(date +%s)
  python3 "$HERE/agents.py" "$BASE" "$MODEL" "$OUT/agentes.json" --conc "${T13_CONC:-1,8,32,64}" 2>&1 | tee "$OUT/logs/T13_agentes.log"
  echo "T13_window,$S,$(date +%s)" >> "$OUT/suite_info.txt"
  snap T13_despues
}

t12() {  # DGX Station community standard test: 8192 in / 1024 out, 1 -> 128 users
  # (same recipe as github.com/catid/dgx_station_benchmarks -> numbers comparable with the public ones)
  local CC=(1 8 16 32 64 128); (( QUICK )) && CC=(1 8)
  snap T12_antes
  for C in "${CC[@]}"; do bench T12 estandar 8192 1024 0 "$C" inf "$(n_req "$C" 8 4)"; saturated && break; done
  snap T12_despues
}

t11() {  # Documents with vision: scanned invoices -> JSON (pages/min and per-field accuracy)
  (( DRYRUN )) && { echo ">>> T11 vision (dry-run)"; return; }
  echo ">>> $(date +%T)  T11 · reading scanned invoices with vision"
  snap T11_antes
  local S; S=$(date +%s)
  python3 "$HERE/invoices_vision.py" "$BASE" "$MODEL" "$OUT/vision.json" --docs "$HERE/data/invoices_clean" \
      --conc "${T11_CONC:-1,4,16,32,64,128}" 2>&1 | tee "$OUT/logs/T11_vision.log"
  echo "T11_window,$S,$(date +%s)" >> "$OUT/suite_info.txt"
  snap T11_despues
}

t8() {   # KV offloading: many different long documents that repeat (more than fit in the GPU)
  local LBL=${T8_LABEL:-base}
  NPREF=${T8_PREFIXES:-160}; (( QUICK )) && NPREF=8
  snap "T8_${LBL}_antes"
  DSMODE=prefix_rep
  bench T8 "$LBL" 256 128 "${T8_PREFIX_LEN:-32768}" "${T8_CONC:-16}" inf $((NPREF * 3))
  DSMODE=
  snap "T8_${LBL}_despues"
}

t7() {   # prefix cache: same input size (16k) with and without an 8k common prompt
  local CC=(1 16 64); (( QUICK )) && CC=(1 16)
  snap T7_antes
  for C in "${CC[@]}"; do
    bench T7 sin_prefijo 16384 512 0 "$C" inf "$(n_req "$C" 12 3)"
    bench T7 con_prefijo 8192 512 8192 "$C" inf "$(n_req "$C" 12 3)"
  done
  snap T7_despues
}

t6() {
  snap T6_antes
  bench T6 velocidad 512 1024 0 1 inf "$( (( QUICK )) && echo 2 || echo 8 )"
  local CC=(1 16 64 150); (( QUICK )) && CC=(1 16)
  for C in "${CC[@]}"; do bench T6 chat 1024 512 0 "$C" inf "$(n_req "$C" 20)"; done
  snap T6_despues
}

# ---- Run ----------------------------------------------------------------------
echo "suite · $TAG · model=$MODEL · max_len=$MAXLEN"
echo "tests: ${TESTS[*]}   QUICK=$QUICK DRYRUN=$DRYRUN ${ONLY:+ONLY=$ONLY}"
echo "results: $OUT"

# warm-up (not measured)
warmup() {
  $DOCKER run --rm --network host "${TOKMNT[@]}" -v "$HF_CACHE:/root/.cache/huggingface" \
     --entrypoint vllm "$BENCH_IMG" bench serve --backend vllm --base-url "$BASE" --model "$MODEL" ${TOKMODE:+--tokenizer-mode "$TOKMODE"} \
     --tokenizer "$TOKARG" --trust-remote-code --dataset-name random --random-input-len 256 \
     --random-output-len 64 --num-prompts 8 --max-concurrency 4 > "$OUT/logs/warmup.log" 2>&1
}
if (( ! DRYRUN )); then
  echo ">>> warm-up..."
  if ! warmup; then
    if [ -n "${TOKMODE:-}" ]; then   # the client does not know that tokenizer-mode: retry with the plain tokenizer
      echo "   (the client rejects --tokenizer-mode $TOKMODE: retrying without it)"; TOKMODE=
      warmup || { echo "!! Warm-up failed. See $OUT/logs/warmup.log"; tail -n 15 "$OUT/logs/warmup.log"; exit 1; }
    else
      echo "!! Warm-up failed. See $OUT/logs/warmup.log"; tail -n 15 "$OUT/logs/warmup.log"; exit 1
    fi
  fi
fi

T0=$(date +%s)
for t in "${TESTS[@]}"; do
  alive || server_down     # it died during the previous test (CUR_T)
  CUR_T=$t
  echo; echo "## $t"
  case $t in
    T1) t1 ;; T2) t2 ;; T3) t3 ;; T4) t4 ;; T5) t5 ;; T6) t6 ;; T7) t7 ;; T8) t8 ;; T9) t9 ;; T11) t11 ;; T12) t12 ;; T13) t13 ;; T14) t14 ;; T15) t15 ;; T16) t16 ;; T17) t17 ;; T18) t18 ;; T19) t19 ;; T4F) t4f ;; T20) t20 ;; T14F) t14f ;; T21) t21 ;; T11D) t11d ;; T22) t22 ;;
    *) echo "Unknown test: $t" ;;
  esac
done

nvidia-smi > "$OUT/nvidia-smi_fin.txt" 2>/dev/null || true
echo
echo "DONE · $NRUNS points · $(( ($(date +%s) - T0) / 60 )) min · results in $OUT"
echo "Next: python3 $HERE/summarize.py $RESULTS_DIR"
