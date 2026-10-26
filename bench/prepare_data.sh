#!/usr/bin/env bash
# prepare_data.sh - download the public datasets the tests use into $DATASETS_DIR. No GPU needed.
#   GSM8K (T16, RAG), ShareGPT and an ~8k-token real-text set (T15), SPEED-Bench throughput_8k (T15),
#   MATH-500 + AIME 2025/2026 (T20), BFCL v3 (T21), FLEURS test clips in 7 languages (ASR).
# Python steps that need huggingface_hub run inside the vLLM image, so the host only needs curl and python3.
# FLEURS and other gated datasets read HF_TOKEN from the environment.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
. "$HERE/common.sh"
D=$DATASETS_DIR; mkdir -p "$D"
run() { $DOCKER run --rm -v "$D:/data" -e HF_TOKEN "$@"; }

echo "== GSM8K (test, 1319 problems; OpenAI's file on GitHub)"
curl -LsSf https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl \
  -o "$D/gsm8k_test.jsonl" && wc -l "$D/gsm8k_test.jsonl"

echo "== ShareGPT (real chat conversations)"
run --entrypoint hf "$VLLM_IMG" download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir /data/sharegpt 2>&1 | tail -1

echo "== ~8k tokens of real text (concatenated ShareGPT conversations), T15 fallback when SPEED-Bench is missing"
python3 - "$D/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json" "$D/real8k.jsonl" <<'PY'
import json, sys
conv = json.load(open(sys.argv[1])); out = open(sys.argv[2], "w"); buf, n = [], 0
for c in conv:
    for t in c.get("conversations", []):
        buf.extend(t.get("value", "").split())
        if len(buf) >= 6000:                       # ~6000 words ~ 8k tokens
            # the instruction stays in Spanish: it is part of the prompts that were measured
            out.write(json.dumps({"prompt": " ".join(buf[:6000]) + "\n\nResume el texto anterior."}) + "\n")
            buf, n = [], n + 1
    if n >= 600: break
print("   real8k:", n, "texts")
PY

echo "== SPEED-Bench throughput_8k (real text with 8k input; NVIDIA)"
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/refs/heads/main/nemo_skills/dataset/speed-bench/prepare.py \
  -o "$D/prepare_speed.py"
run --entrypoint python3 "$VLLM_IMG" /data/prepare_speed.py --config throughput_8k --output_dir /data/speed-bench 2>&1 | tail -3

echo "== MATH-500 and AIME 2025 / 2026 (T20)"
R=$D/reasoning; mkdir -p "$R"; HF=https://huggingface.co/datasets
curl -fsSL "$HF/HuggingFaceH4/MATH-500/resolve/main/test.jsonl" -o "$R/math500.jsonl" || echo "!! MATH-500 not downloaded"
for p in I II; do
  curl -fsSL "$HF/opencompass/AIME2025/resolve/main/aime2025-$p.jsonl" -o "$R/aime2025-$p.jsonl" || echo "!! AIME 2025-$p not downloaded"
done
# AIME 2026 (MathArena) is only published as parquet: ask the Hugging Face rows API for JSON
curl -fsSL "https://datasets-server.huggingface.co/rows?dataset=MathArena%2Faime_2026&config=default&split=train&offset=0&length=100" \
     -o "$R/aime2026_api.json" \
 && python3 -c "import json,sys;d=json.load(open(sys.argv[1]));json.dump([r['row'] for r in d['rows']],open(sys.argv[2],'w'))" \
        "$R/aime2026_api.json" "$R/aime2026.json" \
 || echo "!! AIME 2026 not downloaded (T20 will run with AIME 2025 only)"
python3 - "$R" <<'PY'
import json, os, sys
d = sys.argv[1]
def n(f):
    p = os.path.join(d, f)
    if not os.path.exists(p): return "MISSING"
    t = open(p).read().strip()
    return len(json.loads(t)) if t.startswith("[") else sum(1 for l in t.splitlines() if l.strip())
for f, expected in (("math500.jsonl", 500), ("aime2025-I.jsonl", 15), ("aime2025-II.jsonl", 15), ("aime2026.json", 30)):
    print(f"   {f}: {n(f)} (expected {expected})")
PY

echo "== BFCL v3 (T21, ~10 MB)"
B=$D/bfcl; mkdir -p "$B/possible_answer"
BF=https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main
for c in simple multiple parallel parallel_multiple irrelevance live_simple live_multiple live_parallel live_parallel_multiple live_irrelevance; do
  curl -fsSL --max-time 120 "$BF/BFCL_v3_$c.json" -o "$B/BFCL_v3_$c.json" || echo "!! BFCL_v3_$c.json not downloaded"
  case $c in
    *irrelevance) ;;
    *) curl -fsSL --max-time 120 "$BF/possible_answer/BFCL_v3_$c.json" -o "$B/possible_answer/BFCL_v3_$c.json" \
         || echo "!! possible_answer/BFCL_v3_$c.json not downloaded" ;;
  esac
done
python3 "$HERE/bfcl.py" - - - "$B" --solo-carga

echo "== FLEURS (Google), test split, 60 sentences in each of 7 languages (ASR)"
mkdir -p "$D/asr"
run --entrypoint python3 "$VLLM_IMG" -c "
import tarfile, csv, os
from huggingface_hub import hf_hub_download as h
tok = os.environ.get('HF_TOKEN') or None; n_ok = 0
for lang in ['es_419', 'en_us', 'ca_es', 'fr_fr', 'de_de', 'pt_br', 'it_it']:
    try:
        tsv = h('google/fleurs', f'data/{lang}/test.tsv', repo_type='dataset', token=tok)
        rows = [r for r in csv.reader(open(tsv, encoding='utf-8'), delimiter='\t') if len(r) >= 4]
        want = {}
        for r in rows:
            if r[1] not in want and len(want) < 60: want[r[1]] = r[2]
        tgz = h('google/fleurs', f'data/{lang}/audio/test.tar.gz', repo_type='dataset', token=tok)
        os.makedirs(f'/data/asr/{lang}', exist_ok=True)
        with tarfile.open(tgz) as t:
            for m in t.getmembers():
                b = os.path.basename(m.name)
                if b in want:
                    with open(f'/data/asr/{lang}/{b}', 'wb') as f: f.write(t.extractfile(m).read())
        with open(f'/data/asr/{lang}/verdad.tsv', 'w', encoding='utf-8') as f:
            for k, v in want.items():
                if os.path.exists(f'/data/asr/{lang}/{k}'): f.write(f'{k}\t{v}\n')
        print('FLEURS', lang, len(os.listdir(f'/data/asr/{lang}')) - 1, 'clips'); n_ok += 1
    except Exception as e:
        print('!! FLEURS', lang, repr(e)[:200])
print('FLEURS languages OK:', n_ok)" 2>&1 | grep -E "FLEURS"

echo; echo "== Result:"
ls -la "$D" "$D/sharegpt" "$D/speed-bench" 2>&1 | grep -vE "^total|^d"
wc -l "$D/gsm8k_test.jsonl" "$D/speed-bench/"*.jsonl 2>/dev/null
