# Configs

One file per model or variant that produced results. Each is a bash file sourced by `serve.sh`, `run.sh` and
`suite.sh`:

| Variable | Meaning |
|---|---|
| `TAG` | results folder name (`results/<TAG>/<run>/`); several configs can share a tag |
| `PORT` | server port; configs meant to run side by side use 8000 and 8001 |
| `MODEL` | served model name |
| `HF_REPO`, `MODEL_DIR` | Hugging Face repo and local folder (`download_models.sh`); empty `HF_REPO` = not recorded |
| `ENGINE`, `IMG` | `vllm` or `sglang` server image; `transformers` = no server, `client()` runs the model in a container |
| `SERVE_ARGS` | `vllm serve` arguments as recorded in the run's `server_cmd.txt`, minus host, port and model path |
| `DRAFT_REPO`, `DRAFT_DIR` | speculative-decoding draft, mounted at `/draft` |
| `client()` | workload run by `run.sh <config>` when no tests are given (results folder in `$1`) |

Variants that were measured under the same tag by adding flags (KV offload, MTP) are listed in the config's
header as `SERVE_EXTRA=...`.

| Config | Model | Engine | Notes |
|---|---|---|---|
| `gptoss120b` | gpt-oss-120b (MXFP4) | vLLM | main config, 85 % memory |
| `gptoss120b-eagle3` | gpt-oss-120b + EAGLE-3 draft | vLLM | |
| `gptoss-x1`, `gptoss-x2a`, `gptoss-x2b` | gpt-oss-120b | vLLM | one copy at 84 % vs two copies at 42 % (ports 8000/8001) |
| `gptoss120b-shared` | gpt-oss-120b | vLLM | 55 %, LLM for `paddleocr-vl` and `d1-3b` |
| `plataforma-gptoss`, `plataforma-qwen` | gpt-oss-120b + Qwen3.8-27B FP8 | vLLM | two servers on one GPU (42 % + 33 %) |
| `qwen27b-fp8` | Qwen3.8-27B FP8 | vLLM | text only; T14 offload variant in the header |
| `qwen27b-fp8-vision` | Qwen3.8-27B FP8 | vLLM | with the vision encoder, tag `qwen27b-fp8` (T11, T11D) |
| `qwen27b-fp8-dflash` | Qwen3.8-27B FP8 + DFlash2 draft | vLLM | |
| `qwen27b-nvfp4` | Qwen3.8-27B NVFP4 | vLLM | T8 offload and T6 MTP variants in the header |
| `qwen27b-nvfp4-mbt32k`, `qwen27b-nvfp4-lpt` | Qwen3.8-27B NVFP4 | vLLM | prefill tuning |
| `qwen27b-nvfp4-v028` | Qwen3.8-27B NVFP4 | vLLM (pre-release build) | first smoke run only |
| `qwen27b-bf16` | Qwen3.8-27B BF16 | vLLM | |
| `glm53`, `glm53-ssmbf16`, `glm53-mtp`, `glm53-sinmtp` | GLM-5.3-Flash NVFP4 | vLLM | `glm53-ssmbf16` is the main one |
| `dsv4-flash`, `dsv4-flash-s256` | DeepSeek V4 Flash 0731 FP8 | vLLM | 88 % memory, 8192-token prefill chunks |
| `dsv4-flash-aguja` | DeepSeek V4 Flash 0731 | vLLM | needle test (`needle.py`), same server as `dsv4-flash` |
| `qwen38-flash-next`, `-ngramgpu`, `-nomtp` | Qwen3.8-Flash-Next FP8 | vLLM | n-gram table in CPU RAM / GPU; without MTP |
| `nemotron35-super-vl` | Nemotron 3.5 Super VL NVFP4 | vLLM | MTP 3 |
| `flint-sglang` | Recursive Flint BF16 | SGLang | image built from the model repo's `sglang/` kit |
| `paddleocr-vl` | PaddleOCR-VL-1.6 | vLLM | OCR -> gpt-oss pipeline; `DEGRADED=1` for the degraded invoices |
| `d1-3b` | Liquid d1-3B (+ Laya) | Transformers | e-mail triage vs gpt-oss; `HARD=1` for the hard set |
| `qwen3-embedding-8b`, `qwen3-reranker-8b` | Qwen3 Embedding / Reranker 8B | vLLM (pooling) | RAG; reranker on port 8001 |
| `qwen-image` | Qwen-Image | diffusers | image generation |
| `asr-whisper-large-v3`, `asr-granite-4.0-1b-speech`, `asr-qwen3-asr`, `asr-voxtral-mini` | speech-to-text | vLLM + audio libs | image built from `../docker/vllm-audio` |
| `asr-bark` | Underdog Bark 3.0 Pro | Transformers | English ASR model vLLM does not serve |
| `idle` | none | none | 5 minutes of idle GPU power (tag `reposo`) |

## Runs with more than one server

`run.sh` handles one config. The multi-server runs were started by hand like this:

```bash
# Department platform: chat on gpt-oss, agents and invoices on Qwen, at the same time
./serve.sh configs/plataforma-gptoss.env && ./serve.sh configs/plataforma-qwen.env
ONLY=chat ./suite.sh configs/plataforma-gptoss.env T4 &
ONLY=agentes ./suite.sh configs/plataforma-qwen.env T4 &
sleep 60; d=results/plataforma-docs/$(date +%Y%m%d-%H%M%S); mkdir -p "$d"
timeout 1500 python3 invoices_vision.py http://127.0.0.1:8001 qwen38-27b-fp8 "$d/vision.json" --conc 8 --duration 900
wait

# One gpt-oss at 84 % vs two copies at 42 % with half the users each (16..256 users in total)
T3_CC="16 32 64 128 256" ONLY=chat ./run.sh configs/gptoss-x1.env T3
./serve.sh configs/gptoss-x2a.env && ./serve.sh configs/gptoss-x2b.env
for c in 16 32 64 128 256; do
  T3_CC=$((c / 2)) ONLY=chat ./suite.sh configs/gptoss-x2a.env T3 &
  T3_CC=$((c / 2)) ONLY=chat ./suite.sh configs/gptoss-x2b.env T3 &
  wait
done

# OCR pipeline and e-mail triage: both need gpt-oss on port 8000
./serve.sh configs/gptoss120b-shared.env
./run.sh configs/paddleocr-vl.env; DEGRADED=1 ./run.sh configs/paddleocr-vl.env
./run.sh configs/d1-3b.env; HARD=1 ./run.sh configs/d1-3b.env
./serve.sh configs/gptoss120b-shared.env stop

# RAG: reranker first, then the embedding model with its client
./serve.sh configs/qwen3-reranker-8b.env && ./run.sh configs/qwen3-embedding-8b.env
./serve.sh configs/qwen3-reranker-8b.env stop
```
