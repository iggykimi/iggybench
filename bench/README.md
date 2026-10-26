# Benchmark harness

What ran on the server. It needs Linux, an NVIDIA GPU, Docker and the `vllm/vllm-openai:v0.32.0` image; the
clients themselves need only Python 3.10+ and its standard library.

| Script | Role |
|---|---|
| `serve.sh <config>` | starts the model server in Docker from a config in `configs/`, waits for `/health` and records the exact command and image digest |
| `suite.sh <config> [tests]` | runs serving tests T1–T24 against a running server and writes `results/<tag>/<run>/` |
| `run.sh <config> [tests]` | serve → suite → stop; restarts the server once with more memory headroom if it dies mid-test, and resumes from that test |
| `prepare_data.sh`, `download_models.sh`, `selfcheck.sh` | datasets, model weights, and checks that need no GPU |
| `common.sh` | shared paths (`IGGY_HOME`, `MODELS_DIR`, `RESULTS_DIR`, ...) and defaults |
| `configs/` | one file per model or variant, with the server arguments used ([configs/README.md](configs/README.md)) |
| `docker/vllm-audio/` | the vLLM image plus the audio libraries the speech models need |

## Test clients

The tests that do not go through `vllm bench serve` use these clients. They talk to an OpenAI-compatible server
(vLLM, or SGLang for one model) and use only the standard library, except the three marked "in container", which
run inside the vLLM image because they need torch.

Output files keep the names and JSON keys of the original runs (Spanish); see `SCHEMA_GLOSSARY.md` for their
meaning and for every script's command line. Prompts sent to the models are unchanged from the runs, so they stay
in Spanish where the test content is Spanish.

| File | Test | What it measures | Inputs | Output |
|---|---|---|---|---|
| `agents.py` | T13 | Tool-using agents: 8 multi-step back-office tasks over 8 tools, checked against the final state; success rate and tasks/min at 1-64 concurrent agents | server | `agentes.json` |
| `coding_sessions.py` | T14 | Long coding-agent sessions growing from 48k to ~100k tokens: TTFT on later turns, prefix-cache hit rate, decode speed per session | server | `sesiones_<label>.json` |
| `invoices_vision.py` | T11, T11D | Scanned invoice → 8-field JSON with a VLM: pages/min, latency, per-field accuracy | server, `data/invoices_clean/` or `data/invoices_degraded/` | `vision.json`, `vision_dificil.json` |
| `gsm8k.py` | T16 | GSM8K accuracy (full test set, also first 200), Wilson 95 % CI | server, `gsm8k_test.jsonl` | `gsm8k.json` |
| `reasoning.py` | T20 | AIME 2025/2026 (2 samples) and MATH-500 with reasoning on, under a global time limit | server, dataset folder | `dificil.json` (+ `_detalle.jsonl`) |
| `bfcl.py` | T21 | Function calling on BFCL v3 single-turn categories, simplified AST checker | server, BFCL data folder | `bfcl.json` (+ `_detalle.jsonl`) |
| `email_hard.py` | T22 | Classification of 240 hard customer emails (category + urgency) | server | `clasif_dificil.json` |
| `asr.py` | T23 | Transcription of FLEURS clips in 7 languages: WER per language, × real time, latency | server, audio folder | `asr_<model>.json` |
| `asr_transformers.py` | T23 | Same as `asr.py` for a model vLLM cannot serve (Parakeet TDT via Transformers); in container | model folder, audio folder | `asr_bark.json` |
| `needle.py` | T24 | Needle in a haystack at 128k, 512k and ~1M tokens, 3 depths: found or not, time | server | `aguja.json` |
| `email_classify.py` | - | 300 emails (or the 240 hard ones) classified by decision models d1-3B and Laya, against gpt-oss; in container | d1 model folder, LLM server | `d1.json`, `correos.json` |
| `invoices_ocr.py` | - | OCR model alone (pages/min) and OCR → text LLM pipeline scored like T11 | OCR server, LLM server, invoice folder | `ocr.json`, `vision.json` |
| `rag.py` | - | Embedding indexing and query throughput, reranker latency, recall sanity check | embedding (+ reranker) server, ShareGPT, GSM8K | `embed.json` |
| `image_gen.py` | - | Image generation with diffusers (Qwen-Image): s/image and images/min by batch and steps; in container | model folder | `imagen.json`, `muestra_<i>.png` |
| `make_degraded_invoices.py` | - | Generates the 30 degraded invoices and their ground truth (seeded, needs Pillow) | none | `dificil_NN.jpg`, `verdad.json` |
| `summarize.py` | - | Text report and one CSV row per measured point over a results folder | results folder | `resumen_runs.csv`, `informe.txt` |
| `minipandas/` | T15 | Minimal pandas stand-in mounted into the vLLM container so `vllm bench serve --dataset-name custom` can read a `.jsonl` | - | - |
