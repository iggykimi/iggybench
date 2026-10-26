# Tests

Each test answers one question. The server was `vllm/vllm-openai:v0.32.0` on the same machine as the client, except
Recursive Flint (SGLang) and Underdog Bark (Transformers). The client was `vllm bench serve` for serving tests and a
stdlib-only Python script for the task tests. The script names are those in [`bench/`](../bench); the raw result
files keep the Spanish names the scripts wrote at the time (see [data.md](data.md)).

## Serving (`bench/suite.sh`)

| ID | Question | Method | Main metric |
|---|---|---|---|
| T1 | What speed does a single user see? | 1 user, 512 input / 1,024 output tokens, 8 requests | output tok/s per user (1 / median TPOT), TTFT |
| T2 | How long does reading a document take? | 1 user, 1k to 128k input tokens, 16 output tokens | input tokens read per second (prompt length / median TTFT) |
| T3 | Where is the hardware limit? | Closed loop at a fixed concurrency of 1 to 200, four request shapes: chat 1,024/512, RAG 8,192/256, agents 8,192 + 8,192 shared prefix / 1,024, reasoning 512/4,096 | throughput at saturation |
| T4 | **How many users meet the latency target?** | Open loop, Poisson arrivals at increasing rates, 90 s per rate. Chat (1,024/512) from 0.5 to 16 req/s; agents (8k prompt + 8k shared prefix / 1,024) from 0.5 to 4 req/s | capacity under SLO |
| T4F | Where exactly is the limit? | Intermediate rates between the last passing and the first failing T4 rate; if a model passed every rate, ×1.5, ×2 and ×3 until it failed | capacity under SLO |
| T5 | Does it handle very long prompts? | 128k, 256k, 512k and 1M tokens where the model's context allows, and 4 users at 128k | TTFT, completion |
| T9 | And a department? | Chat 60 %, RAG 25 % and agents 15 % at the same time for 25, 50, 100, 200 and 400 people, 120 s per level | largest department with all three profiles in SLO |
| T12 | How does it compare with other machines? | The community standard for DGX Station: 8,192/1,024, closed loop at 1, 8, 16, 32, 64 and 128 users | output tok/s; throughput–interactivity curve |
| T15 | Do random tokens mislead? | Real text of about 8k tokens (ShareGPT conversations concatenated) at 1/16/64/128 users, and ShareGPT chat at 16/64/150 | difference from T12 and T3 |
| T17 | How much does one measurement vary? | Three repeats of chat and of 8k/1k at 64 users (gpt-oss-120b, DeepSeek V4 Flash) | mean, SD, coefficient of variation |
| T18 | What if traffic arrives in bursts? | As T4 chat, with gamma arrivals of burstiness 0.5 | latency against Poisson |
| T19 | Does capacity hold over time? | 10 minutes at the highest chat rate that passed in T4 | SLO, failures, power and clock in steady state |

Two safeguards apply to every serving test:

- Early stop. Within a sweep, the harder points are skipped once a point exceeds 3× the SLO, more than 5 % of
  requests fail, or (T4) goodput falls below 50 %.
- Dead-server guard. Before each point the harness checks that the server answers. If it does not, the battery
  stops with exit code 3, `run.sh` restarts the model with 5 points less GPU memory and half the prefill chunk, and
  resumes from that test. No point is measured against a dead server.

## Use cases (Python clients in `bench/`)

| ID | Question | Method | Main metric |
|---|---|---|---|
| T11 | How many invoices per minute, how accurately? | 40 synthetic scanned invoices (3 layouts, slight rotation and noise) to an 8-field JSON; 1 to 128 concurrent (`invoices_vision.py`) | pages/min, field accuracy, perfect invoices |
| T11D | And with bad scans? | 30 invoices from `make_degraded_invoices.py`: 2–4° rotation, 900 px width, heavy JPEG compression, stamps over the totals, handwritten notes, distractor numbers (order number, customer tax ID), multi-line items, mixed date and number formats. Same 8 fields and comparator | same |
| OCR | Specialised OCR or a general vision model? | PaddleOCR-VL-1.6 text extraction, then gpt-oss-120b to JSON, on the same invoices (`invoices_ocr.py`) | pages/min, field accuracy |
| T13 | Can it run back-office agents? | 8 multi-step tasks for a fictitious distributor with 8 tools (orders, stock, invoicing, e-mail), checked automatically; 1, 8, 32 and 64 agents (`agents.py`) | tasks solved, tasks/min |
| T14 | How many developers with a coding agent? | Sessions of 14 turns starting at 48k tokens of context, +~1k per turn (8k in 15 % of turns), 300 output tokens, 2–8 s pauses; 4 to 64 sessions, extended to 96 and 128 in T14F (`coding_sessions.py`) | sessions with P95 TTFT ≤ 8 s and median ≥ 30 tok/s; prefix-cache hit rate |
| T16 | Does quantisation hurt quality? | GSM8K test set (1,319), direct answer, temperature 0 (`gsm8k.py`) | accuracy with Wilson interval |
| T20 | Which model reasons best? | AIME 2025 (OpenCompass) and AIME 2026 (MathArena), 60 problems × 2 attempts, exact integer answer; MATH-500 with normalised answers. Reasoning on (gpt-oss: `reasoning_effort=high`), temperature 0.6, top-p 0.95, up to 32,768 tokens (truncated = wrong), 128 in parallel. Repeated with 65,536 tokens for four models (`reasoning.py`) | accuracy with Wilson intervals |
| T21 | Does it call the right function with the right arguments? | BFCL v3, 10 categories, about 2,800 questions, through the API's `tools` field and vLLM's tool parser; simplified replica of the official AST checker (`bfcl.py`) | accuracy per category |
| T22 | Can it route a real mailbox? | 240 hard e-mails: several topics with a priority rule, pro-forma invoices that are price enquiries, phishing that mentions invoices, auto-replies, quoted threads, irony, chat style, Catalan, English and Portuguese, habitual "URGENT". LLMs without reasoning; d1-3B and Laya (`email_hard.py`, `email_classify.py`) | category and urgency accuracy |
| Classification | Decision model or LLM? | 300 synthetic Spanish e-mails, 6 categories + urgency; d1-3B and Laya one by one and in batches against gpt-oss-120b (`email_classify.py`) | accuracy, latency, e-mails/s |
| RAG | How long do indexing and search take? | Qwen3-Embedding-8B on ~540-token chunks; queries at 1/16/64; Qwen3-Reranker-8B over 50 documents at 1/8/32; a sanity check with GSM8K question → solution pairs (`rag.py`) | chunks/s, ms per query, recall@k |
| Image | Can it generate images? | Qwen-Image in diffusers, 1024², batches 1/4/8, 50 and 25 steps (`image_gen.py`) | s/image, images/min |
| Platform | Do several services fit on one GPU? | gpt-oss-120b (42 % of GPU memory) + Qwen3.8-27B (33 %): chat on gpt-oss, agents on Qwen, invoices at 8 concurrent, for 15 min | SLO of each service with the others running |
| T23 | How well does it transcribe? | FLEURS test split, 60 clips in each of es, en, ca, fr, de, pt, it (420 clips). `/v1/audio/transcriptions`, temperature 0. WER with a common normalisation (lower case, no punctuation); speed as × real time with 1 clip and 32 concurrent (`asr.py`; Bark in batches of 1 and 16 with `asr_transformers.py`) | WER per language, real-time factor |
| T24 | Can it find one fact in a huge document? | Varied filler text (a new seed per request, so the prefix cache cannot help) of 128k, 512k and ~1M tokens with "the access code of room N is XXXX-XXXX" at 10, 50 and 90 % depth; correct if the code appears in the answer. DeepSeek V4 Flash only (`needle.py`) | found, latency |

T6, T7 and T8 (speculative decoding, prefix caching and KV offloading in batch 1) are kept in the harness and the
raw data but are not discussed in the report. T10, a quick quality check from batch 1 that T16 and T20–T22
replaced, is not included.

## Configuration experiments

| Variant | What it isolates |
|---|---|
| Qwen3.8-Flash-Next without MTP | value of multi-token prediction at low and high load |
| gpt-oss-120b with EAGLE-3 | value of a draft head on a small-active-parameter MoE |
| Qwen3.8-27B with DFlash2 (7 draft tokens) | per-user speed against capacity |
| DeepSeek V4 Flash with 256 instead of 128 sequences | whether the sequence limit was the bottleneck |
| Qwen3.8-Flash-Next with the n-gram table in GPU memory | cost of keeping the table in Grace RAM |
| gpt-oss-120b, one instance (84 % memory) against two copies (42 % each) | whether two copies serve more than one |
| gpt-oss-120b sharing the GPU with Qwen3.8-27B and an invoice workload | interference between services |
| Qwen3.8-27B with KV cache offloaded to Grace RAM (T14) | whether offloading helps long sessions |

## What is recorded for each point

The JSON from `vllm bench serve` with aggregates and per-request detail (TTFT, inter-token latencies, lengths);
`runs.csv` with the time window of each point; `gpu.csv` with module and GPU power, utilisation, memory,
temperature and SM clock every second; `vllm_metrics.csv` with queue, KV-cache use, prefix-cache hits and
speculative acceptance every 2 s; `/metrics` snapshots before and after each test; the server command and the full
container log. Batch 2 kept only the aggregates; see [data.md](data.md).
