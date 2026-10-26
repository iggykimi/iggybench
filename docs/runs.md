# What was run

All runs used one GB300 GPU. Servers ran `vllm/vllm-openai:v0.32.0` unless the config says otherwise. The
orchestration at the time was a queue script tied to these specific batches; `bench/run.sh` and `bench/serve.sh`
reproduce one row of the table each. Folder names are the server's local time (UTC+8); the table gives UTC.

Settings column: environment variables passed to `run.sh` / `suite.sh` for that run (current names). `SERVE_EXTRA`
values are the ones written in the config header. Rows without tests ran the config's `client()`.

| Batch | Tag | Config | Tests | Settings | Start (UTC) |
|---|---|---|---|---|---|
| Batch 1 (Oct 20–21) | qwen27b-nvfp4-v028 | qwen27b-nvfp4-v028 | T1 T2 | | 2026-10-20 11:18 |
| Batch 1 (Oct 20–21) | qwen27b-nvfp4 | qwen27b-nvfp4 | T1 T2 T3 T4 T7 | | 2026-10-20 12:03 |
| Batch 1 (Oct 20–21) | qwen27b-bf16 | qwen27b-bf16 | T1 T2 T3 T4 T7 T9 T10 | | 2026-10-20 17:35 |
| Batch 1 (Oct 20–21) | glm53-sinmtp | glm53-sinmtp | T1 T2 T3 T4 T7 T9 T10 | | 2026-10-20 19:10 |
| Batch 1 (Oct 20–21) | glm53-mtp | glm53-mtp | T1 T2 T3 T4 T7 T9 T10 | | 2026-10-20 20:19 |
| Batch 1 (Oct 20–21) | glm53 | glm53 | T1 T2 T3 T4 T7 T9 T10 | | 2026-10-20 21:00 |
| Batch 1 (Oct 20–21) | glm53 | glm53 | T8 | `T8_LABEL=base T8_PREFIXES=120` | 2026-10-20 22:28 |
| Batch 1 (Oct 20–21) | glm53 | glm53 + KV offload | T8 | `T8_LABEL=offload T8_PREFIXES=120` | 2026-10-20 22:43 |
| Batch 1 (Oct 20–21) | glm53 | glm53 + MTP | T6 | | 2026-10-20 22:59 |
| Batch 1 (Oct 20–21) | qwen27b-nvfp4 | qwen27b-nvfp4 | T4 T9 T10 | | 2026-10-20 23:12 |
| Batch 1 (Oct 20–21) | qwen27b-nvfp4 | qwen27b-nvfp4 | T8 | `T8_LABEL=base T8_PREFIXES=220` | 2026-10-20 23:42 |
| Batch 1 (Oct 20–21) | qwen27b-nvfp4 | qwen27b-nvfp4 + KV offload | T8 | `T8_LABEL=offload T8_PREFIXES=220` | 2026-10-20 23:53 |
| Batch 1 (Oct 20–21) | qwen27b-nvfp4 | qwen27b-nvfp4 + MTP | T6 | | 2026-10-21 00:08 |
| Batch 1 (Oct 20–21) | qwen27b-nvfp4-mbt32k | qwen27b-nvfp4-mbt32k | T3 T4 | `ONLY=chat` | 2026-10-21 00:17 |
| Batch 1 (Oct 20–21) | gptoss120b | gptoss120b | T1 T2 T3 T9 T10 | | 2026-10-21 00:41 |
| Batch 2 (Oct 21–22) | dsv4-flash | dsv4-flash | T1 T2 T3 T4 T19 T9 T12 T13 T14 T15 T16 T5 | | 2026-10-21 11:57 |
| Batch 2 (Oct 21–22) | qwen38-flash-next | qwen38-flash-next | T1 T2 T3 T4 T19 T9 T11 T12 T13 T14 T15 T16 T5 | | 2026-10-21 14:08 |
| Batch 2 (Oct 21–22) | qwen38-flash-next-ngramgpu | qwen38-flash-next-ngramgpu | T1 T4 | `ONLY=chat` | 2026-10-21 16:01 |
| Batch 2 (Oct 21–22) | gptoss120b | gptoss120b | T4 T19 T12 T13 T14 T15 T16 T17 T18 | | 2026-10-21 16:19 |
| Batch 2 (Oct 21–22) | qwen27b-fp8 | qwen27b-fp8 | T1 T2 T4 T19 T9 T12 T13 T14 T16 | | 2026-10-21 17:42 |
| Batch 2 (Oct 21–22) | qwen27b-fp8 | qwen27b-fp8 + KV offload | T14 | `T14_LABEL=offload` | 2026-10-21 19:11 |
| Batch 2 (Oct 21–22) | qwen27b-fp8 | qwen27b-fp8-vision | T11 | | 2026-10-21 19:35 |
| Batch 2 (Oct 21–22) | paddleocr+gptoss | paddleocr-vl + gptoss120b-shared | client | | 2026-10-21 19:46 |
| Batch 2 (Oct 21–22) | d1-3b | d1-3b + gptoss120b-shared | client | | 2026-10-21 19:48 |
| Batch 2 (Oct 21–22) | qwen3-embedding-8b | qwen3-embedding-8b + qwen3-reranker-8b | client | | 2026-10-21 19:54 |
| Batch 2 (Oct 21–22) | qwen-image | qwen-image | client | | 2026-10-21 20:03 |
| Batch 2 (Oct 21–22) | plataforma-gptoss | plataforma-gptoss | T4 | `ONLY=chat` (with the two rows below, at once) | 2026-10-21 20:19 |
| Batch 2 (Oct 21–22) | plataforma-qwen | plataforma-qwen | T4 | `ONLY=agentes` | 2026-10-21 20:19 |
| Batch 2 (Oct 21–22) | plataforma-docs | plataforma-qwen | `invoices_vision.py --conc 8 --duration 900` | see `bench/configs/README.md` | 2026-10-21 20:19 |
| Batch 2 (Oct 21–22) | glm53-ssmbf16 | glm53-ssmbf16 | T4 T11 T12 T13 T14 T15 T16 | | 2026-10-21 20:47 |
| Batch 2 (Oct 21–22) | qwen27b-nvfp4-lpt | qwen27b-nvfp4-lpt | T4 | `ONLY=chat` | 2026-10-21 22:02 |
| Batch 2 (Oct 21–22) | dsv4-flash | dsv4-flash | T15 T4F T13 T21 T22 T14F T20 | `T15_REAL_ONLY=1` | 2026-10-21 22:28 |
| Batch 2 (Oct 21–22) | qwen38-flash-next | qwen38-flash-next | T15 T4F T13 T21 T22 T11D T14F T20 | `T15_REAL_ONLY=1` | 2026-10-21 23:30 |
| Batch 2 (Oct 21–22) | gptoss120b | gptoss120b | T1 T2 T4F T13 T21 T22 T14F T20 | | 2026-10-22 00:24 |
| Batch 2 (Oct 21–22) | gptoss-x1 | gptoss-x1 | T3 | `ONLY=chat T3_CC="16 32 64 128 256"` | 2026-10-22 01:03 |
| Batch 2 (Oct 21–22) | gptoss-x2a, gptoss-x2b | gptoss-x2a + gptoss-x2b | T3, 5 runs per copy | `ONLY=chat T3_CC=8`, 16, 32, 64, 128 | 2026-10-22 01:22–01:26 |
| Batch 2 (Oct 21–22) | paddleocr+gptoss-dificil | paddleocr-vl + gptoss120b-shared | client | `DEGRADED=1` | 2026-10-22 01:37 |
| Batch 2 (Oct 21–22) | d1-3b-dificil | d1-3b + gptoss120b-shared | client | `HARD=1` | 2026-10-22 01:41 |
| Batch 2 (Oct 21–22) | qwen27b-fp8 | qwen27b-fp8 | T4F T13 T21 T22 T14F T20 | | 2026-10-22 01:45 |
| Batch 2 (Oct 21–22) | qwen27b-fp8 | qwen27b-fp8-vision | T11D | | 2026-10-22 02:27 |
| Batch 2 (Oct 21–22) | glm53-ssmbf16 | glm53-ssmbf16 | T1 T2 T4F T13 T21 T22 T11D T14F T20 | | 2026-10-22 02:39 |
| Batch 2 (Oct 21–22) | gptoss120b-eagle3 | gptoss120b-eagle3 | T1 | | 2026-10-22 03:39 |
| Batch 2 (Oct 21–22) | gptoss120b-eagle3 | gptoss120b-eagle3 | T4 | `ONLY=chat` | 2026-10-22 03:40 |
| Batch 2 (Oct 21–22) | gptoss120b-eagle3 | gptoss120b-eagle3 | T14 | `T14_SESSIONS=4,16,64` | 2026-10-22 03:53 |
| Batch 2 (Oct 21–22) | qwen38-flash-next-nomtp | qwen38-flash-next-nomtp | T1 | | 2026-10-22 04:12 |
| Batch 2 (Oct 21–22) | qwen38-flash-next-nomtp | qwen38-flash-next-nomtp | T4 T4F | `ONLY=chat` | 2026-10-22 04:13 |
| Batch 2 (Oct 21–22) | dsv4-flash-s256 | dsv4-flash-s256 | T4 T4F | `ONLY=chat` | 2026-10-22 04:43 |
| Queue 4 (Oct 22, morning) | reposo | idle | client | | 2026-10-22 06:22 |
| Queue 4 (Oct 22, morning) | dsv4-flash | dsv4-flash | T14F T17 | | 2026-10-22 06:41 |
| Queue 4 (Oct 22, morning) | qwen38-flash-next | qwen38-flash-next | T11D T14F | | 2026-10-22 07:02 |
| Queue 4 (Oct 22, morning) | gptoss120b | gptoss120b | T4F T14F | | 2026-10-22 07:22 |
| Queue 4 (Oct 22, morning) | paddleocr+gptoss-dificil | paddleocr-vl + gptoss120b-shared | client | `DEGRADED=1` | 2026-10-22 07:53 |
| Queue 4 (Oct 22, morning) | qwen27b-fp8 | qwen27b-fp8-vision | T11D | | 2026-10-22 07:59 |
| Queue 4 (Oct 22, morning) | qwen27b-fp8 | qwen27b-fp8 | T20 | `T20_MAXTOK=65536 T20_SUFFIX=_64k T20_AIME_ONLY=1` | 2026-10-22 08:02 |
| Queue 4 (Oct 22, morning) | glm53-ssmbf16 | glm53-ssmbf16 | T11D T20 | T20 as the row above | 2026-10-22 08:32 |
| Queues 5–9 (Oct 22, afternoon) | qwen27b-fp8-dflash | qwen27b-fp8-dflash | T1 | | 2026-10-22 12:23 |
| Queues 5–9 (Oct 22, afternoon) | qwen27b-fp8-dflash | qwen27b-fp8-dflash | T4 | `ONLY=chat` | 2026-10-22 12:24 |
| Queues 5–9 (Oct 22, afternoon) | qwen27b-fp8-dflash | qwen27b-fp8-dflash | T14 | `T14_SESSIONS=4,16,48` | 2026-10-22 12:37 |
| Queues 5–9 (Oct 22, afternoon) | qwen27b-fp8-dflash | qwen27b-fp8-dflash | T20 | `T20_MAXTOK=65536 T20_SUFFIX=_64k T20_AIME_ONLY=1` | 2026-10-22 12:47 |
| Queues 5–9 (Oct 22, afternoon) | asr | asr-* (first attempt, plain vLLM image) | client | only `gpu.csv` kept | 2026-10-22 12:56 |
| Queues 5–9 (Oct 22, afternoon) | dsv4-flash | dsv4-flash | T1 | | 2026-10-22 13:18 |
| Queues 5–9 (Oct 22, afternoon) | dsv4-flash-aguja | dsv4-flash-aguja | client | | 2026-10-22 13:20 |
| Queues 5–9 (Oct 22, afternoon) | dsv4-flash | dsv4-flash | T20 | `T20_MAXTOK=65536 T20_SUFFIX=_64k T20_AIME_ONLY=1` | 2026-10-22 13:24 |
| Queues 5–9 (Oct 22, afternoon) | asr | asr-whisper-large-v3, asr-granite-4.0-1b-speech, asr-qwen3-asr, asr-voxtral-mini | client, one model after another | | 2026-10-22 13:41 |
| Queues 5–9 (Oct 22, afternoon) | nemotron35-super-vl | nemotron35-super-vl | T1 T21 T22 T11D T13 T12 | | 2026-10-22 14:01 |
| Queues 5–9 (Oct 22, afternoon) | nemotron35-super-vl | nemotron35-super-vl | T4 | `ONLY=chat` | 2026-10-22 14:20 |
| Queues 5–9 (Oct 22, afternoon) | asr (folder `bark-<time>`) | asr-bark | client | | 2026-10-22 14:31 |
| Queues 5–9 (Oct 22, afternoon) | flint-sglang | flint-sglang | T1 T22 T13 | | 2026-10-22 14:36 |
| Queues 5–9 (Oct 22, afternoon) | flint-sglang | flint-sglang | T4 | `ONLY=chat` | 2026-10-22 14:41 |
| Queues 5–9 (Oct 22, afternoon) | flint-sglang | flint-sglang | T12 | | 2026-10-22 14:54 |
| Queues 5–9 (Oct 22, afternoon) | flint-sglang | flint-sglang | T20 | `T20_AIME_ONLY=1` | 2026-10-22 15:04 |

## Exclusions, retries and quirks

- DeepSeek V4 Flash: the first run (92 % GPU memory, 16384-token prefill chunks) crashed in T2 at 128k input; its data was discarded and every `dsv4-flash` run uses 88 % and 8192-token chunks.
- Qwen3.8-Flash-Next was set to 88 % and 8192-token chunks as well, as a precaution after the DeepSeek crash.
- `qwen27b-nvfp4` (Oct 20, 12:03): its T4 was contaminated and is relabelled `T4-descartado` in `runs.csv`; T4 was repeated at 23:12.
- `glm53-sinmtp` and `glm53-mtp` were renamed after the run; their `suite_info.txt` still says `tag=glm53`.
- GLM-5.3-Flash: `--prefix-cache-retention-interval` was rejected at startup, so the main config runs without it; the MTP variant filled 94 % of the KV cache with 16 agents and was not used as the main config.
- `qwen27b-nvfp4-v028` ran on a pre-release vLLM build; the image tag in its config is the one recorded in `suite_info.txt`.
- `dsv4-flash` (Oct 21, 11:57): `suite_info.txt` records the preview checkpoint as tokenizer path; the server mounted the 0731 checkpoint.
- T15 used the ShareGPT-based ~8k-token text (`t15_fuente=sharegpt-8k`) for DeepSeek, Qwen3.8-Flash-Next, gpt-oss and GLM because the SPEED-Bench files were not on the host.
- Recursive Flint: vLLM 0.32.0 and `vllm/vllm-openai:latest` both refused the architecture (4 attempts); it was served with SGLang instead.
- Granite Speech 5.0 2B failed to load in both images (unknown `bias` parameter) and was replaced by Granite 4.0 1B Speech; Granite Speech 4.1 2B was skipped once Granite 4.0 1B worked.
- The first ASR attempt (12:56) used the plain vLLM image; Voxtral failed to start and only `gpu.csv` remains. The second attempt (13:41) used the audio image for all four models.
- Nemotron 3.5 Super VL: T20 was started at the end of its queue but produced no data.
- gpt-oss-120b with the DFlash draft was queued but never measured; DeepSeek V4.1 Flash and MiniMax M3 were dropped before any measurement.
- The queue script also retried a failed startup with alternative flags and restarted a crashed server with more memory headroom. The configs record the flags that actually ran. The queue logs from Batch 2 on show no crash rescue; there is no queue log for Batch 1.

## Frozen Spanish labels

The released raw data and the analysis code read these names, so the port keeps them in Spanish.

| Where | Label | Meaning |
|---|---|---|
| T3/T4/T9 profiles | `chat`, `rag`, `agentes`, `razonamiento` | chat, RAG, agents, reasoning |
| other profiles | `velocidad`, `prefill`, `extremo`, `estandar`, `sin_prefijo`, `con_prefijo` | speed, prefill, extreme context, standard 8k/1k, without / with common prefix |
| profile suffixes | `@real`, `@sharegpt`, `@estacionario`, `@rep1..3`, `@rafagas`, `@<people>` | real text, ShareGPT, steady state, repetition, bursty, department size |
| T8 / T14 labels | `base`, `offload`, `fino` | baseline, KV offload, refined (T14F) |
| `runs.csv` | `T4-descartado` | discarded T4 (see above) |
| `suite_info.txt` keys | `slo_agentes`, `slo_razonamiento`, `detalle_por_peticion`, `fecha`, `t15_fuente` | SLO for agents / reasoning, per-request detail saved, date, T15 text source |
| `metrics/` snapshots | `<test>_antes.txt`, `<test>_despues.txt` | before / after |
| run files | `caido.txt`, `nvidia-smi_inicio.txt`, `nvidia-smi_fin.txt` | server down at test, nvidia-smi at start / end |
| client outputs | `calidad.json`, `agentes.json`, `sesiones_<label>.json`, `dificil[_64k].json`, `clasif_dificil.json`, `vision_dificil.json`, `aguja.json`, `correos.json`, `imagen.json` | quality, agents, sessions, hard reasoning, hard e-mail, degraded invoices, needle, e-mails, images |
| log names | `T10_calidad.log`, `T13_agentes.log`, `T20_dificil*.log`, `T22_clasif.log`, `T11D_vision_dificil.log` | |
| result folders | `reposo`, `plataforma-*`, `*-dificil`, `dsv4-flash-aguja`, `glm53-sinmtp` | idle, platform, hard/degraded set, needle, GLM without MTP |
| data files | `verdad.json`, `verdad.tsv` | ground truth (invoices, FLEURS transcripts) |
| T15 real-text prompt | `Resume el texto anterior.` | "Summarize the text above." (part of the measured prompts) |
