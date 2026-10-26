# Schema glossary

The benchmark ran with Spanish file names, JSON keys and CSV columns, and the released raw data uses them. The
clients in this folder were translated to English, but everything they write to disk keeps the original names.
This file maps those names to English. Names that already read as English (`conc`, `rtfx`, `ttft_p50`, `in`, `out`,
`rate`, `api`, `pipeline`, `embedding`, `reranker`, `ok`, `error`, `id`, `ref`, `hyp`, `pred`) are not listed.

Common suffixes: `_pct` percent, `_s` seconds (or "per second" after a count: `correos_s` = emails/s), `_ms`
milliseconds, `_min` per minute, `_media` mean, `_p50`/`_p95` percentiles, `ic95_pct` Wilson 95 % confidence interval.

## Output files

| File | Meaning | Written by |
|---|---|---|
| `agentes.json` | tool-using agents (T13) | `agents.py` |
| `aguja.json` | needle in a haystack (T24) | `needle.py` |
| `asr_<model>.json`, `asr_bark.json` | transcription (T23) | `asr.py`, `asr_transformers.py` |
| `bfcl.json`, `bfcl_detalle.jsonl` | BFCL summary, per-question detail (T21) | `bfcl.py` |
| `clasif_dificil.json` | hard email classification (T22) | `email_hard.py` |
| `d1.json`, `correos.json` | email classification results, email corpus used | `email_classify.py` |
| `dificil.json`, `dificil_64k.json`, `dificil*_detalle.jsonl` | AIME / MATH-500 summary and per-request detail (T20; `_64k` = 64k-token run) | `reasoning.py` |
| `vision.json`, `vision_dificil.json` | invoices read by a VLM (T11 clean set, T11D degraded set); also the OCR + LLM pipeline | `invoices_vision.py`, `invoices_ocr.py` |
| `ocr.json` | OCR-only throughput | `invoices_ocr.py` |
| `embed.json` | embeddings and reranker | `rag.py` |
| `gsm8k.json` | GSM8K (T16) | `gsm8k.py` |
| `imagen.json`, `muestra_<i>.png` | image generation results, sample images | `image_gen.py` |
| `sesiones_<label>.json` | long coding-agent sessions (T14) | `coding_sessions.py` |
| `resumen_runs.csv`, `informe.txt` | one row per measured point; text report (the report text is English now) | `summarize.py` |
| `calidad.json` | quick quality check (T10, first round only). Read by `summarize.py`; its writer is not published | (obsolete) |

## Input files

| Name | Meaning | Read by |
|---|---|---|
| `verdad.json` | invoice ground truth (file name → fields) | `invoices_vision.py`, `invoices_ocr.py` |
| `factura_NN.jpg`, `dificil_NN.jpg` | clean / degraded invoice images | same |
| `verdad.tsv` | ASR references, one `<wav file>\t<text>` per line, in `<audio_dir>/<fleurs_lang>/` | `asr.py`, `asr_transformers.py` |
| `aime2025-I.jsonl`, `aime2025-II.jsonl`, `aime2026.json`, `math500.jsonl` | T20 datasets in `<data_dir>` | `reasoning.py` |
| `runs.csv` (`tag, runid, test, profile, in, out, prefix, conc, rate, n, start, end, rc, file`), `gpu.csv`, `vllm_metrics.csv`, `metrics/<block>_antes.txt` / `_despues.txt` (before / after) | per-run files written by suite.sh | `summarize.py` |

## Invoice fields (`verdad.json` and the JSON the model is asked for)

| Key | Meaning |
|---|---|
| `numero` | invoice number |
| `fecha` | invoice date (YYYY-MM-DD) |
| `proveedor` | supplier legal name |
| `nif_proveedor` | supplier tax ID |
| `cliente` | customer name (not scored) |
| `base_imponible` | tax base (net amount) |
| `iva_pct` | VAT rate % |
| `total` | invoice total |
| `num_lineas` | number of line items |

## JSON keys

| Key | Meaning | Written by |
|---|---|---|
| `modelo` | model name | all |
| `niveles` | one entry per concurrency / batch / session-count level | agents, asr, asr_transformers, invoices_*, coding_sessions |
| `segundos`, `minutos` | wall time of the level / run | several |
| `peticiones` | requests sent | invoices_*, rag |
| `fallos` | failed requests (count) | several |
| `errores` | error count, or the first error messages (list) | several |
| `detalle` | per-item detail | agents, invoices_vision |
| `ejemplos` | sample transcriptions | asr, asr_transformers |
| `tareas` | number of tasks (top level) / tasks run (level) | agents |
| `aciertos_pct`, `tareas_min` | success %, tasks per minute | agents |
| `seg_tarea_p50`, `seg_tarea_p95` | seconds per task | agents |
| `herramientas`, `herramientas_media` | tool calls (per episode, mean per task) | agents |
| `tokens_salida`, `tokens_salida_media` | output tokens (per item, mean) | agents, email_* |
| `por_tarea` | per task id: `"passed/total"` | agents |
| `tarea`, `respuesta` | task id, final answer (first 200 chars) | agents |
| `tokens_por_frase` | tokens per filler sentence (calibrated with /tokenize) | needle |
| `pruebas` | list of trials | needle, image_gen, calidad.json |
| `tokens_objetivo`, `tokens_reales` | target and actual prompt tokens | needle |
| `profundidad`, `acierta` | needle depth (0-1), found the code | needle |
| `aciertos` | `"found/total"` (needle), correct count (gsm8k, calidad.json) | needle, gsm8k |
| `audios`, `horas_audio`, `horas_audio_por_hora` | clips, hours of audio, audio hours per wall hour | asr, asr_transformers |
| `wer_pct`, `wer_medio_pct`, `wer_en_pct` | WER per FLEURS language folder, overall, English only | asr, asr_transformers |
| `idioma` | language folder of an example | asr, asr_transformers |
| `motor`, `lote` | inference engine, batch size | asr_transformers |
| `categorias`, `preguntas`, `respondidas`, `acierto_pct` | per category: questions, answered, accuracy | bfcl |
| `media_categorias_pct`, `media_no_live_pct`, `media_live_pct` | mean accuracy over categories (all, non-live, live) | bfcl |
| `cat`, `estado`, `llamadas`, `texto`, `tokens` | detail line: category, status, tool calls, text, output tokens | bfcl |
| `correos`, `reparto`, `urgentes` | number of emails, emails per category, urgent emails | email_hard, email_classify |
| `acierto_categoria_pct`, `acierto_urgencia_pct`, `acierto_ambos_pct` | accuracy on category, urgency, both | email_hard, email_classify |
| `peor_tipo` | three worst templates: `[accuracy %, template start]` | email_hard |
| `modo` | thinking mode used | email_hard |
| `correos_s`, `correos_s_1en1`, `correos_s_lote32`, `correos_s_lote128` | emails/s (parallel, one at a time, batches of 32 / 128) | email_* |
| `error_lote32`, `error_lote128` | batch error | email_classify |
| `fallos_formato` | answers that were not valid JSON | email_* |
| `carga_s`, `memoria_gib`, `memoria_pico_gib` | model load time, peak GPU memory | email_classify, image_gen |
| `concurrencia` | concurrency | email_classify |
| `fallos` (in `d1`/`laya`) | misclassified emails: `{id, esperado (expected), d1/laya (predicted)}` | email_classify |
| `texto`, `categoria`, `urgente`, `tipo` | email text, category, urgent flag, template index | `correos.json` |
| `muestras_aime`, `temperatura`, `razonamiento` | AIME samples per problem, temperature, reasoning mode | reasoning |
| `conjuntos`, `problemas`, `sin_responder` | per dataset, problems, unanswered requests | reasoning |
| `truncadas`, `con_razonamiento_pct` | truncated answers, % of answers with reasoning | reasoning |
| `tokens_mediana`, `tokens_p90`, `tokens_generados`, `tok_s_medio` | output tokens median / p90, total, mean tok/s | reasoning |
| `conjunto`, `rep`, `acierto`, `truncada`, `razona` | detail line: dataset, sample index, correct, truncated, reasoned | reasoning |
| `documentos`, `paginas_min` | distinct invoices, pages per minute | invoices_* |
| `in_tok_media`, `out_tok_media` | mean prompt / output tokens | invoices_* |
| `acierto_campos_pct`, `facturas_perfectas_pct`, `acierto_por_campo` | field accuracy, invoices with all 8 fields right, per field | invoices_* |
| `campos`, `fallan` | per-field result, failing fields | invoices_vision |
| `lat_ocr_p50_s`, `ejemplo_salida` | OCR part of the latency, sample OCR output | invoices_ocr |
| `indexacion`, `trozos_s`, `tok_por_trozo`, `min_1M_trozos` | indexing levels, chunks/s, tokens per chunk, minutes for 1M chunks | rag |
| `consultas`, `consultas_s` | query levels, queries/s | rag |
| `calidad_embeddings`, `calidad_reranker`, `acierto@1_de_50` | quality checks; reranker hit@1 among 50 | rag |
| `rerank`, `rerank_s`, `docs_s`, `rerank_error` | rerank levels, reranks/s, docs/s, error | rag |
| `preguntas`, `ic95_pct`, `acierto_200_pct`, `por_pregunta` | questions, 95 % CI, accuracy on first 200, 0/1 per question | gsm8k |
| `diffusers`, `resolucion` | diffusers origin and version, resolution | image_gen |
| `pasos`, `s_por_llamada`, `s_por_imagen`, `imagenes_min` | steps, seconds per call / image, images per minute | image_gen |
| `etiqueta`, `turnos_por_sesion` | run label, turns per session | coding_sessions |
| `sesiones`, `turnos`, `turnos_min` | concurrent sessions, turns completed, turns per minute | coding_sessions |
| `contexto_medio`, `contexto_max` | mean / max prompt tokens | coding_sessions |
| `ttft_primer_turno_p50`, `cache_pct`, `dec_tok_s_p50`, `dec_tok_s_p5` | first-turn TTFT, % prompt tokens from cache, decode tok/s per session | coding_sessions |
| `caso`, `tokens_salida` | use case, output tokens | calidad.json |

## Value strings written to output files

| Value | Meaning | Where |
|---|---|---|
| `pedido`, `incidencia`, `factura`, `devolucion`, `stock`, `otro` | order, delivery incident, invoice, return, stock/price enquiry, other | email categories |
| `incidencia_rotura`, `stock_pendientes`, `facturacion`, `devolucion`, `emails_retrasados`, `cliente_inexistente`, `comparar_almacenes`, `pedido_mayor` | breakage incident, pending-order stock, invoicing, return, emails for delayed orders, unknown customer, compare warehouses, largest order | `agentes.json` task ids |
| `demasiados pasos` | too many steps | `agentes.json` error |
| `sin tiempo` | out of time (global limit) | `estado` in `bfcl_detalle.jsonl`, `dificil_detalle.jsonl` |
| `sin razonamiento`, `por defecto` | thinking disabled, model default | `modo` in `clasif_dificil.json` |
| `activado (chat_template_kwargs)`, `modo por defecto del modelo (rechazó chat_template_kwargs)` | thinking enabled via chat_template_kwargs; model default (server rejected chat_template_kwargs) | `razonamiento` in `dificil.json` |
| `contenedor <version>`, `GitHub <version>` | diffusers from the container / from GitHub source | `imagen.json` |
| `d1-3B (3,1B)` | model label (3.1B parameters, Spanish decimal comma) | `d1.json` |
| `SI`, `NO`, `-` | yes, no, no SLO for this profile | `cumple_slo` in `resumen_runs.csv` (the text report prints yes/no) |

## `resumen_runs.csv` columns (`summarize.py`)

| Column | Meaning |
|---|---|
| `modelo`, `perfil`, `prefijo` | model tag, load profile, shared-prefix tokens |
| `fallos`, `duracion_s` | failed requests, duration |
| `usuario_tok_s_p50`, `usuario_tok_s_p95` | per-user tok/s (1000 / TPOT) |
| `gpu_w_media`, `modulo_w_media`, `gpu_temp_max`, `gpu_util_media` | mean GPU power, mean module power, max GPU temperature, mean GPU utilization |
| `out_tok_s_por_w` | output tok/s per watt |
| `cumple_slo` | meets the profile's SLO (`SI`/`NO`/`-`) |
| `cola_max` | max queued requests |
| `tokens_cacheados` | prompt tokens served from the prefix cache |
| `mtp_acept_pct` | MTP / speculative decoding acceptance % |
| `wh_por_mtok`, `eur_por_mtok` | Wh and EUR of electricity per million output tokens |
| `sm_mhz_min`, `sm_mhz_med` | min / median SM clock |
| `eur_mtok_total` | EUR per million tokens including amortization |

Profile values in `perfil` (from `runs.csv`): `velocidad` raw speed, `chat`, `rag`, `agentes` agents, `razonamiento`
reasoning, `estandar` standard 8k/1k, `sin_prefijo` / `con_prefijo` without / with shared prefix, `base` / `offload`,
`<profile>@<n>` for T9 department size or T15 dataset.

## Environment variables

| Variable | Meaning | Script |
|---|---|---|
| `GPU_TZ` | time zone of the machine that wrote `gpu.csv`, when summarizing elsewhere | `summarize.py` |
| `CAPEX_EUR`, `CAPEX_ANOS`, `PRECIO_KWH` | machine price (default 140000), amortization years (3), EUR/kWh (0.15) | `summarize.py` |
| `FONT_DIR` | directory with `DejaVuSans*.ttf` (default `/usr/share/fonts/truetype/dejavu/`) | `make_degraded_invoices.py` |

## CLI

Flags and positionals as `suite.sh` passes them. The Spanish flag names of the original runs are still accepted as
aliases.

| Script (original name) | Positional arguments | Flags (English, Spanish alias) |
|---|---|---|
| `agents.py` (`agentes.py`) | `base_url model out.json` | `--conc 1,8,32,64`, `--max-steps` (`--max-pasos`) 15 |
| `needle.py` (`aguja.py`) | `base_url model out.json` | `--lengths` (`--longitudes`) 128000,512000,1000000, `--depths` (`--prof`) 0.1,0.5,0.9 |
| `asr.py` | `base_url model out.json audio_dir` | `--conc 1,32` |
| `asr_transformers.py` (`bark_asr.py`) | `model_dir audio_dir out.json` | none |
| `bfcl.py` | `base_url model out.json bfcl_dir` | `--conc 64`, `--time-limit-min` (`--limite-min`) 20, `--max-irrelevance` (`--max-irrel`) 240, `--max-tokens 8192`, `--load-only` (`--solo-carga`) |
| `email_hard.py` (`clasif_dificil.py`) | `base_url model out.json` | `--conc 32` |
| `email_classify.py` (`d1.py`) | `d1_dir out_dir llm_url llm_model [hard]` (5th positional `dificil` also accepted) | none |
| `reasoning.py` (`dificil.py`) | `base_url model out.json data_dir` | `--conc 128`, `--time-limit-min` (`--limite-min`) 50, `--max-tokens 32768`, `--reps 2`, `--aime-only` (`--solo-aime`) |
| `invoices_vision.py` (`docvision.py`) | `base_url model out.json` | `--conc 1,4,16,32,64`, `--docs` (default `data/invoices_clean`; T11D passes `data/invoices_degraded`, was `docs_dificil`), `--duration 0` |
| `invoices_ocr.py` (`ocr.py`) | `ocr_url ocr_model llm_url llm_model results_dir` | `--conc 1,8,32,64`, `--conc-pipeline` (`--conc-tuberia`) 1,16,32, `--docs` (default `data/invoices_clean`) |
| `rag.py` (`embed.py`) | `emb_url emb_model rerank_url\|- rerank_model\|- out.json sharegpt.json gsm8k.jsonl` | none |
| `gsm8k.py` | `base_url model out.json gsm8k_test.jsonl` | `--n` (default: all) |
| `image_gen.py` (`imagen.py`) | `model_dir out_dir` | none |
| `coding_sessions.py` (`sesiones.py`) | `base_url model out.json` | `--sessions` (`--sesiones`) 4,8,16,32,64, `--turns` (`--turnos`) 14, `--start-tokens` (`--inicio`) 48000, `--output-tokens` (`--salida`) 300, `--max-sec` (`--max-seg`) 600, `--label` (`--etiqueta`) |
| `make_degraded_invoices.py` (`gen_facturas_dificiles.py`) | `out_dir [n=30]` | env `FONT_DIR` |
| `summarize.py` (`analizar.py`) | `[results_dir]` (default `./results` next to the script) | env vars above |

Scripts without argparse print their usage when called with too few arguments or `-h`/`--help`.
`email_classify.py` imports `email_hard.py` (was `clasif_dificil.py`) from its own folder; `image_gen.py` and
`email_classify.py` look for GitHub code under `/pylib` inside the container (same mount as before).
