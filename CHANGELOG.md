# Changelog

## v1.0 (2026-10-26)

First public release: harness, analysis code, result tables, figures, report and raw data.

### Corrections relative to the internal report

The first write-up of these results was an internal report in Spanish, finished on 22 October 2026. For this
release every number was recomputed from the raw files with the code in `analysis/`, including the part of the
quality analysis that had been done without code. These statements changed:

| Internal report | This release | Why |
|---|---|---|
| The five main models are within 5 points on business tasks | Within 4 points on GSM8K, hard e-mails and clean invoices; the agent test spans 12.5 points (one task in eight) | The earlier claim did not hold for every task it covered |
| Qwen3.8-27B is the most accurate model (AIME 91.7 %) | That score is at 64k tokens against 32k for most others; the difference with the 32k leaders is not significant | Mixed budgets |
| Nemotron 3.5 and DeepSeek V4 Flash are best at e-mail routing | GLM-5.3-Flash also scored 100 %, at a ninth of the speed | Omission |
| Degraded invoices: vision models 90–93 % of fields | 88.8–93.8 %, one request at a time | The earlier figures came from 32 concurrent requests, which re-read four invoices |
| gpt-oss-120b costs €0.20 per million tokens | €0.18 | Now computed at its fine-grid capacity (18 req/s) rather than 16 req/s |
| EAGLE-3 makes one gpt-oss-120b user 19 % slower | 22 % slower | Same-batch baseline |
| Qwen3.8-Flash-Next with its n-gram table in GPU memory: −3 % for one user | +4 % (within noise) | Baseline run of the same batch |
| Energy per token from 1 to 128 users: gpt-oss-120b 1.68 → 0.24 J | 2.43 → 0.31 J per output token in the 8k/1k sweep | Defined on one test (T12) |
| Little's law predicts capacity within 10 % for three models | Within 10 % for two (gpt-oss-120b, DeepSeek V4 Flash); 20 % high for Qwen3.8-27B | Same inputs for every model (medians at the first failing rate) |
| Highest module power 1,289 W | 1,301 W (Qwen3.8-27B with DFlash2, chat at 12 req/s) | The late runs were not included before |
| "Independent review" of the figures | Removed | The cross-check was automated, not independent |

### Changes to the harness

- Translated to English: identifiers, comments, help text and log messages. File names, JSON keys and profile
  labels written to disk keep their original Spanish names so that the released data and the code match
  (`bench/SCHEMA_GLOSSARY.md`).
- The batch-specific queue script was replaced by `serve.sh` and `run.sh`, which run one config at a time.
  `docs/runs.md` records what was run in each batch.
- The quick-quality test T10 from batch 1 was removed; T16 and T20–T22 replaced it.
- Server-specific paths, user names and ports became variables in `bench/common.sh`.
