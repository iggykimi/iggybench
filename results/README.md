# Result tables

Every number in the README and the report comes from one of these files. They are regenerated from the raw data by
the scripts in [`../analysis`](../analysis); each row has a `source` column (or a `file` column in `points.csv`)
with the raw file it was computed from. Column names carry their unit (`_ms`, `_s`, `_pct`, `_tok_s`, `_w`).

| Files | Produced by | Content |
|---|---|---|
| `points.csv` | `point_stats.py` | Every serving measurement: latency percentiles with bootstrap intervals, SLO checks, energy |
| `perf_single_user.csv`, `perf_prompt_reading.csv` | `performance.py` | One-user speed (T1); prompt reading from 1k to 1M tokens (T2, T5) |
| `perf_capacity_points.csv`, `perf_capacity.csv` | `performance.py` | Every T4/T4F/T19 point with its SLO check; capacity intervals per model and profile |
| `perf_department.csv` | `performance.py` | Mixed chat, RAG and agent load (T9) |
| `perf_coding_points.csv`, `perf_coding_capacity.csv` | `performance.py` | Coding-agent sessions (T14, T14F) |
| `perf_frontier_t12.csv`, `perf_frontier_summary.csv` | `performance.py` | The 8k/1k sweep (T12) and throughput at fixed per-user speeds |
| `perf_energy_cost.csv`, `perf_power.csv` | `performance.py` | Energy per token and cost at three operating points; peak and idle power |
| `perf_bandwidth.csv`, `perf_little_law.csv`, `perf_kv_cache.csv` | `performance.py` | Bandwidth used by one user; capacity predicted by Little's law; KV-cache size per configuration |
| `quality_*.csv` | `quality.py` | GSM8K, AIME and MATH-500 (with pairwise tests), BFCL, agents, e-mails, invoices, speech, needle, RAG, image generation |
| `config_*.csv` | `config.py` | Speculative decoding, two copies, GPU sharing, sequence limit, n-gram table placement, repeatability, real text, bursts |
