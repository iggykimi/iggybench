# Metrics

Every metric used in the README, the report and `results/`. Latencies are absolute, measured at the client on the
same machine, and include queueing.

## Latency and throughput

| Metric | Definition |
|---|---|
| TTFT | Time from sending a request to receiving its first output token. Includes queueing and prompt processing (prefill). |
| TPOT | Time per output token after the first: (end-to-end time − TTFT) / (output tokens − 1), per request, as computed by `vllm bench serve`. |
| ITL | Time between two consecutive output chunks. With speculative decoding a chunk carries several tokens, so ITL is not TPOT. |
| E2E | End-to-end time of a request. |
| P50, P95, P99 | Percentiles over the requests of one point. Capacity uses P95. |
| Output tok/s | Output tokens per second of the whole GPU over the point's duration. Called throughput. |
| Tok/s per user | 1,000 / median TPOT in ms. The speed one user sees once the answer starts. Called interactivity in InferenceX. |
| Read tok/s | Prompt tokens / median TTFT with one user. A lower bound on prefill speed, because TTFT also includes the first decode step. |

## Capacity

| Metric | Definition |
|---|---|
| SLO | Service-level objective on P95 latency over one point. Chat: TTFT ≤ 2 s and TPOT ≤ 50 ms (≥ 20 tok/s per user). RAG: TTFT ≤ 5 s, TPOT ≤ 50 ms. Agents: TTFT ≤ 5 s, TPOT ≤ 40 ms. A point also needs zero failed requests. |
| Capacity | The highest tested Poisson arrival rate that met the SLO, reported with the next rate tested, which failed: "18–20 req/s". The true limit lies between the two. |
| Users | Arrival rate × 60, assuming one request per user per minute. A planning convention, not a measured head count. |
| Coding-session capacity | The highest number of concurrent sessions with P95 TTFT ≤ 8 s on follow-up turns, median decode speed ≥ 30 tok/s and no errors. |
| Department capacity | The largest department size at which chat, RAG and agent requests all met their own SLO at once. |
| MLPerf check | The same points re-checked with MLPerf Inference's P99 limits (conversational 2 s / 200 ms, interactive 450 ms / 40 ms) as an outside reference. Not a certification. |
| Fluidity index | Share of tokens delivered within a deadline that carries over unused slack (Etalon; 2 s for the first token, 50 ms per token after). Computed from per-request ITLs. |

## Quality

| Metric | Definition |
|---|---|
| Accuracy | Correct answers / questions. Truncated answers (hit the token budget) count as wrong. |
| Wilson interval | 95 % score interval for a proportion. For AIME, where each problem gets two attempts, I also give a conservative interval with n = number of problems, since the two attempts are not independent. |
| Pairwise tests | Exact McNemar on GSM8K, where per-question results exist. Elsewhere a two-proportion z test (or Fisher's exact test for small counts), which is unpaired and therefore conservative. p-values are Holm-corrected over the pairs compared. |
| Field accuracy | Invoices: share of the 8 fields (number, date, supplier, supplier tax ID, net amount, VAT rate, total, number of lines) that match the ground truth after normalisation. |
| WER | Word error rate after lower-casing and stripping punctuation. This is simpler than Whisper's English normaliser, so numbers that a model writes as digits count as errors. |
| Real-time factor | Seconds of audio transcribed per second of wall time. |

## Power, energy and cost

| Metric | Definition |
|---|---|
| Module power | Power of the GPU module as reported by `nvidia-smi` every second, averaged over the point's time window. It is not wall power: it leaves out power supplies and fans, and it is not the 1,300 W GPU power limit. |
| J per output token | Module energy over the window (mean power × duration) / output tokens. |
| EUR per Mtok | (machine price / (3 years × hours used) + €0.15 per kWh × measured kW) / million output tokens per hour. All cost is assigned to output tokens. Assumptions and sources: [cost-model.md](cost-model.md). |

## Statistics

- Bootstrap intervals. Latency percentiles get a 95 % bootstrap interval from 1,000 resamples of the requests in
  the point, where per-request data exists. The resampling uses one random generator with a fixed seed, walking the
  runs in a fixed order, so the intervals are reproducible.
- Run-to-run variation. Three repeats of two points (T17) and a 10-minute run (T19). Between repeats the
  coefficient of variation was 0.2–1.4 % for throughput and up to 3.3 % for P95 latency, so differences under 3 %
  are treated as noise.
- Per-request TPOT for speculative models. `point_stats.py` also computes a per-request TPOT as the mean ITL,
  for intervals and fluidity. For models with MTP or a draft model, ITL is measured per chunk of accepted tokens,
  which inflates that column 1.7–2.5×. Capacity and every headline number use vLLM's own TPOT, which is correct.
