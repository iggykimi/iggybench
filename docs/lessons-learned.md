# Lessons learned

Notes from running a benchmark on borrowed hardware with a fixed end time. Most of these cost me hours; some cost
results.

## The machine and the engine

Leave memory outside the KV cache. A vendor service held 16 GB of GPU memory the whole time. DeepSeek V4 Flash
with 92 % of memory reserved had about 0.5 GB free, and its sparse-attention indexer, which allocates in proportion
to prefill chunk × context, crashed the server on the first 128k-token prompt. 88 % and 8,192-token prefill chunks
was stable. The same setting was then applied to Qwen3.8-Flash-Next as a precaution.

A dead server keeps answering the benchmark. After that crash the battery went on sending requests to a server
that was gone, and produced numbers. I added a check before every point (stop with exit code 3 if the server does
not answer) and an automatic restart with 5 points less memory and half the prefill chunk, resuming from the test
that failed. Those first invalid points were discarded and are not in the release.

Check the configuration before blaming the model. A colleague had measured Qwen3.8-27B NVFP4 at 4.3 tok/s. The
cause was `--enforce-eager` on an old vLLM image. With vLLM 0.32, CUDA graphs and native FP4 kernels for this GPU,
the same model gave hundreds of tokens per second.

Quantisation is not free speed. For Qwen3.8-27B the NVFP4 build gave about half the BF16 throughput under
realistic load. FP8 was the best choice on this GPU.

MTP can cost more than it gives. On GLM-5.3-Flash, multi-token prediction filled the KV cache at 16 agents for
no throughput gain, so the main GLM configuration runs without it. On Qwen3.8-Flash-Next it added 52 % for one user
and one rate step of chat capacity. EAGLE-3 on gpt-oss-120b made one user 22 % slower.

Server clocks lie about where you are. The server ran on UTC+8. Every join between client time windows and GPU
telemetry needs that offset; the analysis code takes it from one constant.

## The test design

Random tokens flatter speculative decoding and MoE routing. Qwen3.8-Flash-Next's MTP acceptance at one user was
71 % on random-token prompts and 42 % on real text. A draft model's acceptance went from 29 % in chat with random
tokens to 81 % in coding sessions. Any speculative-decoding number needs its workload next to it.

Saturated tests do not rank models. GSM8K at 95–97 % and an 8-task agent test at 87.5–100 % could not separate
the five main models. AIME and BFCL could, and degraded invoice scans and hard e-mails separated them more than the
clean versions.

Graders fail too. One agent task failed for gpt-oss-120b because it wrote an order number with a Unicode
non-breaking hyphen ("P‑1038"); the grader compared bytes. I fixed the grader and re-ran the test for every model.
A second task still requires the literal Spanish word "rotura" in the answer, which penalises paraphrase. It is
documented rather than fixed, because the run had ended.

A model that cannot be turned off thinks. GLM-5.3-Flash kept reasoning even when asked not to (about 200 output
tokens per e-mail against 16–18 for the Qwen models), which made it accurate and ten times slower on e-mail and
invoices.

Truncation is a measurement choice. At 32k tokens, 127 of the 128 AIME failures of the five main models were
truncated answers. Qwen3.8-27B went from 71 % to 92 % with 64k. A reasoning score without its budget is not a
number.

## Running it

No package index on the server. The machine had no PyPI access, so every client is standard-library Python,
datasets came from GitHub and Hugging Face, and missing libraries were mounted into the vLLM container. `vllm bench
serve --dataset-name custom` needs pandas, which the container lacks; [`bench/minipandas`](../bench/minipandas) is
the 34-line stand-in that made the real-text test possible.

Plan for the queue, not the tests. A queue of model batteries with an end time, early stopping for saturated
sweeps and a backup every 30 minutes finished more than planned: DeepSeek's battery took 125 minutes against 240
estimated. Even so, two queues died at start-up on the last afternoon, and one reasoning test (AIME for
Nemotron) was skipped so that the last model could finish before the booking ended.

Write down what is not comparable as you go. Late additions ran on a coarser rate grid, one on a different
engine, and AIME at two budgets. Keeping that list during the run made the report's caveats a copy, not a
reconstruction.
