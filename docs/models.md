# Models

All models ran on the one GB300 GPU, one at a time unless a test says otherwise, served by
`vllm/vllm-openai:v0.32.0`, except Recursive Flint (SGLang, built from the model repository's own kit) and
Underdog Bark (Transformers). The exact server arguments of every run are in its `server_cmd.txt` in the raw data
and, cleaned of host-specific parts, in [`bench/configs/`](../bench/configs).

Hugging Face repository ids were recorded for the specialists, the three models added on the last day and
Qwen3.8-27B FP8 (`Qwen/Qwen3.8-27B-FP8`); gpt-oss-120b is the public `openai/gpt-oss-120b`. For GLM-5.3-Flash,
DeepSeek V4 Flash, Qwen3.8-Flash-Next and the batch-1 Qwen3.8-27B variants the download logs were not kept, and the
configs say so rather than guess.

## Language models

| Model | Architecture | Weights | Configuration used | Why it is here |
|---|---|---|---|---|
| gpt-oss-120b (OpenAI) | MoE, 117B total / 5.1B active, alternating dense and sliding-window attention | MXFP4, about 65 GB | 85 % of GPU memory, 256 sequences, 131k context, prefix caching, `openai` tool and reasoning parsers | Fits in 128 GB, so it can be compared with a DGX Spark; the most efficient model in batch 1 |
| Qwen3.8-27B (Alibaba) | dense 27B, Gated DeltaNet linear attention mixed with full attention; vision encoder | FP8 (main), NVFP4 and BF16 in batch 1 | 85 %, 256 sequences, 262k context, FP8 KV cache, SSM state in BF16 | The dense reference; the model most used by the DGX Station community |
| Qwen3.8-Flash-Next (Alibaba) | MoE, 125B / 6B active, 512 experts, plus a 51B n-gram memory table; three Gated DeltaNet layers per sparse-attention layer; MTP | FP8, 173 GB | n-gram table in Grace RAM (`--engram-config cpu_offload`), MTP 3 tokens, 88 %, 256 sequences, 8,192-token prefill chunks | Uses the Grace CPU's memory through NVLink-C2C |
| DeepSeek V4 Flash 0731 (DeepSeek) | MoE, 284B / 13B active, sparse attention | FP8, 156 GB | FP8 KV cache, block size 256, 128 sequences, 1M context, 88 % memory, 8,192-token prefill chunks | Aimed at coding and agents; the only model configured for 1M tokens |
| GLM-5.3-Flash (Z.ai) | MoE, 320B / 18B active, 34 linear-attention and 11 sparse-attention layers | NVFP4, 191 GB | no MTP (MTP filled the KV cache in batch 1 for no gain), SSM state in BF16, FP8 KV cache, 88 % | Hybrid linear attention at scale |
| Qwen3.8-27B + DFlash2 | the dense model with the z-lab DFlash2 draft model | FP8 + draft | `--speculative-config {"method": "dflash", "num_speculative_tokens": 7}` | What a draft model buys per user and costs in capacity |
| Recursive Flint (RecursiveSI) | MoE, 40B / 2.5B active, 256 experts with 6 per token, 513-token sliding window | BF16, about 80 GB | SGLang, 80 % memory, 262k context | Advertised as the fastest sub-50B MoE per user |
| Nemotron 3.5 Super VL (NVIDIA) | MoE, 120B / 12B active, Mamba-2 + MoE + attention; vision | NVFP4, about 70 GB | MTP 3 tokens, 85 %, 131k context | Similar active size to gpt-oss, with vision |

## Specialists

| Model | Task | Size | License | Repository |
|---|---|---|---|---|
| PaddleOCR-VL-1.6 | OCR (vision-language) | 0.9B | Apache 2.0 | `PaddlePaddle/PaddleOCR-VL-1.6` |
| Liquid d1-3B | classification without text generation | 3.1B | LFM Open License | `LiquidAI/d1-3B` |
| Laya multilingual | classification without text generation (mmBERT encoder) | 322M | Apache 2.0 | `convaiinnovations/laya` |
| Qwen3-Embedding-8B | embeddings | 8B | Apache 2.0 | `Qwen/Qwen3-Embedding-8B` |
| Qwen3-Reranker-8B | reranking | 8B | Apache 2.0 | `Qwen/Qwen3-Reranker-8B` |
| Qwen-Image | text to image (diffusion) | 20B + encoder | Apache 2.0 | `Qwen/Qwen-Image` |
| Whisper large-v3 | transcription | 1.55B | MIT | `openai/whisper-large-v3` |
| Voxtral Mini 3B | transcription and audio understanding | 3B | Apache 2.0 | `mistralai/Voxtral-Mini-3B-2507` |
| Qwen3-ASR 1.7B | transcription | 1.7B | Apache 2.0 | `Qwen/Qwen3-ASR-1.7B` |
| Granite 4.0 1B Speech | transcription | 1B | Apache 2.0 | `ibm-granite/granite-4.0-1b-speech` |
| Underdog Bark 3.0 Pro | transcription (fine-tuned Parakeet TDT 0.6B) | 0.6B | CC BY 4.0 | `ConwayResearch/Underdog-Bark-3.0-Pro` |

The speech models ran with 30 % of GPU memory each, from a vLLM image with `soundfile`, `librosa` and
`mistral-common` added ([`bench/docker/vllm-audio`](../bench/docker/vllm-audio)); the stock image cannot decode
audio.

## Considered and not measured

| Model | Reason |
|---|---|
| DeepSeek V4.1 Flash (NVFP4) | Its 183 GiB n-gram table exhausted host RAM while loading, and kernel autotuning stalled |
| MiniMax M3 (NVFP4) | Fitted only with 40 GB of weights on the CPU and a 64k context, which would not represent the model |
| Granite Speech 5.0 | Did not load in vLLM 0.32 (unknown `bias` parameter); replaced by Granite 4.0 1B Speech |
| gpt-oss-120b + DFlash | Dropped before measuring; the draft model was tested on Qwen3.8-27B instead |
| Qwen-Image-2.1, FLUX.3 Klein | Licences restrict commercial use |

## Configuration notes

These come from the runs and are written up in [lessons-learned.md](lessons-learned.md):

- Leave GPU memory outside the KV cache. With a vendor service holding 16 GB, reserving 92 % left DeepSeek without
  room for its sparse-attention indexer at 128k-token prompts. 88 % with 8,192-token prefill chunks was stable.
- FP8 was the best precision for Qwen3.8-27B on this GPU. The NVFP4 build (W4A4) gave about half the BF16
  throughput under realistic load in batch 1.
- Prefix caching on hybrid models only hits whole blocks (8,576 tokens for GLM), and keeping the linear layers'
  state in BF16 allowed more concurrent sequences.
