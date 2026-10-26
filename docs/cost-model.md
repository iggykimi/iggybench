# Cost model

The cost figures are the output of a simple model, not a quote. This page lists every input so that anyone can
replace them with their own.

## Formula

```
EUR per million output tokens = (machine price / (3 years × hours used per year) + 0.15 EUR/kWh × measured kW)
                                / (output tokens per hour / 10^6)
```

- **Machine price:** €140,000, the midpoint of the market range below; €90,000 is given as the low end.
- **Amortization:** three years, no residual value.
- **Hours used:** 24/7 (8,760 h per year) by default. "Office hours" spreads the same amortization over 176 hours a
  month, which multiplies the cost by about 4.
- **Electricity:** €0.15 per kWh, applied to the measured GPU-module power at the operating point. Wall power,
  cooling and the data-centre overhead (PUE) are not included.
- **Throughput:** the measured output tokens per second at the operating point. All cost goes to output tokens;
  input tokens are free in this model.

Not included: staff, maintenance, rack space, network, VAT and financing.

The operating point used in the README is "chat at capacity": the highest Poisson rate that met the chat SLO
([perf_capacity.csv](../results/perf_capacity.csv)). [perf_energy_cost.csv](../results/perf_energy_cost.csv) also
gives the T12 peak-throughput point and the 10-minute steady-state run. Full utilisation 24/7 is optimistic for
interactive chat, which follows working hours.

## Machine price

DGX Station GB300 list or street prices from several system builders, before VAT, collected in September and
October 2026 from reseller listings (pi3g/qdna.fr, BetterClaw, buyzero). I did not record the URLs and did not
verify the prices with the vendors.

| Builder | Price before VAT |
|---|---|
| Supermicro | ≈ €88,000 |
| MSI (WS300) | ≈ €90,000 |
| Exxact | ≈ €95,000 |
| ASUS (ET900N G3, the unit tested) | €95,750–110,500 |
| Gigabyte | ≈ €99,500 |
| HP | ≈ €110,000 |
| Dell | €151,000–173,000 |

## API prices for comparison

OpenRouter listings on 22 October 2026, in US dollars per million tokens, not converted. "Chat-equivalent" weights
two input tokens per output token, the ratio of the chat test, and assigns the result to the output token.

| Model | Lowest provider (in / out) | Typical provider (in / out) | Chat-equivalent $/Mtok out | Listing |
|---|---|---|---|---|
| gpt-oss-120b | 0.03 / 0.15 | 0.15 / 0.60 (Groq, Together, Bedrock) | 0.21–0.90 | [openrouter.ai/openai/gpt-oss-120b](https://openrouter.ai/openai/gpt-oss-120b) |
| Qwen3.8-27B | 0.039 / 1.25 | 0.20 / 2.50 (DeepInfra, Mancer, Phala) | 1.33–2.90 | [openrouter.ai/qwen/qwen3.8-27b](https://openrouter.ai/qwen/qwen3.8-27b) |
| Qwen3.8-Flash-Next | 0.15 / 0.47 (single provider) | same | 0.77 | [openrouter.ai/qwen/qwen3.8-flash](https://openrouter.ai/qwen/qwen3.8-flash); the hosted checkpoint may differ |
| DeepSeek V4 Flash | 0.0475 / 0.1568 | 0.14 / 0.28 (Together, Cohere, Parasail) | 0.25–0.56 | [openrouter.ai/deepseek/deepseek-v4-flash-0731](https://openrouter.ai/deepseek/deepseek-v4-flash-0731) |
| GLM-5.3-Flash | 0.0234 / 0.0978 | 0.15 / 0.50 (Z.ai) | 0.14–0.80 | [openrouter.ai/z-ai/glm-5.3-flash](https://openrouter.ai/z-ai/glm-5.3-flash) |

API prices do not include the discount many providers give for cached input, and they change often.

Transcription API prices (OpenAI whisper-1 and gpt-4o-transcribe, $0.18–0.36 per hour of audio) are from
[diyai.io](https://diyai.io/ai-tools/speech-to-text/openai-whisper-api-pricing-2026/), not from OpenAI's own page.

## What the model says

With these inputs, amortization is 96–98 % of the cost, so the cost per token depends almost entirely on how much
the machine is used and on the model's throughput, and very little on electricity. Against API prices (euros against dollars, unconverted)
the machine came out ahead for gpt-oss-120b and the two Qwen models under continuous use, and behind for DeepSeek V4
Flash and GLM-5.3-Flash; in office hours it was behind for all of them.
