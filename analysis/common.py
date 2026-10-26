"""Shared helpers: locating raw runs, model names, and the small statistics used across the analysis."""

import json
import math
import os
from pathlib import Path

RAW = Path(os.environ.get("IGGY_RAW", Path(__file__).resolve().parents[1] / "raw"))
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Server clock was UTC+8. Run folders are named with server time.
SERVER_UTC_OFFSET_H = 8

# Main models first (full test battery), then the three added on the last day (reduced battery).
MODELS = {
    "gptoss120b": "gpt-oss-120b",
    "qwen27b-fp8": "Qwen3.8-27B",
    "qwen38-flash-next": "Qwen3.8-Flash-Next",
    "dsv4-flash": "DeepSeek V4 Flash",
    "glm53-ssmbf16": "GLM-5.3-Flash",
    "qwen27b-fp8-dflash": "Qwen3.8-27B + DFlash2",
    "flint-sglang": "Recursive Flint",
    "nemotron35-super-vl": "Nemotron 3.5 Super VL",
}
MAIN = list(MODELS)[:5]
LATE = list(MODELS)[5:]

# Service-level objectives on P95 latency (ms): (TTFT, TPOT). Profile names are the labels used in the raw data.
SLO = {"chat": (2000, 50), "rag": (5000, 50), "agentes": (5000, 40)}
CODING_SLO = {"ttft_p95_s": 8.0, "decode_tok_s_p50": 30.0}


def batch_of(run: str) -> str:
    """Batch label from a run folder name (server time)."""
    ts = run.split("-", 1)[1] if run.startswith("bark-") else run
    if ts < "20261021-19":
        return "batch1"
    if ts < "20261022-14":
        return "batch2"
    if ts < "20261022-20":
        return "queue4"
    return "queue5-9"


def runs(tag: str) -> list[Path]:
    return sorted(p for p in (RAW / "results" / tag).glob("*") if p.is_dir())


def latest(tag: str, filename: str) -> Path | None:
    """Most recent run of `tag` that contains `filename`."""
    hits = [r / filename for r in runs(tag) if (r / filename).exists()]
    return hits[-1] if hits else None


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: Path) -> str:
    """Path relative to the raw root, for the `source` column of every table."""
    return path.relative_to(RAW).as_posix()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proportion and Wilson 95 % interval, in percent."""
    if not n:
        return (math.nan, math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * p, 100 * (c - h), 100 * (c + h))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def write_csv(df, name: str) -> Path:
    """Write a results table with a fixed float format so diffs stay readable."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    df.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")
    return path
