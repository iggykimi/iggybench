#!/usr/bin/env python3
"""Per-point statistics for every serving measurement (`vllm bench serve` result) in the raw data.

Reads results/<tag>/<run>/{runs.csv, json/*.json, gpu.csv} and writes results/points.csv, one row per measured point:
  - P50/P95/P99 with bootstrap 95 % intervals (E2E always; TTFT and TPOT when per-request detail was saved)
  - SLO compliance: ours (P95) and MLPerf's (P99 conversational 2 s / 200 ms, interactive 450 ms / 40 ms)
  - fluidity-index (Etalon) when per-request inter-token latencies exist
  - energy: mean module power and joules per output and per total token over the point's time window

Batch 2 was summarised on the server before the per-request detail and GPU telemetry were deleted to save space.
For those points the server-side statistics (raw/derived/batch2_point_stats.csv, produced by the same computation)
replace the columns that cannot be recomputed here.

Usage: python analysis/point_stats.py [--boot 1000]
"""

import argparse
import csv
import datetime as dt
import math
import sys

import numpy as np
import pandas as pd

from common import RAW, SERVER_UTC_OFFSET_H, batch_of, load_json, write_csv

OUR_SLO = {
    "chat": (2000, 50),
    "rag": (5000, 50),
    "agentes": (5000, 40),
    "razonamiento": (None, 40),
    "velocidad": (None, 50),
}
MLPERF = {
    "mlperf_conv": (2000, 200),
    "mlperf_inter": (450, 40),
}  # P99 TTFT ms, P99 TPOT ms (Llama-2-70B)
FLUID_FIRST_S, FLUID_NEXT_S = (
    2.0,
    0.05,
)  # fluidity-index deadlines: 2 s to first token, 50 ms between tokens

# Server-side batch-2 columns (Spanish, as written on the server) -> names used here.
BATCH2_COLUMNS = {
    "e2e_p50_ic_inf": "e2e_p50_ci_low",
    "e2e_p50_ic_sup": "e2e_p50_ci_high",
    "e2e_p95_ic_inf": "e2e_p95_ci_low",
    "e2e_p95_ic_sup": "e2e_p95_ci_high",
    "e2e_p99_ic_inf": "e2e_p99_ci_low",
    "e2e_p99_ic_sup": "e2e_p99_ci_high",
    "detalle": "per_request_detail",
    "ttft_p50_req": "ttft_p50_req",
    "ttft_p50_ic_inf": "ttft_p50_ci_low",
    "ttft_p50_ic_sup": "ttft_p50_ci_high",
    "ttft_p95_req": "ttft_p95_req",
    "ttft_p95_ic_inf": "ttft_p95_ci_low",
    "ttft_p95_ic_sup": "ttft_p95_ci_high",
    "ttft_p99_req": "ttft_p99_req",
    "ttft_p99_ic_inf": "ttft_p99_ci_low",
    "ttft_p99_ic_sup": "ttft_p99_ci_high",
    "tpot_p50_req": "tpot_p50_req",
    "tpot_p50_ic_inf": "tpot_p50_ci_low",
    "tpot_p50_ic_sup": "tpot_p50_ci_high",
    "tpot_p95_req": "tpot_p95_req",
    "tpot_p95_ic_inf": "tpot_p95_ci_low",
    "tpot_p95_ic_sup": "tpot_p95_ci_high",
    "tpot_p99_req": "tpot_p99_req",
    "tpot_p99_ic_inf": "tpot_p99_ci_low",
    "tpot_p99_ic_sup": "tpot_p99_ci_high",
    "mlperf_conv_cumple_pct": "mlperf_conv_pass_pct",
    "mlperf_conv_pasa_p99": "mlperf_conv_p99_ok",
    "mlperf_inter_cumple_pct": "mlperf_inter_pass_pct",
    "mlperf_inter_pasa_p99": "mlperf_inter_p99_ok",
    "fluidez_media": "fluidity_mean",
    "fluidez_pct_peticiones_090": "fluidity_pct_requests_ge_090",
    "vatios_medios": "power_w_mean",
    "julios": "energy_j",
    "J_por_token_salida": "j_per_output_token",
    "J_por_token_total": "j_per_total_token",
}

rng = np.random.default_rng(2026)
BOOT = 1000


def boot_ci(x, q):
    """Bootstrap 95 % interval of the q-th percentile (0-100). Returns (estimate, low, high)."""
    x = np.asarray([v for v in x if v is not None and not math.isnan(v)], float)
    if len(x) < 5 or BOOT <= 0:
        return (float(np.percentile(x, q)) if len(x) else None, None, None)
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    b = np.percentile(x[idx], q, axis=1)
    return (
        float(np.percentile(x, q)),
        float(np.percentile(b, 2.5)),
        float(np.percentile(b, 97.5)),
    )


def fluidity(ttft, itls):
    """Fluidity-index (Etalon): share of tokens delivered within their deadline, carrying accumulated slack."""
    met = 0
    total = 1 + len(itls)
    if ttft <= FLUID_FIRST_S:
        met += 1
        slack = FLUID_FIRST_S - ttft
    else:
        slack = 0.0
    for gap in itls:
        deadline = FLUID_NEXT_S + slack
        if gap <= deadline:
            met += 1
            slack = deadline - gap
        else:
            slack = 0.0
    return met / total


def load_gpu(path):
    """gpu.csv -> (epoch[], watts[]), using module power when present, else GPU power."""
    if not path.exists():
        return None
    times, watts = [], []
    off = dt.timedelta(hours=SERVER_UTC_OFFSET_H)
    with open(path, errors="ignore") as fh:
        for row in csv.reader(fh):
            try:
                t = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f") - off
                w = float(row[6]) if len(row) > 6 and row[6].strip() not in ("", "[N/A]") else float(row[1])
                times.append(t.replace(tzinfo=dt.timezone.utc).timestamp())
                watts.append(w)
            except (ValueError, IndexError):
                continue
    return (np.array(times), np.array(watts)) if times else None


def energy(gpu, start, end):
    if gpu is None:
        return (None, None)
    t, w = gpu
    m = (t >= start) & (t <= end)
    if m.sum() < 3:
        return (None, None)
    return float(w[m].mean()), float(w[m].mean() * (end - start))  # mean W, joules


def point_row(tag, rdir, r, d, gpu):
    profile = r["profile"].split("@")[0]
    row = {
        "model": tag,
        "run": r["runid"],
        "batch": batch_of(rdir.name),
        "test": r["test"],
        "profile": r["profile"],
        "input_tokens": int(r["in"]),
        "prefix_tokens": int(r["prefix"]),
        "output_tokens": int(r["out"]),
        "concurrency": r["conc"],
        "rate_req_s": r["rate"],
        "n": int(r["n"]),
        "completed": d.get("completed"),
        "failed": d.get("failed"),
        "duration_s": d.get("duration"),
        "output_tok_s": d.get("output_throughput"),
        "total_tok_s": d.get("total_token_throughput"),
        "req_s": d.get("request_throughput"),
        "ttft_p50": d.get("p50_ttft_ms") or d.get("median_ttft_ms"),
        "ttft_p95": d.get("p95_ttft_ms"),
        "ttft_p99": d.get("p99_ttft_ms"),
        "tpot_p50": d.get("p50_tpot_ms") or d.get("median_tpot_ms"),
        "tpot_p95": d.get("p95_tpot_ms"),
        "tpot_p99": d.get("p99_tpot_ms"),
        "itl_p99": d.get("p99_itl_ms"),
        "tok_in": d.get("total_input_tokens"),
        "tok_out": d.get("total_output_tokens"),
        "max_concurrent_requests": d.get("max_concurrent_requests"),
        "start": float(r["start"]),
        "end": float(r["end"]),
        "file": f"results/{tag}/{rdir.name}/json/{r['file']}",
    }
    row["tok_s_per_user"] = 1000 / row["tpot_p50"] if row["tpot_p50"] else None
    e2e = [x * 1000 for x in (d.get("latencies") or [])]
    for q in (50, 95, 99):
        v, lo, hi = boot_ci(e2e, q) if e2e else (None, None, None)
        row[f"e2e_p{q}"], row[f"e2e_p{q}_ci_low"], row[f"e2e_p{q}_ci_high"] = v, lo, hi
    ttfts, itls = d.get("ttfts"), d.get("itls")
    row["per_request_detail"] = bool(ttfts)
    if ttfts:
        tt = [x * 1000 for x in ttfts]
        tp = [((sum(i) / len(i)) * 1000) if i else None for i in (itls or [[]] * len(tt))]
        for name, series in (("ttft", tt), ("tpot", tp)):
            for q in (50, 95, 99):
                v, lo, hi = boot_ci(series, q)
                (
                    row[f"{name}_p{q}_req"],
                    row[f"{name}_p{q}_ci_low"],
                    row[f"{name}_p{q}_ci_high"],
                ) = v, lo, hi
        for k, (ttl, tpl) in MLPERF.items():
            ok = sum(1 for a, b in zip(tt, tp) if a <= ttl and (b or 0) <= tpl)
            row[f"{k}_pass_pct"] = 100 * ok / len(tt)
            row[f"{k}_p99_ok"] = (
                (np.percentile(tt, 99) <= ttl and np.percentile([b for b in tp if b], 99) <= tpl) if any(tp) else None
            )
        if itls:
            fl = [fluidity(a, i) for a, i in zip(ttfts, itls)]
            row["fluidity_mean"] = float(np.mean(fl))
            row["fluidity_pct_requests_ge_090"] = 100 * float(np.mean(np.array(fl) >= 0.9))
    # without per-request detail, MLPerf compliance is checked on the aggregate P99
    if not ttfts and row["ttft_p99"] is not None and row["tpot_p99"] is not None:
        for k, (ttl, tpl) in MLPERF.items():
            row[f"{k}_p99_ok"] = row["ttft_p99"] <= ttl and row["tpot_p99"] <= tpl
    slo = OUR_SLO.get(profile)
    if slo and row["completed"] and not row["failed"]:
        row["slo_ok"] = (slo[0] is None or (row["ttft_p95"] or 1e9) <= slo[0]) and (
            slo[1] is None or (row["tpot_p95"] or 1e9) <= slo[1]
        )
    w, j = energy(gpu, row["start"], row["end"])
    row["power_w_mean"] = w
    row["energy_j"] = j
    row["j_per_output_token"] = j / row["tok_out"] if j and row["tok_out"] else None
    row["j_per_total_token"] = j / (row["tok_out"] + row["tok_in"]) if j and row["tok_out"] and row["tok_in"] else None
    return row


def collect():
    rows = []
    # fixed traversal order: the bootstrap draws from one generator, so the intervals depend on it
    for runs_csv in sorted((RAW / "results").glob("*/*/runs.csv"), key=lambda p: p.as_posix()):
        rdir = runs_csv.parent
        tag = rdir.parent.name
        gpu = load_gpu(rdir / "gpu.csv")
        with open(runs_csv, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                jp = rdir / "json" / r["file"]
                if not jp.exists():
                    continue
                try:
                    d = load_json(jp)
                except ValueError:
                    continue
                rows.append(point_row(tag, rdir, r, d, gpu))
    return rows


def overlay_batch2(rows):
    """Batch-2 points lost their per-request detail and GPU telemetry on the server; take those columns from the
    statistics computed there (raw/derived/batch2_point_stats.csv, same computation as this script)."""
    with open(RAW / "derived" / "batch2_point_stats.csv", encoding="utf-8") as fh:
        server = {
            (s["modelo"], s["ejecucion"], s["test"], s["perfil"], s["conc"], s["rate"], s["tok_in"]): s
            for s in csv.DictReader(fh)
        }
    for row in rows:
        row["stats_source"] = "local"
        if row["per_request_detail"] or row["power_w_mean"] is not None:
            continue
        key = (row["model"], row["run"], row["test"], row["profile"], row["concurrency"], row["rate_req_s"])
        s = server.get(key + (str(row["tok_in"]),))
        if s is None:
            continue
        for src, dst in BATCH2_COLUMNS.items():
            v = s[src]
            row[dst] = None if v == "" else (v == "True" if v in ("True", "False") else float(v))
        row["stats_source"] = "server"
    return rows


def main():
    global BOOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (0 = no intervals, faster)")
    BOOT = ap.parse_args().boot
    rows = collect()
    if not rows:
        sys.exit(f"no results under {RAW}")
    out = write_csv(pd.DataFrame(overlay_batch2(rows)), "points.csv")
    print(f"{len(rows)} points -> {out}")


if __name__ == "__main__":
    main()
