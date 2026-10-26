#!/usr/bin/env python3
"""Serving-performance tables: single-user speed, prompt reading, capacity under SLO, mixed department load,
coding-agent sessions, throughput-interactivity frontier, energy and cost.

Reads results/points.csv (from point_stats.py) and the session files (sesiones_*.json) in the raw data.
Writes results/perf_*.csv. Every row keeps the raw file it comes from in `source`.
"""

import re

import numpy as np
import pandas as pd

from common import CODING_SLO, LATE, MAIN, MODELS, RAW, RESULTS, SLO, load_json, rel, write_csv

# Cost model. The machine price is the midpoint of the market range in docs/cost-model.md; 90 000 EUR is the low end.
PRICE_EUR = 140_000
PRICE_LOW_EUR = 90_000
YEARS = 3
EUR_PER_KWH = 0.15
OFFICE_HOURS_PER_MONTH = 176
HOURS_PER_MONTH = 730

# Points excluded from capacity, with the reason (kept in perf_capacity_points.csv with the flag).
EXCLUDED = {
    ("glm53-ssmbf16", "chat", 0.5): "first point after server start ran during warm-up; re-measured at higher rates",
}


def load_points():
    p = pd.read_csv(RESULTS / "points.csv", dtype={"run": str, "concurrency": str})
    p["rate"] = pd.to_numeric(p.rate_req_s, errors="coerce")
    p["conc"] = pd.to_numeric(p.concurrency, errors="coerce")
    p["base_profile"] = p.profile.str.split("@").str[0]
    p["source"] = p.file
    return p


def latest_run(df):
    """Rows of the most recent run in `df` (one model, one test)."""
    return df[df.run == df.run.max()]


def battery(p, m, test):
    """Points of `test` for model `m` from its battery: batch 2 and queue 4 for the five main models, the last
    afternoon for the three late ones. Later re-runs of a main model (e.g. T1 before the needle test) are left out."""
    batches = ("batch2", "queue4") if m in MAIN else ("queue5-9",)
    return p[(p.model == m) & (p.test == test) & p.batch.isin(batches)]


def single_user(p):
    rows = []
    for m in MODELS:
        x = battery(p, m, "T1")
        if x.empty:
            continue
        r = latest_run(x).iloc[0]
        rows.append(
            dict(
                model=m,
                model_name=MODELS[m],
                tok_s_per_user=1000 / r.tpot_p50,
                tpot_p50_ms=r.tpot_p50,
                ttft_p50_ms=r.ttft_p50,
                ttft_p95_ms=r.ttft_p95,
                requests=r.n,
                power_w_mean=r.power_w_mean,
                run=r.run,
                source=r.source,
            )
        )
    return pd.DataFrame(rows)


def prompt_reading(p):
    """T2 (1 user, 1k-128k input, 16 output tokens) and T5 (128k-1M)."""
    rows = []
    for m in MODELS:
        for test in ("T2", "T5"):
            x = battery(p, m, test)
            if x.empty:
                continue
            for _, r in latest_run(x).sort_values("input_tokens").iterrows():
                rows.append(
                    dict(
                        model=m,
                        model_name=MODELS[m],
                        test=test,
                        input_tokens=r.input_tokens,
                        users=r.concurrency,
                        requests=r.n,
                        completed=r.completed,
                        ttft_p50_ms=r.ttft_p50,
                        ttft_p95_ms=r.ttft_p95,
                        read_tok_s=r.input_tokens / (r.ttft_p50 / 1000),
                        run=r.run,
                        source=r.source,
                    )
                )
    return pd.DataFrame(rows)


def slo_check(x):
    lim = x.base_profile.map(lambda s: SLO[s])
    x = x.assign(ttft_limit_ms=lim.str[0], tpot_limit_ms=lim.str[1])
    x["meets_latency"] = (x.ttft_p95 <= x.ttft_limit_ms) & (x.tpot_p95 <= x.tpot_limit_ms)
    x["meets_slo"] = x.meets_latency & (x.failed == 0)
    return x


def capacity_points(p):
    x = p[p.test.isin(["T4", "T4F", "T19"]) & p.model.isin(MODELS) & (p.batch != "batch1")].copy()
    x = slo_check(x)
    x["excluded"] = [
        EXCLUDED.get((r.model, r.base_profile, r.rate), "") if r.test != "T19" else "" for r in x.itertuples()
    ]
    x["users_at_1_req_per_min"] = x.rate * 60
    cols = [
        "model",
        "test",
        "base_profile",
        "rate",
        "users_at_1_req_per_min",
        "n",
        "completed",
        "failed",
        "req_s",
        "output_tok_s",
        "ttft_p50",
        "ttft_p95",
        "ttft_p95_ci_low",
        "ttft_p95_ci_high",
        "ttft_p99",
        "tpot_p50",
        "tpot_p95",
        "tpot_p99",
        "meets_latency",
        "meets_slo",
        "mlperf_conv_p99_ok",
        "mlperf_inter_p99_ok",
        "power_w_mean",
        "j_per_output_token",
        "duration_s",
        "batch",
        "run",
        "excluded",
        "source",
    ]
    x = x[cols].rename(
        columns={
            "base_profile": "profile",
            "rate": "rate_req_s",
            "ttft_p50": "ttft_p50_ms",
            "ttft_p95": "ttft_p95_ms",
            "ttft_p99": "ttft_p99_ms",
            "tpot_p50": "tpot_p50_ms",
            "tpot_p95": "tpot_p95_ms",
            "tpot_p99": "tpot_p99_ms",
        }
    )
    x.insert(1, "model_name", x.model.map(MODELS))
    return x.sort_values(["model", "profile", "test", "rate_req_s"])


def interval(levels, ok):
    """Capacity interval: highest level that passes, and the first level above it that fails (NaN if none)."""
    levels, ok = np.asarray(levels, float), np.asarray(ok, bool)
    if not ok.any():
        return (0.0, levels.min() if len(levels) else np.nan)
    last = levels[ok].max()
    above = levels[(levels > last) & ~ok]
    return (last, above.min() if len(above) else np.nan)


def capacity(points):
    rows = []
    for m in MODELS:
        for prof in ("chat", "agentes"):
            x = points[(points.model == m) & (points.profile == prof) & points.test.isin(["T4", "T4F"])]
            x = x[x.excluded == ""]
            if x.empty:
                continue
            last, first = interval(x.rate_req_s, x.meets_slo)
            fail = x[x.rate_req_s == first]
            why = ""
            if len(fail):
                f = fail.iloc[-1]
                parts = []
                if f.ttft_p95_ms > SLO[prof][0]:
                    parts.append(f"TTFT P95 {f.ttft_p95_ms:.0f} ms")
                if f.tpot_p95_ms > SLO[prof][1]:
                    parts.append(f"TPOT P95 {f.tpot_p95_ms:.1f} ms")
                if f.failed:
                    parts.append(f"{int(f.failed)} failed requests")
                why = ", ".join(parts)
            steady = points[(points.model == m) & (points.profile == prof) & (points.test == "T19")]
            s = steady.iloc[-1] if len(steady) else None
            rows.append(
                dict(
                    model=m,
                    model_name=MODELS[m],
                    profile=prof,
                    grid="fine" if (x.test == "T4F").any() else "coarse",
                    last_pass_req_s=last,
                    first_fail_req_s=first,
                    users_low=last * 60,
                    users_high=first * 60,
                    fails_on=why,
                    mlperf_conv_max_req_s=x[x.mlperf_conv_p99_ok == True].rate_req_s.max(),  # noqa: E712
                    steady_req_s=s.rate_req_s if s is not None else np.nan,
                    steady_requests=s.n if s is not None else np.nan,
                    steady_failed=s.failed if s is not None else np.nan,
                    steady_ttft_p95_ms=s.ttft_p95_ms if s is not None else np.nan,
                    steady_tpot_p95_ms=s.tpot_p95_ms if s is not None else np.nan,
                    tok_s_per_user_at_last_pass=1000 / x[x.rate_req_s == last].tpot_p50_ms.iloc[-1]
                    if last in set(x.rate_req_s)
                    else np.nan,
                    points=len(x),
                    source=";".join(sorted(set(x.source.str.rsplit("/", n=2).str[0]))),
                )
            )
    return pd.DataFrame(rows)


def department(p):
    """T9: chat 60 %, RAG 25 %, agents 15 % at once; a level passes only if every profile meets its SLO."""
    x = p[(p.test == "T9") & p.model.isin(MODELS)].copy()
    x["people"] = x.profile.str.split("@").str[1].astype(int)
    x = slo_check(x)
    rows = []
    for m, g in x.groupby("model"):
        g = g[g.run == g.run.max()]
        lv = g.groupby("people").meets_slo.all()
        last, first = interval(lv.index, lv.values)
        failing = g[(g.people == first) & ~g.meets_slo].base_profile.tolist()
        rows.append(
            dict(
                model=m,
                model_name=MODELS[m],
                last_pass_people=last,
                first_fail_people=first,
                failing_profiles=",".join(failing),
                max_tested_people=g.people.max(),
                batch=g.batch.iloc[0],
                source=g.source.iloc[0].rsplit("/", 2)[0],
            )
        )
    return pd.DataFrame(rows)


def coding_sessions():
    """T14: coding-agent sessions of 14 turns, 48k growing to ~100k tokens of context, with prefix caching."""
    rows = []
    for m in MODELS:
        for f in sorted((RAW / "results" / m).glob("*/sesiones_*.json")):
            d = load_json(f)
            if (d.get("etiqueta") or "base") == "offload":
                continue  # KV offloading variant, reported with the configuration tests
            for lv in d["niveles"]:
                rows.append(
                    dict(
                        model=m,
                        model_name=MODELS[m],
                        grid=d.get("etiqueta") or "base",
                        sessions=lv["sesiones"],
                        turns=lv["turnos"],
                        errors=lv["errores"],
                        seconds=lv["segundos"],
                        context_mean_tokens=lv["contexto_medio"],
                        context_max_tokens=lv["contexto_max"],
                        ttft_first_turn_p50_s=lv["ttft_primer_turno_p50"],
                        ttft_p50_s=lv["ttft_p50"],
                        ttft_p95_s=lv["ttft_p95"],
                        prefix_cache_pct=lv["cache_pct"],
                        decode_tok_s_p50=lv["dec_tok_s_p50"],
                        decode_tok_s_p5=lv["dec_tok_s_p5"],
                        run=f.parent.name,
                        source=rel(f),
                    )
                )
    s = pd.DataFrame(rows)
    s["meets_slo"] = (
        (s.ttft_p95_s <= CODING_SLO["ttft_p95_s"])
        & (s.decode_tok_s_p50 >= CODING_SLO["decode_tok_s_p50"])
        & (s.errors == 0)
    )
    cap = []
    for m, g in s.groupby("model", sort=False):
        last, first = interval(g.sessions, g.meets_slo)
        four = g[g.sessions == 4]
        cap.append(
            dict(
                model=m,
                model_name=MODELS[m],
                last_pass_sessions=last,
                first_fail_sessions=first,
                max_tested_sessions=g.sessions.max(),
                decode_tok_s_p50_at_4=four.decode_tok_s_p50.iloc[-1] if len(four) else np.nan,
                source=";".join(sorted(set(g.source))),
            )
        )
    return s.sort_values(["model", "sessions", "run"]), pd.DataFrame(cap)


def frontier(p):
    """T12 (8k input, 1k output, closed loop at 1-128 users): the community standard sweep."""
    x = p[(p.test == "T12") & p.model.isin(MODELS) & (p.batch != "batch1")].copy()
    x = x[x.run == x.groupby("model").run.transform("max")]
    out = x[
        [
            "model",
            "conc",
            "n",
            "completed",
            "failed",
            "output_tok_s",
            "total_tok_s",
            "tok_s_per_user",
            "tpot_p50",
            "tpot_p95",
            "ttft_p50",
            "ttft_p95",
            "power_w_mean",
            "j_per_output_token",
            "j_per_total_token",
            "run",
            "source",
        ]
    ].rename(
        columns={
            "conc": "users",
            "tpot_p50": "tpot_p50_ms",
            "tpot_p95": "tpot_p95_ms",
            "ttft_p50": "ttft_p50_ms",
            "ttft_p95": "ttft_p95_ms",
        }
    )
    out.insert(1, "model_name", out.model.map(MODELS))
    out = out.sort_values(["model", "users"])
    rows = []
    for m, g in out.groupby("model", sort=False):
        g = g.sort_values("tok_s_per_user")
        r = dict(
            model=m,
            model_name=MODELS[m],
            max_output_tok_s=g.output_tok_s.max(),
            users_at_max=g.loc[g.output_tok_s.idxmax(), "users"],
        )
        for level in (20, 30, 50, 100):
            inside = g.tok_s_per_user.min() <= level <= g.tok_s_per_user.max()
            r[f"output_tok_s_at_{level}_per_user"] = (
                float(np.interp(level, g.tok_s_per_user, g.output_tok_s)) if inside else np.nan
            )
        rows.append(r)
    return out, pd.DataFrame(rows)


def eur_per_hour(watts, price=PRICE_EUR, hours_used_per_month=HOURS_PER_MONTH):
    capex = price / (YEARS * 12 * hours_used_per_month)
    return capex + EUR_PER_KWH * watts / 1000


def cost(points, cap, front):
    """EUR per million output tokens at three operating points, 24/7 use unless stated."""
    rows = []
    for m in MODELS:
        chat = cap[(cap.model == m) & (cap.profile == "chat")]
        cand = []
        f = front[front.model == m]
        if len(f):
            cand.append(("T12 peak throughput", f.loc[f.output_tok_s.idxmax()]))
        if len(chat):
            rate = chat.last_pass_req_s.iloc[0]
            c = points[
                (points.model == m)
                & (points.profile == "chat")
                & points.test.isin(["T4", "T4F"])
                & (points.rate_req_s == rate)
                & points.meets_slo
            ]
            if len(c):
                cand.append(("chat at capacity", c.iloc[-1]))
            s = points[(points.model == m) & (points.test == "T19")]
            if len(s):
                cand.append(("chat steady state (10 min)", s.iloc[-1]))
        for label, r in cand:
            if pd.isna(r.power_w_mean):
                continue
            tok_h = r.output_tok_s * 3600 / 1e6
            rows.append(
                dict(
                    model=m,
                    model_name=MODELS[m],
                    operating_point=label,
                    output_tok_s=r.output_tok_s,
                    power_w_mean=r.power_w_mean,
                    j_per_output_token=r.j_per_output_token,
                    eur_per_mtok_out=eur_per_hour(r.power_w_mean) / tok_h,
                    eur_per_mtok_out_low_price=eur_per_hour(r.power_w_mean, PRICE_LOW_EUR) / tok_h,
                    eur_per_mtok_out_office_hours=eur_per_hour(
                        r.power_w_mean, hours_used_per_month=OFFICE_HOURS_PER_MONTH
                    )
                    / tok_h,
                    electricity_share_pct=100 * EUR_PER_KWH * r.power_w_mean / 1000 / eur_per_hour(r.power_w_mean),
                    source=r.source,
                )
            )
    return pd.DataFrame(rows)


def power(p):
    """Highest one-point mean module power per model, idle power, and lowest SM clock seen."""
    rows = []
    for m in MODELS:
        x = p[(p.model == m) & p.power_w_mean.notna()]
        if x.empty:
            continue
        top = x.loc[x.power_w_mean.idxmax()]
        rows.append(
            dict(
                model=m,
                model_name=MODELS[m],
                max_point_power_w=top.power_w_mean,
                at=f"{top.test} {top.profile} conc={top.concurrency} rate={top.rate_req_s}",
                source=top.source,
            )
        )
    idle = RAW / "results" / "reposo"
    for f in sorted(idle.glob("*/gpu.csv")):
        w = pd.read_csv(f, header=None, skipinitialspace=True)
        module = pd.to_numeric(w[6], errors="coerce") if w.shape[1] > 6 else pd.to_numeric(w[1], errors="coerce")
        rows.append(
            dict(
                model="idle",
                model_name="idle (no model loaded)",
                max_point_power_w=module.mean(),
                at=f"mean of {module.notna().sum()} one-second samples",
                source=rel(f),
            )
        )
    return pd.DataFrame(rows)


def kv_cache():
    """KV-cache size vLLM reported at each server start, per queue entry of the last (cumulative) queue log.
    SGLang and Transformers servers do not print this line."""
    log = RAW / "logs" / "queue5-9" / "cola.log"
    section, util, rows = None, None, []
    with open(log, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if m := re.search(r"#{6,} ([^#]+?) #{6,}$", line.strip()):
                section = m.group(1)
            elif m := re.search(r"--gpu-memory-utilization=([0-9.]+)", line):
                util = float(m.group(1))
            elif m := re.search(r"GPU KV cache size: ([0-9,]+) tokens", line):
                rows.append(
                    dict(
                        queue_entry=section,
                        gpu_memory_utilization=util,
                        kv_cache_tokens=int(m.group(1).replace(",", "")),
                        source=rel(log),
                    )
                )
    return pd.DataFrame(rows).drop_duplicates(["queue_entry", "gpu_memory_utilization", "kv_cache_tokens"])


def little_law(points):
    """Chat capacity predicted from two numbers: sequences the server admits at once (max-num-seqs) and the time a
    512-token answer takes when the server is full (TTFT + 511 x TPOT, medians at the first failing rate)."""
    rows = []
    for m in MAIN:
        x = points[(points.model == m) & (points.profile == "chat") & points.test.isin(["T4", "T4F"])]
        last, first = interval(x.rate_req_s, x.meets_slo)
        f = x[x.rate_req_s == first].iloc[-1]
        cmd = (RAW / f.source).parents[1] / "server_cmd.txt"
        seqs = int(re.search(r"--max-num-seqs (\d+)", cmd.read_text()).group(1))
        answer_s = f.ttft_p50_ms / 1000 + 511 * f.tpot_p50_ms / 1000
        rows.append(
            dict(
                model=m,
                model_name=MODELS[m],
                max_num_seqs=seqs,
                ttft_p50_ms=f.ttft_p50_ms,
                tpot_p50_ms=f.tpot_p50_ms,
                predicted_req_s=seqs / answer_s,
                measured_last_pass_req_s=last,
                measured_first_fail_req_s=first,
                source=f.source,
            )
        )
    return pd.DataFrame(rows)


# Weight bytes read per generated token at batch size 1: active parameters x bytes per parameter (MXFP4 and NVFP4
# with their block scales ~0.56 B, FP8 1 B). Models with speculative decoding are left out: they verify several
# tokens per weight read.
ACTIVE = {
    "gptoss120b": (5.1e9, 0.56),
    "qwen27b-fp8": (27e9, 1.0),
    "dsv4-flash": (13e9, 1.0),
    "glm53-ssmbf16": (18e9, 0.56),
}
HBM_TB_S = 7.1


def bandwidth(single):
    rows = []
    for m, (params, bpp) in ACTIVE.items():
        tok_s = single[single.model == m].tok_s_per_user.iloc[0]
        gb = params * bpp / 1e9
        rows.append(
            dict(
                model=m,
                model_name=MODELS[m],
                active_params_b=params / 1e9,
                bytes_per_param=bpp,
                gb_per_token=gb,
                tok_s_per_user=tok_s,
                tb_s_used=gb * tok_s / 1000,
                share_of_hbm_pct=100 * gb * tok_s / 1000 / HBM_TB_S,
            )
        )
    return pd.DataFrame(rows)


def main():
    p = load_points()
    single = single_user(p)
    write_csv(single, "perf_single_user.csv")
    write_csv(bandwidth(single), "perf_bandwidth.csv")
    write_csv(kv_cache(), "perf_kv_cache.csv")
    write_csv(prompt_reading(p), "perf_prompt_reading.csv")
    pts = capacity_points(p)
    write_csv(pts, "perf_capacity_points.csv")
    cap = capacity(pts)
    write_csv(cap, "perf_capacity.csv")
    write_csv(little_law(pts), "perf_little_law.csv")
    write_csv(department(p), "perf_department.csv")
    sess, scap = coding_sessions()
    write_csv(sess, "perf_coding_points.csv")
    write_csv(scap, "perf_coding_capacity.csv")
    front, fsum = frontier(p)
    write_csv(front, "perf_frontier_t12.csv")
    write_csv(fsum, "perf_frontier_summary.csv")
    write_csv(cost(pts, cap, front), "perf_energy_cost.csv")
    write_csv(power(p), "perf_power.csv")
    print("main:", MAIN, "late:", LATE)


if __name__ == "__main__":
    main()
