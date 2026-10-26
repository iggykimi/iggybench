#!/usr/bin/env python3
"""Configuration experiments: speculative decoding, one instance vs two copies, sharing the GPU with other models,
more concurrent sequences, n-gram table placement, run-to-run repeatability, real text vs random tokens, and bursty
arrivals.

Reads results/points.csv, raw/derived/*_resumen_runs.csv (vLLM counters summarised on the server) and the vLLM
/metrics snapshots taken before and after each test block. Writes results/config_*.csv.
"""

import numpy as np
import pandas as pd

from common import MODELS, RAW, RESULTS, SLO, write_csv
from performance import interval


def load():
    p = pd.read_csv(RESULTS / "points.csv", dtype={"run": str, "concurrency": str})
    p["rate"] = pd.to_numeric(p.rate_req_s, errors="coerce")
    p["conc"] = pd.to_numeric(p.concurrency, errors="coerce")
    p["base_profile"] = p.profile.str.split("@").str[0]
    lim = p.base_profile.map(lambda s: SLO.get(s, (np.nan, np.nan)))
    p["meets_slo"] = (p.ttft_p95 <= lim.str[0]) & (p.tpot_p95 <= lim.str[1]) & (p.failed == 0)
    return p


def chat_capacity(p, tag, runs=None):
    x = p[(p.model == tag) & (p.base_profile == "chat") & p.test.isin(["T4", "T4F"])]
    if runs is not None:
        x = x[x.run.isin(runs)]
    return interval(x.rate, x.meets_slo) if len(x) else (np.nan, np.nan)


def t1_speed(p, tag):
    x = p[(p.model == tag) & (p.test == "T1")]
    return 1000 / x.sort_values("run").tpot_p50.iloc[-1] if len(x) else np.nan


def speculative(p):
    """Each method against the same model without it. Capacity compares the same rate grid on both sides."""
    cases = [
        ("MTP (3 tokens)", "qwen38-flash-next", "qwen38-flash-next-nomtp", None),
        ("EAGLE-3", "gptoss120b-eagle3", "gptoss120b", ["20261021-084142", "20261022-001945"]),
        ("DFlash2 (7 tokens)", "qwen27b-fp8-dflash", "qwen27b-fp8", ["20261022-014242"]),
    ]
    rows = []
    for method, var, base, base_runs in cases:
        v_last, v_first = chat_capacity(p, var)
        b_last, b_first = chat_capacity(p, base, base_runs)
        rows.append(
            dict(
                method=method,
                model=var,
                baseline=base,
                single_user_tok_s=t1_speed(p, var),
                baseline_single_user_tok_s=t1_speed(p, base),
                single_user_change_pct=100 * (t1_speed(p, var) / t1_speed(p, base) - 1),
                chat_last_pass_req_s=v_last,
                chat_first_fail_req_s=v_first,
                baseline_chat_last_pass_req_s=b_last,
                baseline_chat_first_fail_req_s=b_first,
            )
        )
    return pd.DataFrame(rows)


def acceptance():
    """Draft-token acceptance (accepted / proposed) from vLLM's own counters, per test block."""
    rows = []
    for f in sorted((RAW / "derived").glob("*_resumen_runs.csv")):
        d = pd.read_csv(f, dtype={"runid": str, "conc": str, "rate": str})
        d = d[d.mtp_acept_pct.notna()]
        for r in d.to_dict("records"):
            rows.append(
                dict(
                    model=r["modelo"],
                    run=r["runid"],
                    test=r["test"],
                    profile=r["perfil"],
                    input_tokens=r["in"],
                    concurrency=r["conc"],
                    rate_req_s=r["rate"],
                    acceptance_pct=r["mtp_acept_pct"],
                    source=f"derived/{f.name}",
                )
            )
    for before in sorted((RAW / "results").glob("*/*/metrics/*_antes.txt"), key=lambda q: q.as_posix()):
        after = before.with_name(before.name.replace("_antes", "_despues"))
        if not after.exists():
            continue
        delta = {k: counter(after, k) - counter(before, k) for k in ("accepted_tokens", "draft_tokens")}
        if delta["draft_tokens"] <= 0:
            continue
        run = before.parents[1]
        rows.append(
            dict(
                model=run.parent.name,
                run=run.name,
                test=before.name.replace("_antes.txt", ""),
                profile="block",
                acceptance_pct=100 * delta["accepted_tokens"] / delta["draft_tokens"],
                source=before.relative_to(RAW).as_posix(),
            )
        )
    # the batch-1 and batch-2 server summaries both list the batch-1 runs
    return pd.DataFrame(rows).drop_duplicates(
        ["model", "run", "test", "profile", "input_tokens", "concurrency", "rate_req_s"]
    )


def counter(path, key):
    name = f"vllm:spec_decode_num_{key}_total"
    total = 0.0
    with open(path, errors="ignore") as fh:
        for line in fh:
            if not line.startswith("#") and line.split("{")[0].split(" ")[0] == name:
                total += float(line.split()[-1])
    return total


def two_copies(p):
    """One gpt-oss instance vs two copies on the same GPU, closed-loop chat (T3) at the same total users."""
    one = p[p.model == "gptoss-x1"].set_index("conc")
    rows = []
    for half in ("gptoss-x2a", "gptoss-x2b"):
        p.loc[p.model == half, "total_users"] = p.loc[p.model == half, "conc"] * 2
    a = p[p.model == "gptoss-x2a"].set_index("total_users")
    b = p[p.model == "gptoss-x2b"].set_index("total_users")
    for users in sorted(one.index):
        if users not in a.index or users not in b.index:
            continue
        two = a.loc[users].output_tok_s + b.loc[users].output_tok_s
        rows.append(
            dict(
                total_users=users,
                one_instance_tok_s=one.loc[users].output_tok_s,
                two_copies_tok_s=two,
                change_pct=100 * (two / one.loc[users].output_tok_s - 1),
                one_instance_tpot_p50_ms=one.loc[users].tpot_p50,
                two_copies_tpot_p50_ms=max(a.loc[users].tpot_p50, b.loc[users].tpot_p50),
                source=one.loc[users].file,
            )
        )
    return pd.DataFrame(rows)


def paired_capacity(p, label, variant, base, base_runs):
    v = p[(p.model == variant) & (p.base_profile == "chat") & p.test.isin(["T4", "T4F"])]
    b = p[(p.model == base) & (p.base_profile == "chat") & p.test.isin(["T4", "T4F"]) & p.run.isin(base_runs)]
    rows = []
    for rate in sorted(set(v.rate) | set(b.rate)):
        vr, br = v[v.rate == rate], b[b.rate == rate]
        rows.append(
            dict(
                experiment=label,
                rate_req_s=rate,
                variant_meets_slo=vr.meets_slo.iloc[-1] if len(vr) else None,
                variant_ttft_p95_ms=vr.ttft_p95.iloc[-1] if len(vr) else np.nan,
                variant_tpot_p95_ms=vr.tpot_p95.iloc[-1] if len(vr) else np.nan,
                baseline_meets_slo=br.meets_slo.iloc[-1] if len(br) else None,
                baseline_ttft_p95_ms=br.ttft_p95.iloc[-1] if len(br) else np.nan,
                baseline_tpot_p95_ms=br.tpot_p95.iloc[-1] if len(br) else np.nan,
                source=(vr.file.iloc[-1] if len(vr) else br.file.iloc[-1]),
            )
        )
    return rows


def variants(p):
    rows = []
    rows += paired_capacity(
        p, "gpt-oss sharing the GPU with two other models", "plataforma-gptoss", "gptoss120b", ["20261022-001945"]
    )
    rows += paired_capacity(
        p,
        "DeepSeek with 256 instead of 128 sequences",
        "dsv4-flash-s256",
        "dsv4-flash",
        ["20261021-195719", "20261022-062829"],
    )
    rows += paired_capacity(
        p,
        "Qwen Flash n-gram table in GPU memory instead of Grace RAM",
        "qwen38-flash-next-ngramgpu",
        "qwen38-flash-next",
        ["20261021-220809", "20261022-073000"],
    )
    return pd.DataFrame(rows)


def repeatability(p):
    x = p[p.test == "T17"].copy()
    x["point"] = x.base_profile + " at " + x.concurrency + " users"
    rows = []
    for (m, point), g in x.groupby(["model", "point"]):
        for col, metric in (("output_tok_s", "output_tok_s"), ("ttft_p95", "ttft_p95_ms"), ("tpot_p95", "tpot_p95_ms")):
            v = g[col].astype(float)
            rows.append(
                dict(
                    model=m,
                    model_name=MODELS.get(m, m),
                    point=point,
                    metric=metric,
                    runs=len(v),
                    mean=v.mean(),
                    sd=v.std(ddof=1),
                    cv_pct=100 * v.std(ddof=1) / v.mean(),
                    values=" / ".join(f"{a:.1f}" for a in v),
                    source=g.file.iloc[0].rsplit("/", 2)[0],
                )
            )
    return pd.DataFrame(rows)


def real_vs_random(p):
    """8k-token prompts of real text (T15, mostly ShareGPT) against random tokens (T12) at the same concurrency."""
    rows = []
    for m in MODELS:
        real = p[(p.model == m) & (p.test == "T15") & (p.profile == "estandar@real")]
        rand = p[(p.model == m) & (p.test == "T12") & p.batch.isin(["batch2", "queue4"])]
        for c in sorted(real.conc.dropna()):
            r, q = real[real.conc == c].iloc[-1], rand[rand.conc == c]
            if q.empty:
                continue
            q = q.iloc[-1]
            rows.append(
                dict(
                    model=m,
                    model_name=MODELS[m],
                    users=c,
                    random_tok_s=q.output_tok_s,
                    real_tok_s=r.output_tok_s,
                    change_pct=100 * (r.output_tok_s / q.output_tok_s - 1),
                    random_tpot_p50_ms=q.tpot_p50,
                    real_tpot_p50_ms=r.tpot_p50,
                    tpot_change_pct=100 * (r.tpot_p50 / q.tpot_p50 - 1),
                    real_input_tokens_mean=r.tok_in / r.completed,
                    source=r.file,
                )
            )
    return pd.DataFrame(rows)


def sharegpt_chat(p):
    """ShareGPT conversations (T15, ~200-token prompts, natural output length) against the closed-loop chat profile
    of T3 (1k random-token prompts, 512 output tokens) at the same concurrency. Not like-for-like: the request shapes
    differ, so this shows direction, not a correction factor."""
    real = p[(p.test == "T15") & (p.profile == "chat@sharegpt")]
    ref = p[(p.test == "T3") & (p.profile == "chat")]
    rows = []
    for r in real.itertuples():
        q = ref[(ref.model == r.model) & (ref.conc == r.conc)].sort_values("run")
        q = q[q.batch == "batch2"] if (q.batch == "batch2").any() else q
        if q.empty:
            continue
        q = q.iloc[-1]
        rows.append(
            dict(
                model=r.model,
                model_name=MODELS[r.model],
                users=r.conc,
                reference_batch=q.batch,
                sharegpt_input_tokens_mean=r.tok_in / r.completed,
                sharegpt_output_tokens_mean=r.tok_out / r.completed,
                t3_tpot_p50_ms=q.tpot_p50,
                sharegpt_tpot_p50_ms=r.tpot_p50,
                tpot_change_pct=100 * (r.tpot_p50 / q.tpot_p50 - 1),
                source=r.file,
            )
        )
    return pd.DataFrame(rows)


def bursts(p):
    """Gamma arrivals with burstiness 0.5 (T18) against Poisson (T4) for gpt-oss at the same rates."""
    b = p[(p.model == "gptoss120b") & (p.test == "T18")]
    q = p[(p.model == "gptoss120b") & (p.test == "T4") & (p.base_profile == "chat") & (p.run == "20261022-001945")]
    rows = []
    for r in b.itertuples():
        m = q[q.rate == r.rate]
        rows.append(
            dict(
                rate_req_s=r.rate,
                bursty_meets_slo=r.meets_slo,
                bursty_ttft_p95_ms=r.ttft_p95,
                bursty_tpot_p95_ms=r.tpot_p95,
                poisson_ttft_p95_ms=m.ttft_p95.iloc[0] if len(m) else np.nan,
                poisson_tpot_p95_ms=m.tpot_p95.iloc[0] if len(m) else np.nan,
                source=r.file,
            )
        )
    return pd.DataFrame(rows)


def main():
    p = load()
    write_csv(speculative(p), "config_speculative.csv")
    write_csv(acceptance(), "config_spec_acceptance.csv")
    write_csv(two_copies(p), "config_two_copies.csv")
    write_csv(variants(p), "config_variants_capacity.csv")
    write_csv(repeatability(p), "config_repeatability.csv")
    write_csv(real_vs_random(p), "config_real_vs_random.csv")
    write_csv(sharegpt_chat(p), "config_sharegpt_chat.csv")
    write_csv(bursts(p), "config_bursts.csv")


if __name__ == "__main__":
    main()
