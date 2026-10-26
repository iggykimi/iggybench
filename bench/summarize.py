#!/usr/bin/env python3
"""
summarize.py — Summarizes the results written by suite.sh (stdlib only).

Usage:  python3 summarize.py [results_dir]      (default ./results next to this script)

Writes:
  resumen_runs.csv   one row per measured point (all metrics)
  informe.txt        findings per model and test (also printed)
Environment: GPU_TZ (time zone of the machine that wrote gpu.csv, if different from this one), CAPEX_EUR,
CAPEX_ANOS (amortization years), PRECIO_KWH (EUR/kWh).
"""

import csv
import glob
import json
import os
import sys
from datetime import datetime

if os.environ.get("GPU_TZ"):  # analysing away from the server: use the time zone of the machine that logged gpu.csv
    import time as _t

    os.environ["TZ"] = os.environ["GPU_TZ"]
    _t.tzset()
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# SLO: (max TTFT p95 in ms, max TPOT p95 in ms) — same as in suite.sh
SLO = {
    "chat": (2000, 50),
    "rag": (5000, 50),
    "agentes": (5000, 40),
    "razonamiento": (None, 40),
    "velocidad": (None, 50),
}
USER_INTERVAL_S = 60  # T4: a real user sends ~1 request per minute


def fnum(x, nd=1):
    if x is None or x == "":
        return "-"
    if isinstance(x, (int, float)):
        return f"{x:,.{nd}f}".replace(",", " ")
    return str(x)


def yn(v):
    return {"SI": "yes", "NO": "no"}.get(v, v)


def load_gpu(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 5:
                continue
            try:
                ts = datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                mod = float(p[6]) if len(p) > 6 and p[6] not in ("[N/A]", "N/A", "") else None
                clk = float(p[5]) if len(p) > 5 and p[5] not in ("[N/A]", "N/A", "") else None
                rows.append((ts, float(p[1]), float(p[2]), float(p[3]), float(p[4]), mod, clk))
            except ValueError:
                continue
    return rows


def gpu_window(gpu, s, e):
    w = [r for r in gpu if s <= r[0] <= e]
    if not w:
        return None, None, None, None
    mods = [r[5] for r in w if r[5] is not None]
    return (
        sum(r[1] for r in w) / len(w),
        max(r[4] for r in w),
        sum(r[2] for r in w) / len(w),
        (sum(mods) / len(mods)) if mods else None,
    )


def clock_window(gpu, s, e):
    c = sorted(r[6] for r in gpu if s <= r[0] <= e and len(r) > 6 and r[6])
    return (c[0], c[len(c) // 2]) if c else (None, None)


CAPEX_EUR = float(os.environ.get("CAPEX_EUR", "140000"))  # machine price
CAPEX_YEARS = float(os.environ.get("CAPEX_ANOS", "3"))  # amortization
CAPEX_EUR_H = CAPEX_EUR / (CAPEX_YEARS * 8760)  # EUR/hour, 24/7 use
# Reference API price ($/Mtok input, output) per model family — Oct 2026, typical providers
API_REF = {
    "gptoss": (0.15, 0.60),
    "qwen27b": (0.04, 1.25),
    "glm53": (0.15, 0.50),
    "dsv4": (0.15, 0.60),
    "dsv41": (0.15, 0.60),
}


def metric_sum(path, key):
    if not os.path.exists(path):
        return None
    tot, found = 0.0, False
    with open(path) as f:
        for line in f:
            if line.startswith("#") or key not in line:
                continue
            name = line.split("{")[0].split(" ")[0]
            if key not in name or name.endswith(("_created", "_bucket", "_sum", "_count")):
                continue
            try:
                tot += float(line.rsplit(" ", 1)[1])
                found = True
            except ValueError:
                pass
    return tot if found else None


def metric_delta(mdir, block, key):
    a = metric_sum(os.path.join(mdir, f"{block}_antes.txt"), key)
    b = metric_sum(os.path.join(mdir, f"{block}_despues.txt"), key)
    return None if a is None or b is None else b - a


PRICE_KWH = float(os.environ.get("PRECIO_KWH", "0.15"))  # EUR/kWh for the cost per million tokens


def load_vm(path):
    """vllm_metrics.csv: 't;key=value;...' every 2 s."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            p = line.strip().split(";")
            try:
                t = float(p[0])
            except ValueError:
                continue
            d = {}
            for kv in p[1:]:
                k, _, v = kv.partition("=")
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
            out.append((t, d))
    return out


def vm_window(vm, s, e):
    before = [x for x in vm if x[0] <= s]
    inside = [x for x in vm if s < x[0] <= e + 3]
    if not inside:
        return {}
    a = (before[-1] if before else inside[0])[1]
    b = inside[-1][1]

    def delta(k):
        return (b[k] - a.get(k, 0.0)) if k in b else None

    kv = [max(x[1].get("kv_cache_usage_perc", 0), x[1].get("gpu_cache_usage_perc", 0)) for x in inside]
    hits, qs = delta("prefix_cache_hits_total"), delta("prefix_cache_queries_total")
    acc, dra = delta("spec_decode_num_accepted_tokens_total"), delta("spec_decode_num_draft_tokens_total")
    return {
        "running_max": max(x[1].get("num_requests_running", 0) for x in inside),
        "cola_max": max(x[1].get("num_requests_waiting", 0) for x in inside),
        "kv_max_pct": 100 * max(kv) if kv else None,
        "tokens_cacheados": hits,
        "cache_hit_pct": (100 * hits / qs) if (hits is not None and qs) else None,
        "preemptions": delta("num_preemptions_total"),
        "mtp_acept_pct": (100 * acc / dra) if (acc is not None and dra) else None,
    }


USE_CASE = {
    "chat": "Internal chat / customer support",
    "rag": "RAG · document lookup",
    "agentes": "Coding and agents (long context + shared prompt)",
    "razonamiento": "Reasoning / long reports",
    "velocidad": "Raw speed",
}

rows = []
blocks = []  # (tag, runid, mdir, block)
quality = {}  # tag -> latest calidad.json (T10, first round only)
vision = {}  # tag -> latest vision.json (T11)
gsm = {}  # tag -> gsm8k.json (T16)
agents = {}  # tag -> latest agentes.json (T13)
sessions = {}  # tag -> {label: sesiones_*.json} (T14)
for runs_csv in sorted(glob.glob(os.path.join(ROOT, "*", "*", "runs.csv"))):
    rdir = os.path.dirname(runs_csv)
    gpu = load_gpu(os.path.join(rdir, "gpu.csv"))
    cq = os.path.join(rdir, "calidad.json")
    if os.path.exists(cq):
        try:
            quality[rdir.split(os.sep)[-2]] = json.load(open(cq))
        except Exception:
            pass
    for sf in glob.glob(os.path.join(rdir, "sesiones_*.json")):
        try:
            sj = json.load(open(sf))
            sessions.setdefault(rdir.split(os.sep)[-2], {})[sj.get("etiqueta") or "base"] = sj
        except Exception:
            pass
    aq = os.path.join(rdir, "agentes.json")
    if os.path.exists(aq):
        try:
            agents[rdir.split(os.sep)[-2]] = json.load(open(aq))
        except Exception:
            pass
    gq = os.path.join(rdir, "gsm8k.json")
    if os.path.exists(gq):
        try:
            gsm[rdir.split(os.sep)[-2]] = json.load(open(gq))
        except Exception:
            pass
    vq = os.path.join(rdir, "vision.json")
    if os.path.exists(vq):
        try:
            vision[rdir.split(os.sep)[-2]] = json.load(open(vq))
        except Exception:
            pass
    vm = load_vm(os.path.join(rdir, "vllm_metrics.csv"))
    mdir = os.path.join(rdir, "metrics")
    for f in glob.glob(os.path.join(mdir, "*_antes.txt")):
        b = os.path.basename(f)[: -len("_antes.txt")]
        tag, runid = rdir.split(os.sep)[-2:]
        blocks.append((tag, runid, mdir, b))
    with open(runs_csv) as f:
        for r in csv.DictReader(f):
            jp = os.path.join(rdir, "json", r["file"])
            d = {}
            if os.path.exists(jp):
                with open(jp) as jf:
                    d = json.load(jf)
            n = int(r["n"])
            done = d.get("completed") or 0
            s, e = float(r["start"]), float(r["end"])
            pw, tmax, util, mw = gpu_window(gpu, s, e)
            p50_tpot, p95_tpot = d.get("p50_tpot_ms") or d.get("median_tpot_ms"), d.get("p95_tpot_ms")
            p95_ttft = d.get("p95_ttft_ms")
            row = {
                "modelo": r["tag"],
                "runid": r["runid"],
                "test": r["test"],
                "perfil": r["profile"],
                "in": int(r["in"]) + int(r["prefix"]),
                "out": int(r["out"]),
                "prefijo": int(r["prefix"]),
                "conc": r["conc"],
                "rate": r["rate"],
                "n": n,
                "ok": done,
                "fallos": n - done,
                "rc": r["rc"],
                "duracion_s": d.get("duration"),
                "req_s": d.get("request_throughput"),
                "goodput_req_s": d.get("request_goodput"),
                "out_tok_s": d.get("output_throughput"),
                "total_tok_s": d.get("total_token_throughput"),
                "ttft_p50": d.get("p50_ttft_ms") or d.get("median_ttft_ms"),
                "ttft_p95": p95_ttft,
                "ttft_p99": d.get("p99_ttft_ms"),
                "tpot_p50": p50_tpot,
                "tpot_p95": p95_tpot,
                "usuario_tok_s_p50": (1000 / p50_tpot) if p50_tpot else None,
                "usuario_tok_s_p95": (1000 / p95_tpot) if p95_tpot else None,
                "e2e_p95_ms": d.get("p95_e2el_ms"),
                "gpu_w_media": pw,
                "modulo_w_media": mw,
                "gpu_temp_max": tmax,
                "gpu_util_media": util,
                "out_tok_s_por_w": (d.get("output_throughput") / pw) if (pw and d.get("output_throughput")) else None,
            }
            # prefill (T2/T5): input tokens / median TTFT
            if row["ttft_p50"] and r["test"] in ("T2", "T5"):
                row["prefill_tok_s"] = row["in"] / (row["ttft_p50"] / 1000)
            else:
                row["prefill_tok_s"] = None
            # does it meet the SLO?
            slo = SLO.get(r["profile"].split("@")[0])
            if slo and done and row["fallos"] == 0:
                t_ok = slo[0] is None or (p95_ttft is not None and p95_ttft <= slo[0])
                p_ok = slo[1] is None or (p95_tpot is not None and p95_tpot <= slo[1])
                row["cumple_slo"] = "SI" if (t_ok and p_ok) else "NO"
            else:
                row["cumple_slo"] = "-" if not slo else "NO"
            if row["goodput_req_s"] is not None and row["req_s"]:
                row["goodput_pct"] = 100 * row["goodput_req_s"] / row["req_s"]
            else:
                row["goodput_pct"] = None
            # server metrics during this point (queue, KV, prefix cache, MTP)
            w = vm_window(vm, s, e)
            for k in (
                "running_max",
                "cola_max",
                "kv_max_pct",
                "tokens_cacheados",
                "cache_hit_pct",
                "preemptions",
                "mtp_acept_pct",
            ):
                row[k] = w.get(k)
            # energy and cost per million output tokens (module power if available)
            pot = mw or pw
            out_tok, dur = d.get("total_output_tokens"), d.get("duration")
            if pot and out_tok and dur:
                wh_mtok = pot * dur / 3600 / out_tok * 1e6
                row["wh_por_mtok"] = wh_mtok
                row["eur_por_mtok"] = wh_mtok / 1000 * PRICE_KWH
            else:
                row["wh_por_mtok"] = row["eur_por_mtok"] = None
            row["itl_p50"], row["itl_p99"] = d.get("p50_itl_ms") or d.get("median_itl_ms"), d.get("p99_itl_ms")
            row["sm_mhz_min"], row["sm_mhz_med"] = clock_window(gpu, s, e)
            tt = d.get("total_token_throughput")
            # total cost per million tokens (input+output) with 24/7 amortization + electricity
            row["eur_mtok_total"] = ((CAPEX_EUR_H + (pot or 0) / 1000 * PRICE_KWH) / (tt * 3600 / 1e6)) if tt else None
            rows.append(row)

if not rows:
    print(f"No results in {ROOT}")
    sys.exit(0)

out_csv = os.path.join(ROOT, "resumen_runs.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

L = []
P = L.append


def table(hdr, data):
    widths = (
        [max(len(str(h)), *(len(str(x[i])) for x in data)) for i, h in enumerate(hdr)]
        if data
        else [len(h) for h in hdr]
    )
    P("  " + "  ".join(str(h).rjust(widths[i]) for i, h in enumerate(hdr)))
    for x in data:
        P("  " + "  ".join(str(v).rjust(widths[i]) for i, v in enumerate(x)))


def cnum(c):
    try:
        return int(c)
    except ValueError:
        return 10**9


for tag in sorted({r["modelo"] for r in rows}):
    R = [r for r in rows if r["modelo"] == tag and r["rc"] == "0"]
    P("=" * 78)
    P(f" MODEL: {tag}")
    P("=" * 78)

    t1 = [r for r in R if r["test"] == "T1"]
    if t1:
        r = t1[-1]
        P(
            f"\n[T1] Raw speed (1 user): {fnum(r['usuario_tok_s_p50'])} tok/s  "
            f"(TTFT {fnum(r['ttft_p50'], 0)} ms, GPU {fnum(r['gpu_w_media'], 0)} W)"
        )

    t2 = sorted([r for r in R if r["test"] == "T2"], key=lambda r: r["in"])
    if t2:
        P("\n[T2] Prefill curve (1 user)")
        table(
            ["input", "TTFT p50 ms", "prefill tok/s", "≈ pages*"],
            [[r["in"], fnum(r["ttft_p50"], 0), fnum(r["prefill_tok_s"], 0), fnum(r["in"] / 500, 0)] for r in t2],
        )
        P("  * ~500 tokens per page of text")

    for test, title in (
        ("T3", "Concurrency to the limit"),
        ("T6", "MTP (compare with T3/T1 of the model without MTP)"),
    ):
        for profile in ("velocidad", "chat", "rag", "agentes", "razonamiento"):
            S = sorted([r for r in R if r["test"] == test and r["perfil"] == profile], key=lambda r: cnum(r["conc"]))
            if not S:
                continue
            P(f"\n[{test}] {title} · {USE_CASE.get(profile, profile)} (in {S[0]['in']} / out {S[0]['out']})")
            table(
                [
                    "conc",
                    "tok/s total",
                    "tok/s output",
                    "tok/s/user",
                    "TTFT p95 ms",
                    "TPOT p95 ms",
                    "goodput %",
                    "running",
                    "queue",
                    "KV %",
                    "cache %",
                    "module W",
                    "€/Mtok",
                    "SLO",
                ],
                [
                    [
                        r["conc"],
                        fnum(r["total_tok_s"], 0),
                        fnum(r["out_tok_s"], 0),
                        fnum(r["usuario_tok_s_p50"]),
                        fnum(r["ttft_p95"], 0),
                        fnum(r["tpot_p95"], 1),
                        fnum(r["goodput_pct"], 0),
                        fnum(r["running_max"], 0),
                        fnum(r["cola_max"], 0),
                        fnum(r["kv_max_pct"], 0),
                        fnum(r["cache_hit_pct"], 0),
                        fnum(r["modulo_w_media"] or r["gpu_w_media"], 0),
                        fnum(r["eur_por_mtok"], 3),
                        yn(r["cumple_slo"]),
                    ]
                    for r in S
                ],
            )
            if profile == "agentes":
                ct = sum(r["tokens_cacheados"] or 0 for r in S)
                P(f"  · Tokens served from the prefix cache in this profile: {fnum(ct, 0)}")
            okc = [cnum(r["conc"]) for r in S if r["cumple_slo"] == "SI"]
            best = max(S, key=lambda r: r["total_tok_s"] or 0)
            if SLO.get(profile):
                P(
                    f"  → Max concurrency within SLO: {max(okc) if okc else 'none'}"
                    f"   · Max throughput: {fnum(best['total_tok_s'], 0)} tok/s (conc {best['conc']})"
                )

    for profile in ("chat", "agentes"):
        S = sorted([r for r in R if r["test"] == "T4" and r["perfil"] == profile], key=lambda r: float(r["rate"]))
        if not S:
            continue
        P(f"\n[T4] Realistic traffic (random arrivals) · {USE_CASE.get(profile, profile)}")
        table(
            [
                "req/s offered",
                "req/s served",
                "goodput %",
                "TTFT p95 ms",
                "TPOT p95 ms",
                "tok/s/user",
                "max running",
                "max queue",
                "max KV %",
                "≈ real users*",
            ],
            [
                [
                    r["rate"],
                    fnum(r["req_s"], 2),
                    fnum(r["goodput_pct"], 0),
                    fnum(r["ttft_p95"], 0),
                    fnum(r["tpot_p95"], 1),
                    fnum(r["usuario_tok_s_p50"]),
                    fnum(r["running_max"], 0),
                    fnum(r["cola_max"], 0),
                    fnum(r["kv_max_pct"], 0),
                    fnum(float(r["rate"]) * USER_INTERVAL_S, 0),
                ]
                for r in S
            ],
        )
        good = [float(r["rate"]) for r in S if (r["goodput_pct"] or 0) >= 95]
        if good:
            P(
                f"  → Max rate with ≥95% of requests within SLO: {max(good)} req/s "
                f"≈ {int(max(good) * USER_INTERVAL_S)} active users (1 request/min each)"
            )
        else:
            P("  → No point reaches 95% goodput")
        P(f"  * assuming each user sends one request every {USER_INTERVAL_S} s")

    t5 = sorted([r for r in R if r["test"] == "T5"], key=lambda r: (cnum(r["conc"]), r["in"]))
    if t5:
        P("\n[T5] Extreme context")
        table(
            ["input", "conc", "TTFT p50 s", "prefill tok/s", "tok/s/user p50", "failures"],
            [
                [
                    r["in"],
                    r["conc"],
                    fnum((r["ttft_p50"] or 0) / 1000, 1),
                    fnum(r["prefill_tok_s"], 0),
                    fnum(r["usuario_tok_s_p50"]),
                    r["fallos"],
                ]
                for r in t5
            ],
        )

    t9 = [r for r in R if r["test"] == "T9"]
    if t9:
        P("\n[T9] Real department · chat + RAG + agents at the same time")
        P("     (per person: 60 % chat 1 question/min · 25 % RAG 1 query/2 min · 15 % coding with agents)")
        data = []
        for n in sorted({int(r["perfil"].split("@")[1]) for r in t9}):
            g = {r["perfil"].split("@")[0]: r for r in t9 if r["perfil"].endswith(f"@{n}")}
            c, rg, a = g.get("chat"), g.get("rag"), g.get("agentes")
            all_ok = all(x and x["cumple_slo"] == "SI" for x in (c, rg, a))
            data.append(
                [
                    n,
                    fnum(c and c["ttft_p95"], 0),
                    fnum(c and c["usuario_tok_s_p50"]),
                    fnum(rg and rg["ttft_p95"], 0),
                    fnum(a and a["ttft_p95"], 0),
                    fnum(a and a["usuario_tok_s_p50"]),
                    "yes" if all_ok else "no",
                ]
            )
        table(
            [
                "people",
                "chat TTFT p95",
                "chat tok/s",
                "RAG TTFT p95",
                "agents TTFT p95",
                "agents tok/s",
                "all within SLO",
            ],
            data,
        )
        oks = [d[0] for d in data if d[-1] == "yes"]
        P(
            f"  → Largest department with a good experience for everyone: {max(oks) if oks else 'none of those tested'} people"
        )

    cq = quality.get(tag)
    if cq:
        P(f"\n[T10] Quick quality check: {cq['aciertos']}/{cq['total']}")
        table(
            ["test", "use case", "ok", "s", "tokens"],
            [
                [x["id"], x["caso"], "yes" if x["ok"] else "no", fnum(x["segundos"]), fnum(x["tokens_salida"], 0)]
                for x in cq["pruebas"]
            ],
        )

    t12 = sorted([r for r in R if r["test"] == "T12"], key=lambda r: cnum(r["conc"]))
    if t12:
        P("\n[T12] DGX Station standard test (8192 input / 1024 output) · comparable with catid/dgx_station_benchmarks")
        table(
            [
                "conc",
                "total output tok/s",
                "tok/s/user",
                "TTFT p50 ms",
                "TTFT p95 ms",
                "TPOT p50 ms",
                "KV %",
                "module W",
            ],
            [
                [
                    r["conc"],
                    fnum(r["out_tok_s"], 0),
                    fnum(r["usuario_tok_s_p50"]),
                    fnum(r["ttft_p50"], 0),
                    fnum(r["ttft_p95"], 0),
                    fnum(r["tpot_p50"], 1),
                    fnum(r["kv_max_pct"], 0),
                    fnum(r["modulo_w_media"], 0),
                ]
                for r in t12
            ],
        )

    for lbl, sq in sorted(sessions.get(tag, {}).items()):
        P(
            f"\n[T14] Long coding sessions ({lbl}) · context 48k → ~100k, ~1k new tokens per turn, {sq['turnos_por_sesion']} turns"
        )
        table(
            [
                "sessions",
                "TTFT p50 s",
                "TTFT p95 s",
                "TTFT 1st turn s",
                "cache %",
                "tok/s per session",
                "mean context",
                "turns/min",
                "errors",
            ],
            [
                [
                    n["sesiones"],
                    fnum(n["ttft_p50"]),
                    fnum(n["ttft_p95"]),
                    fnum(n["ttft_primer_turno_p50"]),
                    fnum(n["cache_pct"]),
                    fnum(n["dec_tok_s_p50"]),
                    fnum(n["contexto_medio"], 0),
                    fnum(n["turnos_min"]),
                    n["errores"],
                ]
                for n in sq["niveles"]
            ],
        )
        okn = [
            n["sesiones"]
            for n in sq["niveles"]
            if (n["ttft_p95"] or 99) <= 8 and (n["dec_tok_s_p50"] or 0) >= 30 and n["errores"] == 0
        ]
        P(
            f"  → Concurrent developers with an agent and a good experience (TTFT p95 ≤ 8 s and ≥ 30 tok/s): {max(okn) if okn else 'none of those tested'}"
        )

    aq = agents.get(tag)
    if aq:
        P(f"\n[T13] Tool-using agents ({aq['tareas']} multi-step back-office tasks, checked automatically)")
        table(
            ["agents at once", "success %", "tasks/min", "s/task p50", "s/task p95", "tool calls/task", "errors"],
            [
                [
                    n["conc"],
                    fnum(n["aciertos_pct"]),
                    fnum(n["tareas_min"]),
                    fnum(n["seg_tarea_p50"]),
                    fnum(n["seg_tarea_p95"]),
                    fnum(n["herramientas_media"]),
                    n["errores"],
                ]
                for n in aq["niveles"]
            ],
        )
        P("  Per task (1 agent): " + ", ".join(f"{k} {v}" for k, v in aq["niveles"][0]["por_tarea"].items()))

    vq = vision.get(tag)
    if vq:
        P(f"\n[T11] Reading scanned invoices with vision ({vq['documentos']} different invoices, 8 fields)")
        table(
            [
                "conc",
                "pages/min",
                "lat p50 s",
                "lat p95 s",
                "fields OK %",
                "perfect invoices %",
                "input tokens",
                "failures",
            ],
            [
                [
                    n["conc"],
                    fnum(n["paginas_min"]),
                    fnum(n["lat_p50_s"]),
                    fnum(n["lat_p95_s"]),
                    fnum(n["acierto_campos_pct"]),
                    fnum(n["facturas_perfectas_pct"]),
                    fnum(n["in_tok_media"], 0),
                    n["fallos"],
                ]
                for n in vq["niveles"]
            ],
        )
        best = max(vq["niveles"], key=lambda n: n["paginas_min"])
        P(
            f"  → Max: {fnum(best['paginas_min'])} pages/min (≈ {fnum(best['paginas_min'] * 60 * 8 / 1000)} thousand pages in an 8 h shift)"
        )
        P(
            "  Accuracy per field (conc 1): "
            + ", ".join(f"{k} {v:.0f}%" for k, v in vq["niveles"][0]["acierto_por_campo"].items())
        )

    t15 = [r for r in R if r["test"] == "T15"]
    if t15:
        P("\n[T15] Check with REAL text against random text (bias of synthetic benchmarks)")
        t12 = {cnum(r["conc"]): r for r in R if r["test"] == "T12"}
        data = []
        for r in sorted([x for x in t15 if x["perfil"] == "estandar@real"], key=lambda r: cnum(r["conc"])):
            b = t12.get(cnum(r["conc"]))
            diff = (
                f"{100 * (r['out_tok_s'] / b['out_tok_s'] - 1):+.0f} %"
                if b and b["out_tok_s"] and r["out_tok_s"]
                else "-"
            )
            data.append(
                [
                    "SPEED-Bench 8k",
                    r["conc"],
                    fnum(b["out_tok_s"], 0) if b else "-",
                    fnum(r["out_tok_s"], 0),
                    diff,
                    fnum(r["ttft_p95"], 0),
                    fnum(r["itl_p99"], 0),
                ]
            )
        t3c = {cnum(r["conc"]): r for r in R if r["test"] == "T3" and r["perfil"] == "chat"}
        for r in sorted([x for x in t15 if x["perfil"] == "chat@sharegpt"], key=lambda r: cnum(r["conc"])):
            b = t3c.get(cnum(r["conc"]))
            data.append(
                [
                    "ShareGPT chat",
                    r["conc"],
                    fnum(b["out_tok_s"], 0) if b else "-",
                    fnum(r["out_tok_s"], 0),
                    "(different lengths)",
                    fnum(r["ttft_p95"], 0),
                    fnum(r["itl_p99"], 0),
                ]
            )
        table(["real data", "conc", "tok/s random", "tok/s real", "difference", "TTFT p95 ms", "ITL p99 ms"], data)
        sh = [r for r in t15 if r["perfil"] == "chat@sharegpt" and r["cumple_slo"] == "SI"]
        if sh:
            P(f"  → Chat with real conversations: up to {max(cnum(r['conc']) for r in sh)} concurrent within SLO")

    t16 = gsm.get(tag)
    if t16:
        ic = t16.get("ic95_pct")
        ic = f" (95 % CI {ic[0]}-{ic[1]})" if ic and ic[0] is not None else ""
        a2 = f" · first 200: {t16['acierto_200_pct']} %" if "acierto_200_pct" in t16 else ""
        P(
            f"\n[T16] GSM8K-{t16['preguntas']}: {t16['acierto_pct']} % correct{ic}{a2} (DGX Station community: ~97-98 % for top models)"
        )

    t17 = [r for r in R if r["test"] == "T17"]
    if t17:
        P("\n[T17] Repeatability (3 repetitions of the same point, conc 64)")
        grp = {}
        for r in t17:
            grp.setdefault(r["perfil"].split("@")[0], []).append(r["out_tok_s"])
        for k, v in grp.items():
            v = [x for x in v if x]
            if len(v) >= 2:
                m = sum(v) / len(v)
                sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
                P(
                    f"  {k}: {fnum(m, 0)} tok/s mean ± {fnum(sd, 0)} (variation {100 * sd / m:.1f} %) · values: {', '.join(fnum(x, 0) for x in v)}"
                )

    t18 = sorted([r for r in R if r["test"] == "T18"], key=lambda r: float(r["rate"]))
    if t18:
        P("\n[T18] Chat with BURSTY arrivals (burstiness 0.5) · compare with T4 (Poisson)")
        t4c = {float(r["rate"]): r for r in R if r["test"] == "T4" and r["perfil"] == "chat"}
        table(
            [
                "req/s",
                "goodput % bursty",
                "goodput % Poisson",
                "TTFT p95 ms bursty",
                "TTFT p95 ms Poisson",
                "ITL p99 ms",
            ],
            [
                [
                    r["rate"],
                    fnum(r["goodput_pct"], 0),
                    fnum(t4c[float(r["rate"])]["goodput_pct"], 0) if float(r["rate"]) in t4c else "-",
                    fnum(r["ttft_p95"], 0),
                    fnum(t4c[float(r["rate"])]["ttft_p95"], 0) if float(r["rate"]) in t4c else "-",
                    fnum(r["itl_p99"], 0),
                ]
                for r in t18
            ],
        )

    t19 = [r for r in R if r["test"] == "T19"]
    if t19:
        P("\n[T19] Steady state: 10 min in a row at the highest chat rate that met the SLO in T4")
        t4c = {float(r["rate"]): r for r in R if r["test"] == "T4" and r["perfil"] == "chat"}
        table(
            [
                "req/s",
                "duration s",
                "meets SLO",
                "goodput %",
                "TTFT p95 ms (T4 90 s)",
                "TPOT p95 ms",
                "ITL p99 ms",
                "mean GPU W",
                "min SM clock MHz",
            ],
            [
                [
                    r["rate"],
                    fnum(r.get("duracion_s"), 0),
                    yn(r["cumple_slo"]),
                    fnum(r["goodput_pct"], 0),
                    f"{fnum(r['ttft_p95'], 0)} ({fnum(t4c[float(r['rate'])]['ttft_p95'], 0) if float(r['rate']) in t4c else '-'})",
                    fnum(r.get("tpot_p95"), 0),
                    fnum(r["itl_p99"], 0),
                    fnum(r.get("gpu_w_media"), 0),
                    fnum(r.get("sm_mhz_min"), 0),
                ]
                for r in t19
            ],
        )

    clk = [r["sm_mhz_min"] for r in R if r.get("sm_mhz_min")]
    med = [r["sm_mhz_med"] for r in R if r.get("sm_mhz_med")]
    if clk and med:
        P(
            f"\n[GPU] SM clock: median {fnum(sorted(med)[len(med) // 2], 0)} MHz · lowest seen {fnum(min(clk), 0)} MHz"
            + ("  ⚠ possible power/thermal throttling" if min(clk) < 0.8 * sorted(med)[len(med) // 2] else "")
        )

    t7 = [r for r in R if r["test"] == "T7"]
    if t7:
        P("\n[T7] Prefix cache · 16k input: no shared prefix vs 8k of shared prompt")
        data = []
        for c in sorted({r["conc"] for r in t7}, key=cnum):
            without = next((r for r in t7 if r["conc"] == c and r["perfil"] == "sin_prefijo"), None)
            with_ = next((r for r in t7 if r["conc"] == c and r["perfil"] == "con_prefijo"), None)
            if not (without and with_):
                continue
            gain = (
                (100 * (1 - with_["ttft_p50"] / without["ttft_p50"]))
                if (without["ttft_p50"] and with_["ttft_p50"])
                else None
            )
            data.append(
                [
                    c,
                    fnum(without["ttft_p50"], 0),
                    fnum(with_["ttft_p50"], 0),
                    fnum(gain, 0),
                    fnum(without["total_tok_s"], 0),
                    fnum(with_["total_tok_s"], 0),
                    fnum(with_["cache_hit_pct"], 0),
                    fnum(with_["tokens_cacheados"], 0),
                ]
            )
        table(
            [
                "conc",
                "TTFT without ms",
                "TTFT with ms",
                "TTFT -%",
                "tok/s without",
                "tok/s with",
                "cache %",
                "cached tokens",
            ],
            data,
        )

    t8 = [r for r in R if r["test"] == "T8"]
    if t8:
        P(f"\n[T8] KV offloading · repeated documents of {t8[0]['prefijo']} tokens (more than fit on the GPU)")
        table(
            [
                "config",
                "requests",
                "TTFT p50 ms",
                "TTFT p95 ms",
                "tok/s total",
                "cache %",
                "cached tokens",
                "duration s",
            ],
            [
                [
                    r["perfil"],
                    r["ok"],
                    fnum(r["ttft_p50"], 0),
                    fnum(r["ttft_p95"], 0),
                    fnum(r["total_tok_s"], 0),
                    fnum(r["cache_hit_pct"], 0),
                    fnum(r["tokens_cacheados"], 0),
                    fnum(r["duracion_s"], 0),
                ]
                for r in t8
            ],
        )
        b = next((r for r in t8 if r["perfil"] == "base"), None)
        o = next((r for r in t8 if r["perfil"] == "offload"), None)
        if b and o and b["ttft_p50"] and o["ttft_p50"] and b["total_tok_s"] and o["total_tok_s"]:
            P(
                f"  → With offload to RAM: TTFT p50 {100 * (o['ttft_p50'] / b['ttft_p50'] - 1):+.0f} %, "
                f"throughput {100 * (o['total_tok_s'] / b['total_tok_s'] - 1):+.0f} %"
            )

    # summary of cache, queue and preemptions over all points
    tot_cache = sum(r["tokens_cacheados"] or 0 for r in R)
    tot_pre = sum(r["preemptions"] or 0 for r in R)
    queue = max((r["cola_max"] or 0) for r in R) if R else 0
    kvm = max((r["kv_max_pct"] or 0) for r in R) if R else 0
    P(
        f"\n[Server] Tokens served from cache: {fnum(tot_cache, 0)} · preemptions: {fnum(tot_pre, 0)}"
        f" · max queue: {fnum(queue, 0)} · max KV usage: {fnum(kvm, 0)} %"
        f" · cost computed at {PRICE_KWH} €/kWh"
    )

    # server metrics per block
    BL = [b for b in blocks if b[0] == tag]
    lines = []
    for _, runid, mdir, blk in sorted(BL, key=lambda b: (b[1], b[3])):
        hits = metric_delta(mdir, blk, "prefix_cache_hits")
        qs = metric_delta(mdir, blk, "prefix_cache_queries")
        pre = metric_delta(mdir, blk, "num_preemptions")
        acc = metric_delta(mdir, blk, "spec_decode_num_accepted_tokens")
        dra = metric_delta(mdir, blk, "spec_decode_num_draft_tokens")
        lines.append(
            [
                blk,
                fnum(100 * hits / qs, 1) if (hits is not None and qs) else "-",
                fnum(pre, 0) if pre is not None else "-",
                fnum(100 * acc / dra, 1) if (acc is not None and dra) else "-",
            ]
        )
    if lines:
        P("\n[Server] vLLM metrics per block")
        table(["block", "prefix cache hit %", "preemptions", "MTP acceptance %"], lines)

    failed = [r for r in rows if r["modelo"] == tag and r["rc"] != "0"]
    if failed:
        P(
            f"\n[!] {len(failed)} points failed (see logs/): "
            + ", ".join(f"{r['test']}/{r['perfil']}/c{r['conc']}" for r in failed[:10])
        )
    P("")


# --------------------------------------------------------------------------------
# Final comparison per use case (one row per model)
def max_slo(R, test, profile):
    ok = [cnum(r["conc"]) for r in R if r["test"] == test and r["perfil"] == profile and r["cumple_slo"] == "SI"]
    return max(ok) if ok else None


def api_ref(tag):
    k = next((k for k in API_REF if tag.replace("-", "").startswith(k)), None)
    if not k:
        return "-"
    i, o = API_REF[k]
    return f"{(2 * i + o) / 3:.2f} $"  # typical chat mix: 2 input tokens per output token


comp = []
for tag in sorted({r["modelo"] for r in rows}):
    R = [r for r in rows if r["modelo"] == tag and r["rc"] == "0"]
    t1 = [r for r in R if r["test"] == "T1"]
    pre = [r["prefill_tok_s"] for r in R if r["test"] == "T2" and r["in"] >= 16384 and r["prefill_tok_s"]]
    t4 = [float(r["rate"]) for r in R if r["test"] == "T4" and r["perfil"] == "chat" and (r["goodput_pct"] or 0) >= 95]
    ag = max_slo(R, "T3", "agentes")
    t9 = {}
    for r in R:
        if r["test"] == "T9":
            n = int(r["perfil"].split("@")[1])
            t9.setdefault(n, []).append(r["cumple_slo"] == "SI")
    dep = [n for n, v in t9.items() if len(v) == 3 and all(v)]
    chat_ok = [
        r for r in R if r["test"] == "T3" and r["perfil"] == "chat" and r["cumple_slo"] == "SI" and r["eur_por_mtok"]
    ]
    cq = quality.get(tag)
    comp.append(
        [
            tag,
            fnum(t1[-1]["usuario_tok_s_p50"], 0) if t1 else "-",
            fnum(sorted(pre)[len(pre) // 2] / 500, 0) if pre else "-",
            max_slo(R, "T3", "chat") or "-",
            f"~{int(max(t4) * 60)}" if t4 else "-",
            max_slo(R, "T3", "rag") or "-",
            f"{max(1, ag // 2)}-{ag}" if ag else "-",
            max(dep) if dep else "-",
            fnum(min(r["eur_por_mtok"] for r in chat_ok), 3) if chat_ok else "-",
            fnum(min(r["eur_mtok_total"] for r in chat_ok if r.get("eur_mtok_total")), 3)
            if any(r.get("eur_mtok_total") for r in chat_ok)
            else "-",
            api_ref(tag),
            f"{cq['aciertos']}/{cq['total']}" if cq else "-",
            (
                f"{gsm[tag]['acierto_pct']} %"
                + (
                    f" ±{(gsm[tag]['ic95_pct'][1] - gsm[tag]['ic95_pct'][0]) / 2:.1f}"
                    if gsm[tag].get("ic95_pct") and gsm[tag]["ic95_pct"][0] is not None
                    else ""
                )
            )
            if tag in gsm
            else "-",
        ]
    )

if comp:
    P("=" * 78)
    P(" COMPARISON PER USE CASE")
    P("=" * 78)
    table(
        [
            "model",
            "tok/s 1 user",
            "pages/s read",
            "chat: concurrent",
            "chat: active users*",
            "RAG: concurrent",
            "active developers**",
            "department (people)",
            "€/Mtok power",
            "€/Mtok total***",
            "API $/Mtok****",
            "quality",
            "GSM8K",
        ],
        comp,
    )
    P(
        f"  *** amortization of {CAPEX_EUR:,.0f} € over {CAPEX_YEARS:g} years used 24/7 ({CAPEX_EUR_H:.2f} €/h) + electricity, at the best chat point within SLO."
        " During office hours only (~176 h/month) multiply the amortization by ~4."
    )
    P("  **** typical API price of that model (Oct 2026), 2:1 input:output mix — reference, not an invoice")
    P("  * sustained requests/s with ≥95 % within SLO × 60 (each user asks ~1 question per minute)")
    P("  ** concurrent agents within SLO; each active developer generates 1-2 requests at a time")
    P(
        "  Concurrent = highest concurrency that meets the profile's SLO (chat TTFT ≤2 s and ≥20 tok/s; "
        "RAG TTFT ≤5 s; agents TTFT ≤5 s and ≥25 tok/s)"
    )

# Standard 8k/1k test: one row per model (aggregate output tok/s per concurrency level)
std = []
for tag in sorted({r["modelo"] for r in rows}):
    t12 = {cnum(r["conc"]): r for r in rows if r["modelo"] == tag and r["test"] == "T12" and r["rc"] == "0"}
    if t12:
        std.append([tag] + [fnum(t12[c]["out_tok_s"], 0) if c in t12 else "-" for c in (1, 8, 16, 32, 64, 128)])
if std:
    P("=" * 78)
    P(" DGX STATION STANDARD TEST · 8k input / 1k output · aggregate output tok/s")
    P("=" * 78)
    table(["model", "C1", "C8", "C16", "C32", "C64", "C128"], std)
    P(
        "  Public references (same machine, different software stack): Qwen3.8-27B FP8 C128 ≈ 5 494 · BF16+MTP C128 ≈ 6 349 ·"
    )
    P(
        "  NVFP4A16 C128 ≈ 3 607 (catid) · MiniMax M3 NVFP4 C1 ≈ 153, C16 ≈ 1 596 (catid) · Qwen3.8-Flash-Next MTP C64 ≈ 5 247 (fizban007)"
    )

# Long coding sessions: one row per model and variant
ses = []
for tag, d in sorted(sessions.items()):
    for lbl, sq in sorted(d.items()):
        okn = [
            n
            for n in sq["niveles"]
            if (n["ttft_p95"] or 99) <= 8 and (n["dec_tok_s_p50"] or 0) >= 30 and n["errores"] == 0
        ]
        b = max(okn, key=lambda n: n["sesiones"]) if okn else None
        ses.append(
            [
                tag,
                lbl,
                b["sesiones"] if b else "-",
                fnum(b["ttft_p95"]) if b else "-",
                fnum(b["cache_pct"]) if b else "-",
                fnum(b["dec_tok_s_p50"]) if b else "-",
                fnum(max(n["turnos_min"] for n in sq["niveles"])),
            ]
        )
if ses:
    P("=" * 78)
    P(" LONG CODING SESSIONS (T14) · how many developers with an agent at once")
    P("=" * 78)
    table(["model", "variant", "sessions OK", "TTFT p95 s", "cache %", "tok/s per session", "max turns/min"], ses)
    P("  OK = TTFT p95 ≤ 8 s on later turns and ≥ 30 tok/s per session (real agents: median TTFT ~2 s, ~35-60 tok/s)")

# Tool-using agents: one row per model
ag = []
for tag, aq in sorted(agents.items()):
    n1 = aq["niveles"][0]
    mx = max(aq["niveles"], key=lambda n: n["tareas_min"])
    ag.append(
        [
            tag,
            fnum(n1["aciertos_pct"]),
            fnum(n1["seg_tarea_p50"]),
            fnum(mx["tareas_min"]),
            mx["conc"],
            fnum(mx["aciertos_pct"]),
        ]
    )
if ag:
    P("=" * 78)
    P(" TOOL-USING AGENTS (T13)")
    P("=" * 78)
    table(
        ["model", "success % (1 agent)", "s/task (1 agent)", "max tasks/min", "agents at once", "success % under load"],
        ag,
    )

# Documents with vision: every vision.json (including the platform test)
vis = []
for vf in sorted(glob.glob(os.path.join(ROOT, "*", "*", "vision.json"))):
    try:
        v = json.load(open(vf))
        m = max(v["niveles"], key=lambda n: n["paginas_min"])
        vis.append(
            [
                vf.split(os.sep)[-3],
                m["conc"],
                fnum(m["paginas_min"]),
                fnum(m["lat_p50_s"]),
                fnum(v["niveles"][0]["acierto_campos_pct"]),
                fnum(v["niveles"][0]["facturas_perfectas_pct"]),
            ]
        )
    except Exception:
        pass
if vis:
    P("=" * 78)
    P(" DOCUMENTS WITH VISION (scanned invoices → JSON)")
    P("=" * 78)
    table(
        [
            "model / test",
            "conc. at max",
            "max pages/min",
            "latency p50 s",
            "fields OK % (1 user)",
            "perfect invoices %",
        ],
        vis,
    )


# ---------------- other use cases (OCR, classification, embeddings, image) ----------------
def latest(name):
    """the most recent json of each model/test: {model_dir: data}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "*", "*", name))):
        try:
            out[f.split(os.sep)[-3]] = json.load(open(f))
        except Exception:
            pass
    return out


oc = []
for tag, o in latest("ocr.json").items():
    for n in o.get("niveles", []):
        oc.append(
            [
                tag,
                n["conc"],
                fnum(n["paginas_min"]),
                fnum(n["lat_p50_s"], 2),
                fnum(n["lat_p95_s"], 2),
                n.get("out_tok_media"),
            ]
        )
if oc:
    P("=" * 78)
    P(" SPECIALISED OCR (transcription only; the OCR + LLM pipeline is in the VISION table)")
    P("=" * 78)
    table(["model", "conc.", "pages/min", "latency p50 s", "p95 s", "output tokens"], oc)

for tag, d in latest("d1.json").items():
    P("=" * 78)
    P(f" EMAIL CLASSIFICATION AND ROUTING ({d.get('correos')} emails, 6 categories + urgency)")
    P("=" * 78)
    b = d.get("llm", {})
    table_rows = []
    for key, name in (("d1", "d1-3B"), ("laya", "Laya 322M")):
        a = d.get(key)
        if not a:
            continue
        if "error" in a:
            P(f"  {name}: ERROR {a['error'][:200]}")
            continue
        table_rows.append(
            [
                f"{name} (one at a time)",
                fnum(a.get("acierto_categoria_pct")),
                fnum(a.get("acierto_urgencia_pct")),
                fnum(a.get("acierto_ambos_pct")),
                fnum(a.get("lat_p50_ms")),
                fnum(a.get("correos_s_1en1")),
                0,
            ]
        )
        table_rows.append([f"{name} (batches of 128)", "", "", "", "", fnum(a.get("correos_s_lote128")), 0])
    if b and "error" not in b:
        table_rows.append(
            [
                f"{b.get('modelo')} (32 at once)",
                fnum(b.get("acierto_categoria_pct")),
                fnum(b.get("acierto_urgencia_pct")),
                fnum(b.get("acierto_ambos_pct")),
                fnum(b.get("lat_p50_ms")),
                fnum(b.get("correos_s")),
                b.get("tokens_salida_media"),
            ]
        )
    if table_rows:
        table(
            ["model", "category OK %", "urgency OK %", "both OK %", "latency p50 ms", "emails/s", "tokens generated"],
            table_rows,
        )

for tag, e in latest("embed.json").items():
    P("=" * 78)
    P(f" RAG: EMBEDDINGS ({e.get('embedding')}) AND RERANKER ({e.get('reranker') or '-'})")
    P("=" * 78)
    table(
        ["indexing conc.", "chunks/s", "tokens/s", "tokens/chunk", "min for 1M chunks"],
        [
            [n["conc"], fnum(n["trozos_s"]), n["tokens_s"], n["tok_por_trozo"], fnum(n["min_1M_trozos"])]
            for n in e.get("indexacion", [])
        ],
    )
    table(
        ["query conc.", "queries/s", "p50 ms", "p95 ms"],
        [
            [n["conc"], fnum(n["consultas_s"]), fnum(n["lat_p50_ms"]), fnum(n["lat_p95_ms"])]
            for n in e.get("consultas", [])
        ],
    )
    if e.get("rerank"):
        table(
            ["rerank (50 docs) conc.", "reranks/s", "docs/s", "p50 ms", "p95 ms"],
            [[n["conc"], fnum(n["rerank_s"], 2), n["docs_s"], n["lat_p50_ms"], n["lat_p95_ms"]] for n in e["rerank"]],
        )
    ce, cr = e.get("calidad_embeddings", {}), e.get("calidad_reranker", {})
    P(
        f"  Quality check (GSM8K, English): embeddings recall@1 {ce.get('recall@1')} % · recall@5 {ce.get('recall@5')} % · "
        f"reranker hit@1 out of 50 {cr.get('acierto@1_de_50')} %"
    )

for tag, im in latest("imagen.json").items():
    P("=" * 78)
    P(f" IMAGE GENERATION ({im.get('modelo')}, {im.get('resolucion')}, diffusers {im.get('diffusers')})")
    P("=" * 78)
    table(
        ["steps", "images per batch", "s per image", "images/min", "peak memory GiB"],
        [
            [
                p["pasos"],
                p["lote"],
                fnum(p.get("s_por_imagen"), 2),
                fnum(p.get("imagenes_min")),
                fnum(p.get("memoria_pico_gib")),
            ]
            for p in im.get("pruebas", [])
            if "error" not in p
        ],
    )
    P(f"  model load {im.get('carga_s')} s · samples: {os.path.join(ROOT, tag)}/*/muestra_*.png")

txt = "\n".join(L)
print(txt)
with open(os.path.join(ROOT, "informe.txt"), "w") as f:
    f.write(txt + "\n")
print(f"\nSaved: {out_csv}\n       {os.path.join(ROOT, 'informe.txt')}")
