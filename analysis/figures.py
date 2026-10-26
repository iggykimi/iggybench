#!/usr/bin/env python3
"""All figures, drawn from the CSV tables in results/. Each figure is written as SVG in a light and a dark variant
(figures/<name>-light.svg, figures/<name>-dark.svg) and as a light PNG.

Usage: python analysis/figures.py
"""

import matplotlib.ticker as mt
import numpy as np
import pandas as pd

from common import LATE, MAIN, MODELS, RESULTS
from style import MARKERS, ORDER, color, figure, log_axis


def read(name):
    return pd.read_csv(RESULTS / f"{name}.csv")


def legend_below(fig, ax, ncol=4):
    h, labels = ax.get_legend_handles_labels()
    fig.legend(h, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.01))


def fig01_frontier(theme):
    d = read("perf_frontier_t12")
    with figure("fig01-frontier", theme, size=(8, 5.4)) as (fig, ax, t):
        for m in ORDER:
            x = d[d.model == m].sort_values("users")
            if x.empty:
                continue
            ax.plot(
                x.tok_s_per_user,
                x.output_tok_s,
                marker=MARKERS[m],
                color=color(m, theme),
                label=MODELS[m],
                markeredgecolor=t["surface"],
                markeredgewidth=1.5,
                markersize=7,
            )
        g = d[d.model == "gptoss120b"]
        for r in g.itertuples():
            ax.annotate(
                f"{r.users}",
                (r.tok_s_per_user, r.output_tok_s),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color=t["text2"],
            )
        ax.axvline(20, color=t["muted"], lw=1)
        ax.text(
            20.6,
            250,
            "20 tok/s per user\n(chat SLO: TPOT ≤ 50 ms)",
            fontsize=8,
            color=t["text2"],
            va="bottom",
        )
        log_axis(ax, "x", [15, 20, 30, 50, 100, 200, 400])
        ax.set_xlabel("Output tokens/s per user (1 / median TPOT, log scale)")
        ax.set_ylabel("Output tokens/s per GPU")
        ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0"))
        ax.set_ylim(0, None)
        ax.legend(loc="upper right", ncol=2)


def fig02_capacity(theme):
    cap = read("perf_capacity")
    cod = read("perf_coding_capacity")
    panels = [
        (
            "Chat: 1k in, 512 out\nP95 TTFT ≤ 2 s, P95 TPOT ≤ 50 ms",
            cap[cap.profile == "chat"],
            "users_low",
            "users_high",
            "Users at one request per minute",
        ),
        (
            "Agents: 16k in (8k shared), 1k out\nP95 TTFT ≤ 5 s, P95 TPOT ≤ 40 ms",
            cap[cap.profile == "agentes"],
            "users_low",
            "users_high",
            "Users at one request per minute",
        ),
        (
            "Coding agents: 48k → 100k context\nP95 TTFT ≤ 8 s, median ≥ 30 tok/s",
            cod,
            "last_pass_sessions",
            "first_fail_sessions",
            "Concurrent sessions",
        ),
    ]
    with figure("fig02-capacity", theme, size=(11, 4.6), ncols=3, sharey=True) as (fig, axes, t):
        ys = {m: i for i, m in enumerate(reversed(ORDER))}
        for ax, (title, d, lo, hi, xlabel) in zip(axes, panels):
            for r in d.itertuples():
                m, y, a, b = r.model, ys[r.model], getattr(r, lo), getattr(r, hi)
                c = color(m, theme)
                coarse = getattr(r, "grid", "fine") == "coarse"
                if not np.isnan(b):
                    ax.plot([a, b], [y, y], color=c, lw=2, ls=(0, (2, 2)) if coarse else "-")
                    ax.plot(b, y, "o", mfc=t["surface"], mec=c, mew=2, ms=8)
                ax.plot(a, y, "o", color=c, mec=t["surface"], mew=1.5, ms=8)
                label = f"{a:,.0f}–{b:,.0f}" if not np.isnan(b) else f"≥ {a:,.0f}"
                ax.annotate(
                    label,
                    (b if not np.isnan(b) else a, y),
                    xytext=(8, -3),
                    textcoords="offset points",
                    fontsize=8,
                    color=t["text2"],
                )
            ax.set_title(title, fontsize=9.5)
            ax.set_xlabel(xlabel)
            ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(list(ys.values()), [MODELS[m] for m in ys])
        fig.text(
            0.01,
            -0.02,
            "Filled dot: highest load that met the SLO. Open dot: next load tested, which failed. "
            "Dashed: coarse rate grid (models added on the last day).",
            fontsize=8,
            color=t["text2"],
        )


def fig03_chat_latency(theme):
    d = read("perf_capacity_points")
    d = d[(d.profile == "chat") & d.test.isin(["T4", "T4F"]) & d.model.isin(MAIN) & (d.excluded.isna())]
    with figure("fig03-chat-latency", theme, size=(11, 4.4), ncols=2) as (fig, axes, t):
        for m in MAIN:
            x = d[d.model == m].sort_values("rate_req_s").drop_duplicates("rate_req_s", keep="last")
            kw = dict(marker=MARKERS[m], color=color(m, theme), label=MODELS[m], mec=t["surface"], mew=1.2, ms=6)
            axes[0].plot(x.rate_req_s, x.ttft_p95_ms / 1000, **kw)
            axes[1].plot(x.rate_req_s, x.tpot_p95_ms, **kw)
        axes[0].axhline(2, color=t["muted"], lw=1)
        log_axis(axes[0], "y", [0.1, 0.3, 1, 2, 5, 10, 30])
        axes[0].set_ylabel("P95 time to first token (s, log scale)")
        axes[0].set_title("P95 TTFT; SLO 2 s")
        axes[1].axhline(50, color=t["muted"], lw=1)
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("P95 time per output token (ms)")
        axes[1].set_title("P95 TPOT; SLO 50 ms")
        for ax in axes:
            ax.set_xlabel("Poisson arrival rate (requests/s)")
            ax.set_xlim(0, 25)
        axes[1].legend(loc="upper right", fontsize=8.5)


def fig04_long_context(theme):
    d = read("perf_prompt_reading")
    needle = read("quality_needle")
    with figure("fig04-long-context", theme, size=(8, 5)) as (fig, ax, t):
        for m in MAIN:
            x = d[(d.model == m) & (d.users.astype(str) == "1")].sort_values("input_tokens")
            x = x.drop_duplicates("input_tokens", keep="last")
            ax.plot(
                x.input_tokens,
                x.ttft_p50_ms / 1000,
                marker=MARKERS[m],
                color=color(m, theme),
                label=MODELS[m],
                mec=t["surface"],
                mew=1.2,
                ms=6,
            )
        nd = needle.groupby("context_tokens").seconds.median().reset_index()
        ax.plot(
            nd.context_tokens,
            nd.seconds,
            ls="none",
            marker="o",
            ms=11,
            mfc="none",
            mec=color("dsv4-flash", theme),
            mew=1.5,
            label="DeepSeek V4 Flash, needle test (9/9 found)",
        )
        log_axis(
            ax,
            "x",
            [1024, 4096, 16384, 65536, 131072, 524288, 1048576],
            ["1k", "4k", "16k", "64k", "128k", "512k", "1M"],
        )
        log_axis(ax, "y", [0.03, 0.1, 0.3, 1, 3, 10, 30, 60])
        ax.set_xlabel("Prompt length (tokens, log scale)")
        ax.set_ylabel("Median time to first token (s, log scale)")
        ax.legend(loc="upper left")


def fig05_energy(theme):
    d = read("perf_frontier_t12")
    with figure("fig05-energy", theme, size=(8, 4.8)) as (fig, ax, t):
        for m in ORDER:
            x = d[d.model == m].sort_values("users")
            if x.empty:
                continue
            ax.plot(
                x.users,
                x.j_per_output_token,
                marker=MARKERS[m],
                color=color(m, theme),
                label=MODELS[m],
                mec=t["surface"],
                mew=1.2,
                ms=6,
            )
        log_axis(ax, "x", [1, 8, 16, 32, 64, 128])
        log_axis(ax, "y", [0.3, 0.5, 1, 2, 4, 8])
        ax.set_xlabel("Concurrent users (8k in, 1k out)")
        ax.set_ylabel("Joules per output token (GPU module, log scale)")
        ax.legend(ncol=2)


def fig06_reasoning(theme):
    d = read("quality_reasoning")
    aime = d[d.set == "aime_combined"].copy()
    math = d[d.set == "math500"].copy()
    rows = []
    for m in ORDER:
        for b in (32768, 65536):
            x = aime[(aime.model == m) & (aime.budget_tokens == b)]
            if len(x):
                rows.append(x.iloc[0])
    aime = pd.DataFrame(rows)
    aime["label"] = [f"{MODELS[r.model]}, {r.budget_tokens // 1024}k" for r in aime.itertuples()]
    with figure("fig06-reasoning", theme, size=(11, 5), ncols=2, gridspec_kw=dict(width_ratios=[1.4, 1])) as (
        fig,
        axes,
        t,
    ):
        for ax, x, title, lim in (
            (axes[0], aime, "AIME 2025 + 2026 (60 problems, 2 attempts each)", (40, 100)),
            (axes[1], math, "MATH-500 (32k budget)", (85, 100)),
        ):
            x = x.reset_index(drop=True)
            labels = x.label if "label" in x else [MODELS[m] for m in x.model]
            y = np.arange(len(x))[::-1]
            for yi, r in zip(y, x.itertuples()):
                c = color(r.model, theme)
                ax.plot(
                    [r.ci95_low_per_problem_pct, r.ci95_high_per_problem_pct],
                    [yi, yi],
                    color=c,
                    lw=6,
                    alpha=0.25,
                    solid_capstyle="butt",
                )
                ax.plot([r.ci95_low_pct, r.ci95_high_pct], [yi, yi], color=c, lw=2)
                hollow = getattr(r, "budget_tokens", 32768) == 65536
                ax.plot(
                    r.accuracy_pct,
                    yi,
                    "o",
                    ms=8,
                    color=c,
                    mfc=t["surface"] if hollow else c,
                    mec=c if hollow else t["surface"],
                    mew=2 if hollow else 1.5,
                )
                ax.annotate(
                    f"{r.accuracy_pct:.1f}",
                    (r.ci95_high_per_problem_pct, yi),
                    xytext=(5, -3),
                    textcoords="offset points",
                    fontsize=8,
                    color=t["text2"],
                )
            ax.set_yticks(y, labels)
            ax.set_xlim(*lim)
            ax.set_xlabel("Accuracy (%)")
            ax.set_title(title, fontsize=9.5)
            ax.grid(axis="y", visible=False)
        fig.text(
            0.01,
            -0.03,
            "Thin line: 95 % Wilson interval per attempt. Wide band: conservative interval per "
            "problem (the two attempts at a problem are not independent). Open dot: 64k-token budget.",
            fontsize=8,
            color=t["text2"],
        )


def heat(ax, values, rows, cols, t, fmt="{:.0f}", cmap="Blues", vmin=None, vmax=None, invert=False):
    im = ax.imshow(values, cmap=cmap + ("_r" if invert else ""), vmin=vmin, vmax=vmax, aspect="auto")
    lo, hi = im.norm.vmin, im.norm.vmax
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=8, color=t["muted"])
                continue
            share = (v - lo) / (hi - lo) if hi > lo else 0
            dark_cell = (1 - share) > 0.55 if invert else share > 0.55
            ax.text(
                j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="#ffffff" if dark_cell else "#0b0b0b"
            )
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def fig07_bfcl(theme):
    d = read("quality_bfcl")
    cats = [
        "simple",
        "multiple",
        "parallel",
        "parallel_multiple",
        "irrelevance",
        "live_simple",
        "live_multiple",
        "live_parallel",
        "live_parallel_multiple",
        "live_irrelevance",
    ]
    models = [m for m in ORDER if m in set(d.model)]
    v = np.array([[d[(d.model == m) & (d.category == c)].accuracy_pct.mean() for c in cats] for m in models])
    with figure("fig07-bfcl", theme, size=(10, 3.8)) as (fig, ax, t):
        heat(ax, v, [MODELS[m] for m in models], [c.replace("_", " ") for c in cats], t, vmin=0, vmax=100)
        ax.set_title("BFCL v3 accuracy by category (%)")


def fig08_invoices(theme):
    d = read("quality_invoices")
    d = d[d.reference_level]
    names = {"vision_llm": "", "ocr+llm": "PaddleOCR-VL + gpt-oss-120b"}
    rows = []
    for r in d[d.pipeline.isin(names)].itertuples():
        label = names[r.pipeline] or MODELS.get(r.model, r.model)
        rows.append((label, r.set, r.field_accuracy_pct, r.model))
    x = pd.DataFrame(rows, columns=["label", "set", "acc", "model"])
    piv = x.pivot_table(index="label", columns="set", values="acc")
    piv = piv.sort_values("degraded")
    with figure("fig08-invoices", theme, size=(8, 3.8)) as (fig, ax, t):
        for i, (label, r) in enumerate(piv.iterrows()):
            ax.plot([r.degraded, r.clean], [i, i], color=t["muted"], lw=1.5)
            ax.plot(
                r.clean,
                i,
                "o",
                ms=8,
                color=t["text2"],
                mec=t["surface"],
                mew=1.5,
                label="40 clean invoices" if i == 0 else None,
            )
            ax.plot(
                r.degraded,
                i,
                "o",
                ms=8,
                color="#e34948" if theme == "light" else "#e66767",
                mec=t["surface"],
                mew=1.5,
                label="30 degraded scans" if i == 0 else None,
            )
            ax.annotate(
                f"{r.degraded:.0f}",
                (r.degraded, i),
                xytext=(-8, -3),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=t["text2"],
            )
        ax.set_yticks(range(len(piv)), piv.index)
        ax.set_xlim(55, 102)
        ax.set_xlabel("Fields extracted correctly (%, 8 fields per invoice, one request at a time)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower right")


def fig09_asr(theme):
    d = read("quality_asr")
    langs = ["es_419", "en_us", "ca_es", "fr_fr", "de_de", "pt_br", "it_it"]
    names = {
        "es_419": "Spanish",
        "en_us": "English",
        "ca_es": "Catalan",
        "fr_fr": "French",
        "de_de": "German",
        "pt_br": "Portuguese",
        "it_it": "Italian",
    }
    pretty = {
        "whisper-large-v3": "Whisper large-v3",
        "voxtral-mini-3b": "Voxtral Mini 3B",
        "qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
        "bark-3.0-pro": "Underdog Bark 3.0 Pro",
        "granite-4.0-1b-speech": "Granite 4.0 1B Speech",
    }
    d = d.sort_values("wer_mean_6_no_ca_pct")
    v = d[[f"wer_{lang}_pct" for lang in langs]].to_numpy(float)
    with figure("fig09-asr", theme, size=(11, 3.6), ncols=2, gridspec_kw=dict(width_ratios=[2.2, 1])) as (fig, axes, t):
        heat(
            axes[0],
            v,
            [pretty.get(m, m) for m in d.model],
            [names[lang] for lang in langs],
            t,
            fmt="{:.1f}",
            cmap="Reds",
            vmin=0,
            vmax=25,
        )
        axes[0].set_title("Word error rate by language, FLEURS (%; lower is better)")
        y = np.arange(len(d))
        axes[1].barh(y, d.rtfx_load, height=0.55, color=t["series"][0])
        for yi, val in zip(y, d.rtfx_load):
            axes[1].annotate(
                f"{val:,.0f}×", (val, yi), xytext=(4, -3), textcoords="offset points", fontsize=8, color=t["text2"]
            )
        axes[1].set_yticks(y, [])
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, d.rtfx_load.max() * 1.25)
        axes[1].set_title("Real-time factor under load")
        axes[1].set_xlabel("Hours of audio per hour")
        axes[1].grid(axis="y", visible=False)


def fig10_speculative(theme):
    d = read("config_speculative")
    on = {"qwen38-flash-next-nomtp": "Qwen3.8-Flash-Next"}
    labels = [f"{r.method}\non {on.get(r.baseline, MODELS.get(r.baseline, r.baseline))}" for r in d.itertuples()]
    y = np.arange(len(d))
    with figure("fig10-speculative", theme, size=(11, 3.8), ncols=2, sharey=True) as (fig, axes, t):
        base_c, var_c = t["muted"], t["series"][0]
        for ax, (b, v, title, unit) in zip(
            axes,
            [
                ("baseline_single_user_tok_s", "single_user_tok_s", "One user: output tokens/s", "tok/s"),
                ("baseline_chat_last_pass_req_s", "chat_last_pass_req_s", "Chat capacity under SLO", "req/s"),
            ],
        ):
            ax.barh(y + 0.18, d[b], height=0.34, color=base_c, label="without")
            ax.barh(y - 0.18, d[v], height=0.34, color=var_c, label="with")
            for yi, bb, vv in zip(y, d[b], d[v]):
                ax.annotate(
                    f"{bb:.0f}" if unit == "tok/s" else f"{bb:g}",
                    (bb, yi + 0.18),
                    xytext=(4, -3),
                    textcoords="offset points",
                    fontsize=8,
                    color=t["text2"],
                )
                ax.annotate(
                    f"{vv:.0f}" if unit == "tok/s" else f"{vv:g}",
                    (vv, yi - 0.18),
                    xytext=(4, -3),
                    textcoords="offset points",
                    fontsize=8,
                    color=t["text2"],
                )
            ax.set_title(title)
            ax.set_xlabel(unit)
            ax.set_xlim(0, max(d[b].max(), d[v].max()) * 1.2)
            ax.grid(axis="y", visible=False)
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        axes[1].legend(loc="lower right")


ALL = [
    fig01_frontier,
    fig02_capacity,
    fig03_chat_latency,
    fig04_long_context,
    fig05_energy,
    fig06_reasoning,
    fig07_bfcl,
    fig08_invoices,
    fig09_asr,
    fig10_speculative,
]


def main():
    for f in ALL:
        for theme in ("light", "dark"):
            f(theme)
    print(f"{len(ALL)} figures, main={len(MAIN)} late={len(LATE)}")


if __name__ == "__main__":
    main()
