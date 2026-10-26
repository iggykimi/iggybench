"""Quality analysis: recomputes every quality table from the raw benchmark JSON files.

Reads, under RAW/results/<tag>/<run>/ (raw file names and keys are Spanish, a frozen schema):
  gsm8k.json (T16), dificil.json / dificil_64k.json and their *_detalle.jsonl when present (T20),
  bfcl.json and bfcl_detalle.jsonl (T21), agentes.json (T13), clasif_dificil.json (T22), d1.json (d1-3B / Laya),
  vision.json / vision_dificil.json (T11 / T11D invoices), embed.json (RAG), imagen.json (Qwen-Image),
  asr/*/asr_*.json (T23) and *aguja/*/aguja.json (T24).
Writes results/quality_*.csv (list in main()). Every row has a `source` column: the raw file it comes from.

Run with IGGY_RAW pointing at the raw data root.
"""

import json
import math
import statistics
from itertools import combinations

import pandas as pd
from scipy import stats

from common import MODELS, RAW, batch_of, holm, latest, load_json, rel, wilson, write_csv

# Display names for the systems that are not one of the LLMs in common.MODELS.
OTHER_NAMES = {
    "paddleocr+gptoss": "PaddleOCR-VL 1.6 + gpt-oss-120b",
    "paddleocr+gptoss-dificil": "PaddleOCR-VL 1.6 + gpt-oss-120b",
    "plataforma-docs": "Qwen3.8-27B (shared GPU, 33 % memory)",
    "liquid-d1-3b": "Liquid d1-3B",
    "laya-322m": "Laya multilingual (322M)",
    "qwen3-embedding-8b": "Qwen3-Embedding-8B + Qwen3-Reranker-8B",
    "qwen-image": "Qwen-Image",
    "whisper-large-v3": "Whisper large-v3",
    "voxtral-mini-3b": "Voxtral Mini 3B",
    "qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
    "bark-3.0-pro": "Underdog Bark 3.0 Pro",
    "granite-4.0-1b-speech": "Granite 4.0 1B Speech",
}
LANGS = ["es_419", "en_us", "ca_es", "fr_fr", "de_de", "pt_br", "it_it"]
INVOICE_FIELDS = ["numero", "fecha", "proveedor", "nif_proveedor", "base_imponible", "iva_pct", "total", "num_lineas"]
BFCL_CATS = [
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


def name(tag: str) -> str:
    return MODELS.get(tag) or OTHER_NAMES.get(tag, tag)


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def detail_of(path):
    """The per-item detail file written next to a summary JSON, if it was copied to the raw data."""
    p = path.with_name(path.stem + "_detalle.jsonl")
    return p if p.exists() else None


def count_from_pct(pct: float, n: int) -> tuple[int, str]:
    """Correct count behind a percentage rounded to 0.1, and whether that count is the only one that rounds to it.

    Most summary JSONs store only the rounded percentage; this is how counts are recovered when no detail file exists.
    """
    hits = [k for k in range(n + 1) if abs(100 * k / n - pct) <= 0.05 + 1e-9]
    k = round(pct * n / 100)
    return k, ("from_pct_unique" if len(hits) == 1 else f"from_pct_ambiguous({hits[0]}-{hits[-1]})")


def upper_median(values):
    """Median as the client scripts compute it: element n//2 of the sorted list."""
    v = sorted(values)
    return v[len(v) // 2] if v else math.nan


def ztest(k1: float, n1: float, k2: float, n2: float) -> float:
    """Two-sided p of the pooled two-proportion z test (counts may be fractional for the per-problem version)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return 2 * stats.norm.sf(abs(k1 / n1 - k2 / n2) / se)


def ci(k: float, n: float) -> dict:
    _, lo, hi = wilson(k, n)
    return {"ci95_low_pct": lo, "ci95_high_pct": hi}


# ----------------------------------------------------------------------------------------------------------- GSM8K
def gsm8k():
    rows, answers = [], {}
    for tag in MODELS:
        path = latest(tag, "gsm8k.json")
        if not path:
            continue
        d = load_json(path)
        answers[tag] = d["por_pregunta"]
        assert sum(d["por_pregunta"]) == d["aciertos"]
        rows.append(
            {
                "model": tag,
                "model_name": name(tag),
                "questions": d["preguntas"],
                "correct": d["aciertos"],
                "accuracy_pct": 100 * d["aciertos"] / d["preguntas"],
                **ci(d["aciertos"], d["preguntas"]),
                "accuracy_first200_pct": d["acierto_200_pct"],
                "errors": d["errores"],
                "source": rel(path),
            }
        )
    pairs = []
    for a, b in combinations(answers, 2):
        only_a = sum(x and not y for x, y in zip(answers[a], answers[b]))
        only_b = sum(y and not x for x, y in zip(answers[a], answers[b]))
        pairs.append(
            {
                "model_a": a,
                "model_b": b,
                "diff_pp": 100 * (sum(answers[a]) - sum(answers[b])) / len(answers[a]),
                "a_right_b_wrong": only_a,
                "a_wrong_b_right": only_b,
                "p_mcnemar_exact": stats.binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0,
            }
        )
    for row, p in zip(pairs, holm([r["p_mcnemar_exact"] for r in pairs])):
        row["p_holm"] = p
        row["significant_holm"] = p < 0.05
    return pd.DataFrame(rows), pd.DataFrame(pairs)


# ----------------------------------------------------------------------------------------------- AIME / MATH-500
def _reasoning_rows(tag, path):
    d = load_json(path)
    det = detail_of(path)
    recs = read_jsonl(det) if det else None
    base = {
        "model": tag,
        "model_name": name(tag),
        "budget_tokens": d["max_tokens"],
        "run_minutes": d["minutos"],
        "run_tok_s": d["tok_s_medio"],
    }
    rows = []
    for set_name in ("aime2025", "aime2026", "math500"):
        c = d["conjuntos"].get(set_name)
        if not c:
            continue
        if recs:
            rs = [r for r in recs if r["conjunto"] == set_name and r["estado"] == "ok"]
            k, basis = sum(r["acierto"] for r in rs), "detail"
            truncated = sum(r["truncada"] for r in rs)
            tokens = [r["tokens"] for r in rs if r.get("tokens")]
            assert abs(100 * k / len(rs) - c["acierto_pct"]) <= 0.05 and truncated == c["truncadas"]
        else:
            k, basis = count_from_pct(c["acierto_pct"], c["respondidas"])
            truncated, tokens = c["truncadas"], None
        rows.append(
            {
                **base,
                "set": set_name,
                "problems": c["problemas"],
                "attempts": c["peticiones"],
                "answered": c["respondidas"],
                "correct": k,
                "count_basis": basis,
                "truncated": truncated,
                "median_completion_tokens": upper_median(tokens) if tokens else c["tokens_mediana"],
                "_tokens": tokens,
                "source": rel(det or path),
            }
        )
    aime = [r for r in rows if r["set"].startswith("aime")]
    if len(aime) == 2:
        tokens = aime[0]["_tokens"] + aime[1]["_tokens"] if recs else None
        rows.append(
            {
                **base,
                "set": "aime_combined",
                **{k: aime[0][k] + aime[1][k] for k in ("problems", "attempts", "answered", "correct", "truncated")},
                "count_basis": aime[0]["count_basis"],
                # The summary JSON only has per-year medians; the combined median needs the per-attempt detail.
                "median_completion_tokens": upper_median(tokens) if tokens else math.nan,
                "_tokens": None,
                "source": aime[0]["source"],
            }
        )
    return rows


def reasoning():
    """T20. A truncated answer (finish_reason == length) counts as wrong; accuracy is over answered attempts.

    AIME runs two attempts per problem, so besides the Wilson interval per attempt there is a conservative one that
    keeps the observed proportion but uses n_eff = number of problems.
    """
    rows = []
    for tag in MODELS:
        for fname in ("dificil.json", "dificil_64k.json"):
            path = latest(tag, fname)
            if path:
                rows += _reasoning_rows(tag, path)
    for r in rows:
        r.pop("_tokens")
        r["accuracy_pct"] = 100 * r["correct"] / r["answered"]
        r.update(ci(r["correct"], r["answered"]))
        _, lo, hi = wilson(r["correct"] / r["answered"] * r["problems"], r["problems"])
        r["ci95_low_per_problem_pct"], r["ci95_high_per_problem_pct"] = lo, hi
        r["wrong_not_truncated"] = r["answered"] - r["correct"] - r["truncated"]
    cols = [
        "model",
        "model_name",
        "budget_tokens",
        "set",
        "problems",
        "attempts",
        "answered",
        "correct",
        "count_basis",
        "accuracy_pct",
        "ci95_low_pct",
        "ci95_high_pct",
        "ci95_low_per_problem_pct",
        "ci95_high_per_problem_pct",
        "truncated",
        "wrong_not_truncated",
        "median_completion_tokens",
        "run_minutes",
        "run_tok_s",
        "source",
    ]
    return pd.DataFrame(rows)[cols]


def reasoning_pairs(table):
    """Every pair of model x budget rows on AIME combined and on MATH-500, Holm-adjusted within each set and method.

    Unpaired z tests (no per-problem detail for most runs), so they are conservative. Pairs with different token
    budgets are kept but flagged: they compare model and budget at once.
    """
    out = []
    for set_name in ("aime_combined", "math500"):
        t = table[table["set"] == set_name].to_dict("records")
        pairs = []
        for a, b in combinations(t, 2):
            pa, pb = a["correct"] / a["answered"], b["correct"] / b["answered"]
            se = math.sqrt(pa * (1 - pa) / a["answered"] + pb * (1 - pb) / b["answered"])
            pairs.append(
                {
                    "set": set_name,
                    "model_a": a["model"],
                    "budget_a_tokens": a["budget_tokens"],
                    "model_b": b["model"],
                    "budget_b_tokens": b["budget_tokens"],
                    "like_for_like": a["budget_tokens"] == b["budget_tokens"],
                    "same_model": a["model"] == b["model"],
                    "correct_a": a["correct"],
                    "correct_b": b["correct"],
                    "diff_pp": 100 * (pa - pb),
                    "diff_ci95_low_pp": 100 * (pa - pb - 1.96 * se),
                    "diff_ci95_high_pp": 100 * (pa - pb + 1.96 * se),
                    "p_z_per_attempt": ztest(a["correct"], a["answered"], b["correct"], b["answered"]),
                    "p_z_per_problem": ztest(pa * a["problems"], a["problems"], pb * b["problems"], b["problems"]),
                }
            )
        for key in ("p_z_per_attempt", "p_z_per_problem"):
            for row, p in zip(pairs, holm([r[key] for r in pairs])):
                row[key.replace("p_z", "p_holm")] = p
        for row in pairs:
            row["significant_conservative"] = row["p_holm_per_problem"] < 0.05
        out += pairs
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------------------ BFCL
def bfcl():
    """T21. Per category, plus three summaries per model:
    mean_10_categories  unweighted mean of the 10 category accuracies (what the client reports as media_categorias_pct,
                        except that the client averages the rounded per-category values);
    mean_6_non_parallel unweighted mean of the 6 categories without parallel calls; CI from the per-category binomial
                        variances (normal approximation);
    irrelevance_pooled  irrelevance + live_irrelevance pooled by question (n = 479): "does not call when it should not".
    """
    rows = []
    for tag in MODELS:
        path = latest(tag, "bfcl.json")
        if not path:
            continue
        d = load_json(path)
        det = detail_of(path)
        recs = read_jsonl(det) if det else None
        cats = {}
        for cat in BFCL_CATS:
            c = d["categorias"][cat]
            if recs:
                rs = [r for r in recs if r["cat"] == cat and r["estado"] == "ok"]
                k, basis = sum(r["ok"] for r in rs), "detail"
                assert len(rs) == c["respondidas"] and abs(100 * k / len(rs) - c["acierto_pct"]) <= 0.05
            else:
                k, basis = count_from_pct(c["acierto_pct"], c["respondidas"])
            cats[cat] = (k, c["respondidas"])
            rows.append(
                {
                    "model": tag,
                    "model_name": name(tag),
                    "category": cat,
                    "parallel": "parallel" in cat,
                    "live": cat.startswith("live"),
                    "questions": c["preguntas"],
                    "answered": c["respondidas"],
                    "correct": k,
                    "count_basis": basis,
                    "accuracy_pct": 100 * k / c["respondidas"],
                    **ci(k, c["respondidas"]),
                    "client_value_pct": c["acierto_pct"],
                    "source": rel(det or path),
                }
            )
        summary = {"model": tag, "model_name": name(tag), "count_basis": basis, "source": rel(det or path)}
        p10 = [k / n for k, n in cats.values()]
        rows.append(
            {
                **summary,
                "category": "mean_10_categories",
                "accuracy_pct": 100 * statistics.mean(p10),
                "client_value_pct": d["media_categorias_pct"],
            }
        )
        six = [cats[c] for c in BFCL_CATS if "parallel" not in c]
        m6 = statistics.mean(k / n for k, n in six)
        se6 = math.sqrt(sum((k / n) * (1 - k / n) / n for k, n in six)) / len(six)
        rows.append(
            {
                **summary,
                "category": "mean_6_non_parallel",
                "questions": sum(n for _, n in six),
                "accuracy_pct": 100 * m6,
                "ci95_low_pct": 100 * (m6 - 1.96 * se6),
                "ci95_high_pct": 100 * (m6 + 1.96 * se6),
            }
        )
        k, n = (sum(x) for x in zip(cats["irrelevance"], cats["live_irrelevance"]))
        rows.append(
            {
                **summary,
                "category": "irrelevance_pooled",
                "questions": n,
                "answered": n,
                "correct": k,
                "accuracy_pct": 100 * k / n,
                **ci(k, n),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- Agents
def agents():
    """T13: 8 tasks; at concurrency c the tasks repeat (max(2c, 8) attempts) at temperature 0, so repeats are not
    independent and the per-attempt CI is optimistic (effective n is 8 tasks).

    The latest agentes.json of each model is the corrected-environment run (batch 2, 22/10 morning). The earlier
    batch-2 runs scored gpt-oss lower because the grader rejected the Unicode hyphen in "P‑1038"; fixed afterwards.
    """
    rows = []
    for tag in MODELS:
        path = latest(tag, "agentes.json")
        if not path:
            continue
        for lv in load_json(path)["niveles"]:
            per_task = {t: tuple(map(int, s.split("/"))) for t, s in lv["por_tarea"].items()}
            passed = sum(k for k, _ in per_task.values())
            assert sum(n for _, n in per_task.values()) == lv["tareas"]
            assert abs(100 * passed / lv["tareas"] - lv["aciertos_pct"]) <= 0.05
            rows.append(
                {
                    "model": tag,
                    "model_name": name(tag),
                    "concurrency": lv["conc"],
                    "tasks": lv["tareas"],
                    "passed": passed,
                    "pass_rate_pct": 100 * passed / lv["tareas"],
                    **ci(passed, lv["tareas"]),
                    "tasks_per_min": lv["tareas_min"],
                    "task_s_p50": lv["seg_tarea_p50"],
                    "task_s_p95": lv["seg_tarea_p95"],
                    "tool_calls_per_task": lv["herramientas_media"],
                    "output_tokens_mean": lv["tokens_salida_media"],
                    "errors": lv["errores"],
                    "failed_tasks": "; ".join(f"{t} {k}/{n}" for t, (k, n) in per_task.items() if k < n),
                    "source": rel(path),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- E-mail
def email_hard():
    """T22: 240 hard synthetic e-mails (20 templates x 12), temperature 0, 32 concurrent, reasoning off.
    An answer without parseable JSON counts as wrong (format_failures)."""
    rows = []
    for tag in MODELS:
        path = latest(tag, "clasif_dificil.json")
        if not path:
            continue
        d = load_json(path)
        n = d["correos"]
        k_cat, basis = count_from_pct(d["acierto_categoria_pct"], n)
        k_urg, _ = count_from_pct(d["acierto_urgencia_pct"], n)
        k_both, _ = count_from_pct(d["acierto_ambos_pct"], n)
        rows.append(
            {
                "model": tag,
                "model_name": name(tag),
                "emails": n,
                "count_basis": basis,
                "category_correct": k_cat,
                "category_accuracy_pct": 100 * k_cat / n,
                "urgency_correct": k_urg,
                "urgency_accuracy_pct": 100 * k_urg / n,
                "both_correct": k_both,
                "both_accuracy_pct": 100 * k_both / n,
                **ci(k_both, n),
                "emails_per_s": d["correos_s"],
                "latency_p50_ms": d["lat_p50_ms"],
                "latency_p95_ms": d["lat_p95_ms"],
                "output_tokens_mean": d["tokens_salida_media"],
                "format_failures": d["fallos_formato"],
                "concurrency": d["conc"],
                "source": rel(path),
            }
        )
    return pd.DataFrame(rows)


def email_classify():
    """d1.py: Liquid d1-3B and Laya (decision models) and gpt-oss-120b on the 300 normal e-mails and on the 240 hard
    ones (same corpus as T22, different urgency instruction). Throughput of d1/Laya is the batch-128 figure."""
    rows = []
    for corpus, tag in (("normal", "d1-3b"), ("hard", "d1-3b-dificil")):
        path = latest(tag, "d1.json")
        d = load_json(path)
        n = d["correos"]
        for key, system in (("d1", "liquid-d1-3b"), ("laya", "laya-322m"), ("llm", "gptoss120b")):
            s = d[key]
            k_cat, basis = count_from_pct(s["acierto_categoria_pct"], n)
            k_urg, _ = count_from_pct(s["acierto_urgencia_pct"], n)
            k_both, _ = count_from_pct(s["acierto_ambos_pct"], n)
            _, cat_lo, cat_hi = wilson(k_cat, n)
            rows.append(
                {
                    "model": system,
                    "model_name": name(system),
                    "corpus": corpus,
                    "emails": n,
                    "count_basis": basis,
                    "category_correct": k_cat,
                    "category_accuracy_pct": 100 * k_cat / n,
                    "category_ci95_low_pct": cat_lo,
                    "category_ci95_high_pct": cat_hi,
                    "urgency_accuracy_pct": 100 * k_urg / n,
                    "both_correct": k_both,
                    "both_accuracy_pct": 100 * k_both / n,
                    **ci(k_both, n),
                    "latency_p50_ms": s["lat_p50_ms"],
                    "latency_p95_ms": s["lat_p95_ms"],
                    "emails_per_s": s.get("correos_s_lote128", s.get("correos_s")),
                    "emails_per_s_one_by_one": s.get("correos_s_1en1"),
                    "memory_gib": s.get("memoria_gib"),
                    "source": rel(path),
                }
            )
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------------------- Invoices
INVOICE_RUNS = [
    ("clean", "vision_llm", "qwen27b-fp8", "vision.json"),
    ("clean", "vision_llm", "qwen38-flash-next", "vision.json"),
    ("clean", "vision_llm", "glm53-ssmbf16", "vision.json"),
    ("clean", "ocr+llm", "paddleocr+gptoss", "vision.json"),
    ("clean", "vision_llm_shared_gpu", "plataforma-docs", "vision.json"),
    ("degraded", "vision_llm", "qwen27b-fp8", "vision_dificil.json"),
    ("degraded", "vision_llm", "qwen38-flash-next", "vision_dificil.json"),
    ("degraded", "vision_llm", "glm53-ssmbf16", "vision_dificil.json"),
    ("degraded", "vision_llm", "nemotron35-super-vl", "vision_dificil.json"),
    ("degraded", "ocr+llm", "paddleocr+gptoss-dificil", "vision.json"),
]


def invoices():
    """T11 (40 clean invoices) and T11D (30 degraded scans). 8 fields per invoice.

    Degraded set: the latest run of each system is the queue-4 (or later) re-run with the corrected NIF ground truth;
    the batch-2 runs scored the NIF against a wrong reference and are not used.
    Each concurrency level repeats invoices cyclically, so levels with more requests than invoices weight the first
    ones more. `reference_level` marks the first level that covers every invoice; the perfect-invoice CI uses
    n_eff = distinct invoices.
    """
    rows, fields = [], []
    for set_name, pipeline, tag, fname in INVOICE_RUNS:
        path = latest(tag, fname)
        if set_name == "degraded":
            assert batch_of(path.parent.name) in ("queue4", "queue5-9"), path
        d = load_json(path)
        docs = d["documentos"]
        ref = next(lv["conc"] for lv in d["niveles"] if lv["peticiones"] >= docs)
        for lv in d["niveles"]:
            f = lv["acierto_por_campo"]
            perfect = lv["facturas_perfectas_pct"]
            _, lo, hi = wilson(perfect / 100 * docs, docs)
            lo = max(lo, 0.0)  # Wilson at k = 0 gives -0.0 from rounding
            rows.append(
                {
                    "set": set_name,
                    "pipeline": pipeline,
                    "model": tag,
                    "model_name": name(tag),
                    "concurrency": lv["conc"],
                    "requests": lv["peticiones"],
                    "documents": docs,
                    "reference_level": lv["conc"] == ref,
                    "field_accuracy_pct": statistics.mean(f.values()),
                    "field_accuracy_no_nif_pct": statistics.mean(v for k, v in f.items() if k != "nif_proveedor"),
                    "amounts_accuracy_pct": (f["base_imponible"] + f["total"]) / 2,
                    "base_accuracy_pct": f["base_imponible"],
                    "total_accuracy_pct": f["total"],
                    "nif_accuracy_pct": f["nif_proveedor"],
                    "lines_accuracy_pct": f["num_lineas"],
                    "perfect_invoices_pct": perfect,
                    "perfect_ci95_low_pct": lo,
                    "perfect_ci95_high_pct": hi,
                    "pages_per_min": lv["paginas_min"],
                    "latency_p50_s": lv["lat_p50_s"],
                    "latency_p95_s": lv["lat_p95_s"],
                    "output_tokens_mean": lv.get("out_tok_media"),
                    "failures": lv["fallos"],
                    "client_field_accuracy_pct": lv["acierto_campos_pct"],
                    "source": rel(path),
                }
            )
            for field in INVOICE_FIELDS:
                fields.append(
                    {
                        "set": set_name,
                        "pipeline": pipeline,
                        "model": tag,
                        "model_name": name(tag),
                        "concurrency": lv["conc"],
                        "requests": lv["peticiones"],
                        "reference_level": lv["conc"] == ref,
                        "field": field,
                        "accuracy_pct": f[field],
                        "source": rel(path),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(fields)


# ------------------------------------------------------------------------------------------------------------- ASR
def asr():
    """T23: FLEURS test, 60 utterances x 7 languages = 420. WER per language from the full-load level (all 420 audios).

    Three overall WERs, because the published table mixes two definitions:
    wer_pooled_7_pct    the client's wer_medio_pct: errors / reference words over all languages (word-weighted);
                        not available for Bark (its client reports per-language WER only);
    wer_mean_7_pct      unweighted mean of the 7 language WERs;
    wer_mean_6_no_ca_pct unweighted mean without Catalan (the per-language word counts needed to pool are not stored).
    RTFx at 1 uses the client's single-stream level (vLLM: conc 1 on a 105-audio subset; Bark: batch 1, English only).
    """
    rows = []
    for path in sorted((RAW / "results" / "asr").glob("*/asr_*.json")):
        d = load_json(path)
        levels = d["niveles"]
        key = "conc" if "conc" in levels[0] else "lote"
        one, load = min(levels, key=lambda x: x[key]), max(levels, key=lambda x: x[key])
        w = load["wer_pct"]
        rows.append(
            {
                "model": d["modelo"],
                "model_name": name(d["modelo"]),
                "engine": "vLLM" if key == "conc" else "Transformers",
                "audios": load["audios"],
                "failures": sum(lv.get("fallos", 0) for lv in levels),
                **{f"wer_{lang}_pct": w[lang] for lang in LANGS},
                "wer_pooled_7_pct": load.get("wer_medio_pct", math.nan),
                "wer_mean_7_pct": statistics.mean(w[lang] for lang in LANGS),
                "wer_mean_6_no_ca_pct": statistics.mean(w[lang] for lang in LANGS if lang != "ca_es"),
                "load_level": f"{'concurrency' if key == 'conc' else 'batch'} {load[key]}",
                "rtfx_1": one["rtfx"],
                "audios_at_1": one["audios"],
                "latency_p50_s_at_1": one["lat_p50_s"],
                "rtfx_load": load["rtfx"],
                "s_per_audio_hour_load": 3600 / load["rtfx"],
                "source": rel(path),
            }
        )
    return pd.DataFrame(rows).sort_values("wer_mean_6_no_ca_pct")


# ---------------------------------------------------------------------------------------------------------- Needle
def needle():
    """T24: one code hidden at 10/50/90 % depth of 128k / 512k / ~1M tokens of filler; found = code in the answer."""
    rows = []
    for path in sorted((RAW / "results").glob("*aguja*/*/aguja.json")):
        d = load_json(path)
        tag = path.parents[1].name.removesuffix("-aguja")
        for t in d["pruebas"]:
            rows.append(
                {
                    "model": tag,
                    "model_name": name(tag),
                    "context_target_tokens": t["tokens_objetivo"],
                    "depth": t["profundidad"],
                    "context_tokens": t.get("tokens_reales"),
                    "found": t["acierta"],
                    "seconds": t["segundos"],
                    "tokens_per_s": t["tokens_reales"] / t["segundos"] if t.get("tokens_reales") else math.nan,
                    "source": rel(path),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------- RAG
def rag():
    """Embedding/reranker throughput, plus the configuration check (GSM8K question -> its solution; easy by design)."""
    tag = "qwen3-embedding-8b"
    path = latest(tag, "embed.json")
    d = load_json(path)
    base = {"model": tag, "model_name": name(tag), "source": rel(path)}
    rows = []
    for lv in d["indexacion"]:
        rows.append(
            {
                **base,
                "phase": "indexing",
                "concurrency": lv["conc"],
                "units_per_s": lv["trozos_s"],
                "tokens_per_s": lv["tokens_s"],
                "tokens_per_chunk": lv["tok_por_trozo"],
                "min_per_million_chunks": 1e6 / lv["trozos_s"] / 60,
                "latency_p50_ms": 1000 * lv["lat_p50_s"],
                "failures": lv["fallos"],
            }
        )
    for lv in d["consultas"]:
        rows.append(
            {
                **base,
                "phase": "query",
                "concurrency": lv["conc"],
                "units_per_s": lv["consultas_s"],
                "latency_p50_ms": lv["lat_p50_ms"],
                "latency_p95_ms": lv["lat_p95_ms"],
                "failures": lv["fallos"],
            }
        )
    for lv in d["rerank"]:
        rows.append(
            {
                **base,
                "phase": "rerank_50_docs",
                "concurrency": lv["conc"],
                "units_per_s": lv["rerank_s"],
                "docs_per_s": lv["docs_s"],
                "latency_p50_ms": lv["lat_p50_ms"],
                "latency_p95_ms": lv["lat_p95_ms"],
                "failures": lv["fallos"],
            }
        )
    for phase, pct, n in (
        ("check_embedding_recall@1", d["calidad_embeddings"]["recall@1"], 300),
        ("check_embedding_recall@5", d["calidad_embeddings"]["recall@5"], 300),
        ("check_reranker_accuracy@1_of_50", d["calidad_reranker"]["acierto@1_de_50"], 100),
    ):
        k, _ = count_from_pct(pct, n)
        rows.append({**base, "phase": phase, "n": n, "correct": k, "accuracy_pct": 100 * k / n, **ci(k, n)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- Image
def image():
    """Qwen-Image 1024x1024: speed only; image quality was not measured."""
    tag = "qwen-image"
    path = latest(tag, "imagen.json")
    rows = [
        {
            "model": tag,
            "model_name": name(tag),
            "steps": t["pasos"],
            "batch": t["lote"],
            "s_per_call": t["s_por_llamada"],
            "s_per_image": t["s_por_imagen"],
            "images_per_min": 60 / t["s_por_imagen"],
            "peak_memory_gib": t["memoria_pico_gib"],
            "source": rel(path),
        }
        for t in load_json(path)["pruebas"]
    ]
    return pd.DataFrame(rows)


def main():
    gsm, gsm_pairs = gsm8k()
    reason = reasoning()
    inv, inv_fields = invoices()
    tables = {
        "quality_gsm8k.csv": gsm,
        "quality_gsm8k_pairs.csv": gsm_pairs,
        "quality_reasoning.csv": reason,
        "quality_reasoning_pairs.csv": reasoning_pairs(reason),
        "quality_bfcl.csv": bfcl(),
        "quality_agents.csv": agents(),
        "quality_email_hard.csv": email_hard(),
        "quality_email_classify.csv": email_classify(),
        "quality_invoices.csv": inv,
        "quality_invoices_fields.csv": inv_fields,
        "quality_asr.csv": asr(),
        "quality_needle.csv": needle(),
        "quality_rag.csv": rag(),
        "quality_image.csv": image(),
    }
    for fname, df in tables.items():
        print(f"{write_csv(df, fname).name}: {len(df)} rows")


if __name__ == "__main__":
    main()
