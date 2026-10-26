#!/usr/bin/env python3
"""
invoices_ocr.py — specialised OCR (PaddleOCR-VL-1.6) against the general VLM of T11 (stdlib only).

Phase A · OCR only: the 40 clean invoices with the prompt "OCR:" at concurrency 1, 8, 32 and 64.
          Measures pages/minute, p50/p95 latency and output tokens.
Phase B · real OCR -> LLM pipeline: each invoice goes through the OCR and its text goes to a text LLM (gpt-oss)
          that returns the same 8-field JSON as T11. Same comparer and same ground truth as T11,
          so the result is reported in the VISION table next to the VLMs.

Usage: python3 invoices_ocr.py <ocr_url> <ocr_model> <llm_url> <llm_model> <results_dir>
                               [--conc 1,8,32,64] [--conc-pipeline 1,16,32] [--docs DIR]
       (--conc-tuberia is accepted as an alias of --conc-pipeline)
Writes <dir>/ocr.json (phase A) and <dir>/vision.json (phase B, T11 format).
"""

import argparse
import base64
import json
import os
import re
import statistics
import threading
import time
import unicodedata
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("ocr")
ap.add_argument("ocr_model")
ap.add_argument("llm")
ap.add_argument("llm_model")
ap.add_argument("out")
ap.add_argument("--conc", default="1,8,32,64")
ap.add_argument("--conc-pipeline", "--conc-tuberia", dest="conc_pipeline", default="1,16,32")
ap.add_argument("--docs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "invoices_clean"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

TRUTH = json.load(open(os.path.join(a.docs, "verdad.json")))
IMGS = {n: base64.b64encode(open(os.path.join(a.docs, n), "rb").read()).decode() for n in sorted(TRUTH)}
NAMES = sorted(IMGS)
FIELDS = ["numero", "fecha", "proveedor", "nif_proveedor", "base_imponible", "iva_pct", "total", "num_lineas"]
PROMPT = (
    "Eres un sistema de extracción de facturas. A partir del texto de la factura (salida de un OCR), devuelve SOLO "
    "un JSON con estas claves: numero (texto), fecha (AAAA-MM-DD), proveedor (razón social del emisor), "
    "nif_proveedor (texto), base_imponible (número con punto decimal), iva_pct (número), total (número con punto "
    "decimal), num_lineas (número de líneas de concepto).\n\nTEXTO DE LA FACTURA:\n"
)


# --- comparer identical to the one in invoices_vision.py (T11) ---
def norm(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    return re.sub(r"[^a-z0-9]", "", "".join(c for c in s if unicodedata.category(c) != "Mn"))


def num(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"[^\d,.\-]", "", str(x))
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    return float(s)


def compare(gt, js):
    ok = {}
    for c in FIELDS:
        try:
            v = js.get(c)
            if c in ("base_imponible", "total"):
                ok[c] = abs(num(v) - gt[c]) < 0.011
            elif c in ("iva_pct", "num_lineas"):
                ok[c] = abs(num(v) - gt[c]) < 0.01
            elif c == "proveedor":
                ok[c] = norm(gt[c])[:12] in norm(v)
            elif c == "nif_proveedor":
                ok[c] = norm(v).removeprefix("es") == norm(gt[c]).removeprefix("es")  # ESB12345678 = B12345678
            else:
                ok[c] = norm(v) == norm(gt[c])
        except Exception:
            ok[c] = False
    return ok


def post(url, body, timeout=600):
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ocr(name):
    body = {
        "model": a.ocr_model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + IMGS[name]}},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ],
    }
    r = post(a.ocr, body)
    return r["choices"][0]["message"].get("content") or "", r.get("usage", {})


def extract(text):
    body = {
        "model": a.llm_model,
        "temperature": 0,
        "max_tokens": 2048,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": PROMPT + text}],
    }
    r = post(a.llm, body)
    txt = r["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    return json.loads(m.group(0)) if m else {}


def load_test(c, n, fn):
    """n tasks shared among c threads; returns (results, seconds)."""
    res, lock, idx = [], threading.Lock(), [0]
    t0 = time.time()

    def worker():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= n:
                return
            t = time.time()
            try:
                r = fn(NAMES[i % len(NAMES)])
                r.update(ok=True)
            except Exception as e:
                r = {"ok": False, "error": str(e)[:200]}
            r["s"] = time.time() - t
            with lock:
                res.append(r)

    hs = [threading.Thread(target=worker) for _ in range(c)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    return res, time.time() - t0


def q(v, p):
    return round(sorted(v)[min(len(v) - 1, int(p * len(v)))], 2) if v else None


# ---------------- Phase A: OCR only ----------------
def ocr_task(name):
    txt, u = ocr(name)
    return {"out_tok": u.get("completion_tokens"), "in_tok": u.get("prompt_tokens"), "chars": len(txt)}


out_a = {"modelo": a.ocr_model, "documentos": len(NAMES), "niveles": []}
sample, _ = ocr(NAMES[0])  # warm-up + sample for the report
out_a["ejemplo_salida"] = sample[:1500]
for c in [int(x) for x in a.conc.split(",")]:
    res, T = load_test(c, max(2 * c, 40), ocr_task)
    ok = [r for r in res if r["ok"]]
    lat = [r["s"] for r in ok]
    lvl = {
        "conc": c,
        "peticiones": len(res),
        "fallos": len(res) - len(ok),
        "segundos": round(T, 1),
        "paginas_min": round(len(ok) / T * 60, 1),
        "lat_p50_s": q(lat, 0.5),
        "lat_p95_s": q(lat, 0.95),
        "out_tok_media": round(statistics.mean(r["out_tok"] for r in ok if r["out_tok"])) if ok else None,
        "in_tok_media": round(statistics.mean(r["in_tok"] for r in ok if r["in_tok"])) if ok else None,
    }
    out_a["niveles"].append(lvl)
    print(
        f"  OCR conc {c:>3}: {lvl['paginas_min']:>7} pages/min · p50 {lvl['lat_p50_s']} s · p95 {lvl['lat_p95_s']} s · "
        f"output tokens {lvl['out_tok_media']} · failures {lvl['fallos']}",
        flush=True,
    )
    json.dump(out_a, open(os.path.join(a.out, "ocr.json"), "w"), ensure_ascii=False, indent=1)
    if lvl["fallos"] > 0.1 * max(lvl["peticiones"], 1) or (lvl["lat_p95_s"] or 0) > 120:
        break


# ---------------- Phase B: OCR -> LLM pipeline (T11 format) ----------------
def pipeline_task(name):
    t = time.time()
    txt, _ = ocr(name)
    t_ocr = time.time() - t
    js = extract(txt)
    return {"campos": compare(TRUTH[name], js), "s_ocr": t_ocr}


out_b = {"modelo": f"{a.ocr_model} + {a.llm_model}", "documentos": len(NAMES), "niveles": []}
for c in [int(x) for x in a.conc_pipeline.split(",")]:
    res, T = load_test(c, max(2 * c, 40), pipeline_task)
    ok = [r for r in res if r["ok"]]
    lat = [r["s"] for r in ok]
    accuracy = {k: (sum(r["campos"][k] for r in ok) / len(ok) if ok else 0) for k in FIELDS}
    lvl = {
        "conc": c,
        "peticiones": len(res),
        "fallos": len(res) - len(ok),
        "segundos": round(T, 1),
        "paginas_min": round(len(ok) / T * 60, 1),
        "lat_p50_s": q(lat, 0.5),
        "lat_p95_s": q(lat, 0.95),
        "lat_ocr_p50_s": q([r["s_ocr"] for r in ok], 0.5),
        "acierto_campos_pct": round(100 * statistics.mean(accuracy.values()), 1),
        "facturas_perfectas_pct": round(100 * sum(all(r["campos"].values()) for r in ok) / len(ok), 1) if ok else 0,
        "acierto_por_campo": {k: round(100 * v, 1) for k, v in accuracy.items()},
        "errores": [r.get("error") for r in res if not r["ok"]][:3],
    }
    out_b["niveles"].append(lvl)
    print(
        f"  OCR+LLM conc {c:>3}: {lvl['paginas_min']:>7} pages/min · p50 {lvl['lat_p50_s']} s (OCR {lvl['lat_ocr_p50_s']} s) · "
        f"fields {lvl['acierto_campos_pct']} % · perfect invoices {lvl['facturas_perfectas_pct']} % · failures {lvl['fallos']}",
        flush=True,
    )
    json.dump(out_b, open(os.path.join(a.out, "vision.json"), "w"), ensure_ascii=False, indent=1)
    if lvl["fallos"] > 0.1 * max(lvl["peticiones"], 1) or (lvl["lat_p95_s"] or 0) > 120:
        break
print("  Saved:", a.out)
