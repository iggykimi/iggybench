#!/usr/bin/env python3
"""
invoices_vision.py — T11 / T11D: reading scanned invoices into JSON with a vision model (stdlib only).

Usage:
  python3 invoices_vision.py <base_url> <model> <out.json> [--conc 1,4,16,32,64] [--docs DIR] [--duration S]

For each concurrency level it sends invoices (JPG ~1240x1754, scan-like) and asks for a JSON with 8 fields.
Measures pages/minute, p50/p95 latency, image tokens and per-field accuracy against verdad.json in DIR
(default data/invoices_clean; T11D uses data/invoices_degraded). Writes vision.json / vision_dificil.json in suite.sh.
Stops as soon as a level is too slow (p95 > 120 s) or more than 10 % of requests fail.
With --duration S and a single level, it repeats invoices for S seconds (constant load, for the platform test).
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
ap.add_argument("base")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--conc", default="1,4,16,32,64")
ap.add_argument("--docs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "invoices_clean"))
ap.add_argument("--duration", type=int, default=0)
a = ap.parse_args()

TRUTH = json.load(open(os.path.join(a.docs, "verdad.json")))
IMGS = {n: base64.b64encode(open(os.path.join(a.docs, n), "rb").read()).decode() for n in sorted(TRUTH)}
NAMES = sorted(IMGS)
PROMPT = (
    "Eres un sistema de extracción de facturas. Devuelve SOLO un JSON con estas claves: "
    "numero (texto), fecha (AAAA-MM-DD), proveedor (razón social del emisor), nif_proveedor (texto), "
    "base_imponible (número con punto decimal), iva_pct (número), total (número con punto decimal), "
    "num_lineas (número de líneas de concepto)."
)
FIELDS = ["numero", "fecha", "proveedor", "nif_proveedor", "base_imponible", "iva_pct", "total", "num_lineas"]


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


def one(name):
    body = {
        "model": a.model,
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + IMGS[name]}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    }
    t = time.time()
    try:
        req = urllib.request.Request(
            a.base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        dt = time.time() - t
        txt = resp["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", txt, re.S)
        js = json.loads(m.group(0)) if m else {}
        u = resp.get("usage", {})
        return {
            "doc": name,
            "ok": True,
            "s": dt,
            "campos": compare(TRUTH[name], js),
            "pred": js,
            "in_tok": u.get("prompt_tokens"),
            "out_tok": u.get("completion_tokens"),
        }
    except Exception as e:
        return {"doc": name, "ok": False, "s": time.time() - t, "error": str(e)[:200]}


def run_level(c, n=None, dur=0):
    res, lock, idx = [], threading.Lock(), [0]
    t0 = time.time()

    def worker():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if (dur and time.time() - t0 > dur) or (not dur and i >= n):
                return
            r = one(NAMES[i % len(NAMES)])
            with lock:
                res.append(r)

    hs = [threading.Thread(target=worker) for _ in range(c)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    T = time.time() - t0
    good = [r for r in res if r["ok"]]
    lat = sorted(r["s"] for r in good)

    def p(q):
        return lat[min(len(lat) - 1, int(q * len(lat)))] if lat else None

    accuracy = {c_: (sum(r["campos"][c_] for r in good) / len(good) if good else 0) for c_ in FIELDS}
    perfect = sum(all(r["campos"].values()) for r in good)
    return {
        "conc": c,
        "peticiones": len(res),
        "fallos": len(res) - len(good),
        "segundos": round(T, 1),
        "paginas_min": round(len(good) / T * 60, 1) if T else 0,
        "lat_p50_s": round(p(0.5), 2) if lat else None,
        "lat_p95_s": round(p(0.95), 2) if lat else None,
        "in_tok_media": round(statistics.mean(r["in_tok"] for r in good if r["in_tok"]), 0) if good else None,
        "out_tok_media": round(statistics.mean(r["out_tok"] for r in good if r["out_tok"]), 0) if good else None,
        "acierto_campos_pct": round(100 * statistics.mean(accuracy.values()), 1),
        "facturas_perfectas_pct": round(100 * perfect / len(good), 1) if good else 0,
        "acierto_por_campo": {k: round(100 * v, 1) for k, v in accuracy.items()},
        "errores": [r.get("error") for r in res if not r["ok"]][:3],
        "detalle": list(
            {
                r["doc"]: {
                    "doc": r["doc"],
                    "fallan": [k for k, v in r["campos"].items() if not v],
                    "pred": r.get("pred"),
                }
                for r in reversed(good)
            }.values()
        ),
    }


levels = [int(x) for x in a.conc.split(",")]
out = {"modelo": a.model, "documentos": len(NAMES), "niveles": []}
for c in levels:
    r = run_level(c, dur=a.duration) if a.duration else run_level(c, n=max(2 * c, len(NAMES)))
    out["niveles"].append(r)
    print(
        f"  conc {c:>3}: {r['paginas_min']:>7} pages/min · p50 {r['lat_p50_s']} s · p95 {r['lat_p95_s']} s · "
        f"fields {r['acierto_campos_pct']} % · perfect invoices {r['facturas_perfectas_pct']} % · "
        f"failures {r['fallos']} · image+prompt tokens {r['in_tok_media']}",
        flush=True,
    )
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    if r["fallos"] > 0.1 * max(r["peticiones"], 1) or (r["lat_p95_s"] or 0) > 120:
        print("  (stopping: this level is no longer usable)")
        break
print("  Saved:", a.out)
