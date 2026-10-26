#!/usr/bin/env python3
"""
email_classify.py — Customer-email classification and routing: decision models (Liquid d1-3B and Laya) against
gpt-oss-120b. Writes <out_dir>/d1.json (scores and speed) and <out_dir>/correos.json (the corpus used).

Corpus: 300 synthetic Spanish emails from the customers of the fictitious distributor (the same as agents.py),
with their category and whether they are urgent. 6 categories: pedido, incidencia, factura, devolucion, stock, otro.
With the extra argument "hard" (or "dificil") the 240 hard emails of email_hard.py are used instead, same flow.

d1-3B  (runs INSIDE the vLLM container with transformers + trust_remote_code):
       one call per email with two questions (choice + noul) -> p50/p95 latency one at a time and decisions/s in
       batches of 32 and 128 (system_one_batch). Zero output tokens.
gpt-oss-120b (vLLM server already running): the same classification asking for a JSON, 32 in parallel.
Code that is not on PyPI (transformers from GitHub, laya) is looked up under /pylib, a read-only mount in the container.

Usage: python3 email_classify.py <d1_dir> <out_dir> <llm_url> <llm_model> [hard]
"""

import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.request

if len(sys.argv) < 5 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
D1, OUT, URL_LLM, MOD_LLM = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)


def q(v, p):
    return sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None


# ----------------------------------------------------------------------------- corpus
rnd = random.Random(2026)
CUSTOMERS = [
    ("Construcciones Moncayo", "Laura"),
    ("Reformas Huerva", "Pablo"),
    ("Talleres Ebro Norte", "Marta"),
    ("Instalaciones Cierzo", "Javier"),
    ("Hostelería Ribera", "Ana"),
    ("Comercial Pirineo", "Sergio"),
]
PRODUCTS = [
    "guantes de nitrilo talla L",
    "tornillos M8 inox",
    "cable de 2,5 mm²",
    "cinta de embalar de 48 mm",
    "palés de 1200x800",
    "rodamientos 6204-2RS",
    "bombillas LED E27",
    "tóner láser negro",
    "disolvente industrial",
]
CITIES = ["Zaragoza", "Huesca", "Teruel", "Calatayud", "Ejea", "Monzón"]
TEMPLATES = {
    "pedido": [
        "Buenos días, queremos haceros un pedido de {n} cajas de {p} con entrega en {c}. ¿Nos confirmáis plazo?",
        "Hola, necesitamos {n} unidades de {p}. Facturar a {cli} como siempre, entrega en nuestro almacén de {c}.",
        "Os paso pedido: {n} x {p}. Mismo precio que el pedido anterior {ped}, por favor.",
        "¿Podéis ampliar el pedido {ped} con {n} cajas más de {p}? Gracias.",
    ],
    "incidencia": [
        "El pedido {ped} ha llegado con {k} cajas de {p} rotas. Os mando fotos, ¿cómo lo solucionamos?",
        "Nos faltan {k} cajas de {p} del pedido {ped}; el albarán dice que venían todas.",
        "El pedido {ped} tenía que llegar ayer a {c} y aún no sabemos nada del transportista.",
        "Nos habéis enviado {p} en vez de lo que pedimos en el {ped}. Hay que cambiarlo.",
    ],
    "factura": [
        "En la factura del pedido {ped} nos habéis cobrado {n} cajas y solo recibimos {k}. ¿Podéis rectificarla?",
        "¿Nos podéis reenviar la factura del pedido {ped}? No la encontramos.",
        "Nos aparece un cargo duplicado de la última factura. Revisadlo por favor.",
        "Necesitamos que la factura del {ped} vaya a nombre de {cli} con el CIF nuevo.",
    ],
    "devolucion": [
        "Queremos devolver {k} cajas de {p} del pedido {ped}, nos sobraron y están sin abrir.",
        "¿Cómo gestionamos la devolución de {p}? No es el modelo que necesitamos y queremos el abono.",
        "Os devolvemos {k} unidades de {p} del {ped}; decidnos si pasa a recogerlas el transportista.",
    ],
    "stock": [
        "¿Tenéis stock de {p}? Necesitaríamos unas {n} cajas.",
        "¿Qué plazo de entrega tenéis ahora mismo para {p} en {c}?",
        "Pasadnos precio de {n} cajas de {p}, por favor, con transporte a {c}.",
    ],
    "otro": [
        "Le ofrecemos un 40 % de descuento en campañas de posicionamiento web para su empresa. ¡Llámenos hoy!",
        "Felicidades, ha sido seleccionado para recibir un premio. Pulse aquí para reclamarlo.",
        "Hola, soy estudiante y estoy haciendo una encuesta sobre logística. ¿Me podrían dedicar 5 minutos?",
        "Os escribimos de la asociación de vecinos para invitaros a la cena de Navidad del polígono.",
    ],
}
URGENT = [
    "Es urgente: tenemos la obra parada hasta que llegue.",
    "Lo necesitamos mañana a primera hora sin falta.",
    "Por favor, respondednos hoy, el cliente nos está reclamando.",
    "Muy urgente, nos quedamos sin material esta tarde.",
]
NOT_URGENT = ["Cuando podáis, no hay prisa.", "Sin urgencia, para la semana que viene.", "Gracias de antemano.", ""]


def make_email(i):
    cat = rnd.choices(list(TEMPLATES), weights=[25, 22, 15, 12, 14, 12])[0]
    cust, name = rnd.choice(CUSTOMERS)
    txt = rnd.choice(TEMPLATES[cat]).format(
        n=rnd.randint(5, 200),
        k=rnd.randint(1, 20),
        p=rnd.choice(PRODUCTS),
        c=rnd.choice(CITIES),
        cli=cust,
        ped=f"P-{rnd.randint(1000, 1099)}",
    )
    urg = cat in ("pedido", "incidencia", "stock") and rnd.random() < 0.4
    txt = f"Asunto: {txt[:40]}...\n\n{txt} {rnd.choice(URGENT) if urg else rnd.choice(NOT_URGENT)}\n\n{name}" + (
        "" if cat == "otro" else f" · {cust}"
    )
    return {"id": i, "texto": txt, "categoria": cat, "urgente": urg}


EMAILS = [make_email(i) for i in range(300)]
CRITERIA = {
    "pedido": "Quiere hacer un pedido nuevo o ampliar uno existente",
    "incidencia": "Problema con una entrega: retraso, rotura, faltan cajas o material equivocado",
    "factura": "Dudas, errores o copias de facturas y cobros",
    "devolucion": "Quiere devolver productos o pedir un abono",
    "stock": "Pregunta disponibilidad, precio o plazo de un producto antes de pedir",
    "otro": "Publicidad, spam o mensajes que no tienen que ver con pedidos",
}
QUESTIONS = {
    "categoria": {
        "type": "choice",
        "instructions": "¿De qué trata este correo de un cliente de una distribuidora?",
        "criteria": CRITERIA,
    },
    "urgente": {
        "type": "noul",
        "instructions": "¿Pide atención urgente hoy (obra parada, plazo inminente, cliente reclamando)?",
    },
}
EXTRA_RULES = ""
if len(sys.argv) > 5 and sys.argv[5] in ("hard", "dificil"):  # hard corpus (email_hard.py), same flow
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import email_hard as EH

    EMAILS, CRITERIA, EXTRA_RULES = EH.EMAILS, EH.CRITERIA, "\n" + EH.RULES
    QUESTIONS = {
        "categoria": {
            "type": "choice",
            "instructions": "¿De qué trata este correo de un cliente de una distribuidora? " + EH.RULES,
            "criteria": CRITERIA,
        },
        "urgente": {
            "type": "noul",
            "instructions": "¿Pide atención urgente para hoy o mañana o hay una obra o actividad parada? "
            "La palabra URGENTE puesta por costumbre en el asunto no cuenta.",
        },
    }
res = {
    "correos": len(EMAILS),
    "reparto": {c: sum(x["categoria"] == c for x in EMAILS) for c in CRITERIA},
    "urgentes": sum(x["urgente"] for x in EMAILS),
}


def save():
    json.dump(res, open(os.path.join(OUT, "d1.json"), "w"), ensure_ascii=False, indent=1)


json.dump(EMAILS, open(os.path.join(OUT, "correos.json"), "w"), ensure_ascii=False, indent=1)


def score(preds):
    cat = sum(p["categoria"] == c["categoria"] for p, c in zip(preds, EMAILS))
    urg = sum(p["urgente"] == c["urgente"] for p, c in zip(preds, EMAILS))
    both = sum(p["categoria"] == c["categoria"] and p["urgente"] == c["urgente"] for p, c in zip(preds, EMAILS))
    return {
        "acierto_categoria_pct": round(100 * cat / len(EMAILS), 1),
        "acierto_urgencia_pct": round(100 * urg / len(EMAILS), 1),
        "acierto_ambos_pct": round(100 * both / len(EMAILS), 1),
    }


# ----------------------------------------------------------------------------- d1-3B
def measure(name, single, batch):
    """Measures a decision model: latency one at a time, accuracy and emails/s in batches of 32 and 128.
    single(text) -> output; batch([texts]) -> [outputs]. Output format shared by d1 and Laya."""
    import torch

    def answer(o):
        return {"categoria": o["answers"]["categoria"]["choice"], "urgente": o["answers"]["urgente"]["noul"] > 0.5}

    for c in EMAILS[:10]:
        single(c["texto"])  # warm-up
    torch.cuda.reset_peak_memory_stats()
    lat, preds = [], []
    for c in EMAILS:  # one at a time (latency of one decision)
        torch.cuda.synchronize()
        t = time.time()
        o = single(c["texto"])
        torch.cuda.synchronize()
        lat.append(time.time() - t)
        preds.append(answer(o))
    r = score(preds)
    r.update(
        {
            "lat_p50_ms": round(1000 * q(lat, 0.5), 1),
            "lat_p95_ms": round(1000 * q(lat, 0.95), 1),
            "correos_s_1en1": round(len(lat) / sum(lat), 1),
        }
    )
    for b in (32, 128):  # batches (throughput)
        try:
            t = time.time()
            n = 0
            for _ in range(2):
                for i in range(0, len(EMAILS), b):
                    batch([c["texto"] for c in EMAILS[i : i + b]])
                    n += len(EMAILS[i : i + b])
            torch.cuda.synchronize()
            r[f"correos_s_lote{b}"] = round(n / (time.time() - t), 1)
        except Exception as e:
            r[f"correos_s_lote{b}"] = None
            r[f"error_lote{b}"] = str(e)[:150]
    r["memoria_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    r["fallos"] = [
        {"id": c["id"], "esperado": [c["categoria"], c["urgente"]], name: [p["categoria"], p["urgente"]]}
        for c, p in zip(EMAILS, preds)
        if p["categoria"] != c["categoria"]
    ][:15]
    return r


def phase_d1():
    from importlib.metadata import version

    v = version("transformers")
    if tuple(int(x) for x in re.findall(r"\d+", v)[:2]) < (5, 14) and os.path.isdir("/pylib/transformers/src"):
        sys.path.insert(0, "/pylib/transformers/src")
        v += " -> GitHub"
    import torch
    from transformers import AutoModel

    t = time.time()
    m = AutoModel.from_pretrained(D1, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    load_s = round(time.time() - t, 1)
    print(f"  d1-3B loaded ({v}) in {load_s} s", flush=True)
    r = {"modelo": "d1-3B (3,1B)", "transformers": v, "carga_s": load_s}
    r.update(
        measure("d1", lambda x: m.system_one(x, QUESTIONS), lambda xs: m.system_one_batch([(x, QUESTIONS) for x in xs]))  # noqa: F821
    )
    del m
    torch.cuda.empty_cache()
    return r


def phase_laya():
    """Laya (Apache 2.0): bidirectional encoder (mmBERT, 322M) with the same decision API as d1.
    Code from GitHub in /pylib/laya (not on PyPI); weights from Hugging Face (convaiinnovations/laya, multilingual subfolder)."""
    for p in ("/pylib/laya/src", "/pylib/laya"):
        if os.path.isdir(p):
            sys.path.insert(0, p)
    import torch  # noqa: F401
    import laya

    t = time.time()
    ag = None
    for attempt in (
        lambda: laya.load("convaiinnovations/laya", subfolder="multilingual", device="cuda"),
        lambda: laya.load("convaiinnovations/laya", subfolder="multilingual"),
        lambda: laya.Router(preload=True),
    ):
        try:
            ag = attempt()
            break
        except TypeError:
            continue
    if hasattr(ag, "to"):
        try:
            ag = ag.to("cuda") or ag
        except Exception:
            pass
    load_s = round(time.time() - t, 1)
    print(f"  Laya loaded in {load_s} s ({type(ag).__name__})", flush=True)

    def batch(xs):
        try:
            return ag.predict_batch(xs, QUESTIONS, batch_size=len(xs))
        except TypeError:
            return ag.predict_batch(xs, QUESTIONS)

    r = {"modelo": "Laya multilingual (322M)", "carga_s": load_s}
    r.update(measure("laya", lambda x: ag.predict(x, QUESTIONS), batch))
    return r


for key, phase in (("d1", phase_d1), ("laya", phase_laya)):
    try:
        res[key] = phase()
        print(f"  {key}:", {k: v for k, v in res[key].items() if k != "fallos"}, flush=True)
    except Exception as e:
        res[key] = {"error": f"{type(e).__name__}: {e}"[:400]}
        print(f"  !! {key}:", e, flush=True)
    save()

# ----------------------------------------------------------------------------- gpt-oss (general LLM)
INSTR = (
    "Clasifica este correo de un cliente de una distribuidora. Categorías:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    + EXTRA_RULES
    + "\nurgente = true si pide atención urgente hoy (obra parada, plazo inminente, cliente reclamando).\n"
    'Devuelve SOLO un JSON: {"categoria": "<una de las categorías>", "urgente": true|false}\n\nCORREO:\n'
)


def llm(c):
    body = {
        "model": MOD_LLM,
        "temperature": 0,
        "max_tokens": 512,
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": INSTR + c["texto"]}],
    }
    req = urllib.request.Request(
        URL_LLM + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    txt = resp["choices"][0]["message"].get("content") or ""
    js = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
    u = resp.get("usage", {})
    return {
        "categoria": str(js.get("categoria", "")).strip().lower(),
        "urgente": bool(js.get("urgente")),
        "s": time.time() - t,
        "out_tok": u.get("completion_tokens", 0),
    }


try:
    preds = [None] * len(EMAILS)
    lock = threading.Lock()
    idx = [0]

    def w():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(EMAILS):
                return
            try:
                preds[i] = llm(EMAILS[i])
            except Exception:
                preds[i] = {"categoria": "?", "urgente": None, "s": None, "out_tok": 0}

    t0 = time.time()
    hs = [threading.Thread(target=w) for _ in range(32)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    T = time.time() - t0
    lat = [p["s"] for p in preds if p["s"]]
    r = score(preds)
    r.update(
        {
            "modelo": MOD_LLM,
            "concurrencia": 32,
            "correos_s": round(len(EMAILS) / T, 1),
            "lat_p50_ms": round(1000 * q(lat, 0.5)) if lat else None,
            "lat_p95_ms": round(1000 * q(lat, 0.95)) if lat else None,
            "tokens_salida_media": round(statistics.mean(p["out_tok"] for p in preds)),
            "fallos_formato": sum(p["categoria"] == "?" for p in preds),
        }
    )
    res["llm"] = r
    print("  gpt-oss:", r, flush=True)
except Exception as e:
    res["llm"] = {"error": str(e)[:300]}
save()
print("  Saved:", OUT)
