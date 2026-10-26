#!/usr/bin/env python3
"""
needle.py — T24: needle in a haystack at extreme context lengths. Hides an access code («El código de acceso de la
sala N es XXXX-XXXX») at 10 %, 50 % and 90 % depth in a document of 128k, 512k and ~1M tokens of varied filler text
(different seed per request so prefix caching does not help), asks for it and records whether the model finds it and
how long it takes. Writes one JSON (aguja.json in the original runs).

Usage: python3 needle.py <base_url> <model> <out.json> [--lengths 128000,512000,1000000] [--depths 0.1,0.5,0.9]
       (--longitudes / --prof are accepted as aliases)
"""

import json
import random
import sys
import time
import urllib.request

if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
BASE, MODEL, OUT = sys.argv[1:4]


def arg(keys, d):
    for k in keys:
        if k in sys.argv:
            return sys.argv[sys.argv.index(k) + 1]
    return d


LENGTHS = [int(x) for x in arg(("--lengths", "--longitudes"), "128000,512000,1000000").split(",")]
DEPTHS = [float(x) for x in arg(("--depths", "--prof"), "0.1,0.5,0.9").split(",")]
SUBJ = [
    "El equipo de mantenimiento",
    "La comisión de compras",
    "El departamento de logística",
    "Un técnico de guardia",
    "La dirección financiera",
    "El responsable de calidad",
    "El almacén de Zaragoza",
    "La oficina de Huesca",
]
VERB = ["revisó", "aprobó", "aplazó", "documentó", "comparó", "auditó", "registró", "actualizó"]
OBJ = [
    "el inventario de palés",
    "las facturas del trimestre",
    "el plan de turnos",
    "los contratos de transporte",
    "el consumo eléctrico de la nave",
    "las incidencias de entrega",
    "el pedido de tornillería",
    "la ruta de reparto",
]


def post(path, body, to=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=to))


def filler(rnd, n_sentences):
    return " ".join(
        f"{rnd.choice(SUBJ)} {rnd.choice(VERB)} {rnd.choice(OBJ)} el día {rnd.randint(1, 28)} "
        f"del mes {rnd.randint(1, 12)} con la referencia {rnd.randint(10000, 99999)}."
        for _ in range(n_sentences)
    )


# calibrate tokens per sentence with the server's own tokenizer
sample = filler(random.Random(0), 400)
try:
    tps = post("/tokenize", {"model": MODEL, "prompt": sample})["count"] / 400
except Exception:
    tps = 22.0
res = {"modelo": MODEL, "tokens_por_frase": round(tps, 2), "pruebas": []}
for L in LENGTHS:
    for p in DEPTHS:
        rnd = random.Random(L * 10 + int(p * 100))
        room = rnd.randint(100, 999)
        code = f"{rnd.randint(1000, 9999)}-{rnd.randint(1000, 9999)}"
        n = int((L - 200) / tps)
        k = int(n * p)
        text = filler(rnd, k) + f" El código de acceso de la sala {room} es {code}. " + filler(rnd, n - k)
        question = (
            f"{text}\n\nPregunta: según el documento anterior, ¿cuál es el código de acceso de la sala {room}? "
            "Responde solo con el código."
        )
        t = time.time()
        try:
            r = post(
                "/v1/chat/completions",
                {
                    "model": MODEL,
                    "temperature": 0,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": question}],
                },
            )
            m = r["choices"][0]["message"]
            txt = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
            row = {
                "tokens_objetivo": L,
                "profundidad": p,
                "tokens_reales": r.get("usage", {}).get("prompt_tokens"),
                "acierta": code in txt,
                "segundos": round(time.time() - t, 1),
                "respuesta": (m.get("content") or "")[:120],
            }
        except Exception as e:
            row = {
                "tokens_objetivo": L,
                "profundidad": p,
                "acierta": False,
                "segundos": round(time.time() - t, 1),
                "error": repr(e)[:200],
            }
        res["pruebas"].append(row)
        print(
            f"  T24 needle {L // 1000}k at {int(p * 100)} %: {'OK' if row['acierta'] else 'MISS'} · "
            f"{row.get('tokens_reales')} tokens · {row['segundos']} s",
            flush=True,
        )
        json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
ok = sum(f["acierta"] for f in res["pruebas"])
res["aciertos"] = f"{ok}/{len(res['pruebas'])}"
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"  T24 {MODEL}: {res['aciertos']} needles found", flush=True)
