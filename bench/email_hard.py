#!/usr/bin/env python3
"""
email_hard.py — T22: HARD customer-email classification, and the hard corpus for d1-3B / Laya
(email_classify.py ... hard).

The 300 emails of email_classify.py reach 100 % with gpt-oss: clean templates, one topic per email. Here, 240 emails
with what a real customer-service inbox looks like (ground truth comes from the template):
  several topics at once (with a priority rule), misleading words (pro-forma invoice = price enquiry), phishing that
  talks about an "invoice", auto-replies, threads with quoted text (>), irony, typos and WhatsApp style,
  Catalan/English/Portuguese, "URGENTE" in the subject out of habit and real emergencies hidden in the body.
The priority rules are given to the model explicitly (as in an internal routing procedure). Each email is sent to the
chat endpoint asking for a JSON with category and urgency; writes one JSON with accuracy and speed
(clasif_dificil.json in suite.sh).

Usage (T22, any LLM already running): python3 email_hard.py <base_url> <model> <out.json> [--conc 32]
"""

import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

CATS = ["pedido", "incidencia", "factura", "devolucion", "stock", "otro"]
CRITERIA = {
    "pedido": "Encarga material nuevo o amplía un pedido (aunque sea condicionado a que haya stock)",
    "incidencia": "Problema con una entrega: retraso, rotura, faltan cajas o material equivocado",
    "factura": "Errores, dudas, copias o cobros de facturas ya emitidas",
    "devolucion": "Quiere devolver productos en buen estado o pedir un abono sin que haya un problema de entrega",
    "stock": "Pregunta disponibilidad, precio, plazo o pide presupuesto o factura proforma antes de comprar",
    "otro": "Publicidad, spam, phishing, respuestas automáticas o mensajes que no tienen que ver con pedidos",
}
RULES = (
    "Reglas: 1) si el correo mezcla varios temas, elige el de mayor prioridad: incidencia > factura > devolucion > "
    "pedido > stock > otro. 2) Ignora el texto citado de mensajes anteriores (líneas que empiezan por '>'). "
    "3) Una factura proforma o un presupuesto es stock. 4) Los avisos de factura de remitentes desconocidos con "
    "enlaces y las respuestas automáticas son otro. 5) Urgente solo si pide atención para hoy o mañana o hay una "
    "obra o actividad parada; la palabra URGENTE puesta por costumbre en el asunto no cuenta."
)

rnd = random.Random(2027)
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
# (category, urgent, template)
TEMPLATES = [
    (
        "incidencia",
        False,
        "Os paso pedido de {n} cajas de {p} para {c}. Por cierto, el {ped} llegó con {k} cajas rotas, ¿cómo lo solucionamos?",
    ),
    (
        "incidencia",
        True,
        "Hola, de lo demás sin prisa, pero lo del {ped} sí corre: falta todo el {p} y tenemos la obra parada desde esta mañana.",
    ),
    (
        "factura",
        False,
        "¿Tenéis {p} para la semana que viene? Ah, y la factura {fac} lleva mal nuestro CIF, rectificadla por favor.",
    ),
    (
        "stock",
        False,
        "Necesitamos una factura proforma de {n} cajas de {p} para que nos aprueben la compra. ¿Precio y plazo a {c}?",
    ),
    (
        "stock",
        False,
        "Asunto: URGENTE\n\nBuenas, ponemos urgente por costumbre, no corre prisa. ¿Qué precio tenéis para {n} cajas de {p}?",
    ),
    (
        "otro",
        False,
        "Estimado cliente: su factura nº {fac} está pendiente de pago. Descárguela y abónela aquí: http://pagos-factura-segura.example/{k}",
    ),
    (
        "otro",
        False,
        "Respuesta automática: estoy fuera de la oficina hasta el día {k}. Para pedidos urgentes llame a centralita.",
    ),
    (
        "devolucion",
        False,
        "Perfecto, gracias. Una cosa más: queremos devolver {k} cajas de {p} que nos sobraron, están sin abrir.\n\n> El {ped} ha llegado con cajas rotas\n> ¿Cómo lo solucionamos?",
    ),
    (
        "pedido",
        False,
        "Gracias por el abono.\n\n> Hemos tramitado el abono de la factura {fac}\n\nYa que estamos, mandadnos {n} cajas de {p} a {c}.",
    ),
    (
        "incidencia",
        False,
        "Genial, tercera semana esperando el {ped}. Seguimos sin {p} en {c}. Fantástico servicio, de verdad.",
    ),
    ("pedido", True, "hola!! nos hace falta q nos mandeis {n} cajas d {p} xa mañana sin falta q si no paramos la obra"),
    ("pedido", False, "Si tenéis {p} en stock mandadnos {n} cajas; si no, avisadnos y lo pedimos a otro."),
    (
        "incidencia",
        False,
        "Nos enviasteis {p} en vez de lo pedido en el {ped}. Queremos devolverlo y que nos lo cambiéis.",
    ),
    (
        "factura",
        False,
        "Bon dia, a la factura de la comanda {ped} ens heu cobrat dues vegades el transport. Podeu revisar-ho?",
    ),
    (
        "incidencia",
        True,
        "Hi, delivery {ped} arrived this morning with {k} boxes of {p} damaged. We need replacements today, our site is stopped.",
    ),
    (
        "devolucion",
        False,
        "Olá, gostaríamos de devolver {k} caixas de {p} da encomenda {ped}. Estão em perfeito estado.",
    ),
    ("stock", False, "¿Me confirmáis si os queda {p}? Es para presupuestar una obra que empieza el mes que viene."),
    (
        "factura",
        False,
        "Os reenvío la factura {fac}: habéis aplicado la tarifa antigua y el importe no coincide con el presupuesto aceptado.",
    ),
    (
        "otro",
        False,
        "Hola, soy de una consultora de eficiencia energética. ¿Os interesaría una auditoría gratuita de vuestra nave de {c}?",
    ),
    (
        "devolucion",
        False,
        "Nos equivocamos nosotros al pedir el {ped}: pedimos {p} y queríamos otra referencia. ¿Podemos devolverlo y que nos abonéis?",
    ),
]


def build_corpus(n=240):
    out = []
    for i in range(n):
        cat, urg, tpl = TEMPLATES[i % len(TEMPLATES)]
        cust, name = rnd.choice(CUSTOMERS)
        txt = tpl.format(
            n=rnd.randint(5, 200),
            k=rnd.randint(2, 28),
            p=rnd.choice(PRODUCTS),
            c=rnd.choice(CITIES),
            ped=f"P-{rnd.randint(1000, 1099)}",
            fac=f"F-2026/{rnd.randint(100, 999)}",
        )
        if not txt.startswith("Asunto:"):
            txt = f"Asunto: {txt.splitlines()[0][:38]}...\n\n{txt}"
        signature = "" if cat == "otro" else f"\n\n{name} · {cust}"
        out.append({"id": i, "tipo": i % len(TEMPLATES), "texto": txt + signature, "categoria": cat, "urgente": urg})
    rnd.shuffle(out)
    return out


EMAILS = build_corpus()
INSTR = (
    "Clasifica este correo de un cliente de una distribuidora. Categorías:\n"
    + "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    + "\n"
    + RULES
    + '\nDevuelve SOLO un JSON: {"categoria": "<una de las categorías>", "urgente": true|false}\n\nCORREO:\n'
)


def score(preds, emails=EMAILS):
    n = len(emails)
    cat = sum(p["categoria"] == c["categoria"] for p, c in zip(preds, emails))
    urg = sum(p["urgente"] == c["urgente"] for p, c in zip(preds, emails))
    both = sum(p["categoria"] == c["categoria"] and p["urgente"] == c["urgente"] for p, c in zip(preds, emails))
    by_type = {}
    for p, c in zip(preds, emails):
        by_type.setdefault(c["tipo"], []).append(p["categoria"] == c["categoria"])
    return {
        "acierto_categoria_pct": round(100 * cat / n, 1),
        "acierto_urgencia_pct": round(100 * urg / n, 1),
        "acierto_ambos_pct": round(100 * both / n, 1),
        "peor_tipo": sorted(((round(100 * sum(v) / len(v)), TEMPLATES[k][2][:60]) for k, v in by_type.items()))[:3],
    }


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
        sys.exit(__doc__)
    BASE, MODEL, OUT = sys.argv[1:4]
    CONC = int(sys.argv[sys.argv.index("--conc") + 1]) if "--conc" in sys.argv else 32
    MODE = {"kwargs": "gpt-oss" not in MODEL}

    def one(c):
        body = {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": INSTR + c["texto"]}],
        }
        if "gpt-oss" in MODEL:
            body["reasoning_effort"] = "low"
        for _ in range(2):
            if MODE["kwargs"]:
                body["chat_template_kwargs"] = {"enable_thinking": False, "thinking": False}
            else:
                body.pop("chat_template_kwargs", None)
            t = time.time()
            try:
                req = urllib.request.Request(
                    BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
                )
                r = json.load(urllib.request.urlopen(req, timeout=300))
                break
            except urllib.error.HTTPError as e:
                if e.code == 400 and "chat_template_kwargs" in body:
                    MODE["kwargs"] = False
                    continue
                return {"categoria": "?", "urgente": None, "s": None, "tok": 0}
            except Exception:
                return {"categoria": "?", "urgente": None, "s": None, "tok": 0}
        else:
            return {"categoria": "?", "urgente": None, "s": None, "tok": 0}
        txt = r["choices"][0]["message"].get("content") or ""
        try:
            js = json.loads(re.findall(r"\{[^{}]*\}", txt, re.S)[-1])
            cat = str(js.get("categoria", "")).strip().lower().replace("ó", "o")
            return {
                "categoria": cat,
                "urgente": bool(js.get("urgente")),
                "s": time.time() - t,
                "tok": (r.get("usage") or {}).get("completion_tokens", 0),
            }
        except Exception:
            return {"categoria": "?", "urgente": None, "s": time.time() - t, "tok": 0}

    preds = [None] * len(EMAILS)
    idx = [0]
    lock = threading.Lock()

    def w():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(EMAILS):
                return
            preds[i] = one(EMAILS[i])

    t0 = time.time()
    hs = [threading.Thread(target=w) for _ in range(CONC)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    Tt = time.time() - t0
    lat = sorted(p["s"] for p in preds if p["s"])
    r = {
        "modelo": MODEL,
        "correos": len(EMAILS),
        "conc": CONC,
        "modo": "sin razonamiento" if MODE["kwargs"] or "gpt-oss" in MODEL else "por defecto",
    }
    r.update(score(preds))
    r.update(
        {
            "correos_s": round(len(EMAILS) / Tt, 1),
            "lat_p50_ms": round(1000 * lat[len(lat) // 2]) if lat else None,
            "lat_p95_ms": round(1000 * lat[int(0.95 * len(lat))]) if lat else None,
            "tokens_salida_media": round(statistics.mean(p["tok"] for p in preds)),
            "fallos_formato": sum(p["categoria"] == "?" for p in preds),
        }
    )
    json.dump(r, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(
        f"  T22 hard classification: category {r['acierto_categoria_pct']} % · urgency {r['acierto_urgencia_pct']} % · "
        f"both {r['acierto_ambos_pct']} % · {r['correos_s']} emails/s · format failures {r['fallos_formato']}",
        flush=True,
    )
