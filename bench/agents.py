#!/usr/bin/env python3
"""
agents.py — T13: tool-using agents (stdlib only).

Simulates the back office of a fictitious distributor (customers, orders, stock, internal documentation) with
8 tools. The model must solve 8 multi-step tasks by calling the tools (find customer -> list orders -> calculate ->
create incident -> send email...). Each task is checked automatically against the final state (incidents created,
emails sent) and the answer. Writes one JSON with one entry per concurrency level (agentes.json in suite.sh).

Usage:
  python3 agents.py <base_url> <model> <out.json> [--conc 1,8,32,64] [--max-steps 15]

  conc 1  -> the 8 tasks once, serially ("clean" success rate)
  conc N  -> N agents working at once (repeating tasks): tasks/min and success rate under load
"""

import argparse
import json
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
ap.add_argument("--conc", default="1,8,32,64")
ap.add_argument("--max-steps", "--max-pasos", dest="max_steps", type=int, default=15)
a = ap.parse_args()

TODAY = "2026-10-21"
CUSTOMERS = {
    "C001": {"nombre": "Talleres Moncayo S.A.", "email": "compras@moncayo.example", "ciudad": "Zaragoza"},
    "C002": {"nombre": "Comercial Pirineo S.A.", "email": "admin@pirineo.example", "ciudad": "Huesca"},
    "C003": {"nombre": "Hostelería Ribera S.L.", "email": "pedidos@ribera.example", "ciudad": "Tudela"},
    "C004": {"nombre": "Industrias Bardenas S.L.", "email": "logistica@bardenas.example", "ciudad": "Tarazona"},
}


def order(i, c, f, e, fe, lines):
    return {"id": i, "cliente_id": c, "fecha_pedido": f, "estado": e, "fecha_entrega": fe, "lineas": lines}


ORDERS = [
    order(
        "P-1031",
        "C001",
        "2026-09-02",
        "entregado",
        "2026-09-05",
        [["Tornillo M8", 400, 0.12], ["Guantes nitrilo", 10, 6.5]],
    ),
    order(
        "P-1052",
        "C001",
        "2026-10-14",
        "entregado",
        "2026-10-17",
        [["Rodamiento 6204", 50, 3.8], ["Tornillo M8", 200, 0.12]],
    ),
    order("P-1060", "C001", "2026-10-18", "pendiente", None, [["Tornillo M8", 600, 0.12]]),
    order(
        "P-1035",
        "C002",
        "2026-09-08",
        "entregado",
        "2026-09-12",
        [["Cable 2,5 mm", 30, 41.0], ["Bombilla LED", 120, 2.9]],
    ),
    order("P-1040", "C002", "2026-09-20", "entregado", "2026-09-24", [["Tóner láser", 6, 88.5]]),
    order("P-1049", "C002", "2026-10-01", "entregado", "2026-10-03", [["Papel A4", 40, 21.0]]),
    order(
        "P-1058", "C002", "2026-10-15", "retrasado", None, [["Tornillo M8", 500, 0.12], ["Guantes nitrilo", 25, 6.5]]
    ),
    order("P-1043", "C003", "2026-09-05", "entregado", "2026-09-10", [["Vasos compostables (caja)", 30, 14.2]]),
    order("P-1061", "C003", "2026-10-19", "pendiente", None, [["Tornillo M8", 250, 0.12]]),
    order(
        "P-1038",
        "C004",
        "2026-09-15",
        "entregado",
        "2026-09-18",
        [["Palé madera", 80, 11.0], ["Disolvente industrial 5 L", 12, 23.4]],
    ),
    order("P-1055", "C004", "2026-10-10", "retrasado", None, [["Junta tórica 40 mm", 300, 0.45]]),
    order("P-1062", "C004", "2026-10-20", "pendiente", None, [["Tornillo M8", 500, 0.12], ["Palé madera", 20, 11.0]]),
]
for p in ORDERS:
    p["importe_sin_iva"] = round(sum(q * pr for _, q, pr in p["lineas"]), 2)
STOCK = {
    ("tornillo m8", "zaragoza"): 1200,
    ("tornillo m8", "huesca"): 300,
    ("guantes nitrilo", "zaragoza"): 40,
    ("guantes nitrilo", "huesca"): 85,
    ("palé madera", "zaragoza"): 150,
    ("rodamiento 6204", "zaragoza"): 75,
}
DOCS = [
    {
        "titulo": "Política de devoluciones",
        "texto": "Se aceptan devoluciones en un plazo máximo de 30 días naturales desde la fecha de entrega del pedido. "
        "Los productos químicos (disolventes, ácidos) no admiten devolución. La mercancía debe estar sin usar.",
    },
    {
        "titulo": "Procedimiento de incidencias",
        "texto": "Toda incidencia por material roto o dañado se registra con la herramienta crear_incidencia indicando el "
        "motivo 'rotura'. Después se notifica al cliente por email incluyendo el número de incidencia asignado.",
    },
    {
        "titulo": "Tarifas de envío",
        "texto": "Envío gratuito en pedidos superiores a 300 € sin IVA. Por debajo, 12 € por pedido.",
    },
    {"titulo": "IVA aplicable", "texto": "Todos los productos del catálogo tributan al 21 % de IVA."},
]


def uni(s):
    """equivalent typography: Unicode dashes (‑ – —) and thin/non-breaking spaces -> ASCII"""
    return re.sub(r"[‐-―−]", "-", str(s)).replace(" ", " ").replace(" ", " ").replace(" ", " ")


def norm(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


# ---------------------------------------------------------------- tools
# Tool names, parameters and descriptions are sent to the model: they stay as they were (Spanish).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "buscar_cliente",
            "description": "Busca clientes por nombre (o parte del nombre).",
            "parameters": {"type": "object", "properties": {"nombre": {"type": "string"}}, "required": ["nombre"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listar_pedidos",
            "description": "Lista los pedidos de un cliente, opcionalmente filtrando por estado (pendiente, entregado, retrasado).",
            "parameters": {
                "type": "object",
                "properties": {"cliente_id": {"type": "string"}, "estado": {"type": "string"}},
                "required": ["cliente_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "listar_clientes",
            "description": "Devuelve todos los clientes con su id.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consultar_stock",
            "description": "Unidades disponibles de un producto en un almacén (Zaragoza o Huesca).",
            "parameters": {
                "type": "object",
                "properties": {"producto": {"type": "string"}, "almacen": {"type": "string"}},
                "required": ["producto", "almacen"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_documentacion",
            "description": "Busca en la documentación interna (políticas y procedimientos).",
            "parameters": {"type": "object", "properties": {"consulta": {"type": "string"}}, "required": ["consulta"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calcular",
            "description": "Evalúa una expresión aritmética, p. ej. '(120.5+30)*1.21'.",
            "parameters": {
                "type": "object",
                "properties": {"expresion": {"type": "string"}},
                "required": ["expresion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crear_incidencia",
            "description": "Registra una incidencia sobre un pedido. Devuelve el número de incidencia.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pedido_id": {"type": "string"},
                    "motivo": {"type": "string"},
                    "descripcion": {"type": "string"},
                },
                "required": ["pedido_id", "motivo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enviar_email",
            "description": "Envía un email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destinatario": {"type": "string"},
                    "asunto": {"type": "string"},
                    "cuerpo": {"type": "string"},
                },
                "required": ["destinatario", "asunto", "cuerpo"],
            },
        },
    },
]


class Environment:
    # Methods t_<tool name> implement each tool; their return values go back to the model verbatim.
    def __init__(self, seed):
        self.incidents, self.emails, self.calls, self.seed = [], [], [], seed

    def run(self, name, args):
        self.calls.append(name)
        try:
            return getattr(self, "t_" + name)(**args)
        except TypeError as e:
            return {"error": f"argumentos no válidos: {e}"}
        except AttributeError:
            return {"error": f"la herramienta {name} no existe"}

    def t_buscar_cliente(self, nombre):
        r = [{"id": k, **v} for k, v in CUSTOMERS.items() if norm(nombre).strip(" .") in norm(v["nombre"])]
        return r or {"error": f"no hay ningún cliente que coincida con '{nombre}'"}

    def t_listar_clientes(self):
        return [{"id": k, "nombre": v["nombre"]} for k, v in CUSTOMERS.items()]

    def t_listar_pedidos(self, cliente_id, estado=None):
        if cliente_id not in CUSTOMERS:
            return {"error": f"cliente {cliente_id} no existe"}
        r = [p for p in ORDERS if p["cliente_id"] == cliente_id and (not estado or norm(estado) == p["estado"])]
        return [{k: v for k, v in p.items() if k != "cliente_id"} for p in r]

    def t_consultar_stock(self, producto, almacen):
        k = (norm(producto).strip(), norm(almacen).strip())
        for (pr, al), n in STOCK.items():
            if pr in k[0] and al in k[1]:
                return {"producto": producto, "almacen": almacen, "unidades": n}
        return {"producto": producto, "almacen": almacen, "unidades": 0}

    def t_buscar_documentacion(self, consulta):
        q = set(re.findall(r"\w{4,}", norm(consulta)))
        sc = sorted(DOCS, key=lambda d: -len(q & set(re.findall(r"\w{4,}", norm(d["titulo"] + " " + d["texto"])))))
        return sc[:2]

    def t_calcular(self, expresion):
        if not re.fullmatch(r"[\d\s.+\-*/()%,]+", expresion):
            return {"error": "solo se admiten números y + - * / ( )"}
        try:
            return {"resultado": round(eval(expresion.replace(",", "."), {"__builtins__": {}}), 4)}
        except Exception as e:
            return {"error": str(e)}

    def t_crear_incidencia(self, pedido_id, motivo, descripcion=""):
        pid = uni(pedido_id).strip()
        if pid not in {p["id"] for p in ORDERS}:  # like a real system: the order must exist
            return {"error": f"el pedido {pedido_id} no existe"}
        i = f"INC-{7000 + self.seed * 10 + len(self.incidents)}"
        self.incidents.append({"id": i, "pedido_id": pid, "motivo": motivo})
        return {"incidencia_id": i}

    def t_enviar_email(self, destinatario, asunto, cuerpo):
        self.emails.append({"to": norm(destinatario).strip(), "texto": uni(asunto + "\n" + cuerpo)})
        return {"ok": True}


def numbers_in(txt):
    t = txt.replace(" ", " ")
    out = []
    for m in re.findall(r"\d[\d.,\s]*\d|\d", t):
        s = m.replace(" ", "")
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
        elif "," in s:
            s = s.replace(",", ".") if len(s.split(",")[-1]) != 3 else s.replace(",", "")
        elif s.count(".") == 1 and len(s.split(".")[-1]) == 3 and len(s) > 4:
            s = s.replace(".", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------- tasks and checks
invoiced_sep = round(
    sum(
        p["importe_sin_iva"]
        for p in ORDERS
        if p["cliente_id"] == "C002" and p["estado"] == "entregado" and p["fecha_entrega"].startswith("2026-09")
    )
    * 1.21,
    2,
)
pending_m8 = sum(q for p in ORDERS if p["estado"] == "pendiente" for pr, q, _ in p["lineas"] if pr == "Tornillo M8")
short_m8 = pending_m8 - STOCK[("tornillo m8", "zaragoza")]


def check_incident(e, r):
    inc = [i for i in e.incidents if i["pedido_id"] == "P-1052" and "rotura" in norm(i["motivo"])]
    return bool(inc) and any(m["to"] == "compras@moncayo.example" and inc[-1]["id"] in m["texto"] for m in e.emails)


def check_stock(e, r):
    n = norm(r)
    return abs(short_m8) in numbers_in(r) and (("no" in n.split()) or "faltan" in n or "insuficiente" in n)


def check_invoiced(e, r):
    return any(abs(x - invoiced_sep) < 0.011 for x in numbers_in(r))


def check_return(e, r):
    n = norm(r)[:400]
    return bool(re.search(r"\bno\b", n)) and "30" in n


def check_delayed(e, r):
    dest = {m["to"]: m["texto"] for m in e.emails}
    return (
        set(dest) == {"admin@pirineo.example", "logistica@bardenas.example"}
        and "P-1058" in dest["admin@pirineo.example"]
        and "P-1055" in dest["logistica@bardenas.example"]
    )


def check_unknown_customer(e, r):
    n = norm(r)
    return not re.search(r"P-\d{4}", r) and any(
        k in n for k in ("no existe", "no encuentr", "no hay", "no figura", "no aparece", "ningun")
    )


def check_gloves(e, r):
    return 125 in numbers_in(r) and "huesca" in norm(r)


def check_largest(e, r):
    return "P-1038" in r and "pale" in norm(r)


# (task id written to the output, prompt sent to the model, check)
TASKS = [
    (
        "incidencia_rotura",
        "Talleres Moncayo nos dice que su pedido entregado más reciente ha llegado con material roto. "
        "Registra la incidencia según nuestro procedimiento interno y avisa al cliente por email con el número de incidencia.",
        check_incident,
    ),
    (
        "stock_pendientes",
        "¿Tenemos en el almacén de Zaragoza suficientes unidades de 'Tornillo M8' para servir todos los pedidos "
        "pendientes de ese producto, de todos los clientes? Responde sí o no y cuántas unidades sobran o faltan.",
        check_stock,
    ),
    (
        "facturacion",
        "¿Cuánto hemos facturado, IVA incluido, a Comercial Pirineo por los pedidos entregados en septiembre de 2026? "
        "Da la cifra en euros con dos decimales.",
        check_invoiced,
    ),
    (
        "devolucion",
        "Hostelería Ribera quiere devolver el pedido P-1043. ¿Es posible según nuestra política? Empieza la respuesta "
        "con sí o no y justifícalo brevemente.",
        check_return,
    ),
    (
        "emails_retrasados",
        "Envía un email a cada cliente que tenga algún pedido en estado 'retrasado', disculpándote e indicando "
        "el número de pedido afectado. No escribas a clientes sin pedidos retrasados.",
        check_delayed,
    ),
    ("cliente_inexistente", "Dime los pedidos pendientes del cliente 'Ferretería Los Arcos'.", check_unknown_customer),
    (
        "comparar_almacenes",
        "¿En qué almacén hay más unidades de 'Guantes nitrilo', Zaragoza o Huesca, y cuántas hay entre los dos?",
        check_gloves,
    ),
    (
        "pedido_mayor",
        "¿Cuál es el pedido de mayor importe sin IVA de Industrias Bardenas y qué productos incluye?",
        check_largest,
    ),
]
SYSTEM_PROMPT = (
    f"Eres el asistente de operaciones de Distribuciones Ejemplo S.L. Hoy es {TODAY}. Usa las herramientas para consultar "
    "datos reales y actuar; no inventes datos. Cuando termines, responde al usuario en español de forma breve."
)


def chat(msgs):
    body = {
        "model": a.model,
        "messages": msgs,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 4096,
    }
    req = urllib.request.Request(
        a.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def episode(idx, seed):
    tid, prompt, check = TASKS[idx % len(TASKS)]
    e = Environment(seed)
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    t0, tok_out, final, err = time.time(), 0, "", None
    try:
        for _ in range(a.max_steps):
            resp = chat(msgs)
            tok_out += resp.get("usage", {}).get("completion_tokens", 0) or 0
            m = resp["choices"][0]["message"]
            tcs = m.get("tool_calls") or []
            if not tcs:
                final = m.get("content") or ""
                break
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
            for tc in tcs:
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                    out = e.run(tc["function"]["name"], args if isinstance(args, dict) else {})
                except json.JSONDecodeError:
                    out = {"error": "los argumentos no son JSON válido"}
                msgs.append(
                    {"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(out, ensure_ascii=False)}
                )
        else:
            err = "demasiados pasos"
    except Exception as ex:
        err = str(ex)[:160]
    ok = False if err else bool(check(e, uni(final)))
    return {
        "tarea": tid,
        "ok": ok,
        "s": round(time.time() - t0, 2),
        "herramientas": len(e.calls),
        "tokens_salida": tok_out,
        "error": err,
        "respuesta": final[:200],
    }


def run_level(c):
    n = len(TASKS) if c == 1 else max(2 * c, len(TASKS))
    res, lock, idx = [], threading.Lock(), [0]
    t0 = time.time()

    def w():
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= n:
                return
            r = episode(i, i)
            with lock:
                res.append(r)

    hs = [threading.Thread(target=w) for _ in range(c)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    T = time.time() - t0
    dur = sorted(r["s"] for r in res)
    by_task = {}
    for r in res:
        by_task.setdefault(r["tarea"], []).append(r["ok"])
    return {
        "conc": c,
        "tareas": len(res),
        "aciertos_pct": round(100 * sum(r["ok"] for r in res) / len(res), 1),
        "tareas_min": round(len(res) / T * 60, 1),
        "seg_tarea_p50": dur[len(dur) // 2],
        "seg_tarea_p95": dur[min(len(dur) - 1, int(0.95 * len(dur)))],
        "herramientas_media": round(statistics.mean(r["herramientas"] for r in res), 1),
        "tokens_salida_media": round(statistics.mean(r["tokens_salida"] for r in res), 0),
        "errores": sum(1 for r in res if r["error"]),
        "por_tarea": {k: f"{sum(v)}/{len(v)}" for k, v in by_task.items()},
        "detalle": res if c == 1 else [r for r in res if not r["ok"]][:5],
    }


out = {"modelo": a.model, "tareas": len(TASKS), "niveles": []}
for c in [int(x) for x in a.conc.split(",")]:
    r = run_level(c)
    out["niveles"].append(r)
    print(
        f"  {c:>3} agents at once: success {r['aciertos_pct']:>5} % · {r['tareas_min']:>6} tasks/min · "
        f"p50 {r['seg_tarea_p50']} s · p95 {r['seg_tarea_p95']} s · {r['herramientas_media']} tool calls/task · "
        f"errors {r['errores']}",
        flush=True,
    )
    if c == 1:
        for d in r["detalle"]:
            print(
                f"     {'OK ' if d['ok'] else 'NO '} {d['tarea']:20s} {d['s']:>6} s  {d['herramientas']} calls  "
                f"{(d['error'] or d['respuesta']).replace(chr(10), ' ')[:110]}"
            )
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    if r["errores"] > 0.2 * r["tareas"] or r["seg_tarea_p95"] > 600:
        print("  (stopping: this level is no longer usable)")
        break
print("  Saved:", a.out)
