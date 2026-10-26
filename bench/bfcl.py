#!/usr/bin/env python3
"""
bfcl.py — T21: function calling with the Berkeley Function Calling Leaderboard (BFCL v3, gorilla-llm), stdlib only.

T13 (8 in-house tasks) is at 100 % for almost every model. BFCL is the public reference for whether a model picks the
right function and fills the arguments correctly. Single-turn (AST) categories are used:
  simple, multiple, parallel, parallel_multiple, irrelevance (must not call anything) and their "live" versions
  (real user requests): live_simple, live_multiple, live_parallel, live_parallel_multiple, live_irrelevance.
Functions are passed as OpenAI API tools (`tools`) and `tool_calls` is read back: this measures the model and vLLM's
tool parser together, which is what an application would use.
Checking: simplified replica of the official AST checker (function name, required arguments, allowed values with the
same string normalization, optional arguments, parallel calls in any order). Good for comparing models with each
other; not identical 1:1 to the official leaderboard. Writes <out>.json and per-question <out>_detalle.jsonl.

Usage: python3 bfcl.py <base_url> <model> <out.json> <bfcl_dir> [--conc 64] [--time-limit-min 20] [--max-irrelevance 240]
       python3 bfcl.py - - - <bfcl_dir> --load-only        (only counts what was loaded)
       (Spanish aliases accepted: --limite-min, --max-irrel, --solo-carga)
"""

import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

if len(sys.argv) < 5 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
BASE, MODEL, OUT, DATA = sys.argv[1:5]


def arg(keys, d):
    for k in keys:
        if k in sys.argv:
            return type(d)(sys.argv[sys.argv.index(k) + 1])
    return d


CONC, LIMIT, MAXIRR, MAXTOK = (
    arg(("--conc",), 64),
    arg(("--time-limit-min", "--limite-min"), 20.0),
    arg(("--max-irrelevance", "--max-irrel"), 240),
    arg(("--max-tokens",), 8192),
)
CATS = [
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
T0 = time.time()
END = T0 + LIMIT * 60
lock = threading.Lock()


def jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# ----------------------------------------------------------------------------- functions -> OpenAI tools
TYPES = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "number": "number",
    "object": "object",
    "list": "array",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
}


def schema(s):
    if not isinstance(s, dict):
        return {"type": "string"}
    o = {}
    for k, v in s.items():
        if k == "type":
            t = TYPES.get(str(v).lower())
            if t:
                o["type"] = t
        elif k == "properties" and isinstance(v, dict):
            o["properties"] = {pk: schema(pv) for pk, pv in v.items()}
        elif k == "items":
            o["items"] = schema(v)
        elif k in ("description", "enum", "required", "default"):
            o[k] = v
    if o.get("type") == "object" and "properties" not in o:
        o["properties"] = {}
    return o


def to_tools(funcs):
    tools, names = [], {}
    for f in funcs:
        n = re.sub(r"[^a-zA-Z0-9_-]", "_", f["name"])[:64]
        names[n] = f["name"]
        p = schema(f.get("parameters") or {"type": "dict", "properties": {}})
        p["type"] = "object"
        tools.append(
            {
                "type": "function",
                "function": {"name": n, "description": f.get("description", "")[:1024], "parameters": p},
            }
        )
    return tools, names


# ----------------------------------------------------------------------------- AST checker (simplified)
def std(s):
    return re.sub(r"[ ,./\-_*^]", "", str(s)).lower().replace("'", '"')


def matches(v, p):
    """does the model's value v match an allowed value p?"""
    if isinstance(p, bool) or isinstance(v, bool):
        return isinstance(v, bool) and isinstance(p, bool) and v == p
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        try:
            return float(v) == float(p)
        except (TypeError, ValueError):
            return False
    if isinstance(p, str):
        return isinstance(v, (str, int, float)) and std(v) == std(p)
    if isinstance(p, (list, tuple)):
        return isinstance(v, (list, tuple)) and len(v) == len(p) and all(matches(a, b) for a, b in zip(v, p))
    if isinstance(p, dict):  # nested dict: each key with its list of allowed values
        if not isinstance(v, dict):
            return False
        return arguments_ok(v, p)
    return v == p


def arguments_ok(given, allowed):
    for k in given:
        if k not in allowed:
            return False  # unexpected argument
    for k, options in allowed.items():
        options = options if isinstance(options, list) else [options]
        if k not in given:
            if "" not in options and None not in options:
                return False  # required and missing
            continue
        if not any(matches(given[k], o) for o in options if o != ""):
            return False
    return True


def call_ok(call, gt):
    ((name, allowed),) = gt.items()
    return call[0] == name and isinstance(call[1], dict) and arguments_ok(call[1], allowed)


def match_calls(calls, gts):
    """parallel calls: each expected call paired with one from the model, in any order (small exhaustive search)"""
    if len(calls) != len(gts):
        return False
    used = [False] * len(calls)

    def bt(i):
        if i == len(gts):
            return True
        for j, c in enumerate(calls):
            if not used[j] and call_ok(c, gts[i]):
                used[j] = True
                if bt(i + 1):
                    return True
                used[j] = False
        return False

    return bt(0)


# ----------------------------------------------------------------------------- data
items = []
for c in CATS:
    qs = jsonl(os.path.join(DATA, f"BFCL_v3_{c}.json"))
    gts = {g["id"]: g["ground_truth"] for g in jsonl(os.path.join(DATA, "possible_answer", f"BFCL_v3_{c}.json"))}
    if "irrelevance" in c:
        random.Random(0).shuffle(qs)
        qs = qs[:MAXIRR]
    for q in qs:
        if "irrelevance" not in c and q.get("id") not in gts:
            continue
        msgs = q["question"][0] if q.get("question") and isinstance(q["question"][0], list) else q.get("question")
        if not msgs or not q.get("function"):
            continue
        items.append((c, q["id"], msgs, q["function"], gts.get(q["id"])))
counts = {c: sum(1 for x in items if x[0] == c) for c in CATS}
print(f"  T21 BFCL: {counts} · {len(items)} questions · {CONC} in parallel · limit {LIMIT:.0f} min", flush=True)
if "--load-only" in sys.argv or "--solo-carga" in sys.argv:
    sys.exit(0)


# ----------------------------------------------------------------------------- requests
def ask(msgs, tools):
    body = {
        "model": MODEL,
        "messages": msgs,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": MAXTOK,
    }
    for _ in range(3):
        left = END - time.time()
        if left < 5:
            return None, "sin tiempo"
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            return json.load(urllib.request.urlopen(req, timeout=min(900, left))), None
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code == 400:
                return None, err  # e.g. a schema the server rejects: no retry
        except Exception as e:
            err = type(e).__name__
        time.sleep(3)
    return None, err


DET = open(OUT.replace(".json", "_detalle.jsonl"), "w") if OUT != "-" else None


def one(it):
    c, iid, msgs, funcs, gt = it
    rec = {"cat": c, "id": iid}
    if time.time() > END - 5:
        rec["estado"] = "sin tiempo"
    else:
        try:
            tools, names = to_tools(funcs)
            r, err = ask(msgs, tools)
            if r is None:
                rec["estado"] = err
            else:
                m = r["choices"][0].get("message", {})
                calls = []
                for tc in m.get("tool_calls") or []:
                    f = tc.get("function", {})
                    a = f.get("arguments")
                    try:
                        a = json.loads(a) if isinstance(a, str) else (a or {})
                    except json.JSONDecodeError:
                        a = None
                    calls.append((names.get(f.get("name"), f.get("name")), a))
                if "irrelevance" in c:
                    ok = not calls
                else:
                    ok = match_calls(calls, gt)
                rec.update(
                    estado="ok",
                    ok=ok,
                    llamadas=[[n, a] for n, a in calls][:6],
                    texto=(m.get("content") or "")[:200],
                    tokens=(r.get("usage") or {}).get("completion_tokens"),
                )
        except Exception as e:
            rec["estado"] = f"error {type(e).__name__}"
    with lock:
        if DET:
            DET.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            DET.flush()
    return rec


res = []
with ThreadPoolExecutor(CONC) as ex:
    for k, r in enumerate(ex.map(one, items), 1):
        res.append(r)
        if k % 250 == 0:
            print(f"    {k}/{len(items)} · {int((time.time() - T0) / 60)} min", flush=True)

out = {"modelo": MODEL, "minutos": round((time.time() - T0) / 60, 1), "conc": CONC, "categorias": {}}
for c in CATS:
    rs = [r for r in res if r["cat"] == c]
    if not rs:
        continue
    answered = [r for r in rs if r.get("estado") == "ok"]
    out["categorias"][c] = {
        "preguntas": len(rs),
        "respondidas": len(answered),
        "errores": len(rs) - len(answered),
        "acierto_pct": round(100 * sum(r["ok"] for r in answered) / len(answered), 1) if answered else None,
    }
accs = [v["acierto_pct"] for v in out["categorias"].values() if v["acierto_pct"] is not None]
out["media_categorias_pct"] = round(sum(accs) / len(accs), 1) if accs else None
nl = [
    v["acierto_pct"] for k, v in out["categorias"].items() if not k.startswith("live") and v["acierto_pct"] is not None
]
lv = [v["acierto_pct"] for k, v in out["categorias"].items() if k.startswith("live") and v["acierto_pct"] is not None]
out["media_no_live_pct"] = round(sum(nl) / len(nl), 1) if nl else None
out["media_live_pct"] = round(sum(lv) / len(lv), 1) if lv else None
if OUT != "-":
    json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
for c, v in out["categorias"].items():
    print(f"  {c:24} {v['acierto_pct']} %  ({v['respondidas']}/{v['preguntas']} answered)")
print(
    f"  T21 finished in {out['minutos']} min · mean over categories {out['media_categorias_pct']} % "
    f"(non-live {out['media_no_live_pct']} %, live {out['media_live_pct']} %)",
    flush=True,
)
