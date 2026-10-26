#!/usr/bin/env python3
"""
reasoning.py — T20: hard quality with extended reasoning (stdlib only).

GSM8K is saturated (~97 % for every model). This uses tests where models still separate:
  · AIME 2025 (30 problems, OpenCompass) and AIME 2026 (30, MathArena): integer answer 0-999; 2 samples per problem
    (mean of 2, as MathArena and vendor reports with several samples do).
  · MATH-500 (500 problems from high-school to olympiad level, HuggingFaceH4): answer in \\boxed{}, normalized string
    comparison (Hendrycks style) plus numeric comparison. May under-count by 1-2 points on odd formats.
Reasoning on (each model's default mode; gpt-oss with reasoning_effort=high), temperature 0.6, top_p 0.95,
at most 32,768 tokens (a truncated answer counts as a failure). Everything runs in parallel (--conc) under a global
time limit: what does not finish in time is reported as "unanswered" and is not counted in the accuracy.
Writes <out>.json and per-request <out>_detalle.jsonl (dificil.json / dificil_64k.json in suite.sh).

Usage: python3 reasoning.py <base_url> <model> <out.json> <data_dir> [--conc 128] [--time-limit-min 50] [--max-tokens 32768]
                            [--reps 2] [--aime-only]
       (Spanish aliases accepted: --limite-min, --solo-aime)
"""

import json
import math
import os
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


CONC, LIMIT, MAXTOK, REPS = (
    arg(("--conc",), 128),
    arg(("--time-limit-min", "--limite-min"), 50.0),
    arg(("--max-tokens",), 32768),
    arg(("--reps",), 2),
)
PROMPT = "{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
T0 = time.time()
END = T0 + LIMIT * 60
lock = threading.Lock()
DET = open(OUT.replace(".json", "_detalle.jsonl"), "w")


def read(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return []
    txt = open(p, encoding="utf-8").read().strip()
    if txt.startswith("["):
        return json.loads(txt)
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def field(it, *ks):
    for k in ks:
        if it.get(k) not in (None, ""):
            return it[k]
    return None


# ----------------------------------------------------------------------------- answers
def boxed(s):
    """content of the last \\boxed{...} (with nested braces) or None"""
    if not s:
        return None
    i = max(s.rfind("\\boxed{"), s.rfind("\\fbox{"))
    if i < 0:
        m = re.findall(r"\\boxed\s+([^\s$]+)", s)
        return m[-1] if m else None
    i = s.index("{", i)
    n, j = 0, i
    while j < len(s):
        n += {"{": 1, "}": -1}.get(s[j], 0)
        if n == 0:
            return s[i + 1 : j]
        j += 1
    return None


def _fix_fracs(s):
    out, i = "", 0
    while i < len(s):
        if s.startswith("\\frac", i):
            i += 5
            out += "\\frac"
            for _ in range(2):
                if i < len(s) and s[i] != "{" and s[i] not in "\\":
                    out += "{" + s[i] + "}"
                    i += 1
            continue
        out += s[i]
        i += 1
    return out


def normalize(s):
    """Hendrycks-style (MATH) normalization"""
    if s is None:
        return None
    s = str(s).strip().replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac").replace("\\left", "").replace("\\right", "")
    s = (
        s.replace("^{\\circ}", "")
        .replace("^\\circ", "")
        .replace("\\$", "")
        .replace("$", "")
        .replace("\\%", "")
        .replace("%", "")
    )
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mathbf)\{([^}]*)\}", r"\1", s)
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if len(s.split("=")) == 2 and len(s.split("=")[0]) <= 2:
        s = s.split("=")[1]
    s = _fix_fracs(s.replace(" ", ""))
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)  # 1,000 -> 1000
    s = s.rstrip(".")
    for u in ("units", "unit", "cm", "degrees", "degree", "inches", "inch", "meters", "dollars"):
        if s.endswith(u) and len(s) > len(u):
            s = s[: -len(u)]
    return s


def value(s):
    """try to convert to a number: 12, -3.5, \\frac{a}{b}, a/b"""
    if s is None:
        return None
    t = s.replace("\\frac", "").replace("{", "(").replace("}", ")").replace(")(", ")/(")
    if not re.fullmatch(r"[-+0-9.()/ ]+", t):
        return None
    try:
        return float(eval(t, {"__builtins__": {}}))  # only digits, parentheses and operators (filtered above)
    except Exception:
        return None


def equal(pred, ref):
    a, b = normalize(pred), normalize(ref)
    if a is None or b is None:
        return False
    if a == b:
        return True
    va, vb = value(a), value(b)
    return va is not None and vb is not None and math.isclose(va, vb, rel_tol=1e-6, abs_tol=1e-9)


# ----------------------------------------------------------------------------- requests
MODE = {"kwargs": True}  # if the server rejects chat_template_kwargs, retry without them and record it


def ask(q):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(q=q)}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": MAXTOK,
    }
    if "gpt-oss" in MODEL:
        body["reasoning_effort"] = "high"
    elif MODE["kwargs"]:
        body["chat_template_kwargs"] = {"enable_thinking": True, "thinking": True}
    for attempt in range(3):
        left = END - time.time()
        if left < 5:
            return None, "sin tiempo"
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        try:
            r = json.load(urllib.request.urlopen(req, timeout=min(3600, left)))
            return r, None
        except urllib.error.HTTPError as e:
            if e.code == 400 and "chat_template_kwargs" in body:
                MODE["kwargs"] = False
                body.pop("chat_template_kwargs")
                continue
            err = f"HTTP {e.code}"
        except Exception as e:
            err = type(e).__name__
        time.sleep(5)
    return None, err


def task(t):
    dset, idx, rep, q, ref = t
    if time.time() > END - 5:
        rec = {"conjunto": dset, "id": idx, "rep": rep, "estado": "sin tiempo"}
    else:
        t1 = time.time()
        r, err = ask(q)
        if r is None:
            rec = {"conjunto": dset, "id": idx, "rep": rep, "estado": err}
        else:
            ch = r["choices"][0]
            msg = ch.get("message", {})
            cont = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
            pred = boxed(cont) or boxed(reasoning[-3000:])
            if ch.get("finish_reason") == "length":
                ok = False  # truncated: counts as a failure
            elif dset.startswith("aime"):
                m = re.findall(r"\d+", pred or "")
                ok = bool(m) and int(m[-1]) == int(ref)
            else:
                ok = equal(pred, ref)
            u = r.get("usage") or {}
            rec = {
                "conjunto": dset,
                "id": idx,
                "rep": rep,
                "estado": "ok",
                "acierto": ok,
                "pred": (pred or "")[:200],
                "ref": str(ref)[:200],
                "truncada": ch.get("finish_reason") == "length",
                "tokens": u.get("completion_tokens"),
                "razona": bool(reasoning) or (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) > 0,
                "s": round(time.time() - t1, 1),
            }
    with lock:
        DET.write(json.dumps(rec, ensure_ascii=False) + "\n")
        DET.flush()
    return rec


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


# ----------------------------------------------------------------------------- datasets
tasks, nprob = [], {}
aime25 = read("aime2025-I.jsonl") + read("aime2025-II.jsonl")
aime26 = read("aime2026.json")
math5 = [] if ("--aime-only" in sys.argv or "--solo-aime" in sys.argv) else read("math500.jsonl")
for dset, items in (("aime2026", aime26), ("aime2025", aime25)):
    ok = [
        (i, field(it, "problem", "question", "Question"), field(it, "answer", "Answer")) for i, it in enumerate(items)
    ]
    ok = [
        (i, q, re.sub(r"\^\{?\\circ\}?|\$", "", str(a)).strip() if a is not None else None) for i, q, a in ok
    ]  # e.g. 336^\circ
    ok = [x for x in ok if x[1] and x[2] is not None and re.fullmatch(r"\s*\d+\s*", str(x[2]))]
    nprob[dset] = len(ok)
    for rep in range(REPS):  # first samples of everything first, then the second ones
        tasks += [(dset, i, rep, q, str(a).strip()) for i, q, a in ok]
ok = [
    (
        i,
        field(it, "problem", "question", "Question"),
        field(it, "answer", "Answer") or boxed(field(it, "solution") or ""),
    )
    for i, it in enumerate(math5)
]
ok = [x for x in ok if x[1] and x[2]]
nprob["math500"] = len(ok)
tasks += [("math500", i, 0, q, a) for i, q, a in ok]
# AIME (long) first so no long tails are left at the end; MATH fills the gaps
print(
    f"  T20: {nprob} problems · {len(tasks)} requests · {CONC} in parallel · limit {LIMIT:.0f} min · max {MAXTOK} tokens",
    flush=True,
)

done = []
with ThreadPoolExecutor(CONC) as ex:
    for k, rec in enumerate(ex.map(task, tasks), 1):
        done.append(rec)
        if k % 50 == 0 or k == len(tasks):
            n_ok = sum(1 for r in done if r.get("estado") == "ok")
            print(f"    {k}/{len(tasks)} · answered {n_ok} · {int((time.time() - T0) / 60)} min", flush=True)

T = time.time() - T0
out = {
    "modelo": MODEL,
    "minutos": round(T / 60, 1),
    "conc": CONC,
    "max_tokens": MAXTOK,
    "muestras_aime": REPS,
    "temperatura": 0.6,
    "top_p": 0.95,
    "razonamiento": "reasoning_effort=high"
    if "gpt-oss" in MODEL
    else (
        "activado (chat_template_kwargs)"
        if MODE["kwargs"]
        else "modo por defecto del modelo (rechazó chat_template_kwargs)"
    ),
    "conjuntos": {},
}
tok_total = sum(r.get("tokens") or 0 for r in done)
out["tokens_generados"] = tok_total
out["tok_s_medio"] = round(tok_total / T, 1) if T else None
for dset in ("aime2026", "aime2025", "math500"):
    rs = [r for r in done if r["conjunto"] == dset]
    if not rs:
        continue
    answered = [r for r in rs if r.get("estado") == "ok"]
    k = sum(r["acierto"] for r in answered)
    toks = sorted(r["tokens"] for r in answered if r.get("tokens"))
    out["conjuntos"][dset] = {
        "problemas": nprob[dset],
        "peticiones": len(rs),
        "respondidas": len(answered),
        "sin_responder": len(rs) - len(answered),
        "acierto_pct": round(100 * k / len(answered), 1) if answered else None,
        "ic95_pct": wilson(k, len(answered)),
        "truncadas": sum(r["truncada"] for r in answered),
        "con_razonamiento_pct": round(100 * sum(r["razona"] for r in answered) / len(answered), 1)
        if answered
        else None,
        "tokens_mediana": toks[len(toks) // 2] if toks else None,
        "tokens_p90": toks[int(len(toks) * 0.9)] if toks else None,
    }
    c = out["conjuntos"][dset]
    print(
        f"  {dset}: {c['acierto_pct']} % (95 % CI {c['ic95_pct'][0]}-{c['ic95_pct'][1]}) · {c['respondidas']}/{c['peticiones']} answered · "
        f"{c['truncadas']} truncated · median {c['tokens_mediana']} tokens",
        flush=True,
    )
json.dump(out, open(OUT, "w"), indent=1, ensure_ascii=False)
print(
    f"  T20 finished in {out['minutos']} min · {tok_total} tokens generated ({out['tok_s_medio']} tok/s on average) · reasoning: {out['razonamiento']}"
)
