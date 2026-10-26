#!/usr/bin/env python3
"""
gsm8k.py — T16: GSM8K quality check, comparable with the DGX Station community results (stdlib only).

Full GSM8K test set (1319 problems; accuracy on the first 200 is also reported, comparable with the earlier round),
direct answer mode (no extended reasoning), temperature 0, 64 in parallel, Wilson 95 % CI. The last number of the
answer is compared with the official solution. Writes one JSON (gsm8k.json in suite.sh).

Usage: python3 gsm8k.py <base_url> <model> <out.json> <gsm8k_test.jsonl> [--n 200]
"""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

if len(sys.argv) < 5 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
BASE, MODEL, OUT, DATA = sys.argv[1:5]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100000  # default: full test set (1319)
items = [json.loads(line) for line in open(DATA) if line.strip()][:N]
INSTR = (
    "Resuelve el problema. Razona brevemente paso a paso y termina con una última línea exactamente así: "
    "#### <número>\n\n"
)


def num(s):
    s = s.replace(",", "").replace("$", "").strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return float(m[-1]) if m else None


def one(it):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": INSTR + it["question"]}],
    }
    t = time.time()
    try:
        req = urllib.request.Request(
            BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            c = json.load(r)["choices"][0]["message"].get("content") or ""
        m = re.search(r"####\s*([^\n]+)", c)
        pred = num(m.group(1)) if m else num(c[-200:])
        gold = num(it["answer"].split("####")[-1])
        return {"ok": pred is not None and gold is not None and abs(pred - gold) < 1e-6, "s": time.time() - t}
    except Exception as e:
        return {"ok": False, "s": time.time() - t, "error": str(e)[:120]}


t0 = time.time()
with ThreadPoolExecutor(64) as ex:
    res = list(ex.map(one, items))
ac = sum(r["ok"] for r in res)


def wilson(k, n, z=1.96):  # 95 % confidence interval for a proportion
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]


a200 = sum(r["ok"] for r in res[:200])
out = {
    "modelo": MODEL,
    "preguntas": len(res),
    "aciertos": ac,
    "acierto_pct": round(100 * ac / len(res), 1),
    "ic95_pct": wilson(ac, len(res)),
    "acierto_200_pct": round(100 * a200 / min(200, len(res)), 1),
    "errores": sum(1 for r in res if r.get("error")),
    "segundos": round(time.time() - t0, 1),
    "por_pregunta": [int(r["ok"]) for r in res],
}
json.dump(out, open(OUT, "w"), indent=1)
print(
    f"  GSM8K-{len(res)}: {out['acierto_pct']} % (95 % CI {out['ic95_pct'][0]}-{out['ic95_pct'][1]}) · "
    f"first 200: {out['acierto_200_pct']} % · errors {out['errores']} · {out['segundos']} s"
)
