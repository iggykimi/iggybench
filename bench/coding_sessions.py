#!/usr/bin/env python3
"""
coding_sessions.py — T14: long coding-agent sessions (prefix caching in conversations that keep growing).

Mimics what studies of real coding-agent workloads measure (TraceLab 2026, vLLM AgentX):
  · each session starts with ~48k tokens (agent instructions + repository files)
  · each turn adds ~1k new tokens (a tool result) and, now and then, a large file (~8k);
    the context grows to ~90-110k tokens
  · the model answers ~300 tokens and its real answer is appended to the history
  · there is a tool pause between turns (2-8 s)
  · ~95 % of each request is already in the cache if the server keeps it

Measures, per number of concurrent sessions: time to first token on later turns (p50/p95), on the first turn,
% of tokens served from cache (usage.prompt_tokens_details.cached_tokens), generation speed per session and
turns/minute. Stops when the experience is no longer usable. Writes one JSON (sesiones_<label>.json in suite.sh).

Usage: python3 coding_sessions.py <base_url> <model> <out.json> [--sessions 4,8,16,32,64] [--turns 14]
                                  [--start-tokens 48000] [--output-tokens 300] [--max-sec 600] [--label L]
       (Spanish aliases accepted: --sesiones, --turnos, --inicio, --salida, --max-seg, --etiqueta)
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--sessions", "--sesiones", dest="sessions", default="4,8,16,32,64")
ap.add_argument("--turns", "--turnos", dest="turns", type=int, default=14)
ap.add_argument(
    "--start-tokens", "--inicio", dest="start_tokens", type=int, default=48000, help="initial tokens of each session"
)
ap.add_argument("--output-tokens", "--salida", dest="output_tokens", type=int, default=300)
ap.add_argument("--max-sec", "--max-seg", dest="max_sec", type=int, default=600, help="time limit per level")
ap.add_argument("--label", "--etiqueta", dest="label", default="")
a = ap.parse_args()

WORDS = (
    "def class return import self value result config parser request response client server error retry "
    "timeout buffer stream index cache token model input output tensor batch layer weight update check assert "
    "logger path file read write open close json yaml http route handler user order invoice stock total "
    "price quantity customer status pending delivered"
).split()


def text(seed, n_words):
    r = random.Random(seed)
    lines = []
    while sum(len(ln.split()) for ln in lines) < n_words:
        k = r.randint(6, 14)
        ind = "    " * r.randint(0, 3)
        lines.append(ind + " ".join(r.choice(WORDS) for _ in range(k)) + (":" if r.random() < 0.2 else ""))
    return "\n".join(lines)


# ~0.75 words per token -> n_tok * 0.75 words
def block(seed, n_tok):
    return text(seed, int(n_tok * 0.75))


SYSTEM_BASE = (
    "Eres un agente de programación. Trabajas sobre el repositorio que se muestra. Responde con el "
    "siguiente paso y el cambio de código propuesto.\n\n"
)


def request(msgs, max_tokens):
    body = {
        "model": a.model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "low",
    }
    req = urllib.request.Request(
        a.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    ttft = None
    txt = []
    usage = {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except ValueError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                de = ch.get("delta", {})
                piece = (de.get("content") or "") + (de.get("reasoning_content") or de.get("reasoning") or "")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    txt.append(de.get("content") or "")
    tot = time.time() - t0
    out_tok = usage.get("completion_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    dec = (out_tok - 1) / (tot - ttft) if ttft and tot > ttft and out_tok > 1 else None
    return {
        "ttft": ttft,
        "total": tot,
        "prompt": usage.get("prompt_tokens") or 0,
        "cached": cached,
        "out": out_tok,
        "dec_tok_s": dec,
        "texto": "".join(txt),
    }


def session(sid, t_end, res, lock):
    r = random.Random(sid * 7919)
    msgs = [
        {"role": "system", "content": SYSTEM_BASE + block(sid * 1000, a.start_tokens)},
        {"role": "user", "content": "Tarea: " + block(sid * 1000 + 1, 400)},
    ]
    for turn in range(a.turns):
        if time.time() > t_end:
            break
        try:
            x = request(msgs, a.output_tokens)
        except Exception as e:
            with lock:
                res.append({"sesion": sid, "turno": turn, "error": str(e)[:150]})
            return
        x.update({"sesion": sid, "turno": turn})
        reply = x.pop("texto")
        with lock:
            res.append(x)
        msgs.append({"role": "assistant", "content": reply or "ok"})
        n_new = 8000 if r.random() < 0.15 else r.randint(600, 1500)  # tool result
        msgs.append(
            {"role": "user", "content": "Resultado de la herramienta:\n" + block(sid * 1000 + 10 + turn, n_new)}
        )
        time.sleep(r.uniform(2, 8))  # tool execution


def run_level(n):
    res, lock = [], threading.Lock()
    t0 = time.time()
    t_end = t0 + a.max_sec
    hs = [threading.Thread(target=session, args=(1000 * n + i, t_end, res, lock)) for i in range(n)]
    for i, h in enumerate(hs):
        h.start()
        time.sleep(min(0.5, 10 / n))  # staggered start
    [h.join() for h in hs]
    T = time.time() - t0
    ok = [x for x in res if "error" not in x and x.get("ttft") is not None]
    later = [x for x in ok if x["turno"] > 0]
    first = [x for x in ok if x["turno"] == 0]

    def q(v, p):
        return sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None

    tt = [x["ttft"] for x in later]
    dec = [x["dec_tok_s"] for x in ok if x["dec_tok_s"]]
    pt = sum(x["prompt"] for x in later)
    ct = sum(x["cached"] for x in later)
    return {
        "sesiones": n,
        "turnos": len(ok),
        "errores": len(res) - len(ok),
        "segundos": round(T, 1),
        "turnos_min": round(len(ok) / T * 60, 1),
        "contexto_medio": round(statistics.mean(x["prompt"] for x in ok)) if ok else None,
        "contexto_max": max((x["prompt"] for x in ok), default=None),
        "ttft_primer_turno_p50": round(q([x["ttft"] for x in first], 0.5), 2) if first else None,
        "ttft_p50": round(q(tt, 0.5), 2) if tt else None,
        "ttft_p95": round(q(tt, 0.95), 2) if tt else None,
        "cache_pct": round(100 * ct / pt, 1) if pt else None,
        "dec_tok_s_p50": round(q(dec, 0.5), 1) if dec else None,
        "dec_tok_s_p5": round(q(dec, 0.05), 1) if dec else None,
    }


out = {"modelo": a.model, "etiqueta": a.label, "turnos_por_sesion": a.turns, "niveles": []}
for n in [int(x) for x in a.sessions.split(",")]:
    r = run_level(n)
    out["niveles"].append(r)
    print(
        f"  {n:>3} sessions: TTFT later turns p50 {r['ttft_p50']} s / p95 {r['ttft_p95']} s · first turn "
        f"{r['ttft_primer_turno_p50']} s · cache {r['cache_pct']} % · {r['dec_tok_s_p50']} tok/s per session · "
        f"mean context {r['contexto_medio']} (max {r['contexto_max']}) · {r['turnos_min']} turns/min · "
        f"errors {r['errores']}",
        flush=True,
    )
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
    if r["errores"] > 0.2 * max(r["turnos"], 1) or (r["ttft_p95"] or 0) > 60 or (r["dec_tok_s_p5"] or 99) < 5:
        print("  (stopping: with more sessions the experience is no longer usable)")
        break
print("  Saved:", a.out)
