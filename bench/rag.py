#!/usr/bin/env python3
"""
rag.py — Embeddings and reranker for RAG (Qwen3-Embedding / Qwen3-Reranker on vLLM). Stdlib only.
Writes one JSON (embed.json in the original runs).

1. Indexing: real ~500-token chunks (ShareGPT), 32 per request, at concurrency 1/4/16/32 for ~60 s.
   -> chunks/s, tokens/s and minutes to index 1 million chunks.
2. Queries: one short question per request at concurrency 1/16/64 -> p50/p95 latency (ms) and queries/s.
3. Quality check (that the setup is not broken): 300 GSM8K questions against their 300 solutions.
   Embeddings: recall@1 and @5 by cosine. Reranker: question + 50 solutions (1 correct) -> hit@1.
4. Reranker: latency of reranking 50 documents (~300 tokens) at concurrency 1/8/32.

Usage: python3 rag.py <emb_url> <emb_model> <rerank_url|-> <rerank_model|-> <out.json> <sharegpt.json> <gsm8k.jsonl>
"""

import json
import math
import random
import sys
import threading
import time
import urllib.request

if len(sys.argv) < 8 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
URL_E, MOD_E, URL_R, MOD_R, OUT, SHAREGPT, GSM = sys.argv[1:8]
res = {"embedding": MOD_E, "reranker": None if MOD_R == "-" else MOD_R}


def save():
    json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)


def q(v, p):
    return sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else None


def post(url, path, body, timeout=300):
    req = urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def embed(texts):
    r = post(URL_E, "/v1/embeddings", {"model": MOD_E, "input": texts})
    return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])], r.get("usage", {}).get(
        "prompt_tokens", 0
    )


INSTR_Q = "Instruct: Given a question, retrieve the passage that answers it\nQuery: "


def load_test(c, dur, fn, n_max=None):
    """c threads calling fn() for dur seconds (or n_max calls). Returns a list of (seconds, info, ok)."""
    out, lock, cnt = [], threading.Lock(), [0]
    t_end = time.time() + dur

    def w():
        while time.time() < t_end:
            with lock:
                if n_max and cnt[0] >= n_max:
                    return
                cnt[0] += 1
            t = time.time()
            try:
                info = fn()
                ok = True
            except Exception as e:
                info = str(e)[:150]
                ok = False
            with lock:
                out.append((time.time() - t, info, ok))

    t0 = time.time()
    hs = [threading.Thread(target=w) for _ in range(c)]
    [h.start() for h in hs]
    [h.join() for h in hs]
    return out, time.time() - t0


# ---------- real corpus (ShareGPT) in chunks of ~375 words (~500 tokens) ----------
print("  loading ShareGPT...", flush=True)
conv = json.load(open(SHAREGPT))
words, chunks = [], []
for c in conv[:6000]:
    for t in c.get("conversations", []):
        words.extend(t.get("value", "").split())
        while len(words) >= 375:
            chunks.append(" ".join(words[:375]))
            words = words[375:]
    if len(chunks) >= 12000:
        break
del conv
print(f"  {len(chunks)} chunks", flush=True)
rnd = random.Random(1)

# ---------- 1. indexing ----------
embed(chunks[:8])  # warm-up
res["indexacion"] = []
for c in (1, 4, 16, 32):

    def batch():
        i = rnd.randrange(0, len(chunks) - 32)
        _, tok = embed(chunks[i : i + 32])
        return tok

    r, T = load_test(c, 60, batch)
    ok = [x for x in r if x[2]]
    tps = sum(x[1] for x in ok) / T
    cps = 32 * len(ok) / T
    lvl = {
        "conc": c,
        "peticiones": len(r),
        "fallos": len(r) - len(ok),
        "trozos_s": round(cps, 1),
        "tokens_s": round(tps),
        "tok_por_trozo": round(sum(x[1] for x in ok) / max(32 * len(ok), 1)),
        "min_1M_trozos": round(1e6 / cps / 60, 1) if cps else None,
        "lat_p50_s": round(q([x[0] for x in ok], 0.5) or 0, 2),
    }
    res["indexacion"].append(lvl)
    save()
    print(
        f"  indexing conc {c:>3}: {lvl['trozos_s']} chunks/s · {lvl['tokens_s']} tok/s · 1M chunks in "
        f"{lvl['min_1M_trozos']} min · failures {lvl['fallos']}",
        flush=True,
    )

# ---------- 2. queries ----------
gsm = [json.loads(line) for line in open(GSM) if line.strip()][:300]
questions = [g["question"] for g in gsm]
solutions = [g["answer"].split("####")[0].strip() for g in gsm]
res["consultas"] = []
for c in (1, 16, 64):

    def one():
        embed([INSTR_Q + rnd.choice(questions)])
        return 0

    r, T = load_test(c, 30, one)
    ok = [x[0] * 1000 for x in r if x[2]]
    lvl = {
        "conc": c,
        "consultas_s": round(len(ok) / T, 1),
        "lat_p50_ms": round(q(ok, 0.5) or 0, 1),
        "lat_p95_ms": round(q(ok, 0.95) or 0, 1),
        "fallos": len(r) - len(ok),
    }
    res["consultas"].append(lvl)
    save()
    print(
        f"  queries conc {c:>3}: {lvl['consultas_s']}/s · p50 {lvl['lat_p50_ms']} ms · p95 {lvl['lat_p95_ms']} ms",
        flush=True,
    )


# ---------- 3. embedding quality ----------
def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


try:
    E_q = [norm(v) for i in range(0, 300, 50) for v in embed([INSTR_Q + p for p in questions[i : i + 50]])[0]]
    E_d = [norm(v) for i in range(0, 300, 50) for v in embed(solutions[i : i + 50])[0]]
    r1 = r5 = 0
    try:
        import numpy as np  # fast if the host has numpy

        S = np.array(E_q) @ np.array(E_d).T
        rows = [list(np.argsort(-S[i])[:5]) for i in range(300)]
    except ImportError:  # stdlib (~1-2 min)
        rows = [sorted(range(300), key=lambda j: -sum(a * b for a, b in zip(eq, E_d[j])))[:5] for eq in E_q]
    for i, s in enumerate(rows):
        r1 += s[0] == i
        r5 += i in s
    res["calidad_embeddings"] = {"recall@1": round(100 * r1 / 300, 1), "recall@5": round(100 * r5 / 300, 1)}
    print(
        f"  embedding quality (GSM8K question -> solution, 300): recall@1 {res['calidad_embeddings']['recall@1']} % · "
        f"recall@5 {res['calidad_embeddings']['recall@5']} %",
        flush=True,
    )
except Exception as e:
    res["calidad_embeddings"] = {"error": str(e)[:200]}
save()

# ---------- 4. reranker ----------
if MOD_R != "-":
    PRE = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INS = "Given a question, retrieve the passage that answers it"

    def rerank(i, docs_idx):
        body = {
            "model": MOD_R,
            "query": f"{PRE}<Instruct>: {INS}\n<Query>: {questions[i]}\n",
            "documents": [f"<Document>: {solutions[j]}{SUF}" for j in docs_idx],
        }
        r = post(URL_R, "/v1/rerank", body)
        best = max(r["results"], key=lambda x: x["relevance_score"])["index"]
        return docs_idx[best] == i

    def candidates(i):
        d = rnd.sample([j for j in range(300) if j != i], 49) + [i]
        rnd.shuffle(d)
        return d

    try:
        hits = sum(rerank(i, candidates(i)) for i in range(100))
        res["calidad_reranker"] = {"acierto@1_de_50": round(100 * hits / 100, 1)}
        print(f"  reranker quality (100 questions, 50 candidates): hit@1 {hits} %", flush=True)
        res["rerank"] = []
        for c in (1, 8, 32):

            def one():
                i = rnd.randrange(300)
                return rerank(i, candidates(i))

            r, T = load_test(c, 45, one)
            ok = [x[0] * 1000 for x in r if x[2]]
            lvl = {
                "conc": c,
                "rerank_s": round(len(ok) / T, 2),
                "docs_s": round(50 * len(ok) / T),
                "lat_p50_ms": round(q(ok, 0.5) or 0),
                "lat_p95_ms": round(q(ok, 0.95) or 0),
                "fallos": len(r) - len(ok),
            }
            res["rerank"].append(lvl)
            save()
            print(
                f"  rerank 50 docs conc {c:>3}: {lvl['rerank_s']}/s ({lvl['docs_s']} docs/s) · p50 {lvl['lat_p50_ms']} ms · "
                f"p95 {lvl['lat_p95_ms']} ms · failures {lvl['fallos']}",
                flush=True,
            )
    except Exception as e:
        res["rerank_error"] = str(e)[:300]
        print("  !! reranker:", e)
save()
print("  Saved:", OUT)
