#!/usr/bin/env python3
"""
asr_transformers.py — T23 for Underdog Bark 3.0 Pro (Parakeet TDT 0.6B, English only) run with Transformers, since
vLLM cannot serve it. Same FLEURS clips and the same normalization/WER as asr.py:
  · batch 1: the 60 English clips, one at a time (latency)
  · batch 16: the 420 clips of the 7 languages (throughput; outside English it is expected to fail: English only)
Runs inside a container with torch, transformers and soundfile. Writes one JSON (asr_bark.json in the original runs).

Usage: python3 asr_transformers.py <model_dir> <audio_dir> <out.json>
"""

import json
import os
import re
import sys
import time
import unicodedata
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, ParakeetForTDT

try:
    import soxr
except Exception:
    soxr = None

if len(sys.argv) < 4 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
MOD, DIR, OUT = sys.argv[1:4]
NAME = "bark-3.0-pro"
LANGS = ["es_419", "en_us", "ca_es", "fr_fr", "de_de", "pt_br", "it_it"]


def norm(t):
    t = unicodedata.normalize("NFKC", str(t)).lower()
    t = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def word_errors(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)], len(r)


def load(p):
    w, sr = sf.read(p, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != 16000:
        if soxr:
            w = soxr.resample(w, sr, 16000)
        else:
            import librosa

            w = librosa.resample(w, orig_sr=sr, target_sr=16000)
    return np.concatenate([np.zeros(4000, np.float32), w.astype(np.float32)])  # 0.25 s of silence, as in the model card


items = []
for c in LANGS:
    v = os.path.join(DIR, c, "verdad.tsv")
    if os.path.exists(v):
        for line in open(v, encoding="utf-8"):
            f, ref = line.rstrip("\n").split("\t", 1)
            if os.path.exists(os.path.join(DIR, c, f)):
                items.append((os.path.join(DIR, c, f), c, ref))
audio = {p: load(p) for p, _, _ in items}
dur = {p: (len(a) - 4000) / 16000 for p, a in audio.items()}

proc = AutoProcessor.from_pretrained(MOD)
model = ParakeetForTDT.from_pretrained(MOD, dtype=torch.bfloat16).to("cuda").eval()


def transcribe(batch):
    inp = proc([audio[p] for p in batch], sampling_rate=16000, return_tensors="pt", padding=True).to(
        "cuda", torch.bfloat16
    )
    with torch.inference_mode():
        out = model.generate(**inp)
    ids = out.sequences if hasattr(out, "sequences") else out
    return proc.batch_decode(ids, skip_special_tokens=True)


transcribe([items[0][0]])  # warm-up
res = {"modelo": NAME, "motor": "transformers", "audios": len(items), "niveles": []}
for batch_n, sel in [(1, [it for it in items if it[1] == "en_us"]), (16, items)]:
    hyps, lat, t0, err = [], [], time.time(), None
    try:
        for i in range(0, len(sel), batch_n):
            t = time.time()
            hyps += transcribe([it[0] for it in sel[i : i + batch_n]])
            torch.cuda.synchronize()
            lat.append(time.time() - t)
    except Exception as e:
        err = repr(e)[:300]
    T = time.time() - t0
    by_lang = {}
    for it, h in zip(sel, hyps):
        e, n = word_errors(it[2], h)
        acc = by_lang.setdefault(it[1], [0, 0])
        acc[0] += e
        acc[1] += n
    seg = sum(dur[it[0]] for it in sel[: len(hyps)])
    lat.sort()
    level = {
        "lote": batch_n,
        "audios": len(hyps),
        "segundos": round(T, 1),
        "rtfx": round(seg / T, 1) if T else None,
        "lat_p50_s": round(lat[len(lat) // 2], 3) if lat else None,
        "wer_pct": {k: round(100 * v[0] / max(v[1], 1), 1) for k, v in by_lang.items()},
        "wer_en_pct": round(100 * by_lang["en_us"][0] / max(by_lang["en_us"][1], 1), 1) if "en_us" in by_lang else None,
        "error": err,
        "ejemplos": [{"idioma": it[1], "ref": it[2], "hyp": h} for it, h in list(zip(sel, hyps))[::10]][:8],
    }
    res["niveles"].append(level)
    print(
        f"  T23 {NAME} batch {batch_n}: English WER {level['wer_en_pct']} % · {level['rtfx']}× real time · "
        f"p50 {level['lat_p50_s']} s per batch · clips {len(hyps)}/{len(sel)} · per language {level['wer_pct']}"
        + (f" · ERROR {err}" if err else ""),
        flush=True,
    )
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
