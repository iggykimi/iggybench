#!/usr/bin/env python3
"""
asr.py — T23: speech transcription. Sends FLEURS clips (Google, test split, 60 sentences per language) to a vLLM
server's /v1/audio/transcriptions endpoint; if the model does not have it, falls back to /v1/chat/completions with the
audio attached (input_audio) and an instruction to transcribe verbatim. Measures word error rate (WER) per language
and speed (audio seconds processed per second). Same normalization for reference and hypothesis: lowercase, no
punctuation or symbols, single spaces. Writes one JSON (asr_<model>.json in the original runs).

Audio layout: <audio_dir>/<fleurs_lang>/verdad.tsv (one "<file>\t<reference text>" per line) plus the WAV files.

Usage: python3 asr.py <base_url> <model> <out.json> <audio_dir> [--conc 1,32]
"""

import base64
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import wave

if len(sys.argv) < 5 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
BASE, MODEL, OUT, DIR = sys.argv[1:5]
CONCS = [int(x) for x in (sys.argv[sys.argv.index("--conc") + 1] if "--conc" in sys.argv else "1,32").split(",")]
# FLEURS folder -> (ISO code, language name used in the chat fallback prompt)
LANGS = {
    "es_419": ("es", "español"),
    "en_us": ("en", "inglés"),
    "ca_es": ("ca", "catalán"),
    "fr_fr": ("fr", "francés"),
    "de_de": ("de", "alemán"),
    "pt_br": ("pt", "portugués"),
    "it_it": ("it", "italiano"),
}
MODE = {"api": "transcriptions"}


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


def duration(p):
    # FLEURS ships WAVs the wave module cannot always open (e.g. float32); the RIFF header is read by hand
    try:
        with wave.open(p) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        pass
    try:
        h = open(p, "rb").read(1 << 16)
        i = 12
        br = None
        while h[:4] == b"RIFF" and i + 8 <= len(h):
            cid, n = h[i : i + 4], int.from_bytes(h[i + 4 : i + 8], "little")
            if cid == b"fmt ":
                br = int.from_bytes(h[i + 16 : i + 20], "little")
            if cid == b"data" and br:
                if n in (0, 0xFFFFFFFF):
                    n = os.path.getsize(p) - i - 8
                return n / br
            i += 8 + n + (n & 1)
    except Exception:
        pass
    return os.path.getsize(p) / 32000.0  # last resort: 16 kHz, 16 bit


def transcribe_api(path, lang):
    bnd = uuid.uuid4().hex
    audio = open(path, "rb").read()
    parts = [("model", MODEL), ("language", lang), ("temperature", "0"), ("response_format", "json")]
    body = b"".join(f'--{bnd}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in parts)
    body += (
        (
            f'--{bnd}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + audio
        + f"\r\n--{bnd}--\r\n".encode()
    )
    req = urllib.request.Request(
        BASE + "/v1/audio/transcriptions", data=body, headers={"Content-Type": f"multipart/form-data; boundary={bnd}"}
    )
    return json.load(urllib.request.urlopen(req, timeout=600)).get("text", "")


def transcribe_chat(path, lang):
    name = next((v[1] for v in LANGS.values() if v[0] == lang), lang)
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64.b64encode(open(path, "rb").read()).decode(), "format": "wav"},
                    },
                    {
                        "type": "text",
                        "text": f"Transcribe literalmente este audio en {name}. Devuelve solo la transcripción.",
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or ""


def one(item):
    path, lang, ref = item
    t = time.time()
    try:
        if MODE["api"] == "transcriptions":
            try:
                txt = transcribe_api(path, lang)
            except urllib.error.HTTPError as e:
                if e.code in (400, 404, 405, 422, 500, 501):
                    MODE["api"] = "chat"
                    txt = transcribe_chat(path, lang)
                else:
                    raise
        else:
            txt = transcribe_chat(path, lang)
        return {"ok": True, "s": time.time() - t, "txt": txt}
    except Exception as e:
        return {"ok": False, "s": time.time() - t, "txt": "", "error": repr(e)[:200]}


items = []
for folder, (lang, _) in LANGS.items():
    v = os.path.join(DIR, folder, "verdad.tsv")
    if not os.path.exists(v):
        continue
    for line in open(v, encoding="utf-8"):
        f, ref = line.rstrip("\n").split("\t", 1)
        p = os.path.join(DIR, folder, f)
        if os.path.exists(p):
            items.append((p, lang, ref, folder))
if not items:
    sys.exit("!! no audio files in " + DIR)
dur = {it[0]: duration(it[0]) for it in items}
res = {"modelo": MODEL, "audios": len(items), "horas_audio": round(sum(dur.values()) / 3600, 3), "niveles": []}

print(f"  asr.py {MODEL}: {len(items)} clips · {res['horas_audio']} h", flush=True)
for c in CONCS:
    sel = items if c > 1 else items[::4]  # at concurrency 1, one clip in four (latency)
    t0 = time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        outs = list(ex.map(lambda it: one(it[:3]), sel))
    T = time.time() - t0
    by_lang = {}
    for it, o in zip(sel, outs):
        e, n = word_errors(it[2], o["txt"]) if o["ok"] else (len(norm(it[2]).split()),) * 2
        acc = by_lang.setdefault(it[3], [0, 0, 0])
        acc[0] += e
        acc[1] += n
        acc[2] += 1
    lat = sorted(o["s"] for o in outs if o["ok"])
    audio_s = sum(dur[it[0]] for it in sel)
    level = {
        "conc": c,
        "audios": len(sel),
        "fallos": sum(not o["ok"] for o in outs),
        "segundos": round(T, 1),
        "rtfx": round(audio_s / T, 1),
        "horas_audio_por_hora": round(audio_s / T, 1),
        "lat_p50_s": round(lat[len(lat) // 2], 2) if lat else None,
        "lat_p95_s": round(lat[int(0.95 * len(lat))], 2) if lat else None,
        "wer_pct": {k: round(100 * v[0] / max(v[1], 1), 1) for k, v in by_lang.items()},
        "wer_medio_pct": round(
            100 * sum(v[0] for v in by_lang.values()) / max(sum(v[1] for v in by_lang.values()), 1), 1
        ),
        "api": MODE["api"],
        "errores": [o.get("error") for o in outs if not o["ok"]][:3],
        "ejemplos": [
            {"idioma": it[3], "ref": it[2], "hyp": o["txt"]} for it, o in list(zip(sel, outs))[:: max(1, len(sel) // 7)]
        ][:7],
    }
    res["niveles"].append(level)
    print(
        f"  T23 {MODEL} conc {c}: mean WER {level['wer_medio_pct']} % · {level['rtfx']}× real time · "
        f"p50 {level['lat_p50_s']} s · failures {level['fallos']} · API {MODE['api']} · per language {level['wer_pct']}",
        flush=True,
    )
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
