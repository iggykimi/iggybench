#!/usr/bin/env python3
"""
image_gen.py — Image generation (Qwen-Image or another diffusers pipeline) on the GPU. Runs INSIDE the vLLM
container (it has torch). If the container's diffusers lacks the pipeline, it uses the source code downloaded from
GitHub into /pylib/diffusers/src (a read-only mount in the container).

Measures, at 1024x1024: seconds per image and images/minute with batches of 1, 4 and 8, at 50 steps (default quality)
and at 25 steps (fast mode). Saves 4 sample images (poster with Spanish text, product photo, etc.).
Writes <out_dir>/imagen.json and <out_dir>/muestra_<i>.png.

Usage: python3 image_gen.py <model_dir> <out_dir>
"""

import importlib
import json
import os
import sys
import time

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
MODEL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
cls = json.load(open(os.path.join(MODEL, "model_index.json")))["_class_name"]


def load_diffusers():
    try:
        import diffusers

        if hasattr(diffusers, cls):
            return diffusers, "contenedor " + diffusers.__version__
    except Exception:
        pass
    for k in [k for k in sys.modules if k == "diffusers" or k.startswith("diffusers.")]:
        del sys.modules[k]
    sys.path.insert(0, "/pylib/diffusers/src")
    import diffusers

    importlib.reload(diffusers)
    return diffusers, "GitHub " + diffusers.__version__


import torch  # noqa: E402

diffusers, origin = load_diffusers()
print(f"  diffusers {origin} · pipeline {cls}", flush=True)
res = {
    "modelo": os.path.basename(MODEL.rstrip("/")),
    "pipeline": cls,
    "diffusers": origin,
    "resolucion": "1024x1024",
    "pruebas": [],
}


def save():
    json.dump(res, open(os.path.join(OUT, "imagen.json"), "w"), ensure_ascii=False, indent=1)


t = time.time()
pipe = getattr(diffusers, cls).from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
pipe.set_progress_bar_config(disable=True)
res["carga_s"] = round(time.time() - t, 1)
save()
print(f"  loaded in {res['carga_s']} s · GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

PROMPTS = [
    'Cartel publicitario de una ferretería con el texto grande "REBAJAS DE OTOÑO -30%" y debajo "Ferretería Casco Viejo, Zaragoza", estilo moderno, colores cálidos',
    "Fotografía de producto para tienda online: una taladradora inalámbrica amarilla sobre fondo blanco, iluminación de estudio, alta definición",
    'Infografía sencilla con tres pasos numerados: "1 Pedido", "2 Preparación", "3 Entrega", iconos planos, fondo claro',
    "Interior luminoso de una oficina moderna con plantas, gente trabajando con portátiles, fotografía realista, luz natural",
]
kw = {"negative_prompt": " ", "true_cfg_scale": 4.0} if "QwenImage" in cls else {}


def generate(n, steps, seed=42):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt=[PROMPTS[i % len(PROMPTS)] for i in range(n)],
        width=1024,
        height=1024,
        num_inference_steps=steps,
        generator=g,
        **kw,
    ).images


try:
    generate(1, 10)  # warm-up
except TypeError:
    kw = {}
    generate(1, 10)  # pipeline without negative_prompt/true_cfg
torch.cuda.synchronize()
for steps, batch, reps in ((50, 1, 3), (50, 4, 2), (50, 8, 1), (25, 1, 3), (25, 4, 2)):
    torch.cuda.reset_peak_memory_stats()
    try:
        ts = []
        for r in range(reps):
            t = time.time()
            imgs = generate(batch, steps, seed=42 + r)
            torch.cuda.synchronize()
            ts.append(time.time() - t)
            if steps == 50 and batch == 4 and r == 0:
                for i, im in enumerate(imgs):
                    im.save(os.path.join(OUT, f"muestra_{i}.png"))
        s = sum(ts) / len(ts)
        p = {
            "pasos": steps,
            "lote": batch,
            "s_por_llamada": round(s, 2),
            "s_por_imagen": round(s / batch, 2),
            "imagenes_min": round(60 * batch / s, 1),
            "memoria_pico_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        }
    except Exception as e:
        p = {"pasos": steps, "lote": batch, "error": str(e)[:200]}
    res["pruebas"].append(p)
    save()
    print("  ", p, flush=True)
print("  Saved:", OUT)
