"""
make_degraded_invoices.py — generates the 30 synthetic HARD invoices of T11D (invented companies) plus verdad.json,
same schema as data/invoices_clean/. Deterministic (random.seed(2027)); needs Pillow and the DejaVu Sans fonts.

The 40 clean invoices score 99-100 % on every system: this set puts together the problems of real scans
(each invoice has several):
  · bad scan: 2-4° rotation, low resolution (900 px wide), heavy JPEG compression, grain, blur, grey paper
  · "PAGADO"/"CONTABILIZADO" stamp over the totals and handwritten notes
  · distractors: order and delivery-note numbers formatted like the invoice number, the customer's tax ID, due date
    and delivery-note date, "pendiente de pago" after an advance payment, trade name different from the legal name
  · per-line discounts and early-payment discount (the tax base is no longer the sum of quantity × price)
  · long descriptions split over two lines (they count as ONE line), 8-16 lines
  · varied date and number formats (03.02.2026, 3-feb-26, February 3, 2026; 1.234,56 / 1,234.56)
Usage: python3 make_degraded_invoices.py <out_dir> [n=30]
       FONT_DIR=<dir with DejaVuSans*.ttf> (default /usr/share/fonts/truetype/dejavu/)
"""

import json
import os
import random
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
OUT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
os.makedirs(OUT, exist_ok=True)
random.seed(2027)
F = os.environ.get("FONT_DIR", "/usr/share/fonts/truetype/dejavu/")


def font(n, s):
    return ImageFont.truetype(os.path.join(F, n), s)


SUPPLIERS = [
    ("Suministros Ebro S.L.", "EBROSUM"),
    ("Talleres Moncayo S.A.", "MONCAYO PRO"),
    ("Distribuciones Turia S.L.", "TURIA DIST"),
    ("Logística Cierzo S.L.", "CIERZO LOG"),
    ("Papelería Delicias S.L.", "DELIPAPEL"),
    ("Electro Aragón Norte S.L.", "EAN ELECTRO"),
    ("Maderas del Jalón S.A.", "JALÓN WOOD"),
    ("Química Huerva S.L.", "HUERVAQUIM"),
    ("Envases Cinca S.A.", "CINCAPACK"),
    ("Iberian Fasteners Ltd.", "IBERFAST"),
    ("Recambios Arba S.L.", "ARBA PARTS"),
    ("Portucargo Lda.", "PORTUCARGO"),
]
CUSTOMERS = [
    ("Comercial Pirineo S.A.", "A50123456"),
    ("Hostelería Ribera S.L.", "B50987654"),
    ("Industrias Bardenas S.L.", "B31555777"),
]
ITEMS = [
    "Tornillo M8 inox (caja 100)",
    "Cable 2,5 mm² libre de halógenos, rollo de 100 m, color azul, norma UNE",
    "Palé madera 1200x800 tratado NIMF-15",
    "Guantes nitrilo T-L (caja)",
    "Papel A4 80 g (caja 5 paq.)",
    "Tóner láser negro",
    "Disolvente industrial 5 L",
    "Servicio de mantenimiento preventivo trimestral de instalación eléctrica en nave 3",
    "Cinta embalar 48 mm",
    "Etiquetas térmicas 100x150",
    "Bombilla LED E27 10 W",
    "Junta tórica 40 mm",
    "Rodamiento 6204-2RS",
    "Horas técnico desplazado (incluye kilometraje y dietas según tarifa vigente)",
    "Caja cartón doble canal",
    "Transporte urgente península 24 h",
]
MONTHS_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]


def eur(x, english=False):
    s = f"{x:,.2f}"
    return s if english else s.replace(",", "X").replace(".", ",").replace("X", ".")


def tax_id():
    return random.choice("ABJ") + f"{random.randint(10000000, 99999999)}"


def split_line(t, n=46):
    if len(t) <= n:
        return [t]
    i = t.rfind(" ", 0, n)
    i = i if i > 0 else n
    return [t[:i], t[i + 1 :]]


def invoice(i):
    (supplier, brand), (cust, cust_id) = random.choice(SUPPLIERS), random.choice(CUSTOMERS)
    english = supplier.endswith(("Ltd.", "Lda.")) or random.random() < 0.15
    d, m = random.randint(1, 28), random.randint(1, 12)
    num = f"{random.choice(['F', 'FV', 'FAC', 'A'])}{random.choice(['', '-'])}2026{random.choice(['-', '/'])}{random.randint(100, 9999):04d}"
    vat = random.choice([21, 21, 21, 10, 4])
    lines = []
    for _ in range(random.randint(8, 16)):
        q = random.randint(1, 40)
        p = round(random.uniform(0.8, 180), 2)
        disc = random.choice([0, 0, 0, 5, 10, 15])
        lines.append((random.choice(ITEMS), q, p, disc, round(q * p * (1 - disc / 100), 2)))
    subtotal = round(sum(ln[4] for ln in lines), 2)
    pp = random.choice([0, 0, 2, 3])
    base = round(subtotal * (1 - pp / 100), 2)
    vat_amount = round(base * vat / 100, 2)
    total = round(base + vat_amount, 2)
    advance = round(total * random.choice([0, 0, 0.3, 0.5]), 2)
    gt = {
        "numero": num,
        "fecha": f"2026-{m:02d}-{d:02d}",
        "proveedor": supplier,
        "nif_proveedor": tax_id(),
        "cliente": cust,
        "base_imponible": base,
        "iva_pct": vat,
        "total": total,
        "num_lineas": len(lines),
    }
    date_txt = random.choice(
        [f"{d:02d}.{m:02d}.2026", f"{d}-{MONTHS_ES[m - 1]}-26", f"2026/{m:02d}/{d:02d}", f"{d:02d}/{m:02d}/2026"]
    )
    if english:
        date_txt = f"{MONTHS[m - 1]} {d}, 2026"
    order, dnote = (
        f"PED-2026-{random.randint(1000, 9999)}",
        f"ALB{random.choice(['-', ''])}2026-{random.randint(1000, 9999)}",
    )
    due = f"{min(d + random.randint(1, 20), 28):02d}/{m % 12 + 1:02d}/2026"

    W, H = 1240, 1754
    img = Image.new("RGB", (W, H), (246, 244, 238))
    dr = ImageDraw.Draw(img)
    B, R, S, X, HAND = (
        font("DejaVuSans-Bold.ttf", 30),
        font("DejaVuSans.ttf", 20),
        font("DejaVuSans.ttf", 17),
        font("DejaVuSans-Bold.ttf", 54),
        font("DejaVuSans-Oblique.ttf", 26),
    )
    x0 = 80
    dr.text((x0, 50), brand, font=X, fill=(30, 30, 30))  # large trade name
    dr.text(
        (x0, 120),
        f"{supplier} · NIF {gt['nif_proveedor']}" if not english else f"{supplier} · VAT ES{gt['nif_proveedor']}",
        font=S,
        fill="black",
    )
    dr.text((x0, 145), "Pol. Ind. Inventado, nave 7 · 50197 Zaragoza", font=S, fill="black")
    title = "INVOICE" if english else random.choice(["FACTURA", "FACTURA ORIGINAL", "Factura"])
    dr.text((780, 55), title, font=B, fill="black")
    box = [
        (f"{'Invoice no.' if english else 'Nº factura'}:", num),
        (f"{'Date' if english else 'Fecha'}:", date_txt),
        (f"{'Order' if english else 'Su pedido'}:", order),
        (f"{'Delivery note' if english else 'Albarán'}:", dnote),
        (f"{'Due date' if english else 'Vencimiento'}:", due),
    ]
    random.shuffle(box)
    for k, (a, b) in enumerate(box):
        dr.text((780, 105 + 28 * k), a, font=S, fill="black")
        dr.text((940, 105 + 28 * k), b, font=S, fill="black")
    dr.text((x0, 255), "Cliente:" if not english else "Bill to:", font=S, fill="black")
    dr.text((x0, 280), cust, font=R, fill="black")
    dr.text((x0, 308), f"NIF {cust_id} · Zaragoza", font=S, fill="black")
    y = 380
    dr.line((x0, y, W - 80, y), fill="black", width=2)
    header = (
        ("Concepto" if not english else "Description", x0),
        ("Cant.", 640),
        ("Precio", 730),
        ("Dto.", 860),
        ("Importe", 980),
    )
    for t, xx in header:
        dr.text((xx, y + 8), t, font=R, fill="black")
    y += 44
    dr.line((x0, y, W - 80, y), fill="black", width=1)
    for a, q, p, disc, amount in lines:
        y += 8
        pieces = split_line(a)
        for k, tz in enumerate(pieces):
            dr.text((x0, y + 22 * k), tz, font=S, fill="black")
        dr.text((640, y), str(q), font=S, fill="black")
        dr.text((730, y), eur(p, english), font=S, fill="black")
        dr.text((860, y), f"{disc} %" if disc else "", font=S, fill="black")
        dr.text((980, y), eur(amount, english), font=S, fill="black")
        y += 22 * len(pieces) + 6
    y += 20
    dr.line((620, y, W - 80, y), fill="black", width=1)
    rows = [("Suma" if not english else "Subtotal", subtotal)]
    if pp:
        rows.append((f"Dto. pronto pago {pp} %", -round(subtotal - base, 2)))
    rows += [
        ("Base imponible" if not english else "Tax base", base),
        (f"IVA {vat} %" if not english else f"VAT {vat} %", vat_amount),
        ("TOTAL FACTURA" if not english else "TOTAL", total),
    ]
    if advance:
        rows += [("Anticipo recibido", -advance), ("Pendiente de pago", round(total - advance, 2))]
    ytot = y
    for t, v in rows:
        y += 10
        f_ = B if t.startswith("TOTAL") else R
        dr.text((620, y), t, font=f_, fill="black")
        dr.text((980, y), eur(v, english) + (" EUR" if english else " €"), font=f_, fill="black")
        y += 34
    dr.text(
        (x0, H - 170),
        "Forma de pago: transferencia a 30 días. IBAN ES00 0000 0000 0000 0000 0000 (ficticio)",
        font=S,
        fill="black",
    )
    dr.text(
        (x0, H - 140), f"{supplier} · Reg. Mercantil de Zaragoza, tomo 0000, folio 00 (ficticio)", font=S, fill="black"
    )
    dr.text(
        (x0, H - 110),
        "Documento sintético generado para pruebas. Empresas y datos inventados.",
        font=S,
        fill=(90, 90, 90),
    )
    # stamp over the totals and a handwritten note
    color = None
    if random.random() < 0.6:
        txt = random.choice(["PAGADO", "CONTABILIZADO", "RECIBIDO"])
        color = random.choice([(200, 30, 30), (30, 60, 170)])
        stamp = Image.new("RGBA", (520, 130), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stamp)
        sd.rectangle((4, 4, 515, 125), outline=color + (170,), width=6)
        sd.text((30, 25), txt, font=font("DejaVuSans-Bold.ttf", 64), fill=color + (150,))
        stamp = stamp.rotate(random.uniform(-18, 18), expand=True)
        img.paste(stamp, (random.randint(560, 720), ytot + random.randint(-10, 90)), stamp)
    if random.random() < 0.5:
        dr.text(
            (random.randint(500, 760), random.randint(300, 360)),
            random.choice(["OK revisar IVA", "conforme - JL", "pasar a contab."]),
            font=HAND,
            fill=(40, 40, 120),
        )
    # bad scan
    img = img.rotate(random.uniform(2, 4) * random.choice([-1, 1]), expand=True, fillcolor=(235, 233, 228))
    img = img.resize((900, int(img.height * 900 / img.width)))
    noise = Image.effect_noise(img.size, 40).convert("RGB")
    img = Image.blend(img, noise, random.uniform(0.08, 0.14)).filter(ImageFilter.GaussianBlur(random.uniform(0.6, 1.0)))
    if color is None:
        img = img.convert("L")
    name = f"dificil_{i:02d}.jpg"
    img.save(os.path.join(OUT, name), quality=random.randint(28, 40))
    return name, gt


truth = {}
for i in range(N):
    n, g = invoice(i)
    truth[n] = g
json.dump(truth, open(os.path.join(OUT, "verdad.json"), "w"), ensure_ascii=False, indent=1)
print("ok", N)
