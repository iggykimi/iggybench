# Invoice sets

Two sets of synthetic Spanish invoices used by the document tests. Each folder holds the JPG images and a
`verdad.json` with the ground truth per file (field names in `../SCHEMA_GLOSSARY.md`). Eight fields are scored:
`numero`, `fecha`, `proveedor`, `nif_proveedor`, `base_imponible`, `iva_pct`, `total`, `num_lineas`.

| Folder | Contents | Used by |
|---|---|---|
| `invoices_clean/` | 40 invoices (`factura_00.jpg` … `factura_39.jpg`), 1240x1754, clean scan-like rendering | T11, OCR pipeline |
| `invoices_degraded/` | 30 invoices (`dificil_00.jpg` … `dificil_29.jpg`), 900 px wide: rotation, blur, noise, heavy JPEG compression, stamps over the totals, handwritten notes, distractor numbers and dates, discounts, mixed date and number formats | T11D |

Everything in them is invented: company names, addresses, line items and amounts. The IBAN printed on the invoices
is a placeholder (`ES00 0000 …`, marked "ficticio"). Tax IDs (NIF) are random strings in the Spanish format; one
could match a real company's NIF by chance.

The degraded set is produced by `../make_degraded_invoices.py` with a fixed seed (`random.seed(2027)`). Running it
again regenerates the same `verdad.json`; the images can differ slightly with other Pillow or DejaVu font versions.
The script that generated the clean set is not available, so that set is published as data only.

In 7 clean invoices (`factura_12`, `13`, `24`, `29`, `31`, `33`, `35`) the customer name was changed after the
benchmark ran, to "Demo Cliente S.L.", in both the image and `verdad.json`. The customer field (`cliente`) is not
one of the scored fields, so the published results do not depend on it. The other pixels of those images are
unchanged apart from JPEG re-encoding.
