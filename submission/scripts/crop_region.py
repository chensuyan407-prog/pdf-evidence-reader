#!/usr/bin/env python3
"""Crop one page region to a PNG for visual reading, measure its ink, and
record where the crop came from.

Usage:
    python crop_region.py PARSED_DIR --page 1 --bbox 50,230,290,420 [--dpi 220]
    python crop_region.py PARSED_DIR --page 1 --region p1-r12
    python crop_region.py PARSED_DIR --equation 8          # crop of an equation by its number

--bbox is in PDF points (origin top-left), as in doc.json. --page is the
physical page number (1-based), not the printed page label. The source PDF is
taken from doc.json meta.source_file (or --pdf) and must have the SHA-256
recorded in doc.json.

Writes PARSED_DIR/crops/<name>.png and <name>.png.prov.json (provenance:
source PDF SHA-256, page, page size, bbox, dpi, crop SHA-256). Prints JSON
{crop, crop_rel, page, bbox, dpi, ink_ratio, blank, provenance}. blank=true:
the region holds (almost) no ink - nothing can be read there, do not guess.

Image evidence item for result.json (crop path relative to result.json is fine):
  {"field": F, "type": "image_region", "page": 1, "bbox": [...], "crop": "parsed/<stem>/crops/....png",
   "provenance": <the provenance object printed here>,
   "transcription": "exact text you read", "read_by": "agent_vision", "legibility": "clear|partial|illegible"}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_pdf  # noqa: E402
import provenance  # noqa: E402
from parse_pdf import fitz  # noqa: E402


def find_pdf(doc_json, parsed, explicit=None):
    want = doc_json.get("meta", {}).get("sha256")
    cands = [explicit, doc_json.get("meta", {}).get("source_file")]
    cands += [os.path.join(parsed, "..", "..", n) for n in os.listdir(os.path.join(parsed, "..", "..")) if n.lower().endswith(".pdf")] \
        if os.path.isdir(os.path.join(parsed, "..", "..")) else []
    for c in cands:
        if c and os.path.exists(c) and (not want or provenance.sha256_file(c) == want):
            return c
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parsed")
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--bbox", default=None)
    ap.add_argument("--region", default=None)
    ap.add_argument("--equation", default=None, help="equation number, e.g. 8")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--pad", type=float, default=0.0, help="extra margin (pt) around the bbox")
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--out", default=None, help="directory for crops (default PARSED/crops)")
    a = ap.parse_args()
    doc_json = json.load(open(os.path.join(a.parsed, "doc.json"), encoding="utf-8"))
    pdf = find_pdf(doc_json, a.parsed, a.pdf)
    if not pdf:
        print(json.dumps({"error": "source PDF not found or its SHA-256 differs from doc.json meta.sha256; pass --pdf"}))
        sys.exit(2)
    bbox, page_no = None, a.page
    if a.region:
        for p in doc_json.get("pages", []):
            for r in p.get("image_regions", []) or []:
                if r["id"] == a.region:
                    bbox, page_no = r["bbox"], p["page"]
        if bbox is None:
            print(json.dumps({"error": f"region {a.region} not found"}))
            sys.exit(1)
    elif a.equation:
        eq = next((e for e in doc_json.get("equations", []) if str(e.get("number")) == str(a.equation)), None)
        if eq is None:
            print(json.dumps({"error": f"equation ({a.equation}) not in doc.json equations"}))
            sys.exit(1)
        bbox, page_no = eq["bbox"], eq["page"]
    elif a.bbox:
        bbox = [float(v) for v in a.bbox.split(",")]
    if page_no is None:
        print(json.dumps({"error": "--page is required with --bbox"}))
        sys.exit(1)
    d = fitz.open(pdf)
    if not 1 <= page_no <= d.page_count:
        print(json.dumps({"error": f"page {page_no} outside 1..{d.page_count}"}))
        sys.exit(1)
    page = d[page_no - 1]
    clip = (fitz.Rect(bbox) + (-a.pad, -a.pad, a.pad, a.pad)) & page.rect if bbox else page.rect
    if bbox and fitz.Rect(bbox) != fitz.Rect(bbox) & page.rect:
        print(json.dumps({"warning": "bbox extended beyond the page; clipped to the page"}), file=sys.stderr)
    cb = [round(v, 2) for v in (clip.x0, clip.y0, clip.x1, clip.y1)]
    mask, _ = parse_pdf.ink_mask(page, a.dpi, clip)
    out_dir = a.out or os.path.join(a.parsed, "crops")
    os.makedirs(out_dir, exist_ok=True)
    name = f"p{page_no}_{int(clip.x0)}_{int(clip.y0)}_{int(clip.x1)}_{int(clip.y1)}_{a.dpi}.png"
    path = os.path.join(out_dir, name)
    page.get_pixmap(dpi=a.dpi, clip=clip).save(path)
    rec = provenance.make_record(pdf, page_no, cb, a.dpi, path, pdf_sha=doc_json.get("meta", {}).get("sha256"))
    provenance.write_sidecar(path, rec)
    ink = float(mask.mean()) if mask is not None else None
    print(json.dumps({"crop": path, "crop_rel": os.path.relpath(path, a.parsed).replace("\\", "/"), "page": page_no,
                      "bbox": cb, "dpi": a.dpi, "ink_ratio": None if ink is None else round(ink, 4),
                      "blank": None if ink is None else ink < 0.002, "provenance": rec}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
