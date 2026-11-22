#!/usr/bin/env python3
"""Provenance records that bind a crop image to its source PDF.

A record (written next to the crop as <crop>.prov.json and returned by
crop_region.py) holds: source PDF SHA-256, physical page (1-based), page
count and size, bbox in PDF points, render dpi, crop PNG SHA-256 and pixel
size. check_crop() re-validates a crop against a record and, when the source
PDF is available, re-renders the region and compares pixels.

Paths are stored relative to the crop's own directory, so a moved result
folder can still be checked.
"""
import hashlib
import json
import os

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

PROV_VERSION = 1


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def render_clip(pdf_path, page_no, bbox, dpi):
    d = fitz.open(pdf_path)
    page = d[page_no - 1]
    clip = fitz.Rect(bbox) & page.rect
    return page.get_pixmap(dpi=dpi, clip=clip), page


def make_record(pdf_path, page_no, bbox, dpi, crop_path, pdf_sha=None):
    d = fitz.open(pdf_path)
    page = d[page_no - 1]
    rec = {
        "provenance_version": PROV_VERSION,
        "tool": "crop_region.py",
        "source_sha256": pdf_sha or sha256_file(pdf_path),
        "source_name": os.path.basename(pdf_path),
        "page": page_no,
        "page_count": d.page_count,
        "page_size": [round(page.rect.width, 2), round(page.rect.height, 2)],
        "bbox": [round(v, 2) for v in bbox],
        "dpi": dpi,
        "crop_file": os.path.basename(crop_path),
        "crop_sha256": sha256_file(crop_path),
    }
    try:
        pix = fitz.Pixmap(crop_path)
        rec["crop_px"] = [pix.width, pix.height]
    except Exception:
        pass
    return rec


def write_sidecar(crop_path, rec):
    p = crop_path + ".prov.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, ensure_ascii=False, indent=1)
    return p


def load_sidecar(crop_path):
    p = crop_path + ".prov.json"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def pixel_diff(pix_a, path_b):
    """Mean absolute difference (0-255) between a pixmap and a PNG file; None if sizes differ."""
    b = fitz.Pixmap(path_b)
    a = pix_a
    if a.n != b.n:
        a = fitz.Pixmap(fitz.csRGB, a) if a.n < 3 else a
        b = fitz.Pixmap(fitz.csRGB, b) if b.n < 3 else b
        if a.alpha:
            a = fitz.Pixmap(a, 0)
        if b.alpha:
            b = fitz.Pixmap(b, 0)
    if (a.width, a.height, a.n) != (b.width, b.height, b.n):
        return None
    sa, sb = a.samples, b.samples
    if len(sa) != len(sb):
        return None
    step = max(1, len(sa) // 200000)
    tot = sum(abs(sa[i] - sb[i]) for i in range(0, len(sa), step))
    return tot / len(range(0, len(sa), step))


def check_crop(evidence, crop_path, rec, known_pdf_shas, pdf_lookup, page_sizes=None):
    """Return (level, problems, info).

    level: 'bound+rendered' (record consistent and pixels re-rendered from the
    PDF match), 'bound' (record consistent, PDF not available to re-render),
    'none' (no record), or 'mismatch' (problems listed).
    """
    problems, info = [], {}
    if rec is None:
        return "none", [], {"note": "no provenance record: the crop cannot be tied to a PDF page"}
    if not crop_path or not os.path.exists(crop_path):
        return "mismatch", ["crop file not found"], info
    cur = sha256_file(crop_path)
    if rec.get("crop_sha256") and cur != rec["crop_sha256"]:
        problems.append("crop file was replaced or edited after it was made (SHA-256 differs from its provenance record)")
    pg = evidence.get("page")
    if pg is not None and int(pg) != int(rec.get("page", -1)):
        problems.append(f"evidence cites page {pg} but the crop was cut from page {rec.get('page')}")
    eb = evidence.get("bbox")
    if eb and rec.get("bbox") and any(abs(float(a) - float(b)) > 1.5 for a, b in zip(eb, rec["bbox"])):
        problems.append(f"evidence bbox {eb} differs from the crop's bbox {rec['bbox']}")
    if known_pdf_shas and rec.get("source_sha256") not in known_pdf_shas:
        problems.append("crop comes from a different PDF (source SHA-256 not among this case's documents)")
    size = rec.get("page_size") or (page_sizes or {}).get(int(rec.get("page", 0)))
    bb = rec.get("bbox") or eb
    if size and bb and (bb[0] < -1 or bb[1] < -1 or bb[2] > size[0] + 1 or bb[3] > size[1] + 1):
        problems.append(f"bbox {bb} lies outside the page ({size[0]} x {size[1]} pt)")
    if rec.get("page_count") and not (1 <= int(rec.get("page", 0)) <= int(rec["page_count"])):
        problems.append(f"page {rec.get('page')} outside 1..{rec['page_count']}")
    if problems:
        return "mismatch", problems, info
    pdf = pdf_lookup(rec.get("source_sha256")) if pdf_lookup else None
    if pdf:
        try:
            pix, _ = render_clip(pdf, int(rec["page"]), rec["bbox"], int(rec.get("dpi", 150)))
            diff = pixel_diff(pix, crop_path)
            info["pixel_mean_abs_diff"] = None if diff is None else round(diff, 3)
            if diff is None or diff > 3.0:
                return "mismatch", ["crop pixels do not match a fresh render of the cited PDF region"], info
            return "bound+rendered", [], info
        except Exception as ex:
            info["render_error"] = str(ex)
    return "bound", [], info
