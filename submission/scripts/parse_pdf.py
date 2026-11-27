#!/usr/bin/env python3
"""Layout-aware scientific PDF parser with page/block provenance.

Usage:
    python parse_pdf.py INPUT.pdf --out OUTDIR [--password PW] [--preview]
                        [--pages 1-5] [--ocr auto|force|off] [--lang eng+chi_sim]

Writes OUTDIR/doc.json (structured result), document.md (reading-order
markdown with source anchors), blocks.jsonl (one block per line, for grep),
tables/*.{md,csv,html}, qa_report.json and, with --preview, previews/*.png
with the detected blocks and their reading order drawn on each page.

Every block id has the form p{page}-b{index}; every output element keeps the
page number, bbox (PDF points, origin top-left) and the rule that classified it.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import mathlayout  # numbered display equations
except Exception:  # pragma: no cover - parsing still works without it
    mathlayout = None

PARSER_VERSION = "0.4.0"

# ----------------------------------------------------------------- patterns
RE_CAPTION = re.compile(
    r"^\s*(Fig\.|Figure|FIGURE|Fig|Table|TABLE|Tab\.|Scheme|Algorithm|Supplementary\s+(?:Figure|Table)|"
    r"Extended\s+Data\s+(?:Fig\.|Figure|Table)|图|表)\s*(S?(?:[A-Z]\.?)?\d+(?:\.\d+)*[a-zA-Z]?|[IVX]+)\s*[\.:|：．]?",
)
RE_HEAD_NUM = re.compile(
    r"^\s*((?:\d+\.)*\d+)\.?\s+([A-Z一-鿿][^\n]{0,120})$"
)
RE_HEAD_ROMAN = re.compile(r"^\s*([IVX]{1,5})\.\s+([A-Z][^\n]{0,120})$")
RE_HEAD_ALPHA = re.compile(r"^\s*([A-H])\.\s+([A-Z][^\n]{0,100})$")
RE_HEAD_APPX = re.compile(r"^\s*([A-Z](?:\.\d+)*)\.?\s+([A-Z][^\n]{1,120})$")
RE_HEAD_APPX_WORD = re.compile(r"^\s*(?:Appendix|APPENDIX|Supplementary\s+(?:Note|Section)|SUPPLEMENTARY\s+NOTE)\s+([A-Z]|\d+)(?:[\.:：]\s*|\s+[-–—]\s*|\s+)([A-Z][^\n]{1,120})$")
RE_HEAD_CN = re.compile(r"^\s*(第[一二三四五六七八九十]+[章节]|[一二三四五六七八九十]+、)\s*(.{1,60})$")
KNOWN_SECTIONS = re.compile(
    r"^\s*(abstract|introduction|background|related work|methods?|materials and methods|"
    r"experimental( section| details)?|results( and discussion)?|discussion|conclusions?|"
    r"acknowledg(e)?ments?|references|bibliography|appendix( [a-z])?|supplementary (information|materials?)|"
    r"data availability|author contributions|competing interests|摘\s*要|引\s*言|参考文献|结\s*论|致\s*谢)\s*:?\s*$",
    re.I,
)
RE_REFS_HEAD = re.compile(r"^\s*((\d+|[IVX]+)\.?\s*)?(references|bibliography|literature cited|参考文献)\s*$", re.I)
RE_EQNUM_LINE = re.compile(r"^\(\s*(?:\d+\.)?\d+[a-z]?\s*\)$")
RE_EQ_NUM = re.compile(r"\(\s*((?:\d+\.)?\d+[a-z]?|[A-Z]\d+)\s*\)\s*$")
RE_PAGE_NUM = re.compile(r"^\s*(page\s*)?(\d{1,4}|[ivxlc]{1,6})(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
RE_REF_SPLIT_BRACKET = re.compile(r"(?=(?:^|\s)\[\d{1,4}\]\s)")
RE_REF_START_NUM = re.compile(r"^\s*(\[\d{1,4}\]|\d{1,4}\.)\s+")
MATH_CHARS = set("=+−-×·∑∏∫∂∇√∞≈≠≤≥±∈∉⊂⊆∀∃→←↔⇒αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ^_()[]{}|/<>")
SENT_END = re.compile(r"[\.\?\!。！？:：;；]\s*[\"'”’)\]]?\s*$")


def norm_ws(s):
    return re.sub(r"\s+", " ", s or "").strip()


def norm_key(s):
    """Normalisation used to match repeated headers/footers across pages."""
    s = norm_ws(s).lower()
    s = re.sub(r"\d+", "#", s)
    return s


# ----------------------------------------------------------------- open
def open_document(path, password=None):
    """Return (doc, error_code, message)."""
    if not os.path.exists(path):
        return None, "file_not_found", f"{path} does not exist"
    try:
        doc = fitz.open(path)
    except Exception as e:  # damaged / not a pdf
        return None, "open_failed", f"PyMuPDF could not open the file: {e}"
    if getattr(doc, "needs_pass", False):
        ok = doc.authenticate(password or "")
        if not ok:
            return None, "encrypted", "PDF is password protected; pass --password or ask a human for the password"
    if doc.page_count == 0:
        return None, "empty", "PDF has zero pages"
    return doc, None, ""


# ----------------------------------------------------------------- spans/lines
MATH_FONT = re.compile(r"^(CM(MI|SY|EX|BX|R|MIB|BSY)\d*|MSBM|MSAM|EUFM|EUSM|.*Math.*|Symbol.*|STIX.*|rsfs.*)", re.I)
SCRIPT_CAPS = {"B": "ℬ", "E": "ℰ", "F": "ℱ", "H": "ℋ", "I": "ℐ", "L": "ℒ", "M": "ℳ", "R": "ℛ"}
DOUBLE_CAPS = {"C": "ℂ", "H": "ℍ", "N": "ℕ", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "Z": "ℤ"}


def _math_alpha(ch, style):
    """Unicode mathematical alphanumeric for one ASCII letter."""
    if not ch.isascii() or not ch.isalpha():
        return ch
    up = ch.isupper()
    off = ord(ch) - (65 if up else 97)
    if style == "bold":
        return chr((0x1D400 if up else 0x1D41A) + off)
    if style == "cal" and up:
        return SCRIPT_CAPS.get(ch, chr(0x1D49C + off))
    if style == "bb" and up:
        return DOUBLE_CAPS.get(ch, chr(0x1D538 + off))
    if style == "frak":
        return chr((0x1D504 if up else 0x1D51E) + off)
    return ch


def span_math(t, font):
    """(latex, unicode) for one span, using the font name to recover the math alphabet."""
    core = t.strip()
    if not core:
        return t, t
    lead, trail = t[:len(t) - len(t.lstrip())], t[len(t.rstrip()):]
    style = None
    if re.match(r"CMSY|EUSM|rsfs", font, re.I) and re.fullmatch(r"[A-Z]+", core):
        style = "cal"
    elif re.match(r"MSBM", font, re.I) and re.fullmatch(r"[A-Z]+", core):
        style = "bb"
    elif re.match(r"EUFM", font, re.I):
        style = "frak"
    elif re.match(r"CMBX|CMMIB|CMBSY", font, re.I) or (re.search(r"bold", font, re.I) and re.search(r"math|stix", font, re.I)):
        style = "bold"
    cmd = {"cal": "\\mathcal", "bb": "\\mathbb", "frak": "\\mathfrak", "bold": "\\mathbf"}.get(style)
    if cmd and re.fullmatch(r"[A-Za-z]+", core):
        lx = f"{cmd}{{{core}}}"
        uni = "".join(_math_alpha(c, style) for c in core)
    else:
        lx = core.replace("{", "\\{").replace("}", "\\}") if re.match(r"CMSY", font, re.I) else core
        uni = core
    return lead + lx + trail, lead + uni + trail


def span_char_count(sp):
    return len(sp.get("text", "").strip())


def build_line(line):
    spans = [s for s in line.get("spans", []) if s.get("text", "") != ""]
    if not spans:
        return None
    # dominant span = most characters
    sizes = Counter()
    for s in spans:
        sizes[round(s["size"], 1)] += max(1, span_char_count(s))
    base = sizes.most_common(1)[0][0]
    dom = max(spans, key=lambda s: (round(s["size"], 1) == base, span_char_count(s)))
    baseline = dom["origin"][1]
    text_parts, marked_parts, latex_parts, umath_parts = [], [], [], []
    bold_chars = italic_chars = total = math_chars = 0
    fonts = Counter()
    n_sup = n_sub = 0
    for s in spans:
        t = s["text"]
        text_parts.append(t)
        c = span_char_count(s)
        total += c
        fname = s.get("font", "")
        fonts[fname] += c
        flags = s.get("flags", 0)
        if flags & 16 or re.search(r"bold|black|heavy|semibold|\.B$|-B$|CMBX", fname, re.I):
            bold_chars += c
        if flags & 2 or re.search(r"italic|oblique|CMMI|CMTI", fname, re.I):
            italic_chars += c
        small = s["size"] < 0.85 * base
        dy = s["origin"][1] - baseline
        lx, um = span_math(t, fname)
        if MATH_FONT.match(fname):
            math_chars += c
        if t.strip() and (flags & 1 or (small and dy < -0.12 * base)):
            marked_parts.append("^{" + t.strip() + "}")
            latex_parts.append("^{" + lx.strip() + "}")
            umath_parts.append("^{" + um.strip() + "}")
            n_sup += 1
        elif t.strip() and small and dy > 0.08 * base:
            marked_parts.append("_{" + t.strip() + "}")
            latex_parts.append("_{" + lx.strip() + "}")
            umath_parts.append("_{" + um.strip() + "}")
            n_sub += 1
        else:
            marked_parts.append(t)
            latex_parts.append(lx)
            umath_parts.append(um)
    d = line.get("dir", (1, 0))
    lead = next((s for s in spans if s["text"].strip()), spans[0])
    lead_bold = bool(lead.get("flags", 0) & 16 or re.search(r"bold|black|heavy|semibold|CMBX", lead.get("font", ""), re.I))
    return {
        "lead_bold": lead_bold,
        "text": "".join(text_parts),
        "marked": "".join(marked_parts),
        "latex": "".join(latex_parts),
        "umath": "".join(umath_parts),
        "math_chars": math_chars,
        "bbox": [round(v, 2) for v in line["bbox"]],
        "size": base,
        "bold": total > 0 and bold_chars / total > 0.6,
        "italic": total > 0 and italic_chars / total > 0.6,
        "fonts": [f for f, _ in fonts.most_common(3)],
        "horizontal": abs(d[0]) > 0.99 and abs(d[1]) < 0.01,
        "n_sup": n_sup,
        "n_sub": n_sub,
    }


def join_lines(lines, key="text"):
    """Join lines of a block, de-hyphenating words split at line ends.
    Returns (text, n_dehyphenated)."""
    out = ""
    n = 0
    for i, ln in enumerate(lines):
        t = ln[key].strip()
        if not t:
            continue
        if not out:
            out = t
            continue
        if re.search(r"[A-Za-z]{2,}-$", out) and re.match(r"^[a-z]", t):
            out = out[:-1] + t
            n += 1
        elif re.search(r"[A-Za-z]{2,}-$", out) and re.match(r"^[A-Z][a-z]", t):
            out = out + t  # compound split at the line end keeps its hyphen ("Batch-" + "Normalized")
        elif re.search(r"[一-鿿]$", out) and re.match(r"^[一-鿿]", t):
            out = out + t  # CJK: no space between wrapped lines
        else:
            out = out + " " + t
    return out, n


def heading_like(text):
    t = norm_ws(text)
    if not t or len(t) > 130:
        return None
    for rx, kind in ((RE_HEAD_APPX_WORD, "appendix_word"), (RE_HEAD_NUM, "numeric"), (RE_HEAD_ROMAN, "roman"), (RE_HEAD_ALPHA, "alpha"),
                     (RE_HEAD_CN, "cn"), (RE_HEAD_APPX, "appendix")):
        m = rx.match(t)
        if m:
            if kind != "cn" and not re.search(r"[A-Za-z]{2,}|[一-鿿]", m.group(2)):
                continue  # "1 N", "2 x": a number followed by a symbol is not a heading
            if kind in ("appendix", "named", "numeric") and re.search(r"(\.\s*){3,}\s*\d+$|\s\d{1,3}$", t) and \
                    not re.search(r"\b(19|20)\d\d$", t) and not RE_LABEL_NUM_END.search(t):
                return None  # table-of-contents entry ("A Details ... 43"), but not "Proof of Lemma 1"
            return kind, m.group(1), m.group(2)
    if KNOWN_SECTIONS.match(t):
        return "named", "", t
    return None


RE_LABEL_NUM_END = re.compile(r"\b(Lemma|Theorem|Proposition|Corollary|Definition|Assumption|Experiment|Study|Phase|Task|"
                              r"Part|Stage|Case|Step|Round|Day|Group|Level|Version|Setting|Scenario|Method|Model|"
                              r"Example|Problem|Question|Hypothesis|Claim|Condition|Appendix|Section|Chapter)\s+\d{1,3}$", re.I)


def visual_lines(lines):
    """Number of printed lines: MuPDF may split one justified line into several 'lines' on the same baseline."""
    n, last = 0, None
    for ln in sorted(lines, key=lambda l: l["bbox"][1]):
        y0, y1 = ln["bbox"][1], ln["bbox"][3]
        if last is None or y0 > last[0] + 0.6 * (last[1] - last[0]):
            n += 1
            last = (y0, y1)
    return n


def make_block(page_no, lines, source="text_layer", conf=None):
    x0 = min(l["bbox"][0] for l in lines)
    y0 = min(l["bbox"][1] for l in lines)
    x1 = max(l["bbox"][2] for l in lines)
    y1 = max(l["bbox"][3] for l in lines)
    text, n_dehy = join_lines(lines, "text")
    marked, _ = join_lines(lines, "marked")
    has_math = any(l.get("math_chars", 0) for l in lines)
    latex = join_lines(lines, "latex")[0] if has_math else None
    umath = join_lines(lines, "umath")[0] if has_math else None
    sizes = [l["size"] for l in lines]
    return {
        "id": None,
        "page": page_no,
        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "lines": lines,
        "text": text,
        "marked": marked if marked != text else None,
        "latex": latex,
        "unicode_math": umath,
        "math_fonts": sorted({f for l in lines for f in l["fonts"] if MATH_FONT.match(f)})[:6] if has_math else None,
        "size": round(statistics.median(sizes), 2),
        "bold": all(l["bold"] for l in lines),
        "fonts": sorted({f for l in lines for f in l["fonts"]})[:4],
        "horizontal": all(l["horizontal"] for l in lines),
        "n_dehyphenated": n_dehy,
        "source": source,
        "ocr_conf": conf,
        "role": "body",
        "role_rule": "default",
        "column": None,
        "order": None,
    }


def split_raw_block(page_no, raw_lines):
    """Split a MuPDF block where a heading / caption line is glued to body text,
    or where a large vertical gap separates paragraphs."""
    groups, cur = [], []
    prev = None
    for ln in raw_lines:
        start_new = False
        if prev is not None:
            gap = ln["bbox"][1] - prev["bbox"][3]
            lh = max(1.0, prev["bbox"][3] - prev["bbox"][1])
            if gap > 0.9 * lh:
                start_new = True
            elif RE_EQNUM_LINE.match(ln["text"].strip()) or RE_EQNUM_LINE.match(prev["text"].strip()):
                start_new = True  # an equation number on its own line is not part of the next paragraph
            elif (heading_like(prev["text"]) and (prev["bold"] != ln["bold"] or abs(prev["size"] - ln["size"]) > 0.5)):
                start_new = True
            elif RE_CAPTION.match(ln["text"]) and not RE_CAPTION.match(cur[0]["text"]) if cur else False:
                start_new = True
            elif abs(prev["size"] - ln["size"]) > 1.5:
                start_new = True
        if start_new and cur:
            groups.append(cur)
            cur = []
        cur.append(ln)
        prev = ln
    if cur:
        groups.append(cur)
    return [make_block(page_no, g) for g in groups]


def extract_text_blocks(page, page_no):
    flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP
    d = page.get_text("dict", flags=flags)
    blocks = []
    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        lines = [build_line(l) for l in b.get("lines", [])]
        lines = [l for l in lines if l and l["text"].strip()]
        if not lines:
            continue
        blocks.extend(split_raw_block(page_no, lines))
    return blocks


# ----------------------------------------------------------------- OCR
def tesseract_langs():
    exe = shutil.which("tesseract")
    if not exe:
        return None, []
    try:
        out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=30)
        langs = [l.strip() for l in (out.stdout + out.stderr).splitlines()[1:] if l.strip() and " " not in l.strip()]
        return exe, langs
    except Exception:
        return exe, []


def ocr_page(page, page_no, workdir, lang_pref, dpi=300, budget_deadline=None):
    """OCR one page with the tesseract CLI. Returns (blocks, info dict)."""
    if budget_deadline is not None and time.time() >= budget_deadline:
        return [], {"ocr": "skipped", "reason": "cumulative time budget reached"}
    exe, langs = tesseract_langs()
    if not exe:
        return [], {"ocr": "unavailable", "reason": "tesseract executable not found on PATH"}
    wanted = [l for l in lang_pref.split("+") if l in langs] or (["eng"] if "eng" in langs else langs[:1])
    lang = "+".join(wanted) if wanted else "eng"
    os.makedirs(workdir, exist_ok=True)
    img = os.path.join(workdir, f"ocr_p{page_no}.png")
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    pix.save(img)
    timeout = 180 if budget_deadline is None else min(180.0, max(1.0, budget_deadline - time.time()))
    try:
        res = subprocess.run([exe, img, "stdout", "-l", lang, "--psm", "3", "tsv"],
                             capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    except Exception as e:
        return [], {"ocr": "failed", "reason": str(e)}
    scale = 72.0 / dpi
    groups = defaultdict(list)
    low_conf = []
    for row in res.stdout.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        text = parts[11]
        if not text.strip():
            continue
        try:
            conf = float(parts[10])
        except ValueError:
            conf = -1
        key = (int(parts[2]), int(parts[3]), int(parts[4]))
        l, t, w, h = (int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9]))
        groups[key].append((l, t, w, h, text, conf))
        if 0 <= conf < 60:
            low_conf.append({"word": text, "conf": conf, "bbox": [round(l * scale, 1), round(t * scale, 1), round((l + w) * scale, 1), round((t + h) * scale, 1)]})
    by_par = defaultdict(list)
    for (blk, par, ln), words in sorted(groups.items()):
        words.sort(key=lambda w: w[0])
        x0 = min(w[0] for w in words) * scale
        y0 = min(w[1] for w in words) * scale
        x1 = max(w[0] + w[2] for w in words) * scale
        y1 = max(w[1] + w[3] for w in words) * scale
        txt = " ".join(w[4] for w in words)
        confs = [w[5] for w in words if w[5] >= 0]
        by_par[(blk, par)].append({
            "text": txt, "marked": txt, "latex": txt, "umath": txt, "math_chars": 0, "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
            "size": round((y1 - y0), 1), "bold": False, "lead_bold": False, "italic": False, "fonts": ["ocr"],
            "horizontal": True, "n_sup": 0, "n_sub": 0, "_conf": statistics.mean(confs) if confs else None,
        })
    blocks = []
    for key, lines in by_par.items():
        confs = [l["_conf"] for l in lines if l["_conf"] is not None]
        b = make_block(page_no, lines, source="ocr", conf=round(statistics.mean(confs), 1) if confs else None)
        blocks.append(b)
    return blocks, {"ocr": "done", "lang": lang, "dpi": dpi, "low_conf_words": low_conf[:200],
                    "n_low_conf_words": len(low_conf)}


def ink_mask(page, dpi=150, clip=None):
    """Boolean ink mask of a rendered page (None if numpy is unavailable)."""
    try:
        import numpy as np
    except ImportError:
        return None, None
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, clip=clip)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    return arr < 160, pix


def xy_cut_regions(mask, scale, min_h_gap=10, min_v_gap=16, min_ink=30):
    """Recursive XY-cut on an ink mask; returns layout regions in PDF points
    (x0, y0, x1, y1, ink_ratio) in reading order (top-down, left-right)."""
    import numpy as np
    out = []

    def longest_gap(flags):
        best, start = (0, 0), None
        for i, f in enumerate(flags):
            if not f:
                if start is None:
                    start = i
            elif start is not None:
                if i - start > best[1] - best[0]:
                    best = (start, i)
                start = None
        return best  # interior gaps only (ends are trimmed before)

    def cut(x0, y0, x1, y1, depth):
        sub = mask[y0:y1, x0:x1]
        if sub.sum() < min_ink:
            return
        ys = np.where(sub.any(axis=1))[0]
        xs = np.where(sub.any(axis=0))[0]
        y0, y1, x0, x1 = y0 + ys[0], y0 + ys[-1] + 1, x0 + xs[0], x0 + xs[-1] + 1
        sub = mask[y0:y1, x0:x1]
        if depth < 14:
            vg = longest_gap(sub.any(axis=0))
            hg = longest_gap(sub.any(axis=1))
            vlen, hlen = vg[1] - vg[0], hg[1] - hg[0]
            if vlen >= min_v_gap and (vlen >= hlen or hlen < min_h_gap):
                m = x0 + (vg[0] + vg[1]) // 2
                cut(x0, y0, m, y1, depth + 1)
                cut(m, y0, x1, y1, depth + 1)
                return
            if hlen >= min_h_gap:
                m = y0 + (hg[0] + hg[1]) // 2
                cut(x0, y0, x1, m, depth + 1)
                cut(x0, m, x1, y1, depth + 1)
                return
        ink = float(sub.mean())
        out.append((round(x0 * scale, 1), round(y0 * scale, 1), round(x1 * scale, 1), round(y1 * scale, 1), round(ink, 4)))

    cut(0, 0, mask.shape[1], mask.shape[0], 0)
    return out


def page_image_and_regions(page, page_no, out_dir, dpi=150):
    """For pages without usable text: save the page image and layout regions
    so an agent with vision can read a region and cite it by coordinates."""
    os.makedirs(os.path.join(out_dir, "page_images"), exist_ok=True)
    rel = f"page_images/page_{page_no:03d}.png"
    mask, pix = ink_mask(page, dpi)
    if pix is None:
        pix = page.get_pixmap(dpi=dpi)
    pix.save(os.path.join(out_dir, rel))
    regions = []
    if mask is not None:
        for k, (x0, y0, x1, y1, ink) in enumerate(xy_cut_regions(mask, 72.0 / dpi)):
            regions.append({"id": f"p{page_no}-r{k + 1}", "page": page_no, "bbox": [x0, y0, x1, y1], "ink_ratio": ink})
    return rel, dpi, regions


def raster_regions_without_text(page, page_no, blocks, pdf_path, pdf_sha, out_dir, lang="eng", dpi=170):
    """Embedded raster images on a page that HAS a text layer but whose picture area holds (almost) no text:
    a table/figure/equation pasted as an image. Their content is invisible to text extraction, so each one gets a
    crop (with a provenance sidecar, usable directly as image_region evidence) and, if tesseract exists, OCR text
    that is reported separately (never mixed into body text)."""
    W, H = page.rect.width, page.rect.height
    out = []
    try:
        infos = page.get_image_info()
    except Exception:
        return out
    seen = []
    tab_caps = [b["bbox"] for b in blocks if re.match(r"^\s*(Table|TABLE|Tab\.|表)\s*[A-Z]?\d*", b.get("text") or "")]
    for im in infos:
        r = fitz.Rect(im["bbox"]) & page.rect
        if r.is_empty:
            continue
        near_tab_cap = any(
            min(r.x1, cb[2]) - max(r.x0, cb[0]) > -20 and
            min(abs(cb[3] - r.y0), abs(r.y1 - cb[1])) <= 140
            for cb in tab_caps
        )
        if near_tab_cap:
            if r.width < 45 or r.height < 20 or r.width * r.height < 0.004 * W * H:
                continue
        elif r.width < 90 or r.height < 45 or r.width * r.height < 0.04 * W * H:
            continue
        if r.width * r.height > 0.85 * W * H:
            continue  # full-page scan: handled by the scanned-page path
        if any(abs(r.x0 - q.x0) < 2 and abs(r.y0 - q.y0) < 2 and abs(r.x1 - q.x1) < 2 for q in seen):
            continue
        seen.append(r)
        inside = sum(len(b["text"]) for b in blocks if bbox_center_in(b["bbox"], [r.x0, r.y0, r.x1, r.y1], pad=0))
        if inside >= 40:
            continue  # text layer covers the picture (vector-like figure labels or an OCR overlay)
        mask, _ = ink_mask(page, 60, r)
        ink = float(mask.mean()) if mask is not None else None
        if ink is not None and ink < 0.004:
            continue
        bb = [round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)]
        rec = {"id": f"p{page_no}-x{len(out) + 1}", "page": page_no, "bbox": bb, "text_layer_chars_inside": inside,
               "ink_ratio": None if ink is None else round(ink, 4), "crop": None, "ocr_text": None,
               "crop_command": f"crop_region.py PARSED --page {page_no} --bbox {','.join(str(v) for v in bb)}"}
        out.append(rec)
    return out


def materialize_raster(page, rec, pdf_path, pdf_sha, out_dir, lang="eng", dpi=170, budget_deadline=None):
    """Crop one raster region (provenance sidecar, same format as crop_region.py) and OCR it when tesseract exists."""
    try:
        import provenance
        r = fitz.Rect(rec["bbox"])
        cdir = os.path.join(out_dir, "crops")
        os.makedirs(cdir, exist_ok=True)
        name = f"p{rec['page']}_{int(r.x0)}_{int(r.y0)}_{int(r.x1)}_{int(r.y1)}_{dpi}.png"
        cpath = os.path.join(cdir, name)
        page.get_pixmap(dpi=dpi, clip=r).save(cpath)
        provenance.write_sidecar(cpath, provenance.make_record(pdf_path, rec["page"], rec["bbox"], dpi, cpath, pdf_sha=pdf_sha))
        rec["crop"], rec["dpi"] = f"crops/{name}", dpi
    except Exception as ex:
        rec["crop_error"] = str(ex)
        return
    if budget_deadline is not None and time.time() >= budget_deadline:
        rec["ocr_error"] = "cumulative time budget reached"
        return
    exe, langs = tesseract_langs()
    if exe:
        timeout = 60 if budget_deadline is None else min(60.0, max(1.0, budget_deadline - time.time()))
        try:
            use = "+".join(l for l in lang.split("+") if l in langs) or "eng"
            res = subprocess.run([exe, os.path.join(out_dir, rec["crop"]), "stdout", "-l", use, "--psm", "6"],
                                 capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
            rec["ocr_text"] = norm_ws(res.stdout) or None
            rec["ocr_note"] = "tesseract on the crop; unverified OCR - check against the crop before citing"
        except Exception as ex:
            rec["ocr_error"] = str(ex)


def page_is_scanned(page, text_blocks):
    chars = sum(len(b["text"]) for b in text_blocks)
    area = page.rect.width * page.rect.height
    img_area = 0.0
    try:
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"]) & page.rect
            img_area += r.width * r.height
    except Exception:
        pass
    return chars < 40 and img_area > 0.3 * area, chars, img_area / area if area else 0


def garbled_ratio(text):
    if not text:
        return 0.0
    bad = sum(1 for c in text if c == "�" or 0xE000 <= ord(c) <= 0xF8FF)
    bad += 4 * len(re.findall(r"\(cid:\d+\)", text))
    return bad / max(1, len(text))


# ----------------------------------------------------------------- layout
def mark_figure_text(blocks, regions, W):
    """Text that belongs to a figure (tick labels, legends, rotated axis titles):
    inside or just around a graphic region, and small or rotated or short."""
    if not regions:
        return 0
    sizes = Counter()
    for b in blocks:
        for ln in b["lines"]:
            sizes[ln["size"]] += len(ln["text"])
    local_body = sizes.most_common(1)[0][0] if sizes else 10.0
    n = 0
    for b in blocks:
        if b["role"] != "body" or RE_CAPTION.match(b["text"]):
            continue
        x0, y0, x1, y1 = b["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for r in regions:
            pad = 16
            if r[0] - pad <= cx <= r[2] + pad and r[1] - pad <= cy <= r[3] + pad:
                small = b["size"] < 0.85 * local_body
                if small or not b["horizontal"] or len(b["text"]) < 25:
                    b["role"] = "figure_text"
                    b["role_rule"] = ("rotated text next to a graphic (axis title)" if not b["horizontal"]
                                      else "small/short text inside a graphic region")
                    b["figure_region"] = [round(v, 1) for v in r]
                    n += 1
                break
    return n


def detect_headers_footers(pages_blocks, page_heights, body_size=None, page_widths=None):
    """Mark running headers/footers/page numbers and rotated margin text."""
    n_pages = len(pages_blocks)
    cand = defaultdict(set)
    for pi, blocks in enumerate(pages_blocks):
        H = page_heights[pi]
        for b in blocks:
            if b["bbox"][3] < 0.085 * H or b["bbox"][1] > 0.915 * H:
                cand[norm_key(b["text"])].add(pi)
    hf = []
    for pi, blocks in enumerate(pages_blocks):
        H = page_heights[pi]
        for b in blocks:
            top = b["bbox"][3] < 0.085 * H
            bottom = b["bbox"][1] > 0.915 * H
            if b["role"] == "figure_text":
                continue
            if not b["horizontal"]:
                W = page_widths[pi] if page_widths else 612
                if b["bbox"][2] < 0.12 * W or b["bbox"][0] > 0.88 * W:
                    b["role"], b["role_rule"] = "margin_note", "rotated text in outer page margin (e.g. arXiv stamp)"
                else:
                    b["role"], b["role_rule"] = "rotated_text", "rotated text inside the page area, not near a detected figure"
                hf.append(b)
                continue
            if not (top or bottom):
                W = page_widths[pi] if page_widths else 612
                # Margin line numbers (e.g. preprint / review drafts: 001, 002, ... in outer margin)
                if b["bbox"][2] < 0.11 * W or b["bbox"][0] > 0.89 * W:
                    toks = b["text"].split()
                    if toks and all(re.match(r"^\d{1,4}$", t) for t in toks):
                        b["role"], b["role_rule"] = "margin_note", "margin line number"
                        hf.append(b)
                        continue
                # Horizontal watermark text (e.g. DRAFT / CONFIDENTIAL / UNDER REVIEW)
                if re.match(r"^\s*(?:draft|confidential|preprint|under\s+review|do\s+not\s+distribute|watermark)\s*$", b["text"], re.I):
                    if body_size is None or b["size"] >= 1.3 * body_size:
                        b["role"], b["role_rule"] = "margin_note", "watermark text"
                        hf.append(b)
                        continue
                continue
            k = norm_key(b["text"])
            repeated = len(cand[k]) >= 2 and len(cand[k]) >= 0.3 * n_pages
            pagenum = bool(RE_PAGE_NUM.match(b["text"]))
            small_short = body_size is not None and b["size"] < 0.9 * body_size and len(b["text"]) < 100
            if repeated or pagenum or (len(b["text"]) < 90 and len(cand[k]) >= 2) or small_short:
                b["role"] = "header" if top else "footer"
                if pagenum:
                    b["role_rule"] = "page number"
                elif repeated or len(cand[k]) >= 2:
                    b["role_rule"] = f"repeats on {len(cand[k])} pages in margin band"
                else:
                    b["role_rule"] = "short small-font line in margin band (not repeated; check)"
                hf.append(b)
    return hf


def find_gutters(blocks, W):
    """Return x positions of column gutters from narrow body blocks."""
    narrow = [b for b in blocks if (b["bbox"][2] - b["bbox"][0]) < 0.58 * W and len(b["text"]) > 15]
    if len(narrow) < 3:
        return []
    nb = 200
    occ = [0.0] * nb
    for b in narrow:
        i0 = max(0, int(b["bbox"][0] / W * nb))
        i1 = min(nb - 1, int(b["bbox"][2] / W * nb))
        h = b["bbox"][3] - b["bbox"][1]
        for i in range(i0, i1 + 1):
            occ[i] += h
    xs = [b["bbox"][0] for b in narrow]
    xe = [b["bbox"][2] for b in narrow]
    left_edge, right_edge = min(xs), max(xe)
    # A gutter may be crossed by a few centred blocks (author lines, e-mail,
    # short headings), so "empty" means: occupied by < 12% of the typical
    # column occupancy, not strictly zero.
    mid = [o for o in occ[int(0.2 * nb):int(0.8 * nb)] if o > 0]
    typical = sorted(mid)[int(0.75 * (len(mid) - 1))] if mid else 0
    low = max(1e-6, 0.2 * typical)
    total_h = sum(b["bbox"][3] - b["bbox"][1] for b in narrow)
    gutters = []
    i = int(0.2 * nb)
    while i < int(0.8 * nb):
        if occ[i] <= low:
            j = i
            while j < nb and occ[j] <= low:
                j += 1
            # the gutter is the widest stretch of minimum occupancy inside the low run
            m = min(occ[i:j])
            best, k = (i, i + 1), i
            while k < j:
                if occ[k] <= m + 0.02 * typical:
                    e = k
                    while e < j and occ[e] <= m + 0.02 * typical:
                        e += 1
                    if e - k > best[1] - best[0]:
                        best = (k, e)
                    k = e
                else:
                    k += 1
            g = (best[0] + best[1]) / 2 / nb * W
            left_h = sum(b["bbox"][3] - b["bbox"][1] for b in narrow if b["bbox"][2] <= g)
            right_h = sum(b["bbox"][3] - b["bbox"][1] for b in narrow if b["bbox"][0] >= g)
            def side_ok(side):
                # a real column: some height, and either several blocks or one column-wide block
                bs = [b for b in narrow if (b["bbox"][2] <= g if side == "L" else b["bbox"][0] >= g)]
                h = sum(b["bbox"][3] - b["bbox"][1] for b in bs)
                return h > 30 and any(len(b["text"]) >= 80 for b in bs) and \
                    (len(bs) >= 2 or any(b["bbox"][2] - b["bbox"][0] >= 0.33 * W for b in bs))
            if left_edge < g < right_edge and side_ok("L") and side_ok("R") and (j - i) >= 2:
                gutters.append(round(g, 1))
            i = j
        else:
            i += 1
    return gutters


OUT_OF_FLOW = ("header", "footer", "margin_note", "figure_text", "rotated_text")


def order_page(blocks, W):
    """Assign column + reading order. Full-width blocks cut the page into
    horizontal bands; inside a band, columns are read left to right."""
    body = [b for b in blocks if b["role"] not in OUT_OF_FLOW]
    gutters = find_gutters([b for b in body if b["role"] != "table"], W)
    tol = 3.0
    spanning, cols = [], []
    for b in body:
        x0, x1 = b["bbox"][0], b["bbox"][2]
        if any(x0 < g - tol and x1 > g + tol for g in gutters):
            b["column"] = "span"
            spanning.append(b)
        else:
            cx = (x0 + x1) / 2
            b["column"] = sum(1 for g in gutters if cx > g)
            cols.append(b)
    spanning.sort(key=lambda b: b["bbox"][1])
    ordered = []
    remaining = sorted(cols, key=lambda b: (b["column"], b["bbox"][1], b["bbox"][0]))
    for s in spanning:
        before = [b for b in remaining if (b["bbox"][1] + b["bbox"][3]) / 2 < s["bbox"][1] + 1]
        ordered.extend(sorted(before, key=lambda b: (b["column"], b["bbox"][1], b["bbox"][0])))
        remaining = [b for b in remaining if b not in before]
        ordered.append(s)
    ordered.extend(sorted(remaining, key=lambda b: (b["column"], b["bbox"][1], b["bbox"][0])))
    margin = [b for b in blocks if b["role"] in OUT_OF_FLOW]
    return ordered, margin, gutters


# ----------------------------------------------------------------- tables
def ruling_segments(page):
    """Horizontal / vertical rules from vector drawings (lines and thin rects)."""
    hs, vs = [], []
    try:
        drawings = page.get_drawings()
    except Exception:
        return hs, vs
    for d in drawings:
        for it in d.get("items", []):
            if it[0] == "l":
                p, q = it[1], it[2]
                if abs(p.y - q.y) < 1.0 and abs(p.x - q.x) > 30:
                    hs.append((min(p.x, q.x), max(p.x, q.x), (p.y + q.y) / 2))
                elif abs(p.x - q.x) < 1.0 and abs(p.y - q.y) > 6:
                    vs.append((min(p.y, q.y), max(p.y, q.y), (p.x + q.x) / 2))
            elif it[0] == "re":
                r = it[1]
                if r.height < 2.5 and r.width > 30:
                    hs.append((r.x0, r.x1, (r.y0 + r.y1) / 2))
                elif r.width < 2.5 and r.height > 6:
                    vs.append((r.y0, r.y1, (r.x0 + r.x1) / 2))
    # merge collinear horizontal pieces
    hs.sort(key=lambda s: (round(s[2]), s[0]))
    merged = []
    for s in hs:
        if merged and abs(merged[-1][2] - s[2]) < 1.0 and s[0] <= merged[-1][1] + 2:
            m = merged[-1]
            merged[-1] = (m[0], max(m[1], s[1]), m[2])
        else:
            merged.append(s)
    return merged, vs


def column_intervals(words, x0, x1, min_gap=7.0):
    """Infer column intervals from whitespace gaps common to all rows."""
    if not words:
        return [(x0, x1)]
    n = int(x1 - x0) + 2
    occ = [0] * n
    for w in words:
        a = max(0, int(w[0] - x0))
        b = min(n - 1, int(w[2] - x0))
        for i in range(a, b + 1):
            occ[i] += 1
    cols, start, i = [], None, 0
    gap_start = None
    for i in range(n):
        if occ[i] > 0:
            if start is None:
                start = i
            elif gap_start is not None and i - gap_start >= min_gap:
                cols.append((x0 + start, x0 + gap_start))
                start = i
            gap_start = None
        else:
            if start is not None and gap_start is None:
                gap_start = i
    if start is not None:
        end = gap_start if gap_start is not None else n - 1
        cols.append((x0 + start, x0 + end))
    return cols or [(x0, x1)]


def group_rows(words, tol=2.5):
    rows = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy = (w[1] + w[3]) / 2
        if rows and abs(rows[-1]["cy"] - cy) <= tol:
            rows[-1]["words"].append(w)
            rows[-1]["cy"] = (rows[-1]["cy"] * (len(rows[-1]["words"]) - 1) + cy) / len(rows[-1]["words"])
        else:
            rows.append({"cy": cy, "words": [w]})
    for r in rows:
        r["words"].sort(key=lambda w: w[0])
    return rows


def phrases(words, max_gap=4.5):
    out = []
    for w in words:
        if out and w[0] - out[-1]["x1"] <= max_gap:
            out[-1]["text"] += " " + w[4]
            out[-1]["x1"] = w[2]
        else:
            out.append({"x0": w[0], "x1": w[2], "text": w[4]})
    return out


RE_NUMTOK = re.compile(r"^[\(\[]?[-−+±~<>≈]?\$?\d[\d,\.]*(?:[eE][-+]?\d+)?[%×x]?[\)\]\*†‡]*(?:\^\S+)?$|^[-–—]$|^n/?a$", re.I)
RE_SCI_DISPLAY = re.compile(r"^[-−]?(?:\d[\d,\.]*\s*[·×x*∗]\s*)?10\^[-−]?\d+$")
RE_CITATION = re.compile(r"^\[\d+(?:\s*[,–-]\s*\d+)*\]$")
SCI_OPS = {"·", "×", "x", "*", "∗", "⋅"}
BIND_OPS = {"·", "×", "⋅", "∗", "+", "−", "=", "/", "±"}
MINUS_SIGNS = {"-", "−", "–"}
SUPERSCRIPT_DIGITS = str.maketrans("0123456789-−+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻⁺")


def is_num(tok):
    t = tok.strip()
    if RE_CITATION.match(t):
        return False  # "[38]" is a citation, not a number
    return bool(RE_NUMTOK.match(t) or RE_SCI_DISPLAY.match(t))


def _plain_number(t):
    return re.fullmatch(r"[-−]?\d[\d,]*(?:\.\d+)?", t) is not None


def _to_float(t):
    try:
        return float(t.replace("−", "-").replace(",", ""))
    except ValueError:
        return None


def table_tokens(page, clip=None):
    """Word-like tokens for table analysis, built from characters.

    Unlike page.get_text("words"), a word breaks at a change of script level,
    so "10" and a raised "19" stay apart, and digits split across fonts
    ("2" "." "3" in CMR/CMMI) are joined. Then, within a line:
      mantissa  op(·×)  10  ^exp   -> one scientific-notation token
      10 ^exp                      -> one token (no mantissa)
      number ^mark (not after 10)  -> number + footnote marker (not an exponent)
      minus  number                -> signed number
    Each token is a tuple (x0, y0, x1, y1, display_text, meta) where meta keeps
    the original characters with their boxes ("parts") and, for numbers, a
    normalised form ("norm").
    """
    flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP
    d = page.get_text("rawdict", flags=flags, clip=clip)
    out = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            spans = [s for s in l.get("spans", []) if s.get("chars")]
            if not spans:
                continue
            dirv = l.get("dir", (1, 0))
            if abs(dirv[0]) < 0.99:
                continue
            sizes = Counter()
            for s in spans:
                sizes[round(s["size"], 1)] += len(s["chars"])
            base = sizes.most_common(1)[0][0]
            dom = [s["origin"][1] for s in spans if round(s["size"], 1) == base]
            baseline = statistics.median(dom) if dom else spans[0]["origin"][1]
            words, cur, prev_x1 = [], None, None
            for s in spans:
                small = s["size"] < 0.85 * base
                dy = s["origin"][1] - baseline
                script = "sup" if (s.get("flags", 0) & 1 or (small and dy < -0.12 * base)) else \
                         ("sub" if small and dy > 0.08 * base else "")
                for ch in s["chars"]:
                    c = ch["c"]
                    bb = ch["bbox"]
                    if c.isspace():
                        cur = None
                        prev_x1 = bb[2]
                        continue
                    gap = None if prev_x1 is None else bb[0] - prev_x1
                    if cur is None or cur["script"] != script or (gap is not None and gap > 0.25 * s["size"]):
                        cur = {"chars": [], "script": script, "size": s["size"], "font": s.get("font", ""),
                               "x0": bb[0], "y0": bb[1], "x1": bb[2], "y1": bb[3]}
                        words.append(cur)
                    cur["chars"].append((c, [round(v, 2) for v in bb]))
                    cur["x0"], cur["y0"] = min(cur["x0"], bb[0]), min(cur["y0"], bb[1])
                    cur["x1"], cur["y1"] = max(cur["x1"], bb[2]), max(cur["y1"], bb[3])
                    prev_x1 = bb[2]
            for w in words:
                w["text"] = "".join(c for c, _ in w["chars"])
            words.sort(key=lambda w: w["x0"])
            lt = []
            i = 0
            while i < len(words):
                w = words[i]
                nxt = words[i + 1:i + 4]
                close = lambda a, b, k=0.6: b["x0"] - a["x1"] <= k * max(a["size"], b["size"])
                group, norm = [w], None
                if (w["script"] == "" and _plain_number(w["text"]) and len(nxt) >= 3 and nxt[0]["text"] in SCI_OPS
                        and nxt[1]["text"] == "10" and nxt[2]["script"] == "sup"
                        and re.fullmatch(r"[-−]?\d+", nxt[2]["text"])
                        and close(w, nxt[0]) and close(nxt[0], nxt[1]) and close(nxt[1], nxt[2], 0.3)):
                    group = [w] + nxt[:3]
                    m, op, e = w["text"], nxt[0]["text"], nxt[2]["text"]
                    norm = {"kind": "scientific", "mantissa": m, "operator": op, "base": "10", "exponent": e}
                elif (w["script"] == "" and w["text"] == "10" and nxt and nxt[0]["script"] == "sup"
                      and re.fullmatch(r"[-−]?\d+", nxt[0]["text"]) and close(w, nxt[0], 0.3)):
                    group = [w, nxt[0]]
                    norm = {"kind": "scientific", "mantissa": None, "operator": None, "base": "10", "exponent": nxt[0]["text"]}
                elif w["text"] in MINUS_SIGNS and w["script"] == "" and nxt and _plain_number(nxt[0]["text"]) \
                        and nxt[0]["x0"] - w["x1"] <= 0.4 * w["size"]:
                    group = [w, nxt[0]]
                    norm = {"kind": "number", "text": "−" + nxt[0]["text"].lstrip("-−")}
                elif w["script"] == "" and nxt and nxt[0]["script"] == "sup" and w["text"] != "10" \
                        and close(w, nxt[0], 0.3) and len(nxt[0]["text"]) <= 3:
                    group = [w, nxt[0]]
                    norm = {"kind": "footnoted", "text": w["text"], "footnote": nxt[0]["text"]}
                elif w["script"] == "" and _plain_number(w["text"]):
                    norm = {"kind": "number", "text": w["text"]}
                if norm and norm["kind"] == "scientific":
                    mant = norm["mantissa"]
                    exp = norm["exponent"].replace("−", "-")
                    disp = (f"{mant} {norm['operator']} 10^{norm['exponent']}" if mant else f"10^{norm['exponent']}")
                    mv = _to_float(mant) if mant else 1.0
                    norm["value"] = mv * 10 ** int(exp) if mv is not None else None
                    norm["canonical"] = f"{mant or '1'}e{exp}"
                    norm["pretty"] = (f"{mant} × " if mant else "") + "10" + exp.translate(SUPERSCRIPT_DIGITS)
                elif norm and norm["kind"] == "footnoted":
                    disp = f"{norm['text']}^{norm['footnote']}"
                    norm["value"] = _to_float(norm["text"]) if _plain_number(norm["text"]) else None
                elif norm and norm["kind"] == "number":
                    disp = norm["text"]
                    norm["value"] = _to_float(norm["text"])
                else:
                    disp = w["text"]
                parts = [{"text": g["text"], "script": g["script"], "bbox": [round(g["x0"], 2), round(g["y0"], 2),
                                                                             round(g["x1"], 2), round(g["y1"], 2)]}
                         for g in group]
                raw = " ".join(g["text"] if not g["script"] else f"{g['script']}:{g['text']}" for g in group)
                lt.append((min(g["x0"] for g in group), min(g["y0"] for g in group),
                           max(g["x1"] for g in group), max(g["y1"] for g in group),
                           disp, {"parts": parts, "raw": raw, "norm": norm, "size": w["size"]}))
                i += len(group)
            # a binary operator with close neighbours on both sides binds them into one expression
            # ("O(n^2 · d)", "85.6 ± 0.3"), so a column can never be split at an operator
            k = 1
            while 0 < k < len(lt) - 1:
                a_, op, b_ = lt[k - 1], lt[k], lt[k + 1]
                lim = 0.6 * op[5]["size"]
                if op[4] in BIND_OPS and op[0] - a_[2] <= lim and b_[0] - op[2] <= lim:
                    merged = (a_[0], min(a_[1], op[1], b_[1]), b_[2], max(a_[3], op[3], b_[3]),
                              f"{a_[4]} {op[4]} {b_[4]}",
                              {"parts": a_[5]["parts"] + op[5]["parts"] + b_[5]["parts"],
                               "raw": f"{a_[5]['raw']} {op[5]['raw']} {b_[5]['raw']}", "norm": None, "size": op[5]["size"]})
                    lt[k - 1:k + 2] = [merged]
                else:
                    k += 1
            out.extend(lt)
    return out


def row_gaps(r):
    ws = r["words"]
    return [b[0] - a[2] for a, b in zip(ws, ws[1:])]


def line_is_prose(r, width):
    """A running-text line: many words, no wide gaps, mostly not numbers."""
    ws = r["words"]
    if len(ws) < 6:
        return False
    gaps = row_gaps(r)
    nums = sum(1 for w in ws if is_num(w[4]))
    span = ws[-1][2] - ws[0][0]
    return max(gaps) < 6.5 and span > 0.55 * width and nums < 0.3 * len(ws)


def line_is_tablelike(r):
    ws = r["words"]
    nums = sum(1 for w in ws if is_num(w[4]))
    gaps = row_gaps(r)
    return nums >= 3 or (len(ws) >= 2 and max(gaps) >= 8) or (nums >= 2 and nums >= 0.4 * len(ws))


def band_rows(words, x0, x1, y0, y1):
    tw = [w for w in words if x0 - 2 <= (w[0] + w[2]) / 2 <= x1 + 2 and y0 < (w[1] + w[3]) / 2 < y1]
    return group_rows(tw)


def align_columns(body_rows, x0, x1, tol=2.0):
    """Columns from token alignment across body rows (left, right or centre edges)."""
    nrows = len(body_rows)
    toks = []
    for ri, r in enumerate(body_rows):
        for wi, w in enumerate(r["words"]):
            toks.append((ri, wi, w))
    cands = []
    for anchor, fn in (("left", lambda w: w[0]), ("right", lambda w: w[2]), ("center", lambda w: (w[0] + w[2]) / 2)):
        vals = sorted(toks, key=lambda t: fn(t[2]))
        cur = []
        for t in vals:
            if cur and fn(t[2]) - fn(cur[-1][2]) > tol:
                cands.append((anchor, cur))
                cur = []
            cur.append(t)
        if cur:
            cands.append((anchor, cur))
    scored = []
    for anchor, members in cands:
        rows = {t[0] for t in members}
        if len(members) != len(rows):  # two tokens of one row in one cluster: not a column
            members = list({t[0]: t for t in members}.values())
        scored.append((len(rows), anchor, members))
    scored.sort(key=lambda s: -s[0])
    claimed = set()
    cols = []
    thresh = max(2, 0.3 * nrows)
    for support, anchor, members in scored:
        new = [t for t in members if (t[0], t[1]) not in claimed]
        if len({t[0] for t in new}) < thresh or len(new) < 0.6 * len(members):
            continue
        for t in new:
            claimed.add((t[0], t[1]))
        cols.append({"anchor": anchor, "x0": min(t[2][0] for t in new), "x1": max(t[2][2] for t in new),
                     "support": len({t[0] for t in new})})
    cols.sort(key=lambda c: c["x0"])
    # merge columns whose spans overlap heavily (same column, mixed alignment)
    merged = []
    for c in cols:
        if merged:
            p = merged[-1]
            ov = min(p["x1"], c["x1"]) - max(p["x0"], c["x0"])
            if ov > 0.5 * min(p["x1"] - p["x0"], c["x1"] - c["x0"]):
                p["x0"], p["x1"] = min(p["x0"], c["x0"]), max(p["x1"], c["x1"])
                p["support"] = max(p["support"], c["support"])
                continue
        merged.append(dict(c))
    # leading label column for unclaimed tokens left of the first column
    lead = [t for t in toks if (t[0], t[1]) not in claimed and merged and t[2][2] <= merged[0]["x0"] + 1]
    if lead and len({t[0] for t in lead}) >= thresh:
        merged.insert(0, {"anchor": "left", "x0": min(t[2][0] for t in lead), "x1": max(t[2][2] for t in lead),
                          "support": len({t[0] for t in lead})})
    return merged


def assign_col(cols, x0, x1):
    """Column for a token: the column whose span contains it, else the nearest column to its left."""
    cx = (x0 + x1) / 2
    for i, c in enumerate(cols):
        if c["x0"] - 1.5 <= cx <= c["x1"] + 1.5:
            return i
    left = [i for i, c in enumerate(cols) if c["x1"] <= x0 + 1.5]
    if left:
        return left[-1]
    return min(range(len(cols)), key=lambda i: abs((cols[i]["x0"] + cols[i]["x1"]) / 2 - cx))


def covered_cols(cols, x0, x1):
    """Columns a body token belongs to: one column if its centre lies in that
    column; several if it sits between columns and overlaps each of them."""
    if not cols:
        return [0]
    cx = (x0 + x1) / 2
    for i, c in enumerate(cols):
        if c["x0"] - 1.5 <= cx <= c["x1"] + 1.5:
            return [i]
    w = max(1e-6, x1 - x0)
    cov = [i for i, c in enumerate(cols) if min(c["x1"], x1) - max(c["x0"], x0) > 0.08 * w]
    if len(cov) >= 2:
        return list(range(cov[0], cov[-1] + 1))
    return [assign_col(cols, x0, x1)]


def assign_header_col(cols, x0, x1):
    """Header tokens are often centred over their column: use midpoints between columns."""
    cx = (x0 + x1) / 2
    for i, c in enumerate(cols):
        lo = (cols[i - 1]["x1"] + c["x0"]) / 2 if i > 0 else -1e9
        hi = (c["x1"] + cols[i + 1]["x0"]) / 2 if i + 1 < len(cols) else 1e9
        if lo <= cx < hi:
            return i
    return len(cols) - 1


def merge_wordspace_columns(cols, body):
    """Two neighbouring columns separated by no more than a word space in every row
    that fills both are one column (right-aligned multi-word labels "400 Million",
    values with a bracketed interval "0.49 (0.462, 0.534)")."""
    changed = True
    while changed and len(cols) > 1:
        changed = False
        for i in range(len(cols) - 1):
            gaps, sizes = [], []
            for r in body:
                a_ = [w for w in r["words"] if assign_col(cols, w[0], w[2]) == i]
                b_ = [w for w in r["words"] if assign_col(cols, w[0], w[2]) == i + 1]
                if a_ and b_:
                    la = max(a_, key=lambda w: w[2])
                    fb = min(b_, key=lambda w: w[0])
                    if is_num(la[4]) and is_num(fb[4]):
                        gaps.append(1e9)  # two numbers side by side are two values, however close
                        continue
                    gaps.append(fb[0] - la[2])
                    sizes += [w[5]["size"] for w in a_ + b_ if len(w) > 5 and isinstance(w[5], dict) and w[5].get("size")]
            if len(gaps) >= 2 and sizes:
                em = statistics.median(sizes)
                if max(gaps) < 0.45 * em:
                    cols[i] = {"anchor": cols[i]["anchor"], "x0": min(cols[i]["x0"], cols[i + 1]["x0"]),
                               "x1": max(cols[i]["x1"], cols[i + 1]["x1"]), "support": max(cols[i]["support"], cols[i + 1]["support"])}
                    del cols[i + 1]
                    changed = True
                    break
    return cols


def merge_label_columns(cols, head, body):
    """A column without any header word whose values are all text continues the
    text column on its left (multi-word row labels such as "GPT-3 Few-Shot")."""
    if len(cols) < 3:
        return cols
    has_head = [False] * len(cols)
    for r in head:
        for w in r["words"]:
            has_head[assign_header_col(cols, w[0], w[2])] = True
    vals = [[] for _ in cols]
    for r in body:
        per = defaultdict(list)
        for w in sorted(r["words"], key=lambda w: w[0]):
            per[assign_col(cols, w[0], w[2])].append(w[4])
        for ci, ws in per.items():
            vals[ci].append(" ".join(ws))  # judge whole cell texts, not word fragments ("[STQ+ 19]")
    textual = [bool(v) and sum(1 for x in v if is_num(x)) <= 0.1 * len(v) for v in vals]
    out = [dict(cols[0])]
    out_text = [textual[0]]
    for i in range(1, len(cols)):
        if not has_head[i] and textual[i] and out_text[-1]:
            out[-1]["x1"] = max(out[-1]["x1"], cols[i]["x1"])
            continue
        out.append(dict(cols[i]))
        out_text.append(textual[i])
    return out


def detect_ruled_tables(page, page_no):
    """Tables delimited by horizontal rules (booktabs or full grids).

    Rules are chained into one table while the band between them contains no
    prose line or caption; columns come from vertical rules when present,
    otherwise from token alignment in the body rows; header rows (above the
    header rule) are mapped onto the leaf columns, and short rules under a
    header phrase (cmidrules) turn it into a spanning group header.
    """
    hs, vs = ruling_segments(page)
    if len(hs) < 2:
        return []
    words = table_tokens(page)
    W = page.rect.width
    groups = []
    for s in sorted(hs, key=lambda s: (s[2], s[0])):
        placed = False
        for g in reversed(groups):
            gx0 = min(r[0] for r in g)
            gx1 = max(r[1] for r in g)
            ov = min(gx1, s[1]) - max(gx0, s[0])
            if ov <= 0.6 * min(gx1 - gx0, s[1] - s[0]):
                continue
            last_y = max(r[2] for r in g)
            if s[2] - last_y > 2:
                rows = band_rows(words, max(gx0, s[0]), min(gx1, s[1]), last_y, s[2])
                width = min(gx1, s[1]) - max(gx0, s[0])
                if any(line_is_prose(r, width) or RE_CAPTION.match(" ".join(w[4] for w in r["words"])) for r in rows):
                    break
            g.append(s)
            placed = True
            break
        if not placed:
            groups.append([s])
    tables = []
    for g in groups:
        maxlen = max(s[1] - s[0] for s in g)
        main = sorted([s for s in g if s[1] - s[0] >= 0.8 * maxlen], key=lambda s: s[2])
        partial = [s for s in g if s[1] - s[0] < 0.8 * maxlen]
        if not main:
            continue
        x0 = min(s[0] for s in main)
        x1 = max(s[1] for s in main)
        y0, y1 = main[0][2], main[-1][2]
        header_rule = main[1][2] if len(main) >= 3 else None
        extended = False
        # no bottom rule: extend downward while lines stay table-like
        if len(main) <= 2:
            below = band_rows(words, x0, x1, y1, page.rect.height)
            last = y1
            pitch = None
            taken = 0
            for r in below:
                text = " ".join(w[4] for w in r["words"])
                top = min(w[1] for w in r["words"])
                gap = top - last
                if (not line_is_tablelike(r)) or line_is_prose(r, x1 - x0) or RE_CAPTION.match(text):
                    break
                if pitch is not None and gap > 2.5 * pitch + 4:
                    break
                pitch = max(4.0, gap) if pitch is None else pitch
                last = max(w[3] for w in r["words"])
                taken += 1
            if taken >= 2:
                if len(main) == 2 and header_rule is None:
                    header_rule = y1
                y1 = last + 1
                extended = True
        if y1 - y0 < 6:
            continue
        rows = band_rows(words, x0, x1, y0 - 1, y1 + 1)
        if len(rows) < 2:
            continue
        if header_rule is not None:
            head = [r for r in rows if r["cy"] < header_rule]
            body = [r for r in rows if r["cy"] >= header_rule]
        else:
            k = 0
            while k < len(rows) and sum(1 for w in rows[k]["words"] if is_num(w[4])) < 0.5 * len(rows[k]["words"]):
                k += 1
            k = max(1, min(k, len(rows) - 1))
            head, body = rows[:k], rows[k:]
        if not body or sum(1 for r in body if line_is_tablelike(r)) < max(1, 0.5 * len(body)):
            continue
        if sum(1 for r in body if line_is_prose(r, x1 - x0)) > 0.2 * len(body):
            continue  # boxed running text (e.g. prompt examples), not a table
        near = band_rows(words, x0, x1, y0 - 40, y0 - 1) + band_rows(words, x0, x1, y1 + 1, y1 + 40)
        near_caps = [" ".join(w[4] for w in r["words"]) for r in near]
        if any(re.match(r"^\s*(Fig\.?|Figure|FIGURE)\s*[A-Z]?\.?\d", t) for t in near_caps) and \
                not any(re.match(r"^\s*(Table|TABLE|Tab\.)\s*[A-Z]?\.?\d", t) for t in near_caps):
            continue  # ruled box captioned as a figure
        inner_v = sorted({round(v[2], 1) for v in vs if v[0] < y1 and v[1] > y0 and x0 - 2 <= v[2] <= x1 + 2})
        if len(inner_v) >= 2:
            edges = inner_v
            if edges[0] - x0 > 5:
                edges = [x0] + edges
            if x1 - edges[-1] > 5:
                edges = edges + [x1]
            cols = [{"anchor": "rules", "x0": a, "x1": b, "support": len(body)} for a, b in zip(edges, edges[1:])]
            col_rule = "vertical rules"
        else:
            cols = align_columns(body, x0, x1)
            col_rule = "token alignment across body rows"
            # sparse columns (few filled rows) survive only if a header sits over them
            inside = lambda cx: any(c["x0"] - 2 <= cx <= c["x1"] + 2 for c in cols)
            under_rule = lambda w, r: any(p[0] - 2 <= (w[0] + w[2]) / 2 <= p[1] + 2 and 0 < p[2] - r["cy"] < 14 for p in partial)
            hw = sorted([w for r in head for w in r["words"] if not inside((w[0] + w[2]) / 2) and not under_rule(w, r)],
                        key=lambda w: w[0])
            spans = []
            for w in hw:
                if spans and w[0] <= spans[-1][1] + 3:
                    spans[-1][1] = max(spans[-1][1], w[2])
                else:
                    spans.append([w[0], w[2]])
            for a, b in spans:
                orphans = [w for r in body for w in r["words"] if a - 3 <= (w[0] + w[2]) / 2 <= b + 3
                           and not inside((w[0] + w[2]) / 2) and len(covered_cols(cols, w[0], w[2])) < 2]
                if orphans:
                    cols.append({"anchor": "header", "x0": min(min(w[0] for w in orphans), a), "x1": max(max(w[2] for w in orphans), b),
                                 "support": len(orphans)})
            cols.sort(key=lambda c: c["x0"])
            cols = merge_wordspace_columns(cols, body)
            cols = merge_label_columns(cols, head, body)
        if len(cols) < 2:
            continue
        ncol = len(cols)
        grid, cells, records = [], [], []
        # ---- header rows
        header_cells = []  # per header row: list of (c0, c1, text)
        for ri, r in enumerate(head):
            slots = {}
            spans = []
            for w in r["words"]:
                wcx = (w[0] + w[2]) / 2
                pr = next((p for p in partial if p[0] - 2 <= wcx <= p[1] + 2 and 0 < p[2] - r["cy"] < 14), None)
                if pr is not None:
                    under = [i for i, c in enumerate(cols) if c["x0"] < pr[1] - 1 and c["x1"] > pr[0] + 1]
                    if under:
                        key = (under[0], under[-1])
                        slots.setdefault(key, []).append(w)
                        continue
                ci = assign_header_col(cols, w[0], w[2])
                slots.setdefault((ci, ci), []).append(w)
            row_cells = []
            for (c0, c1), ws in sorted(slots.items()):
                ws.sort(key=lambda w: w[0])
                row_cells.append((c0, c1, " ".join(w[4] for w in ws),
                                  [round(min(w[0] for w in ws), 2), round(min(w[1] for w in ws), 2),
                                   round(max(w[2] for w in ws), 2), round(max(w[3] for w in ws), 2)]))
            header_cells.append(row_cells)
        for ri, row_cells in enumerate(header_cells):
            flat = [""] * ncol
            for c0, c1, text, bb in row_cells:
                if flat[c0] not in ("", None):
                    flat[c0] = flat[c0] + " " + text
                else:
                    flat[c0] = text
                for k in range(c0 + 1, c1 + 1):
                    flat[k] = None
                cells.append({"row": ri, "col": c0, "text": text, "colspan": c1 - c0 + 1, "rowspan": 1, "bbox": bb, "header": True})
            grid.append(flat)
        paths = []
        for ci in range(ncol):
            path = []
            for row_cells in header_cells:
                for c0, c1, text, bb in row_cells:
                    if c0 <= ci <= c1 and text and text not in path:
                        path.append(text)
            paths.append(path)
        # ---- body rows
        nh = len(head)
        n_spanning = 0
        for bi, r in enumerate(body):
            slots = defaultdict(list)
            for w in r["words"]:
                cov = covered_cols(cols, w[0], w[2])
                slots[(cov[0], cov[-1])].append(w)
            # a spanning token wins over single-column slots it covers only if those are empty
            for (c0, c1) in [k for k in slots if k[1] > k[0]]:
                if any(k != (c0, c1) and c0 <= k[0] <= c1 for k in slots):
                    ws = slots.pop((c0, c1))
                    for w in ws:
                        slots[(assign_col(cols, w[0], w[2]),) * 2].append(w)
            flat = [""] * ncol
            for (c0, c1), ws in sorted(slots.items()):
                ws.sort(key=lambda w: w[0])
                text = " ".join(w[4] for w in ws)
                flat[c0] = text
                for k in range(c0 + 1, c1 + 1):
                    flat[k] = None
                metas = [w[5] for w in ws if len(w) > 5]
                cell = {"row": nh + bi, "col": c0, "text": text, "colspan": c1 - c0 + 1, "rowspan": 1,
                        "bbox": [round(min(w[0] for w in ws), 2), round(min(w[1] for w in ws), 2),
                                 round(max(w[2] for w in ws), 2), round(max(w[3] for w in ws), 2)],
                        "raw": " | ".join(m["raw"] for m in metas),
                        "parts": [p for m in metas for p in m["parts"]]}
                if len(metas) == 1 and metas[0].get("norm"):
                    cell["norm"] = metas[0]["norm"]
                if c1 > c0:
                    n_spanning += 1
                    cell["span_rule"] = "token centred between columns and overlapping each of them"
                cells.append(cell)
            grid.append(flat)
        # key (row-label) columns: leading columns whose body values are mostly non-numeric
        key_cols = []
        for ci in range(ncol):
            vals = [grid[nh + bi][ci] for bi in range(len(body)) if grid[nh + bi][ci]]
            if vals and sum(1 for v in vals if is_num(v.split()[0])) < 0.3 * len(vals):
                key_cols.append(ci)
            else:
                break
        names = []
        for ci in range(ncol):
            n = " / ".join(paths[ci]) or f"col{ci}"
            while n in names:
                n += "'"
            names.append(n)
        for bi in range(len(body)):
            rec = {"_row": nh + bi}
            for ci in range(ncol):
                v = grid[nh + bi][ci]
                if v is None:  # covered by a cell spanning from the left
                    k = ci
                    while k > 0 and grid[nh + bi][k] is None:
                        k -= 1
                    v = grid[nh + bi][k]
                    rec.setdefault("_shared", {})[names[ci]] = names[k]
                rec[names[ci]] = v
            records.append(rec)
        tables.append({
            "id": None, "page": page_no, "bbox": [round(x0, 2), round(y0 - 2, 2), round(x1, 2), round(y1 + 2, 2)],
            "n_rows": len(grid), "n_cols": ncol, "rows": grid, "cells": cells,
            "header": {"n_header_rows": nh, "names": names},
            "columns": [{"index": ci, "path": paths[ci], "name": names[ci], "x0": round(cols[ci]["x0"], 2),
                         "x1": round(cols[ci]["x1"], 2), "align": cols[ci]["anchor"]} for ci in range(ncol)],
            "key_columns": key_cols,
            "records": records,
            "caption": None, "caption_block": None, "continued_from": None,
            "detector": f"ruling lines ({len(main)} full rules, {len(partial)} partial; columns from {col_rule}"
                        + ("; body extended below last rule" if extended else "") + ")",
            "span_inference": "header spans from short rules (cmidrules) under the header phrase; body spans "
                              "when a token is centred between columns and overlaps each; None marks covered cells",
            "n_spanning_body_cells": n_spanning,
        })
    return tables


def extract_tables(page, page_no, allow_fallback=True):
    out = detect_ruled_tables(page, page_no)
    if out or not allow_fallback:
        return out, None
    if not hasattr(page, "find_tables"):
        return out, None
    try:
        tabs = page.find_tables()
    except Exception as e:
        return out, f"find_tables failed: {e}"
    for ti, t in enumerate(tabs.tables if hasattr(tabs, "tables") else tabs):
        try:
            rows = t.extract()
        except Exception:
            continue
        if not rows or len(rows) < 2 or max(len(r) for r in rows) < 2:
            continue
        ncol = max(len(r) for r in rows)
        rows = [list(r) + [None] * (ncol - len(r)) for r in rows]
        nonempty = sum(1 for r in rows for c in r if c not in (None, ""))
        if nonempty < 4:
            continue
        cells = []
        for ri, r in enumerate(rows):
            for ci, c in enumerate(r):
                if c is None:
                    continue
                colspan = 1
                k = ci + 1
                while k < ncol and r[k] is None and (ri == 0 or rows[ri - 1][k] is not None or rows[ri - 1][ci] is None):
                    colspan += 1
                    k += 1
                rowspan = 1
                k = ri + 1
                while k < len(rows) and rows[k][ci] is None and colspan == 1 and ci == 0:
                    rowspan += 1
                    k += 1
                cells.append({"row": ri, "col": ci, "text": norm_ws(c), "colspan": colspan, "rowspan": rowspan})
        try:
            hdr = t.header
            header = {"names": [norm_ws(x) if x else "" for x in hdr.names], "external": bool(hdr.external)}
        except Exception:
            header = None
        out.append({
            "id": None,
            "page": page_no,
            "bbox": [round(v, 2) for v in t.bbox],
            "n_rows": len(rows),
            "n_cols": ncol,
            "rows": [[norm_ws(c) if c is not None else None for c in r] for r in rows],
            "cells": cells,
            "header": header,
            "caption": None,
            "caption_block": None,
            "continued_from": None,
            "span_inference": "None cells from PyMuPDF treated as merged-cell continuations (heuristic)",
        })
    return out, None


def table_to_markdown(t):
    rows = [[(c if c is not None else "") for c in r] for r in t["rows"]]
    if not rows:
        return ""
    w = len(rows[0])
    esc = lambda s: s.replace("|", "\\|")
    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |", "|" + "---|" * w]
    for r in rows[1:]:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)


def table_to_html(t):
    spans = {(c["row"], c["col"]): c for c in t["cells"]}
    n_header = (t.get("header") or {}).get("n_header_rows") or 1
    out = ["<table>"]
    for ri, row in enumerate(t["rows"]):
        out.append("  <tr>")
        tag = "th" if ri < n_header else "td"
        for ci, val in enumerate(row):
            if val is None:
                continue  # covered by a spanning cell
            c = spans.get((ri, ci), {"colspan": 1, "rowspan": 1})
            attrs = ""
            if c["colspan"] > 1:
                attrs += f' colspan="{c["colspan"]}"'
            if c["rowspan"] > 1:
                attrs += f' rowspan="{c["rowspan"]}"'
            out.append(f"    <{tag}{attrs}>{html.escape(val)}</{tag}>")
        out.append("  </tr>")
    out.append("</table>")
    return "\n".join(out)


def bbox_center_in(b, box, pad=2):
    cx = (b[0] + b[2]) / 2
    cy = (b[1] + b[3]) / 2
    return box[0] - pad <= cx <= box[2] + pad and box[1] - pad <= cy <= box[3] + pad


# ----------------------------------------------------------------- roles
def math_ratio(t):
    t = t.replace(" ", "")
    if not t:
        return 0
    return sum(1 for c in t if c in MATH_CHARS) / len(t)


def classify_roles(ordered, body_size, page_heights, first_page_no):
    """Assign title/author/abstract/heading/caption/equation/footnote/reference roles."""
    in_refs = False
    seen_abstract = False
    title_done = False
    # title: largest font on first page upper half
    first = [b for b in ordered if b["page"] == first_page_no and b["bbox"][1] < 0.5 * page_heights[first_page_no]]
    if first:
        maxsize = max(b["size"] for b in first)
        if maxsize >= body_size * 1.3:
            for b in first:
                if b["size"] >= maxsize - 0.3 and len(b["text"]) < 300:
                    b["role"], b["role_rule"] = "title", f"largest font on page 1 ({b['size']}pt)"
                    title_done = True
    for i, b in enumerate(ordered):
        if b["role"] in ("title", "table"):
            continue
        t = norm_ws(b["text"])
        H = page_heights[b["page"]]
        if RE_REFS_HEAD.match(t):
            b["role"], b["role_rule"] = "heading", "references heading"
            in_refs = True
            continue
        if in_refs:
            if heading_like(t) and (b["bold"] or b["size"] > body_size * 1.05) and not RE_REF_START_NUM.match(t):
                in_refs = False
            else:
                b["role"], b["role_rule"] = "reference", "after references heading"
                continue
        cm = RE_CAPTION.match(t)
        if cm:
            has_sep = bool(re.search(r"[\.:|：．]\s*$", cm.group(0))) and not re.match(r"^\s*(Fig|Tab)\.\s*\d+\s*$", cm.group(0))
            lead_bold = bool(b["lines"]) and b["lines"][0].get("lead_bold")
            smaller = b["size"] < body_size * 0.97
            if has_sep or lead_bold or b["bold"] or (smaller and "。" not in t[:200]):
                why = "separator" if has_sep else ("bold label" if (lead_bold or b["bold"]) else "smaller font")
                b["role"], b["role_rule"] = "caption", f"caption prefix (Figure/Table/图/表 N) + {why}"
                continue
        hl = heading_like(t)
        nvis = visual_lines(b["lines"]) if b.get("lines") else 1
        short = nvis <= 2 and len(t) < 130
        tiny = b["size"] < body_size * 0.9 and not b["bold"]
        if hl and short and not tiny and (b["bold"] or b["size"] > body_size * 1.05 or hl[0] in ("named", "roman", "cn", "appendix_word")
                                          or (hl[0] == "numeric" and not SENT_END.search(t))):
            b["role"], b["role_rule"] = "heading", f"{hl[0]} heading pattern"
            b["heading"] = {"kind": hl[0], "number": hl[1], "title": norm_ws(hl[2])}
            if re.match(r"^\s*abstract", t, re.I) or re.match(r"^\s*摘\s*要", t):
                seen_abstract = True
            continue
        # unnumbered heading set apart by style only (bold or larger, one printed line, own block):
        # "Experimental Setup", "RELATED WORK". Not on page 1 above the abstract (author names, affiliations).
        words = t.split()
        if (nvis == 1 and 1 <= len(words) <= 10 and 3 <= len(t) <= 80 and re.match(r"^[A-Z]", t)
                and not re.search(r"[\.,;:]$", t) and not re.search(r"\d{2,}|@|http|[=<>±×]", t)
                and (seen_abstract or b["page"] != first_page_no)
                and (b["bold"] and b["size"] >= body_size * 0.95 or b["size"] >= body_size * 1.15
                     or (t.isupper() and len(t) >= 6 and b["size"] >= body_size * 0.95 and b["bold"]))
                and sum(1 for w in words if w[:1].isupper() or w.lower() in STOP_IN_TITLE) >= 0.6 * len(words)
                and i + 1 < len(ordered) and ordered[i + 1]["page"] == b["page"]
                and ordered[i + 1]["role"] not in ("title", "table") and len(ordered[i + 1]["text"]) > 60
                and 0 <= ordered[i + 1]["bbox"][1] - b["bbox"][3] < 3 * b["size"]):
            b["role"], b["role_rule"] = "heading", "unnumbered heading: bold/larger single line followed by body text"
            b["heading"] = {"kind": "styled", "number": "", "title": t, "size": b["size"]}
            continue
        if re.match(r"^\s*(abstract|摘\s*要)\s*[:：\.—-]", t, re.I) or (not seen_abstract and re.match(r"^\s*abstract\b", t, re.I)):
            b["role"], b["role_rule"] = "abstract", "starts with 'Abstract'"
            seen_abstract = True
            continue
        if (b["size"] < body_size * 0.92 and b["bbox"][1] > 0.72 * H and len(t) < 600
                and (re.match(r"^\s*([\*†‡§¶]|\d{1,2}\s|[a-z]\s)", t) or (b.get("marked") or "").startswith("^{"))):
            b["role"], b["role_rule"] = "footnote", "small font, bottom of page, starts with marker"
            continue
        m = RE_EQ_NUM.search(t)
        mr = math_ratio(t)
        mathfont = any(re.search(r"math|cmmi|cmsy|cmex|symbol|stix", f, re.I) for f in b["fonts"])
        if len(t) < 260 and ((m and (mr > 0.12 or mathfont or "=" in t)) or (mr > 0.3 and len(t) > 3)):
            b["role"] = "equation"
            b["role_rule"] = "math-symbol density" + (" + trailing equation number" if m else "")
            b["equation_number"] = m.group(1) if m else None
            continue
    # authors / affiliations: blocks between title and abstract on first page
    if title_done:
        for b in ordered:
            if b["page"] != first_page_no:
                break
            if b["role"] in ("abstract",) or (b["role"] == "heading" and re.match(r"abstract|摘", b["text"], re.I)):
                break
            if b["role"] == "body" and b["bbox"][1] < 0.6 * page_heights[first_page_no]:
                prev_title = any(x["role"] == "title" and x["bbox"][3] <= b["bbox"][1] + 1 for x in ordered if x["page"] == first_page_no)
                if prev_title and len(b["text"]) < 400:
                    b["role"], b["role_rule"] = "author_affiliation", "between title and abstract on page 1"


STOP_IN_TITLE = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs", "vs.", "via", "with"}


def heading_level(h):
    if h["kind"] == "appendix_word":
        return 1
    if h["kind"] == "numeric":
        return h["number"].count(".") + 1
    if h["kind"] in ("roman", "named", "cn"):
        return 1
    if h["kind"] == "alpha":
        return 2
    if h["kind"] == "appendix":
        return h["number"].count(".") + 1
    return 1


def split_references(blocks):
    text = " ".join(norm_ws(b["text"]) for b in blocks)
    refs = []
    if re.search(r"\[\d{1,4}\]\s", text):
        parts = [p.strip() for p in RE_REF_SPLIT_BRACKET.split(text) if p.strip()]
    else:
        parts = []
        for b in blocks:
            cur = ""
            for ln in b["lines"]:
                t = ln["text"].strip()
                if RE_REF_START_NUM.match(t) and cur:
                    parts.append(cur)
                    cur = t
                else:
                    cur = (cur + " " + t).strip()
            if cur:
                parts.append(cur)
    for p in parts:
        # locate the block that contains the start of the entry
        key = norm_ws(p)[:40]
        src = next((b for b in blocks if key[:25] in norm_ws(b["text"])), blocks[0] if blocks else None)
        m = RE_REF_START_NUM.match(p)
        refs.append({
            "label": m.group(1).strip("[].") if m else None,
            "text": p,
            "page": src["page"] if src else None,
            "block_id": src["id"] if src else None,
        })
    return refs


# ----------------------------------------------------------------- main parse
def cache_key(sha, pages, ocr_mode, lang, preview):
    cfg = json.dumps({"v": PARSER_VERSION, "pages": pages, "ocr": ocr_mode, "lang": lang, "preview": bool(preview)},
                     sort_keys=True)
    return f"{sha[:24]}_{hashlib.sha256(cfg.encode()).hexdigest()[:12]}", cfg


def parse(path, out_dir, password=None, pages=None, ocr_mode="auto", lang="eng+chi_sim", preview=False, cache_dir=None, budget_deadline=None):
    """Parse a PDF into out_dir. With cache_dir, a previous result for the same
    PDF content + parser version + options is copied instead of re-parsing."""
    t0 = time.time()
    if isinstance(pages, str):
        pages = parse_pages(pages)
    if cache_dir and os.path.exists(path):
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        key, cfg = cache_key(sha, pages, ocr_mode, lang, preview)
        hit = os.path.join(cache_dir, key)
        if os.path.exists(os.path.join(hit, "doc.json")):
            if os.path.abspath(hit) != os.path.abspath(out_dir):
                shutil.copytree(hit, out_dir, dirs_exist_ok=True)
            r = json.load(open(os.path.join(out_dir, "doc.json"), encoding="utf-8"))
            r["meta"]["source_file"] = os.path.abspath(path)
            r["meta"]["cache"] = {"hit": True, "key": key, "config": json.loads(cfg), "seconds": round(time.time() - t0, 3)}
            write_json(os.path.join(out_dir, "doc.json"), r)
            return r
        r = parse(path, out_dir, password, pages, ocr_mode, lang, preview, None, budget_deadline=budget_deadline)
        if r.get("status") != "failed":
            r["meta"]["cache"] = {"hit": False, "key": key, "config": json.loads(cfg)}
            write_json(os.path.join(out_dir, "doc.json"), r)
            os.makedirs(cache_dir, exist_ok=True)
            shutil.copytree(out_dir, hit, dirs_exist_ok=True)
        return r
    os.makedirs(out_dir, exist_ok=True)
    warnings = []
    doc, err, msg = open_document(path, password)
    meta = {
        "source_file": os.path.abspath(path),
        "parser": f"pdf-evidence parse_pdf.py {PARSER_VERSION}",
        "pymupdf": getattr(fitz, "VersionBind", "?"),
    }
    if os.path.exists(path):
        with open(path, "rb") as fh:
            meta["sha256"] = hashlib.sha256(fh.read()).hexdigest()
    if doc is None:
        result = {"status": "failed", "error": err, "message": msg, "meta": meta,
                  "human_takeover": "Provide a decrypted/valid PDF or the password; nothing was extracted."}
        write_json(os.path.join(out_dir, "doc.json"), result)
        write_json(os.path.join(out_dir, "qa_report.json"), {"status": "failed", "error": err, "message": msg})
        return result
    meta.update({"page_count": doc.page_count, "metadata": {k: v for k, v in (doc.metadata or {}).items() if v},
                 "was_encrypted": bool(getattr(doc, "is_encrypted", False))})
    try:
        labels = [doc[i].get_label() for i in range(doc.page_count)]
        if any(labels):
            meta["page_labels"] = labels
    except Exception:
        pass
    page_idx = list(range(doc.page_count))
    if pages is not None:
        page_idx = [p - 1 for p in pages if 1 <= p <= doc.page_count]
        if not page_idx:
            msg = f"requested pages {pages} outside 1..{doc.page_count}"
            result = {"status": "failed", "error": "no_valid_pages", "message": msg, "meta": meta,
                      "pages_parsed": [], "pages_not_parsed": list(range(1, doc.page_count + 1)),
                      "warnings": [{"code": "no_valid_pages", "detail": msg}]}
            write_json(os.path.join(out_dir, "doc.json"), result)
            write_json(os.path.join(out_dir, "qa_report.json"), {"status": "failed", "error": "no_valid_pages", "message": msg})
            return result
    pages_blocks, page_info, page_heights, all_tables, figures = [], [], [], [], []
    ocr_pages, need_ocr_unresolved = [], []
    for pi in page_idx:
        page = doc[pi]
        pno = pi + 1
        W, H = page.rect.width, page.rect.height
        blocks = extract_text_blocks(page, pno)
        scanned, chars, img_frac = page_is_scanned(page, blocks)
        g = garbled_ratio(" ".join(b["text"] for b in blocks))
        info = {"page": pno, "width": round(W, 2), "height": round(H, 2), "text_layer_chars": chars,
                "image_coverage": round(img_frac, 3), "garbled_ratio": round(g, 4), "source": "text_layer"}
        if (scanned and ocr_mode == "auto") or ocr_mode == "force" or (g > 0.05 and ocr_mode == "auto"):
            reason = "no text layer (scanned)" if scanned else ("garbled font encoding" if g > 0.05 else "forced")
            oblocks, oinfo = ocr_page(page, pno, os.path.join(out_dir, "_ocr"), lang, budget_deadline=budget_deadline)
            info["ocr"] = oinfo
            info["ocr_reason"] = reason
            if oblocks:
                blocks = oblocks
                info["source"] = "ocr"
                ocr_pages.append(pno)
            else:
                need_ocr_unresolved.append(pno)
                warnings.append({"page": pno, "code": "ocr_needed_but_unavailable",
                                 "detail": f"{reason}; {oinfo.get('reason', '')}. Text for this page is missing or unreliable. "
                                           f"Page image and layout regions are provided for visual reading."})
                try:
                    rel, dpi_used, regs = page_image_and_regions(page, pno, out_dir)
                    info["page_image"] = rel
                    info["page_image_dpi"] = dpi_used
                    info["image_regions"] = regs
                    info["source"] = "image_only"
                except Exception as ex:
                    info["page_image_error"] = str(ex)
        elif g > 0.01:
            warnings.append({"page": pno, "code": "possible_garbled_text", "detail": f"garbled char ratio {g:.3f}"})
        # PyMuPDF's find_tables fallback is slow; only try it where a table caption exists
        has_tab_cap = any(re.match(r"^\s*(Table|TABLE|Tab\.|表)\s*[A-Z]?\.?\d", ln["text"]) for b in blocks for ln in b["lines"])
        info["has_equation_number"] = any(RE_EQNUM_LINE.match(ln["text"].strip()) or RE_EQ_NUM.search(ln["text"])
                                          for b in blocks for ln in b["lines"])
        tables, terr = extract_tables(page, pno, allow_fallback=has_tab_cap) if info["source"] == "text_layer" else ([], None)
        if terr:
            warnings.append({"page": pno, "code": "table_detection_unavailable", "detail": terr})
        for t in tables:
            for b in blocks:
                if bbox_center_in(b["bbox"], t["bbox"]) and not RE_CAPTION.match(b["text"]):
                    b["role"], b["role_rule"] = "table", "inside detected table bbox"
                    b["table_ref"] = len(all_tables) + tables.index(t)
        all_tables.extend(tables)
        try:
            for im in page.get_image_info():
                bb = [round(v, 2) for v in im["bbox"]]
                if (bb[2] - bb[0]) * (bb[3] - bb[1]) > 0.01 * W * H and info["source"] == "text_layer":
                    figures.append({"page": pno, "bbox": bb, "kind": "raster", "caption": None})
        except Exception:
            pass
        try:
            if hasattr(page, "cluster_drawings") and info["source"] == "text_layer":
                for r in page.cluster_drawings():
                    if r.width > 0.15 * W and r.height > 0.08 * H:
                        tb = [round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2)]
                        if not any(abs(tb[1] - t["bbox"][1]) < 15 and abs(tb[0] - t["bbox"][0]) < 15 for t in tables if t["page"] == pno):
                            figures.append({"page": pno, "bbox": tb, "kind": "vector", "caption": None})
        except Exception:
            pass
        # graphic regions (for attaching tick labels / legends / axis titles to figures)
        regions = []
        if info["source"] == "text_layer":
            try:
                rects = [fitz.Rect(im["bbox"]) for im in page.get_image_info()]
                if hasattr(page, "cluster_drawings"):
                    rects += list(page.cluster_drawings())
                for r in rects:
                    r = r & page.rect
                    if r.width < 40 or r.height < 30 or r.width > 0.95 * W:
                        continue
                    rb = [r.x0, r.y0, r.x1, r.y1]
                    if any(min(rb[2], t["bbox"][2]) - max(rb[0], t["bbox"][0]) > 0 and
                           min(rb[3], t["bbox"][3]) - max(rb[1], t["bbox"][1]) > 0.3 * r.height for t in tables):
                        continue
                    regions.append(rb)
            except Exception:
                pass
        info["n_figure_text_blocks"] = mark_figure_text(blocks, regions, W)
        if info["source"] == "text_layer":
            rr = raster_regions_without_text(page, pno, blocks, path, meta.get("sha256"), out_dir, lang=lang)
            if rr:
                info["raster_regions"] = rr
                warnings.append({"page": pno, "code": "raster_content_not_in_text_layer",
                                 "detail": f"{len(rr)} embedded image region(s) with no text layer inside "
                                           f"({', '.join(x['id'] + ' bbox ' + str(x['bbox']) for x in rr)}); their content "
                                           f"(table/figure/equation pasted as an image) is not in the extracted text. "
                                           f"See pages[].raster_regions (crop / crop_command) and tables_as_images."})
        pages_blocks.append(blocks)
        page_info.append(info)
        page_heights.append(H)

    pre_sizes = Counter()
    for blocks in pages_blocks:
        for b in blocks:
            for ln in b["lines"]:
                pre_sizes[ln["size"]] += len(ln["text"])
    hf = detect_headers_footers(pages_blocks, page_heights, pre_sizes.most_common(1)[0][0] if pre_sizes else None,
                                [p["width"] for p in page_info])
    page_h = {p["page"]: p["height"] for p in page_info}
    ordered_all, margin_all = [], []
    for pi, blocks in enumerate(pages_blocks):
        W = page_info[pi]["width"]
        ordered, margin, gutters = order_page(blocks, W)
        page_info[pi]["column_gutters"] = gutters
        page_info[pi]["n_columns"] = len(gutters) + 1
        for k, b in enumerate(ordered):
            b["id"] = f"p{b['page']}-b{k + 1}"
        km = kg = 0
        for b in margin:
            if b["role"] == "figure_text":
                kg += 1
                b["id"] = f"p{b['page']}-g{kg}"
            else:
                km += 1
                b["id"] = f"p{b['page']}-m{km}"
        ordered_all.extend(ordered)
        margin_all.extend(margin)
    for k, b in enumerate(ordered_all):
        b["order"] = k + 1

    body_sizes = Counter()
    for b in ordered_all:
        for ln in b["lines"]:
            body_sizes[ln["size"]] += len(ln["text"])
    body_size = body_sizes.most_common(1)[0][0] if body_sizes else 10.0
    meta["body_font_size"] = body_size
    first_page_no = page_idx[0] + 1 if page_idx else 1
    classify_roles(ordered_all, body_size, page_h, first_page_no)

    # ---- sections
    sections, stack = [], []
    for b in ordered_all:
        if b["role"] == "heading":
            h = b.get("heading") or {"kind": "named", "number": "", "title": norm_ws(b["text"])}
            lvl = heading_level(h)
            if h["kind"] == "styled" and stack:
                # unnumbered heading: level relative to the enclosing heading, by font size
                top = stack[-1]
                if top["_kind"] in ("styled", "named") and abs(b["size"] - top["_size"]) < 0.5:
                    lvl = top["level"]  # same style as an unnumbered/named heading: sibling
                elif b["size"] <= top["_size"] + 0.3:
                    lvl = top["level"] + 1
                else:
                    lvl = max(1, top["level"] - (1 if top["_kind"] == "styled" else 0))
            while stack and stack[-1]["level"] >= lvl:
                stack.pop()
            sec = {"id": f"s{len(sections) + 1}", "level": lvl, "number": h["number"] or None,
                   "title": h["title"], "page": b["page"], "block_id": b["id"],
                   "parent": stack[-1]["id"] if stack else None, "rule": b["role_rule"],
                   "_kind": h["kind"], "_size": b["size"]}
            sections.append(sec)
            stack.append(sec)
            b["section_path"] = [s["title"] for s in stack]
        else:
            b["section_path"] = [s["title"] for s in stack]
            b["section_id"] = stack[-1]["id"] if stack else None
    for sec in sections:
        sec["kind"] = sec.pop("_kind")
        sec.pop("_size", None)
    # numbering sanity check: report jumps such as 2 -> 4
    nums = [s["number"] for s in sections if s["number"] and re.match(r"^\d+$", s["number"])]
    for a, c in zip(nums, nums[1:]):
        if int(c) - int(a) > 1:
            warnings.append({"code": "section_number_gap", "detail": f"top-level section {a} followed by {c}; a heading may be missed"})

    def overlap_frac(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        return ix * iy / area
    figures = [f for f in figures if not any(t["page"] == f["page"] and overlap_frac(f["bbox"], t["bbox"]) > 0.5 for t in all_tables)]
    # ---- captions -> figures/tables
    captions = [b for b in ordered_all if b["role"] == "caption"]
    for t_i, t in enumerate(all_tables):
        t["id"] = f"T{t_i + 1}"
        cands = [c for c in captions if c["page"] == t["page"] and re.match(r"^\s*(Table|TABLE|Tab\.|表)", c["text"])]
        if cands:
            c = min(cands, key=lambda c: min(abs(c["bbox"][3] - t["bbox"][1]), abs(c["bbox"][1] - t["bbox"][3])))
            t["caption"], t["caption_block"] = norm_ws(c["text"]), c["id"]
    # cross-page table continuation: explicit "Table 2 (continued)" captions, then caption-less tables at a page top
    def cap_number(cap):
        m = RE_CAPTION.match(cap or "")
        return (m.group(1)[:3].lower(), m.group(2)) if m else None
    for k, c in enumerate(all_tables):
        if c.get("caption") and re.search(r"\b(continued|cont\.|contd)\b|\(cont", c["caption"], re.I):
            prev = [a for a in all_tables[:k] if a.get("caption") and not a.get("continued_from")
                    and cap_number(a["caption"]) == cap_number(c["caption"])]
            if prev:
                c["continued_from"] = prev[-1]["id"]
    for a, c in zip(all_tables, all_tables[1:]):
        if c["page"] == a["page"] + 1 and c["n_cols"] == a["n_cols"] and not c["caption"]:
            Ha = page_h[a["page"]]
            Hc = page_h[c["page"]]
            if a["bbox"][3] > 0.6 * Ha and c["bbox"][1] < 0.4 * Hc:
                c["continued_from"] = a["id"]
                c["caption"] = (a["caption"] or "") + " (continued)"
    for f_i, f in enumerate(figures):
        f["id"] = f"F{f_i + 1}"
        cands = [c for c in captions if c["page"] == f["page"] and re.match(r"^\s*(Fig|FIG|Figure|FIGURE|图|Scheme|Extended)", c["text"])]
        if cands:
            c = min(cands, key=lambda c: abs(c["bbox"][1] - f["bbox"][3]))
            f["caption"], f["caption_block"] = norm_ws(c["text"]), c["id"]
    # raster regions (pictures without text layer) next to a caption: "Table 3" pasted as an image
    # crops are made at parse time for table-like / uncaptioned pictures (capped); captioned figures get crop_command
    tables_as_images = []
    n_crops = 0
    for info in page_info:
        for x in info.get("raster_regions", []) or []:
            caps = [c for c in captions if c["page"] == info["page"]]
            c = min(caps, key=lambda c: min(abs(c["bbox"][3] - x["bbox"][1]), abs(c["bbox"][1] - x["bbox"][3]))) if caps else None
            if c is not None and min(abs(c["bbox"][3] - x["bbox"][1]), abs(c["bbox"][1] - x["bbox"][3])) > 60:
                c = None
            is_table = (c is not None and bool(re.match(r"^\s*(Table|TABLE|Tab\.|表)", c["text"]))
                        and not any(t.get("caption_block") == c["id"] for t in all_tables))
            if (is_table or c is None) and n_crops < 12:
                materialize_raster(doc[x["page"] - 1], x, path, meta.get("sha256"), out_dir, lang=lang, budget_deadline=budget_deadline)
                n_crops += 1
            if c is None:
                continue
            x["caption"], x["caption_block"] = norm_ws(c["text"]), c["id"]
            if is_table:
                tables_as_images.append({"id": f"TI{len(tables_as_images) + 1}", "page": x["page"], "bbox": x["bbox"],
                                         "caption": x["caption"], "caption_block": c["id"], "crop": x["crop"],
                                         "region_id": x["id"], "ocr_text": x.get("ocr_text"),
                                         "note": "table is an embedded image: no cells in the text layer; read the crop "
                                                 "and cite it as image_region evidence (provenance sidecar next to the crop)"})
    # figure captions without a detected graphic still count as figure locations
    linked = {f.get("caption_block") for f in figures}
    for c in captions:
        if c["id"] not in linked and re.match(r"^\s*(Fig|FIG|Figure|FIGURE|图|Scheme|Extended)", c["text"]):
            figures.append({"id": f"F{len(figures) + 1}", "page": c["page"], "bbox": None, "kind": "caption_only",
                            "caption": norm_ws(c["text"]), "caption_block": c["id"]})

    equations = []
    eq_dir = os.path.join(out_dir, "equations")

    def eq_crop(page_no, bbox, tag):
        try:
            os.makedirs(eq_dir, exist_ok=True)
            pg = doc[page_no - 1]
            r = (fitz.Rect(bbox) + (-4, -4, 4, 4)) & pg.rect
            name = f"eq_p{page_no}_{tag}.png"
            pg.get_pixmap(dpi=200, clip=r).save(os.path.join(eq_dir, name))
            return f"equations/{name}", [round(v, 2) for v in (r.x0, r.y0, r.x1, r.y1)]
        except Exception:
            return None, None

    # 1) number-anchored equations (pieces gathered across blocks, glyphs decoded, 2D layout rebuilt)
    anchored = []
    if mathlayout is not None:
        for info in page_info:
            if info.get("source") != "text_layer" or not info.get("has_equation_number"):
                continue
            try:
                anchored += mathlayout.numbered_equations(doc, doc[info["page"] - 1], info["page"], info.get("column_gutters"))
            except Exception as ex:
                warnings.append({"page": info["page"], "code": "equation_layout_error", "detail": str(ex)})
    for e in anchored:
        part_ids = []
        for b in ordered_all:
            if b["page"] == e["page"] and bbox_center_in(b["bbox"], e["bbox"], pad=2) and b["role"] in ("body", "equation", "caption"):
                if b["role"] == "caption":
                    continue
                b["role"], b["role_rule"] = "equation", f"part of number-anchored equation ({e['number']})"
                b["equation_number"] = e["number"]
                part_ids.append(b["id"])
        crop, crop_bbox = eq_crop(e["page"], e["bbox"], "n" + re.sub(r"\W", "_", e["number"]))
        unc = list(e["uncertain_symbols"])
        equations.append({
            "number": e["number"], "page": e["page"], "bbox": e["bbox"], "expr_bbox": e["expr_bbox"],
            "number_bbox": e["number_bbox"], "block_ids": part_ids, "block_id": part_ids[0] if part_ids else None,
            "text": " ".join(next(b["text"] for b in ordered_all if b["id"] == i) for i in part_ids),
            "expr_latex": e["expr_latex"], "expr_unicode": e["expr_unicode"],
            "latex": f"{e['expr_latex']} ({e['number']})", "unicode_math": f"{e['expr_unicode']} ({e['number']})",
            "marked": f"{e['expr_unicode']} ({e['number']})",
            "transcription_status": e["transcription_status"] if crop or e["transcription_status"] == "auto" else e["transcription_status"],
            "uncertain_symbols": unc, "resolved_glyphs": e["resolved_glyphs"],
            "crop": crop, "crop_bbox": crop_bbox, "source": "number_anchored", "rule": e["rule"],
            "uncertainty": (["transcription rebuilt from glyph positions; check against the crop"] +
                            [u.get("note") or f"unmapped glyph {u.get('glyph')}" for u in unc]),
        })
    taken = [(e["page"], e["bbox"]) for e in equations]
    # 2) single-block equations without an anchored counterpart
    for b in ordered_all:
        if b["role"] != "equation" or b.get("equation_number") and any(e["number"] == b.get("equation_number") and e["page"] == b["page"] for e in equations):
            continue
        if any(pg == b["page"] and bbox_center_in(b["bbox"], bb, pad=2) for pg, bb in taken):
            continue
        unc = []
        if not b.get("math_fonts"):
            unc.append("no recognisable math font names: bold/calligraphic/blackboard alphabets cannot be recovered from the text layer")
        if "�" in b["text"]:
            unc.append("glyphs without Unicode mapping in the text")
        crop, crop_bbox = eq_crop(b["page"], b["bbox"], b["id"].split("-")[1])
        status = "partial" if unc else "auto"
        equations.append({"block_id": b["id"], "block_ids": [b["id"]], "page": b["page"], "bbox": b["bbox"],
                          "number": b.get("equation_number"), "text": b["text"], "marked": b.get("marked") or b["text"],
                          "latex": mathlayout.latex_plain(b.get("latex")) if mathlayout is not None else b.get("latex"),
                          "unicode_math": b.get("unicode_math"), "math_fonts": b.get("math_fonts"),
                          "transcription_status": status if crop or status == "auto" else "crop_only",
                          "uncertain_symbols": [{"note": u} for u in unc], "crop": crop, "crop_bbox": crop_bbox,
                          "source": "single_block", "rule": b["role_rule"],
                          "uncertainty": ["LaTeX/Unicode rebuilt from font names; fractions, roots and stacked limits inside a "
                                          "single block are not reconstructed - compare with the crop"] + unc})
    equations.sort(key=lambda e: (e["page"], e["bbox"][1]))
    footnotes = [{"block_id": b["id"], "page": b["page"], "text": norm_ws(b["text"])} for b in ordered_all if b["role"] == "footnote"]
    references = split_references([b for b in ordered_all if b["role"] == "reference"])
    eq_nums = sorted({int(e["number"]) for e in equations if e.get("number") and re.match(r"^\d+$", e["number"])})
    eq_nums = [str(n) for n in eq_nums]
    for a, c in zip(eq_nums, eq_nums[1:]):
        if int(c) - int(a) > 1:
            warnings.append({"code": "equation_number_gap", "detail": f"equation ({a}) followed by ({c}); an equation may be missed"})

    # ---- paragraphs: merge body blocks split across columns / pages
    paragraphs = []
    for b in ordered_all:
        if b["role"] not in ("body", "abstract"):
            continue
        t = norm_ws(b["text"])
        if paragraphs:
            p = paragraphs[-1]
            same_sec = p["section_id"] == b.get("section_id")
            between = [x["role"] for x in ordered_all[p["last_order"]:b["order"] - 1]]
            skippable = all(r in ("footnote", "caption", "table", "figure") for r in between)
            contiguous_page = b["page"] - p["sources"][-1]["page"] <= 1
            if contiguous_page and same_sec and p["role"] == b["role"] and not SENT_END.search(p["text"]) and re.match(r"^[a-z(\[,;]", t) \
                    and skippable:
                joiner = "" if p["text"].endswith("-") and re.match(r"^[a-z]", t) else " "
                text = (p["text"][:-1] if joiner == "" else p["text"]) + joiner + t
                p["text"] = text
                p["sources"].append({"block_id": b["id"], "page": b["page"], "bbox": b["bbox"]})
                p["continued"] = True
                if between:
                    p.setdefault("skipped_over", []).extend(between)
                p["last_order"] = b["order"]
                continue
        paragraphs.append({"id": f"para{len(paragraphs) + 1}", "role": b["role"], "section_id": b.get("section_id"),
                           "section_path": b.get("section_path"), "text": t,
                           "sources": [{"block_id": b["id"], "page": b["page"], "bbox": b["bbox"]}],
                           "continued": False, "last_order": b["order"]})
    for p in paragraphs:
        p.pop("last_order", None)

    if need_ocr_unresolved:
        status = "partial"
    elif not ordered_all:
        status = "failed"
        warnings.append({"code": "no_text", "detail": "no text extracted from any page"})
    else:
        status = "success"
    if any(w["code"] == "possible_garbled_text" for w in warnings):
        status = "partial" if status == "success" else status
    pages_not_parsed = [i + 1 for i in range(doc.page_count) if i not in set(page_idx)]
    if pages_not_parsed:
        warnings.append({"code": "partial_page_selection",
                         "detail": f"parsed pages {[i + 1 for i in page_idx]}; skipped pages {pages_not_parsed}"})

    for b in ordered_all + margin_all:
        b.pop("table_ref", None)
    result = {
        "status": status,
        "meta": meta,
        "pages_parsed": [i + 1 for i in page_idx],
        "pages_not_parsed": [i + 1 for i in range(doc.page_count) if i not in set(page_idx)],
        "warnings": warnings,
        "pages": page_info,
        "ocr_pages": ocr_pages,
        "unresolved_pages": need_ocr_unresolved,
        "sections": sections,
        "blocks": [slim(b) for b in ordered_all],
        "headers_footers": [slim(b) for b in margin_all],
        "paragraphs": paragraphs,
        "tables": all_tables,
        "tables_as_images": tables_as_images,
        "figures": figures,
        "equations": equations,
        "footnotes": footnotes,
        "references": references,
        "stats": {
            "n_blocks": len(ordered_all), "n_sections": len(sections), "n_tables": len(all_tables),
            "n_figures": len(figures), "n_equations": len(equations), "n_references": len(references),
            "n_tables_as_images": len(tables_as_images),
            "n_raster_regions_without_text": sum(len(p.get("raster_regions") or []) for p in page_info),
            "n_footnotes": len(footnotes), "n_headers_footers_removed": len(margin_all),
            "n_dehyphenated": sum(b["n_dehyphenated"] for b in ordered_all),
            "n_cross_block_paragraph_joins": sum(len(p["sources"]) - 1 for p in paragraphs),
            "seconds": round(time.time() - t0, 3),
        },
    }
    write_outputs(result, out_dir)
    if preview:
        render_previews(doc, result, os.path.join(out_dir, "previews"))
    return result


def slim(b):
    out = {k: v for k, v in b.items() if k not in ("lines",)}
    out["n_lines"] = len(b["lines"])
    if b.get("marked") is None:
        out.pop("marked", None)
    return out


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)


def write_outputs(r, out_dir):
    write_json(os.path.join(out_dir, "doc.json"), r)
    with open(os.path.join(out_dir, "blocks.jsonl"), "w", encoding="utf-8") as fh:
        for b in r["blocks"] + r["headers_footers"]:
            fh.write(json.dumps({k: b.get(k) for k in ("id", "page", "order", "role", "column", "bbox", "section_path", "text", "marked")},
                                ensure_ascii=False) + "\n")
    tdir = os.path.join(out_dir, "tables")
    os.makedirs(tdir, exist_ok=True)
    for t in r["tables"]:
        with open(os.path.join(tdir, f"{t['id']}.md"), "w", encoding="utf-8") as fh:
            fh.write(f"<!-- {t['id']} page {t['page']} bbox {t['bbox']} caption: {t['caption']} -->\n" + table_to_markdown(t) + "\n")
        with open(os.path.join(tdir, f"{t['id']}.html"), "w", encoding="utf-8") as fh:
            fh.write(table_to_html(t) + "\n")
        with open(os.path.join(tdir, f"{t['id']}.csv"), "w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerows([[c if c is not None else "" for c in row] for row in t["rows"]])
    # markdown in reading order with anchors
    md = [f"<!-- source: {r['meta']['source_file']} | status: {r['status']} | parser: {r['meta']['parser']} -->"]
    tables_by_block = {}
    for t in r["tables"]:
        tables_by_block.setdefault(t.get("caption_block"), []).append(t)
    emitted_tables = set()
    cur_page = None
    for b in r["blocks"]:
        if b["page"] != cur_page:
            cur_page = b["page"]
            md.append(f"\n<!-- ===== page {cur_page} ===== -->")
        anchor = f"<!-- {b['id']} {b['role']} col={b['column']} bbox={b['bbox']} -->"
        text = b.get("marked") or b["text"]
        role = b["role"]
        if role == "title":
            md.append(f"{anchor}\n# {norm_ws(text)}")
        elif role == "heading":
            sec = next((s for s in r["sections"] if s["block_id"] == b["id"]), None)
            lvl = min(6, (sec["level"] if sec else 1) + 1)
            md.append(f"{anchor}\n{'#' * lvl} {norm_ws(b['text'])}")
        elif role == "table":
            t = next((t for t in r["tables"] if t["page"] == b["page"] and bbox_center_in(b["bbox"], t["bbox"])), None)
            if t and t["id"] not in emitted_tables:
                emitted_tables.add(t["id"])
                md.append(f"<!-- {t['id']} table page={t['page']} bbox={t['bbox']} -->\n" + table_to_markdown(t))
        elif role == "equation":
            md.append(f"{anchor}\n$$ {norm_ws(text)} $$" + (f"  (eq. {b.get('equation_number')})" if b.get("equation_number") else ""))
        elif role == "caption":
            md.append(f"{anchor}\n> **{norm_ws(text)}**")
            for t in tables_by_block.get(b["id"], []):
                if t["id"] not in emitted_tables:
                    emitted_tables.add(t["id"])
                    md.append(f"<!-- {t['id']} table page={t['page']} bbox={t['bbox']} -->\n" + table_to_markdown(t))
        elif role == "footnote":
            md.append(f"{anchor}\n[^fn]: {norm_ws(text)}")
        elif role == "reference":
            md.append(f"{anchor}\n- {norm_ws(text)}")
        else:
            md.append(f"{anchor}\n{norm_ws(text)}")
    for t in r["tables"]:
        if t["id"] not in emitted_tables:
            md.append(f"<!-- {t['id']} table page={t['page']} bbox={t['bbox']} -->\n" + table_to_markdown(t))
    if r["headers_footers"]:
        md.append("\n<!-- ===== removed running headers / footers / margin notes ===== -->")
        for b in r["headers_footers"]:
            md.append(f"<!-- {b['id']} {b['role']}: {norm_ws(b['text'])[:120]} -->")
    with open(os.path.join(out_dir, "document.md"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(md) + "\n")
    qa = {"status": r["status"], "warnings": r["warnings"], "unresolved_pages": r["unresolved_pages"],
          "ocr_pages": r["ocr_pages"], "stats": r["stats"],
          "pages": [{k: p.get(k) for k in ("page", "source", "n_columns", "text_layer_chars", "garbled_ratio", "image_coverage")} for p in r["pages"]]}
    write_json(os.path.join(out_dir, "qa_report.json"), qa)


ROLE_COLORS = {"title": (0.8, 0, 0), "heading": (0.9, 0.4, 0), "body": (0, 0.3, 0.9), "abstract": (0, 0.5, 0.5),
               "caption": (0.6, 0, 0.6), "equation": (0, 0.6, 0), "table": (0.5, 0.3, 0), "footnote": (0.4, 0.4, 0.4),
               "reference": (0.2, 0.2, 0.6), "author_affiliation": (0.7, 0.2, 0.4),
               "header": (0.6, 0.6, 0.6), "footer": (0.6, 0.6, 0.6), "margin_note": (0.6, 0.6, 0.6)}


def render_previews(doc, r, pdir, dpi=80):
    os.makedirs(pdir, exist_ok=True)
    tmp = fitz.open()
    tmp.insert_pdf(doc)
    for p in r["pages"]:
        page = tmp[p["page"] - 1]
        for b in r["blocks"] + r["headers_footers"]:
            if b["page"] != p["page"]:
                continue
            col = ROLE_COLORS.get(b["role"], (0, 0, 0))
            rect = fitz.Rect(b["bbox"])
            page.draw_rect(rect, color=col, width=0.8)
            label = f"{b['order'] or '-'} {b['role']}"
            page.insert_text((rect.x0, max(8, rect.y0 - 1)), label, fontsize=6, color=col)
        for t in r["tables"]:
            if t["page"] == p["page"]:
                page.draw_rect(fitz.Rect(t["bbox"]), color=(1, 0.5, 0), width=1.5)
        for g in p.get("column_gutters", []):
            page.draw_line((g, 0), (g, p["height"]), color=(1, 0, 0), width=0.4, dashes="[2] 0")
        page.get_pixmap(dpi=dpi).save(os.path.join(pdir, f"page_{p['page']:03d}.png"))
    tmp.close()


def parse_pages(spec):
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        raise ValueError("empty page specification")
    out = []
    for raw in s.split(","):
        part = raw.strip()
        if not part:
            raise ValueError(f"invalid page range: {spec!r}")
        if "-" in part:
            bits = part.split("-")
            if len(bits) != 2 or not bits[0].strip().isdigit() or not bits[1].strip().isdigit():
                raise ValueError(f"invalid page range: {part!r}")
            a, b = int(bits[0].strip()), int(bits[1].strip())
            if a < 1 or b < a:
                raise ValueError(f"invalid page range: {part!r} (must be 1 <= start <= end)")
            out.extend(range(a, b + 1))
        else:
            if not part.isdigit() or int(part) < 1:
                raise ValueError(f"invalid page number: {part!r}")
            out.append(int(part))
    if not out:
        raise ValueError(f"no pages specified: {spec!r}")
    return sorted(set(out))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--password", default=None)
    ap.add_argument("--pages", default=None, help="e.g. 1-3,7")
    ap.add_argument("--ocr", default="auto", choices=["auto", "force", "off"])
    ap.add_argument("--lang", default="eng+chi_sim")
    ap.add_argument("--preview", action="store_true", help="render PNG previews with block boxes")
    ap.add_argument("--cache", default=None, help="cache directory: reuse a result for the same PDF content, parser version and options")
    ap.add_argument("--budget", type=float, default=None, help="cumulative OCR time budget in seconds")
    a = ap.parse_args()
    try:
        pages_arg = parse_pages(a.pages)
    except ValueError as ex:
        os.makedirs(a.out, exist_ok=True)
        err_doc = {"status": "failed", "error": "invalid_pages", "message": str(ex),
                   "meta": {"source_file": os.path.abspath(a.pdf), "parser": f"pdf-evidence parse_pdf.py {PARSER_VERSION}"}}
        write_json(os.path.join(a.out, "doc.json"), err_doc)
        write_json(os.path.join(a.out, "qa_report.json"), {"status": "failed", "error": "invalid_pages", "message": str(ex)})
        print(json.dumps({"status": "failed", "out": os.path.abspath(a.out), "error": "invalid_pages", "message": str(ex)}, ensure_ascii=False))
        sys.exit(2)
    if a.budget is not None and a.budget <= 0:
        os.makedirs(a.out, exist_ok=True)
        msg = f"invalid --budget: {a.budget} (must be > 0)"
        err_doc = {"status": "failed", "error": "invalid_budget", "message": msg,
                   "meta": {"source_file": os.path.abspath(a.pdf), "parser": f"pdf-evidence parse_pdf.py {PARSER_VERSION}"}}
        write_json(os.path.join(a.out, "doc.json"), err_doc)
        print(json.dumps({"status": "failed", "out": os.path.abspath(a.out), "error": "invalid_budget", "message": msg}, ensure_ascii=False))
        sys.exit(2)
    deadline = (time.time() + a.budget) if a.budget is not None else None
    r = parse(a.pdf, a.out, a.password, pages_arg, a.ocr, a.lang, a.preview, a.cache, budget_deadline=deadline)
    s = r.get("stats", {})
    print(json.dumps({"status": r["status"], "out": os.path.abspath(a.out), "error": r.get("error"),
                      "warnings": len(r.get("warnings", [])), **s}, ensure_ascii=False))
    sys.exit(0 if r["status"] in ("success", "partial") else 2)


if __name__ == "__main__":
    main()
