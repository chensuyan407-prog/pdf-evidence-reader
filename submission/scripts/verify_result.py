#!/usr/bin/env python3
"""Check result.json against the parsed PDF and the original PDF before handing it in.

Usage:
    python verify_result.py [--result /app/output/result.json] [--parsed DIR ...] [--pdf FILE ...]

For every answered contract field the report separates three questions:

  value_equals_parsed_cell  the answer equals what the parser extracted
                            (table_cell evidence is looked up again by row
                            labels + column path);
  source_region_complete    the original PDF region behind that cell holds
                            exactly that value - no expression continuing past
                            the cell edge (e.g. "2.3 ·" followed by "10^19"),
                            no trailing operator, no missing exponent, no unit
                            left outside, no characters inside the box that the
                            cell lacks. True / False / None (PDF not available);
  needs_visual_check        a person (or vision model) must still look.

Evidence kinds:
  table_cell   {field, type:"table_cell", table|table_id, page, row{KEY:VALUE}, column "A/B", value}
  text         {field, page, block_id|table_id, quote}       -> string match only, never semantic
  image_region {field, type:"image_region", page, bbox, crop, provenance, transcription}
               -> the crop is tied to the PDF (SHA-256, page, bbox, crop hash,
                  re-rendered pixels); the transcription itself is NOT checked.

Field verdicts (fail = exit code 1; review = listed in needs_review):
  supported_by_table_cell                ok
  supported_by_equation_record           ok (answer = automatic transcription of the numbered equation, no uncertain symbols)
  equation_needs_visual_check            review (transcription partial / crop only)
  supported_by_parsed_cell_region_unchecked   review (source PDF not available)
  parsed_cell_incomplete                 fail + review (parser output is truncated / suspicious)
  value_mismatch | column_mismatch | row_mismatch | evidence_unsupported   fail
  string_match_only                      review (string found in a verified quote; meaning not checked)
  image_read_unverified                  review (crop bound to the PDF; reading by eye)
  image_read_unbound                     review (no provenance record: cannot tie crop to the page)
  image_provenance_mismatch              fail (wrong page / document / replaced crop / out of bounds)
  value_not_in_evidence | no_evidence | null_without_reason   fail
  unreported_ok                          ok

Older result.json files (evidence {claim, page, block_id, quote}, image evidence
without provenance) are still read; missing provenance lowers the level instead
of failing.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import provenance  # noqa: E402
import table_query  # noqa: E402

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz

FAIL_VERDICTS = {"value_mismatch", "column_mismatch", "row_mismatch", "evidence_unsupported", "parsed_cell_incomplete",
                 "image_provenance_mismatch", "value_not_in_evidence", "no_evidence", "null_without_reason", "answer_incomplete"}
REVIEW_VERDICTS = {"equation_needs_visual_check", "string_match_only", "image_read_unverified", "image_read_unbound", "parsed_cell_incomplete",
                   "supported_by_parsed_cell_region_unchecked", "answer_incomplete"}
META_KEYS = {"case_id", "status", "sources", "evidence", "uncertainty", "unreported", "conflicts", "run_status",
             "change_log", "verification", "field_status", "_contract_fields", "_contract_raw", "_fill_instructions"}
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
RE_SCI = re.compile(r"([-−]?\d[\d,]*(?:\.\d+)?)\s*[·×x*∗⋅]\s*10\s*(?:\^\s*\{?\s*([-−]?\d+)\s*\}?|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))")
RE_POW = re.compile(r"(?<![\d.])10\s*(?:\^\s*\{?\s*([-−]?\d+)\s*\}?|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))")
RE_E = re.compile(r"([-−]?\d+(?:\.\d+)?)[eE]([-+]?\d+)")
RE_NUM = re.compile(r"[-−]?\d+(?:[.,]\d+)*")
OPS = "·×*∗⋅^+−/="
UNIT_RE = re.compile(r"^(%|‰|km/s|km\.?/sec\.?|m/s|FLOPs?|GB|MB|KB|ms|s|h|M|B|K|pc|kpc|Mpc|K|°C|dB|Hz|kHz|MHz|GHz|eV|keV|MeV|GeV)\b", re.I)


def norm(s):
    s = (s or "").replace("­", "")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = re.sub(r"[\s　]+", "", s).lower()
    return s.replace("’", "'").replace("“", '"').replace("”", '"').replace("−", "-").replace("–", "-")


def values_in(text):
    """Numeric values in a string, scientific notation aware ('2.3 × 10¹⁹', '2.3·10^19', '2.3e19', '10^20')."""
    t = str(text or "")
    vals, used = [], []
    for m in RE_SCI.finditer(t):
        e = (m.group(2) or (m.group(3) or "").translate(SUP)).replace("−", "-")
        vals.append(float(m.group(1).replace("−", "-").replace(",", "")) * 10 ** int(e))
        used.append(m.span())
    for m in RE_E.finditer(t):
        if not any(a <= m.start() < b for a, b in used):
            vals.append(float(m.group(1).replace("−", "-")) * 10 ** int(m.group(2)))
            used.append(m.span())
    for m in RE_POW.finditer(t):
        if not any(a <= m.start() < b for a, b in used):
            e = (m.group(1) or (m.group(2) or "").translate(SUP)).replace("−", "-")
            vals.append(10.0 ** int(e))
            used.append(m.span())
    for m in RE_NUM.finditer(t):
        if not any(a <= m.start() < b for a, b in used):
            try:
                vals.append(float(m.group(0).replace("−", "-").replace(",", "")))
            except ValueError:
                pass
    return vals


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def cell_value(cell_text, cell_norm=None):
    if cell_norm and cell_norm.get("value") is not None:
        return cell_norm["value"]
    v = values_in(re.sub(r"\^\S+$", "", cell_text or ""))  # drop a trailing footnote marker
    return v[0] if v else None


def same_value(answer, cell_text, cell_norm=None):
    """Answer equals the cell: identical text, or every number in the answer equals the cell's number."""
    if norm(answer) == norm(cell_text):
        return True
    cv = cell_value(cell_text, cell_norm)
    av = values_in(answer)
    return cv is not None and bool(av) and all(close(x, cv) for x in av)


def incomplete_reasons(text, what="cell"):
    """Signs that a value was cut off, judged from the string alone."""
    t = (text or "").strip()
    out = []
    if not t or t in ("-", "–", "—"):
        return out
    # '*' and '+' are left out: after a number they are usually markers ("88.3*", "2D+"), not operators
    if re.search(r"[·×∗⋅^−/=(\[{]\s*$", t) or re.search(r"\s[x-]\s*$", t):
        out.append(f"{what} ends with an operator or opening bracket: {t!r}")
    if re.match(r"^\s*[·×*∗⋅^/=]", t):
        out.append(f"{what} starts with an operator: {t!r}")
    if re.search(r"[·×*∗⋅x]\s*10\s*$", t):
        out.append(f"{what} ends with '× 10' and no exponent: {t!r}")
    for o, c in ("()", "[]", "{}"):
        if t.count(o) != t.count(c):
            out.append(f"{what} has unbalanced {o}{c}: {t!r}")
    return out


class Sources:
    """Locates the original PDFs by SHA-256 (moved folders still work)."""

    def __init__(self, res, rdir, docs, extra):
        self.by_sha = {}
        cands = list(extra or [])
        for d in docs:
            cands.append(d.get("meta", {}).get("source_file"))
        for s in res.get("sources", []) or []:
            cands.append(s.get("file"))
        for base in (rdir, os.path.dirname(rdir), os.path.dirname(os.path.dirname(rdir))):
            cands += glob.glob(os.path.join(base, "*.pdf")) + glob.glob(os.path.join(base, "*", "*.pdf"))
        self.cands = [c for c in dict.fromkeys(c for c in cands if c)]
        self.known = {d.get("meta", {}).get("sha256") for d in docs if d.get("meta", {}).get("sha256")}
        for s in res.get("sources", []) or []:
            if s.get("sha256"):
                self.known.add(s["sha256"])

    def find(self, sha):
        if not sha:
            return None
        if sha in self.by_sha:
            return self.by_sha[sha]
        for c in self.cands:
            if os.path.exists(c):
                try:
                    if provenance.sha256_file(c) == sha:
                        self.by_sha[sha] = c
                        return c
                except OSError:
                    pass
        self.by_sha[sha] = None
        return None


def region_check(pdf, page_no, bbox, cell_text):
    """Compare the parsed cell with the characters of the original page around its box."""
    reasons = []
    d = fitz.open(pdf)
    page = d[page_no - 1]
    x0, y0, x1, y1 = bbox
    chars = []
    for b in page.get_text("rawdict").get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                for ch in s.get("chars", []):
                    if ch["c"].isspace():
                        continue
                    cb = ch["bbox"]
                    cy = (cb[1] + cb[3]) / 2
                    if y0 - 1.5 <= cy <= y1 + 1.5:
                        chars.append((cb[0], cb[2], ch["c"], s["size"]))
    inside = sorted([c for c in chars if c[0] >= x0 - 0.6 and c[1] <= x1 + 0.6], key=lambda c: c[0])
    right = sorted([c for c in chars if c[0] >= x1 - 0.6 and c not in inside], key=lambda c: c[0])
    left = sorted([c for c in chars if c[1] <= x0 + 0.6 and c not in inside], key=lambda c: -c[1])
    ins_digits = re.sub(r"\D", "", "".join(c[2] for c in inside))
    cell_digits = re.sub(r"\D", "", cell_text or "")
    if ins_digits != cell_digits:
        reasons.append(f"digits inside the cell box on the page ({ins_digits!r}) differ from the parsed cell ({cell_digits!r})")
    size = max([c[3] for c in inside] or [10.0])
    last = (cell_text or "").rstrip()[-1:]
    if right:
        gap = right[0][0] - x1
        nb = right[0]
        nxt = "".join(c[2] for c in right[:12] if c[0] - x1 < 25)
        # a close neighbour continues the value only if something ties them together:
        # the cell ends in an operator, the neighbour is raised/smaller (exponent), is an operator, or touches
        tied = (last and last in OPS + "x(") or nb[3] < 0.85 * size or nb[2] in OPS or gap < 0.15 * size
        if gap < 0.35 * size and tied:
            reasons.append(f"the expression continues right after the cell edge (gap {gap:.1f} pt): next characters {nxt!r}")
        elif gap < 0.45 * size and not (last.isdigit() and (nb[2].isdigit() or nb[2] in "−-+")):
            # only a word space away and not simply the next number: the printed value
            # (e.g. "8.0 Billion") runs past the cell box
            reasons.append(f"text continues within a word space of the cell's right edge (gap {gap:.1f} pt): {nxt!r}; "
                           "the cell may hold only part of the printed value")
        elif gap < 1.2 * size and UNIT_RE.match(nxt):
            reasons.append(f"a unit sits just right of the cell: {nxt!r}")
    if left:
        gap = x0 - left[0][1]
        prv = "".join(c[2] for c in sorted(left[:12], key=lambda c: c[0]) if x0 - c[1] < 25)
        if gap < 0.35 * size and (left[0][2] in OPS or gap < 0.15 * size):
            reasons.append(f"the expression continues left of the cell edge (gap {gap:.1f} pt)")
        elif gap < 0.45 * size and not (left[0][2].isdigit() and (cell_text or " ").lstrip()[:1] in "0123456789−-+"):
            reasons.append(f"text continues within a word space of the cell's left edge (gap {gap:.1f} pt): {prv!r}; "
                           "the cell may hold only part of the printed value")
    return reasons


def iter_evidence(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("evidence", "evidences") and isinstance(v, list):
                for i, e in enumerate(v):
                    if isinstance(e, dict):
                        yield f"{path}.{k}[{i}]", e
            elif k not in ("sources", "_contract_raw", "verification"):
                yield from iter_evidence(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_evidence(v, f"{path}[{i}]")


def ev_field(e):
    return e.get("field") or e.get("claim")


def ev_kind(e):
    t = e.get("type")
    if t in ("table_cell", "image_region", "text", "equation"):
        return t
    if "equation" in e and ("latex" in e or "page" in e) and not e.get("transcription"):
        return "equation"
    if "row" in e and "column" in e:
        return "table_cell"
    if e.get("crop") or e.get("transcription"):
        return "image_region"
    return "text"


def resolve_path(p, bases):
    if not p:
        return None
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for b in bases:
        q = os.path.join(b, p)
        if os.path.exists(q):
            return q
    name = os.path.basename(p.replace("\\", "/"))
    for b in bases:
        hits = glob.glob(os.path.join(b, "**", name), recursive=True)
        if hits:
            return hits[0]
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", default="/app/output/result.json")
    ap.add_argument("--parsed", nargs="*", default=None)
    ap.add_argument("--pdf", nargs="*", default=None, help="source PDF(s), if not where doc.json says")
    ap.add_argument("--out", default=None, help="verification.json path (default: next to result)")
    a = ap.parse_args()
    with open(a.result, encoding="utf-8") as fh:
        res = json.load(fh)
    rdir = os.path.dirname(os.path.abspath(a.result))
    parsed_dirs = a.parsed or [s.get("parsed_dir") for s in res.get("sources", []) if s.get("parsed_dir")]
    parsed_dirs = [d for d in parsed_dirs if d and os.path.exists(os.path.join(d, "doc.json"))]
    if not parsed_dirs:  # moved result folder: look for parsed/<stem>/doc.json next to result.json
        parsed_dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(rdir, "parsed", "*", "doc.json"))]
    docs = []
    for d in parsed_dirs:
        with open(os.path.join(d, "doc.json"), encoding="utf-8") as fh:
            docs.append(json.load(fh))
    src = Sources(res, rdir, docs, a.pdf)
    blocks, tables, pages, page_sizes = {}, {}, {}, {}
    for d in docs:
        for b in d.get("blocks", []) + d.get("headers_footers", []):
            blocks[b["id"]] = blocks.get(b["id"], "") + " " + b["text"] + " " + (b.get("marked") or "")
            pages.setdefault(b["page"], []).append(b["text"])
        for t in d.get("tables", []):
            tables[t["id"]] = " ".join(" ".join(c or "" for c in r) for r in t["rows"]) + " " + (t.get("caption") or "")
            pages.setdefault(t["page"], []).append(tables[t["id"]])
        for p in d.get("pages", []):
            page_sizes[p["page"]] = [p.get("width"), p.get("height")]

    issues, details = [], []
    if not res.get("case_id"):
        issues.append({"code": "missing_case_id"})
    fields = res.get("_contract_fields") or [k for k in res if k not in META_KEYS]
    for f in fields:
        if f not in res:
            issues.append({"code": "missing_contract_field", "field": f})

    by_field = {}
    n_quotes = n_quotes_ok = 0
    search_bases = [rdir] + parsed_dirs
    for path, e in iter_evidence(res):
        kind = ev_kind(e)
        d = {"path": path, "field": ev_field(e), "kind": kind}
        if kind == "table_cell":
            spec = e.get("table_id") or e.get("table")
            col = e["column"] if isinstance(e["column"], str) else "/".join(e["column"])
            rows = [f"{k}={v}" for k, v in (e.get("row") or {}).items()]
            q, qdoc = None, None
            for doc in docs:
                q = table_query.query(doc, spec, rows, col)
                if q["status"] == "ok":
                    qdoc = doc
                    break
            d["lookup"] = {k: q.get(k) for k in ("status", "table_id", "page", "value", "value_pretty", "value_numeric",
                                                 "column_path", "row_key", "cell_bbox", "shared_across")} if q else None
            if not q or q["status"] != "ok":
                d["ok"] = False
                d["reason"] = f"table lookup failed: {q['status'] if q else 'no parsed document'}"
            elif e.get("page") is not None and int(e["page"]) != q["page"]:
                d["ok"] = False
                d["reason"] = f"cited page {e['page']} but the table is on page {q['page']}"
            elif e.get("value") is not None and not same_value(e["value"], q["value"], _norm_of(qdoc, q)):
                d["ok"] = False
                d["reason"] = f"evidence value {e['value']!r} but the cell holds {q['value']!r}"
            else:
                d["ok"] = True
            d["cell_value"] = q.get("value") if q else None
            d["_q"], d["_doc"] = q, qdoc
            if d["ok"]:
                # completeness of the parsed cell against the original page
                cr = incomplete_reasons(q["value"], "parsed cell")
                pdf = src.find(qdoc.get("meta", {}).get("sha256"))
                if pdf and q.get("cell_bbox"):
                    cr += region_check(pdf, q["page"], q["cell_bbox"], q["value"])
                    d["source_region_complete"] = not cr
                else:
                    d["source_region_complete"] = False if cr else None
                    if not pdf:
                        d["region_note"] = "source PDF with the recorded SHA-256 not found; original region not checked"
                d["completeness_reasons"] = cr
                bb = q.get("cell_bbox") or [0, 0, 0, 0]
                d["review_location"] = {"page": q["page"], "bbox": [round(bb[0] - 30, 1), round(bb[1] - 6, 1),
                                                                   round(bb[2] + 30, 1), round(bb[3] + 6, 1)],
                                        "pdf_sha256": qdoc.get("meta", {}).get("sha256"),
                                        "hint": "crop with crop_region.py --page P --bbox x0,y0,x1,y1 and read it"}
        elif kind == "equation":
            num = str(e.get("equation"))
            eqs = [x for d0 in docs for x in d0.get("equations", []) if str(x.get("number")) == num
                   and (e.get("page") is None or int(e["page"]) == x["page"])]
            if not eqs:
                d.update(ok=False, reason=f"equation ({num}) not found on page {e.get('page')}")
            else:
                x = eqs[0]
                d.update(ok=True, equation=num, transcription_status=x.get("transcription_status"),
                         expr_unicode=x.get("expr_unicode") or x.get("unicode_math"), expr_latex=x.get("expr_latex") or x.get("latex"),
                         uncertain_symbols=x.get("uncertain_symbols"), crop=x.get("crop"))
        elif kind == "image_region":
            crop = resolve_path(e.get("crop"), search_bases)
            rec = e.get("provenance") or (provenance.load_sidecar(crop) if crop else None)
            level, probs, info = provenance.check_crop(e, crop, rec, src.known, src.find, page_sizes)
            if crop is None:
                level, probs = "mismatch", ["crop file not found (looked next to result.json and in parsed folders)"]
            d.update(ok=level != "mismatch", provenance_level=level, provenance_problems=probs, provenance_info=info,
                     crop_resolved=crop and os.path.relpath(crop, rdir).replace("\\", "/"),
                     note="transcription read from an image by the agent; not matched against any text layer")
            if probs:
                d["reason"] = "; ".join(probs)
        else:
            n_quotes += 1
            qt = e.get("quote") or e.get("text") or ""
            pg, bid, tid = e.get("page"), e.get("block_id"), e.get("table_id")
            found = None
            if bid and bid in blocks and norm(qt) in norm(blocks[bid]):
                found = f"block {bid}"
            elif tid and tid in tables and norm(qt) in norm(tables[tid]):
                found = f"table {tid}"
            elif pg is not None and int(pg) in pages and norm(qt) in norm(" ".join(pages[int(pg)])):
                found = f"page {pg}"
            if not qt:
                d.update(ok=False, reason="no quote")
            elif found:
                n_quotes_ok += 1
                d.update(ok=True, found_in=found)
            else:
                elsewhere = next((k for k, v in blocks.items() if norm(qt) in norm(v)), None)
                d.update(ok=False, reason=("quote exists but not at the cited location (found in " + elsewhere + ")")
                         if elsewhere else "quote not found in parsed PDF", quote=qt[:200])
            d["quote"] = qt
        details.append(d)
        if not d["ok"]:
            issues.append({"code": "evidence_not_verifiable", "path": path, "detail": d.get("reason")})
        by_field.setdefault(d["field"], []).append(d)

    unreported_txt = json.dumps(res.get("unreported", []), ensure_ascii=False)
    verdicts = {}
    for f in fields:
        val = res.get(f)
        evs = by_field.get(f, [])
        if val in (None, "", [], {}):
            verdicts[f] = {"verdict": "unreported_ok" if f in unreported_txt else "null_without_reason", "needs_review": False}
            continue
        eqv = [e for e in evs if e["kind"] == "equation"]
        tab = [e for e in evs if e["kind"] == "table_cell"]
        txt = [e for e in evs if e["kind"] == "text"]
        img = [e for e in evs if e["kind"] == "image_region"]
        v = {"answer": val, "checks": {"value_equals_parsed_cell": None, "source_region_complete": None,
                                       "needs_visual_check": False}, "reasons": []}
        ans_bad = incomplete_reasons(str(val), "answer")
        if eqv:
            e = eqv[0]
            if not e["ok"]:
                v["verdict"], v["reasons"] = "evidence_unsupported", [e["reason"]]
            else:
                rec = [norm(e.get("expr_unicode") or ""), norm(e.get("expr_latex") or "")]
                matches = any(r and r in norm(str(val)) for r in rec)
                v["checks"]["value_equals_parsed_cell"] = matches
                if not matches:
                    v["verdict"] = "value_mismatch"
                    v["reasons"] = ["answer does not contain the parser's transcription of this equation"]
                elif e.get("transcription_status") == "auto":
                    v["verdict"] = "supported_by_equation_record"
                    v["reasons"] = ["answer equals the automatic transcription (no uncertain symbols were found); "
                                    f"crop {e.get('crop')} available for a visual check"]
                else:
                    v["verdict"] = "equation_needs_visual_check"
                    v["checks"]["needs_visual_check"] = True
                    v["reasons"] = [f"transcription status {e.get('transcription_status')}: "
                                    + "; ".join(u.get("note") or str(u.get("glyph")) for u in (e.get("uncertain_symbols") or []))[:300]]
        elif tab:
            good = [e for e in tab if e["ok"] and same_value(val, e["cell_value"], _norm_of(e["_doc"], e["_q"]))]
            if good:
                e = good[0]
                v["checks"]["value_equals_parsed_cell"] = True
                v["checks"]["source_region_complete"] = e.get("source_region_complete")
                v["cell"] = e["lookup"]
                if e.get("completeness_reasons") or ans_bad:
                    v["verdict"] = "parsed_cell_incomplete"
                    v["reasons"] = e.get("completeness_reasons", []) + ans_bad
                    v["checks"]["needs_visual_check"] = True
                    v["review_location"] = e.get("review_location")
                elif e.get("source_region_complete") is None:
                    v["verdict"] = "supported_by_parsed_cell_region_unchecked"
                    v["reasons"] = [e.get("region_note", "original region not checked")]
                    v["checks"]["needs_visual_check"] = True
                    v["review_location"] = e.get("review_location")
                else:
                    v["verdict"] = "supported_by_table_cell"
            else:
                e = tab[0]
                q = e.get("_q") or {}
                v["checks"]["value_equals_parsed_cell"] = False
                if not e["ok"]:
                    v["verdict"] = "evidence_unsupported"
                    v["reasons"] = [e["reason"]]
                else:
                    v["verdict"] = "value_mismatch"
                    v["reasons"] = [f"answer {val!r} but the cited cell holds {e['cell_value']!r}"]
                if q.get("status") == "ok":
                    doc = e["_doc"]
                    t = next(t for t in doc["tables"] if t["id"] == q["table_id"])
                    cols = table_query.columns_of(t)
                    other_c = [cols[ci]["name"] for ci, c in enumerate(t["rows"][q["row"]])
                               if ci != q["col"] and c and same_value(val, c)]
                    nh = (t.get("header") or {}).get("n_header_rows", 1)
                    k0 = t.get("key_columns", [0])[0] if t.get("key_columns") else 0
                    other_r = [t["rows"][ri][k0] for ri in range(nh, len(t["rows"]))
                               if ri != q["row"] and t["rows"][ri][q["col"]] and same_value(val, t["rows"][ri][q["col"]])]
                    if other_c:
                        v["verdict"] = "column_mismatch"
                        v["reasons"] = [f"answer {val!r} is in column(s) {other_c} of the cited row, not in the cited column "
                                        f"{'/'.join(q['column_path'])} (which holds {q['value']!r})"]
                    elif other_r and v["verdict"] == "value_mismatch":
                        v["verdict"] = "row_mismatch"
                        v["reasons"] = [f"answer {val!r} is in the cited column but in row(s) {other_r[:5]}"]
        elif txt:
            ok_txt = [e for e in txt if e["ok"]]
            if not ok_txt:
                v["verdict"] = "evidence_unsupported"
                v["reasons"] = ["no cited quote was found at its location"]
            else:
                joined = " ".join(e["quote"] for e in ok_txt)
                an = values_in(val)
                present = all(any(close(x, y) for y in values_in(joined)) for x in an) if an else norm(str(val)) in norm(joined)
                if present:
                    v["verdict"] = "string_match_only"
                    v["checks"]["needs_visual_check"] = True
                    multi = max(len(values_in(e["quote"])) for e in ok_txt)
                    v["reasons"] = ["answer string occurs in the verified quotes; meaning/attribution not checked"]
                    if an and multi > 1:
                        v["reasons"].append(f"a quote holds {multi} numbers, so which one is meant is not verified (use table_cell evidence)")
                else:
                    v["verdict"] = "value_not_in_evidence"
                    v["reasons"] = [f"answer {val!r} does not occur in any verified quote for this field"]
            if ans_bad and v.get("verdict") == "string_match_only":
                v["verdict"] = "answer_incomplete"
                v["reasons"] += ans_bad
        elif img:
            levels = [e.get("provenance_level") for e in img]
            v["checks"]["needs_visual_check"] = True
            v["provenance_levels"] = levels
            if any(l == "mismatch" for l in levels):
                v["verdict"] = "image_provenance_mismatch"
                v["reasons"] = [p for e in img for p in e.get("provenance_problems", [])]
            elif all(l == "none" for l in levels):
                v["verdict"] = "image_read_unbound"
                v["reasons"] = ["no provenance record: the crop cannot be tied to the cited PDF page; transcription not verified"]
            else:
                v["verdict"] = "image_read_unverified"
                v["reasons"] = ["crop is tied to the cited PDF page (" + ", ".join(sorted(set(levels))) +
                                "); the value was read from the image and is not machine-checked"]
        else:
            v["verdict"] = "no_evidence"
        v["needs_review"] = v["verdict"] in REVIEW_VERDICTS
        verdicts[f] = v
        if v["verdict"] in FAIL_VERDICTS:
            issues.append({"code": v["verdict"], "field": f, "detail": "; ".join(v.get("reasons", []))})

    for d in details:
        d.pop("_q", None)
        d.pop("_doc", None)
    summary = {
        "checks_performed": {
            "quote_check": "cited quote occurs at cited block/table/page (whitespace, case and hyphenation insensitive)",
            "table_value_check": "table_cell evidence re-looked-up by row labels + column path; answer must equal the cell "
                                 "(scientific notation compared numerically)",
            "table_region_check": "original PDF (matched by SHA-256) re-read around the cell box: expression continuing past "
                                  "the edge, digits missing from the cell, unit left outside, trailing operator / missing exponent",
            "text_value_check": "answer string / numbers occur in verified quotes (string match only, no semantic check)",
            "image_check": "crop tied to PDF SHA-256, page, bbox and crop hash; region re-rendered and compared when the PDF "
                           "is available; the transcription itself is not verified",
        },
        "quotes_checked": n_quotes, "quotes_verified": n_quotes_ok,
        "evidence_items": len(details), "evidence_ok": sum(1 for d in details if d["ok"]),
        "field_verdicts": verdicts,
        "needs_review": [f for f, v in verdicts.items() if v.get("needs_review")],
        "issues": issues, "parsed_docs": parsed_dirs,
        "evidence_checked": len(details), "evidence_verified": sum(1 for d in details if d["ok"]),
    }
    out = a.out or os.path.join(rdir, "verification.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "details": details}, fh, ensure_ascii=False, indent=1)
    res["verification"] = {"evidence_items": len(details), "evidence_ok": summary["evidence_ok"],
                           "field_verdicts": {f: v["verdict"] for f, v in verdicts.items()},
                           "needs_review": summary["needs_review"],
                           "n_issues": len(issues), "report": os.path.basename(out)}
    with open(a.result, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    sys.exit(1 if any(i["code"] in FAIL_VERDICTS or i["code"] in ("evidence_not_verifiable", "missing_contract_field", "missing_case_id")
                      for i in issues) else 0)


def _norm_of(doc, q):
    """Normalised value of the looked-up cell (scientific notation etc.), if the parser stored one."""
    if not doc or not q or q.get("status") != "ok":
        return None
    t = next((t for t in doc.get("tables", []) if t["id"] == q["table_id"]), None)
    if not t:
        return None
    c = next((c for c in t.get("cells", []) if c["row"] == q["row"] and c["col"] == q["col"]), None)
    return (c or {}).get("norm")


if __name__ == "__main__":
    main()
