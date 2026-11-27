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

Field verdicts (hard fail = exit code 1; soft fail / review = warnings, exit 0):
  Exit 1 only for evidence that contradicts the PDF (value/column/row mismatch, quote not in the PDF,
  answer not in its evidence, value without any evidence, foreign crop) or a broken contract (missing
  case_id / contract field). Soft fails (null_without_reason, contract_type_mismatch, answer_incomplete,
  parsed_cell_incomplete) and review verdicts are listed in `warnings` and never block.
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
                 "image_provenance_mismatch", "value_not_in_evidence", "no_evidence", "null_without_reason",
                 "answer_incomplete", "contract_type_mismatch"}
SOFT_VERDICTS = {"null_without_reason", "contract_type_mismatch", "answer_incomplete", "parsed_cell_incomplete"}
HARD_VERDICTS = FAIL_VERDICTS - SOFT_VERDICTS  # evidence contradicts the PDF / value without evidence
REVIEW_VERDICTS = {"equation_needs_visual_check", "string_match_only", "image_read_unverified", "image_read_unbound", "parsed_cell_incomplete",
                   "supported_by_parsed_cell_region_unchecked", "answer_incomplete", "not_determined"}
META_KEYS = {"case_id", "status", "sources", "evidence", "uncertainty", "unreported", "conflicts", "run_status",
             "change_log", "verification", "field_status", "status_reason", "warnings", "_contract_fields",
             "_contract_required", "_contract_schema", "_contract_raw", "_fill_instructions"}
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
RE_SCI = re.compile(r"([-−]?\d[\d,]*(?:\.\d+)?)\s*[·×x*∗⋅]\s*10\s*(?:\^\s*\{?\s*([-−]?\d+)\s*\}?|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))")
RE_POW = re.compile(r"(?<![\d.])10\s*(?:\^\s*\{?\s*([-−]?\d+)\s*\}?|([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))")
RE_E = re.compile(r"([-−]?\d+(?:\.\d+)?)[eE]([-+]?\d+)")
RE_NUM = re.compile(r"[-−]?\d+(?:[.,]\d+)*")
OPS = "·×*∗⋅^+−/="
UNIT_RE = re.compile(r"^(%|‰|km/s|km\.?/sec\.?|m/s|FLOPs?|GB|MB|KB|ms|s|h|M|B|K|pc|kpc|Mpc|K|°C|dB|Hz|kHz|MHz|GHz|eV|keV|MeV|GeV)\b", re.I)
TYPE_MAP = {"int": "integer", "integer": "integer", "float": "number", "number": "number",
            "str": "string", "string": "string", "text": "string", "bool": "boolean", "boolean": "boolean",
            "list": "array", "array": "array", "dict": "object", "object": "object", "null": "null"}
STOP_FIELD_TOKENS = {"value", "rate", "score", "count", "number", "total", "mean", "result", "answer", "field", "data", "metric", "text", "item"}


def norm(s):
    if s is None:
        s = ""
    elif not isinstance(s, str):
        s = json.dumps(s, ensure_ascii=False)
    s = s.replace("­", "")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = re.sub(r"[\s　]+", "", s).lower()
    return s.replace("’", "'").replace("“", '"').replace("”", '"').replace("−", "-").replace("–", "-")


def _atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.flush()
    with open(tmp, encoding="utf-8") as fh:
        json.load(fh)
    os.replace(tmp, path)


def subschema_for_path(schema_map, dotted_path):
    if not isinstance(schema_map, dict) or not dotted_path:
        return None
    parts = dotted_path.split(".")
    cur = schema_map.get(parts[0])
    for part in parts[1:]:
        if not isinstance(cur, dict):
            return None
        props = cur.get("properties")
        if isinstance(props, dict) and part in props:
            cur = props[part]
        else:
            return None
    return cur


def _matches_type(val, tname):
    if tname == "null":
        return val is None
    if tname == "boolean":
        return isinstance(val, bool)
    if tname == "integer":
        return (isinstance(val, int) and not isinstance(val, bool)) or (
            isinstance(val, float) and val.is_integer()
        )
    if tname == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if tname == "string":
        return isinstance(val, str)
    if tname == "array":
        return isinstance(val, list)
    if tname == "object":
        return isinstance(val, dict)
    return True


def validate_value(val, schema, path=""):
    """Recursively validate val against a JSON-Schema-like dict; return list of error strings."""
    if not schema:
        return []
    if isinstance(schema, str):
        t = TYPE_MAP.get(schema.strip().lower())
        schema = {"type": t} if t else {}
    if not isinstance(schema, dict):
        return []
    errs = []
    raw_t = schema.get("type")
    types = []
    if isinstance(raw_t, list):
        types = [TYPE_MAP.get(str(x).strip().lower()) for x in raw_t if TYPE_MAP.get(str(x).strip().lower())]
    elif isinstance(raw_t, str):
        t = TYPE_MAP.get(raw_t.strip().lower())
        if t:
            types = [t]
    elif isinstance(schema.get("properties"), dict):
        types = ["object"]
    elif "items" in schema:
        types = ["array"]
    if types and not any(_matches_type(val, t) for t in types):
        errs.append(f"{path}: expected {'|'.join(types)}, got {type(val).__name__} ({val!r})")
        return errs
    if isinstance(schema.get("enum"), list) and val not in schema["enum"]:
        errs.append(f"{path}: value {val!r} not in enum {schema['enum']!r}")
    if isinstance(val, list) and "items" in schema:
        items_s = schema["items"]
        if isinstance(items_s, dict):
            for idx, item in enumerate(val):
                errs.extend(validate_value(item, items_s, f"{path}[{idx}]"))
        elif isinstance(items_s, list):
            for idx, sub_s in enumerate(items_s):
                if idx < len(val):
                    errs.extend(validate_value(val[idx], sub_s, f"{path}[{idx}]"))
    if isinstance(val, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        req = schema.get("required") if isinstance(schema.get("required"), list) else []
        for rk in req:
            if rk not in val or val[rk] is None:
                errs.append(f"{path}.{rk}: missing required nested field")
        for pk, pschema in props.items():
            if pk in val and val[pk] is not None:
                errs.extend(validate_value(val[pk], pschema, f"{path}.{pk}"))
    return errs


def get_subfield(val, subfield):
    if not subfield:
        return True, val
    cur = val
    for part in str(subfield).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and 0 <= int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def norm_eq(s):
    """Normalise LaTeX/Unicode equations across brace variants (w_{i}^{T} == w_i^T)."""
    t = norm(s).replace(r"\left", "").replace(r"\right", "")
    t = re.sub(r"([_^])\{([A-Za-z0-9\\]+)\}", r"\1\2", t)
    return t.replace("{", "").replace("}", "")


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
    cell_has_digit = any(ch.isdigit() for ch in (cell_text or ""))
    if right:
        gap = right[0][0] - x1
        nb = right[0]
        nxt = "".join(c[2] for c in right[:12] if c[0] - x1 < 25)
        # a close neighbour continues the value only if something ties them together:
        # the cell ends in an operator, the neighbour is raised/smaller (exponent), is an operator, or touches
        tied = (last and last in OPS + "x(") or nb[3] < 0.85 * size or nb[2] in OPS or gap < 0.15 * size
        if gap < 0.35 * size and tied:
            reasons.append(f"the expression continues right after the cell edge (gap {gap:.1f} pt): next characters {nxt!r}")
        elif gap < 0.45 * size and (cell_has_digit or nb[2].isdigit()) and not (last.isdigit() and (nb[2].isdigit() or nb[2] in "−-+")):
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
        elif gap < 0.45 * size and (cell_has_digit or left[0][2].isdigit()) and not (left[0][2].isdigit() and (cell_text or " ").lstrip()[:1] in "0123456789−-+"):
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


def _verify_target(val, evs, field_name, doc_full_norm):
    """Verify a scalar/composite answer against its evidence list (used for top-level fields and subfields)."""
    eqv = [e for e in evs if e["kind"] == "equation"]
    tab = [e for e in evs if e["kind"] == "table_cell"]
    txt = [e for e in evs if e["kind"] == "text"]
    img = [e for e in evs if e["kind"] == "image_region"]
    v = {"answer": val, "checks": {"value_equals_parsed_cell": None, "source_region_complete": None,
                                   "needs_visual_check": False}, "reasons": []}
    ans_bad = incomplete_reasons(val, "answer") if isinstance(val, str) else []
    if eqv:
        e = eqv[0]
        if not e["ok"]:
            v["verdict"], v["reasons"] = "evidence_unsupported", [e["reason"]]
        else:
            rec = [norm(e.get("expr_unicode") or ""), norm(e.get("expr_latex") or "")]
            rec_eq = [norm_eq(e.get("expr_unicode") or ""), norm_eq(e.get("expr_latex") or "")]
            matches = any(r and r in norm(str(val)) for r in rec) or any(r and r in norm_eq(str(val)) for r in rec_eq)
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
            if isinstance(val, (dict, list)):
                def _leaf_vals(obj):
                    if isinstance(obj, dict):
                        for vv in obj.values():
                            yield from _leaf_vals(vv)
                    elif isinstance(obj, list):
                        for vv in obj:
                            yield from _leaf_vals(vv)
                    elif obj not in (None, ""):
                        yield str(obj)
                leaves = list(_leaf_vals(val))
                nj = norm(joined)
                present = (all(any(close(x, y) for y in values_in(joined)) for x in an) if an else False) or (bool(leaves) and all(norm(lf) in nj for lf in leaves if norm(lf)))
            else:
                present = all(any(close(x, y) for y in values_in(joined)) for x in an) if an else norm(str(val)) in norm(joined)
            if present:
                v["verdict"] = "string_match_only"
                v["checks"]["needs_visual_check"] = True
                multi = max(len(values_in(e["quote"])) for e in ok_txt)
                v["reasons"] = ["answer string occurs in the verified quotes; meaning/attribution not checked"]
                if an and multi > 1:
                    v["reasons"].append(f"a quote holds {multi} numbers, so which one is meant is not verified (use table_cell evidence)")
                if an and doc_full_norm:
                    ftoks = [t for t in re.split(r"[^a-z0-9]+", field_name.lower())
                             if len(t) >= 4 and t not in STOP_FIELD_TOKENS]
                    nj = norm(joined)
                    missing_kw = [t for t in ftoks if t not in nj and t in doc_full_norm]
                    if missing_kw:
                        v["checks"]["quote_keyword_mismatch"] = True
                        v["reasons"].append(f"field keyword(s) {missing_kw} appear in the PDF but are absent from the cited quote")
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
    return v


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
    parsed_dirs = a.parsed or [s.get("parsed_dir") for s in res.get("sources", []) if isinstance(s, dict) and s.get("parsed_dir")]
    parsed_dirs = [d for d in parsed_dirs if d and os.path.exists(os.path.join(d, "doc.json"))]
    if not parsed_dirs:  # moved result folder: look for parsed/<stem>/doc.json next to result.json
        parsed_dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(rdir, "parsed", "*", "doc.json"))]
    docs = []
    for d in parsed_dirs:
        with open(os.path.join(d, "doc.json"), encoding="utf-8") as fh:
            docs.append(json.load(fh))
    src = Sources(res, rdir, docs, a.pdf)
    blocks_by_id, tables_by_id, pages_by_no, page_sizes = {}, {}, {}, {}
    doc_text_chunks = []
    for d in docs:
        dsha = (d.get("meta") or {}).get("sha256")
        for b in d.get("blocks", []) + d.get("headers_footers", []):
            bt = b["text"] + " " + (b.get("marked") or "")
            blocks_by_id.setdefault(b["id"], []).append((dsha, bt))
            pages_by_no.setdefault(b["page"], []).append((dsha, b["text"]))
            doc_text_chunks.append(b["text"])
        for t in d.get("tables", []):
            tt = " ".join(" ".join(c or "" for c in r) for r in t["rows"]) + " " + (t.get("caption") or "")
            tables_by_id.setdefault(t["id"], []).append((dsha, tt))
            pages_by_no.setdefault(t["page"], []).append((dsha, tt))
            doc_text_chunks.append(tt)
        for p in d.get("pages", []):
            page_sizes[p["page"]] = [p.get("width"), p.get("height")]
    doc_full_norm = norm(" ".join(doc_text_chunks))

    issues, details, contract_gaps = [], [], []
    if res.get("case_id") in (None, ""):
        issues.append({"code": "missing_case_id"})
    fields = res.get("_contract_fields") or [k for k in res if k not in META_KEYS]
    req_fields = set(res["_contract_required"]) if isinstance(res.get("_contract_required"), list) else set(fields)
    schema_map = res.get("_contract_schema") if isinstance(res.get("_contract_schema"), dict) else {}
    for f in fields:
        if f not in res:
            issues.append({"code": "missing_contract_field", "field": f})
            if f in req_fields:
                contract_gaps.append({"field": f, "code": "missing_contract_field", "detail": "required contract field missing"})

    by_field = {}
    n_quotes = n_quotes_ok = 0
    search_bases = [rdir] + parsed_dirs
    for path, e in iter_evidence(res):
        kind = ev_kind(e)
        d = {"path": path, "field": ev_field(e), "subfield": e.get("subfield"), "kind": kind}
        esrc = e.get("source_sha256")
        doc_cands = ([doc for doc in docs if (doc.get("meta") or {}).get("sha256") == esrc] if esrc else []) or docs
        if kind == "table_cell":
            spec = e.get("table_id") or e.get("table")
            col = e["column"] if isinstance(e["column"], str) else "/".join(e["column"])
            rows = [f"{k}={v}" for k, v in (e.get("row") or {}).items()]
            q, qdoc = None, None
            for doc in doc_cands:
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
            eqs = [x for d0 in doc_cands for x in d0.get("equations", []) if str(x.get("number")) == num
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
            nqt = norm(qt)
            found = None
            if bid and bid in blocks_by_id and any(nqt in norm(bt) for dsha, bt in blocks_by_id[bid] if not esrc or dsha == esrc):
                found = f"block {bid}"
            elif tid and tid in tables_by_id and any(nqt in norm(tt) for dsha, tt in tables_by_id[tid] if not esrc or dsha == esrc):
                found = f"table {tid}"
            elif pg is not None and int(pg) in pages_by_no and any(nqt in norm(pt) for dsha, pt in pages_by_no[int(pg)] if not esrc or dsha == esrc):
                found = f"page {pg}"
            if not qt:
                d.update(ok=False, reason="no quote")
            elif found:
                n_quotes_ok += 1
                d.update(ok=True, found_in=found)
            else:
                elsewhere = next((k for k, lst in blocks_by_id.items() if any(nqt in norm(bt) for dsha, bt in lst if not esrc or dsha == esrc)), None)
                d.update(ok=False, reason=("quote exists but not at the cited location (found in " + elsewhere + ")")
                         if elsewhere else "quote not found in parsed PDF", quote=qt[:200])
            d["quote"] = qt
        details.append(d)
        if not d["ok"]:
            issues.append({"code": "evidence_not_verifiable", "path": path, "detail": d.get("reason")})
        by_field.setdefault(d["field"], []).append(d)

    unreported_list = res.get("unreported", []) if isinstance(res.get("unreported"), list) else []
    unreported_txt = json.dumps(unreported_list, ensure_ascii=False)
    verdicts = {}
    for f in fields:
        val = res.get(f)
        evs = by_field.get(f, [])
        if f in META_KEYS:
            # contract names a key the Skill maintains itself (case_id, status, evidence, sources, uncertainty...)
            ok = f != "case_id" or (val is not None and val != "")
            verdicts[f] = {"verdict": "meta_field" if ok else "null_without_reason", "needs_review": False,
                           "reasons": ["maintained by the Skill scripts (identifier / status / evidence / provenance lists)"]}
            if not ok:
                issues.append({"code": "null_without_reason", "field": f, "detail": "case_id is empty"})
                contract_gaps.append({"field": f, "code": "null_without_reason", "detail": "case_id is empty"})
            continue
        if val in (None, "", [], {}):
            un_items = [u for u in unreported_list if isinstance(u, dict) and u.get("field") == f]
            if un_items:
                if any(u.get("kind") == "not_determined" for u in un_items):
                    verdicts[f] = {"verdict": "not_determined", "needs_review": True,
                                   "reasons": [u.get("reason") for u in un_items if u.get("reason")]}
                else:
                    verdicts[f] = {"verdict": "unreported_ok", "needs_review": False}
            elif f in unreported_txt:
                verdicts[f] = {"verdict": "unreported_ok", "needs_review": False}
            elif f not in req_fields:
                verdicts[f] = {"verdict": "optional_omitted", "needs_review": False,
                               "reasons": ["optional contract field omitted"]}
            else:
                verdicts[f] = {"verdict": "null_without_reason", "needs_review": False}
                contract_gaps.append({"field": f, "code": "null_without_reason", "detail": "required contract field is empty without reason"})
            if verdicts[f]["verdict"] in FAIL_VERDICTS:
                issues.append({"code": verdicts[f]["verdict"], "field": f, "detail": "; ".join(verdicts[f].get("reasons", [])) or verdicts[f]["verdict"]})
            continue

        # Validate against contract type/nested schema if defined
        f_schema = schema_map.get(f)
        schema_errs = validate_value(val, f_schema, f) if f_schema else []
        if schema_errs:
            issues.append({"code": "contract_type_mismatch", "field": f, "detail": "; ".join(schema_errs)})
            contract_gaps.append({"field": f, "code": "contract_type_mismatch", "detail": "; ".join(schema_errs)})

        sub_evs = [e for e in evs if e.get("subfield")]
        if sub_evs:
            by_sf = {}
            for e in sub_evs:
                by_sf.setdefault(e["subfield"], []).append(e)
            sf_verdicts = {}
            for sf, sevs in by_sf.items():
                found_sf, sf_val = get_subfield(val, sf)
                if not found_sf:
                    sf_verdicts[sf] = {"verdict": "evidence_unsupported", "needs_review": False,
                                       "checks": {}, "reasons": [f"subfield {sf!r} missing from answer"]}
                else:
                    sf_verdicts[sf] = _verify_target(sf_val, sevs, f"{f}.{sf}", doc_full_norm)
            if isinstance(val, dict):
                for k in val:
                    if val[k] not in (None, "", [], {}) and k not in by_sf and not any(s.startswith(k + ".") for s in by_sf):
                        whole_evs = [e for e in evs if not e.get("subfield")]
                        if not whole_evs:
                            sf_verdicts[k] = {"verdict": "no_evidence", "needs_review": False,
                                              "checks": {}, "reasons": [f"subfield {k!r} has no evidence"]}
            all_sf = list(sf_verdicts.values())
            failing_sf = [sv for sv in all_sf if sv["verdict"] in FAIL_VERDICTS]
            review_sf = [sv for sv in all_sf if sv.get("needs_review")]
            chosen = failing_sf[0] if failing_sf else (review_sf[0] if review_sf else all_sf[0])
            v = {
                "answer": val,
                "verdict": chosen["verdict"],
                "needs_review": any(sv.get("needs_review") for sv in all_sf),
                "checks": dict(chosen.get("checks") or {},
                               quote_keyword_mismatch=any((sv.get("checks") or {}).get("quote_keyword_mismatch") for sv in all_sf)),
                "reasons": [r for sv in all_sf for r in sv.get("reasons", [])],
                "subfield_verdicts": {sf: sv["verdict"] for sf, sv in sf_verdicts.items()},
            }
        else:
            v = _verify_target(val, evs, f, doc_full_norm)
        if schema_errs:
            v.setdefault("reasons", []).extend(schema_errs)
            if v.get("verdict") not in HARD_VERDICTS:
                v["verdict"] = "contract_type_mismatch"
        verdicts[f] = v
        if v["verdict"] in FAIL_VERDICTS:
            issues.append({"code": v["verdict"], "field": f, "detail": "; ".join(v.get("reasons", []))})

    for d in details:
        d.pop("_q", None)
        d.pop("_doc", None)
    review_reasons = {f: v.get("reasons", []) for f, v in verdicts.items() if v.get("needs_review") and v.get("reasons")}
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
        "review_reasons": review_reasons,
        "contract_gaps": contract_gaps,
        "issues": issues, "parsed_docs": parsed_dirs,
        "evidence_checked": len(details), "evidence_verified": sum(1 for d in details if d["ok"]),
    }
    out = a.out or os.path.join(rdir, "verification.json")
    _atomic_write_json(out, {"summary": summary, "details": details})
    blocking = any(i["code"] in HARD_VERDICTS or i["code"] in ("evidence_not_verifiable", "missing_contract_field", "missing_case_id")
                   for i in issues)
    warnings = [f"{i.get('field') or i.get('path')}: {i['code']}" for i in issues if i["code"] in SOFT_VERDICTS]
    warnings += [f"{f}: {v.get('verdict')} (review)" for f, v in verdicts.items() if v.get("needs_review")]
    warnings += [f"{f}: quote/metric keyword mismatch (review)" for f, v in verdicts.items()
                 if (v.get("checks") or {}).get("quote_keyword_mismatch")]
    summary["warnings"] = warnings
    summary["blocking"] = blocking
    _atomic_write_json(out, {"summary": summary, "details": details})
    if not blocking and res.get("status") == "pending_agent_fill":
        res["status"] = "success"  # review items are warnings; fill_result.py finalize sets the final status
    res.pop("_contract_raw", None)
    res.pop("_fill_instructions", None)
    res["verification"] = {"evidence_items": len(details), "evidence_ok": summary["evidence_ok"],
                           "field_verdicts": {f: v["verdict"] for f, v in verdicts.items()},
                           "needs_review": summary["needs_review"],
                           "review_reasons": review_reasons,
                           "contract_gaps": contract_gaps,
                           "warnings": warnings, "blocking": blocking,
                           "n_issues": len(issues), "report": os.path.basename(out)}
    _atomic_write_json(a.result, res)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    sys.exit(1 if blocking else 0)


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
