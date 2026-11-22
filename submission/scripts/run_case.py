#!/usr/bin/env python3
"""One-shot case runner: locate the case's PDFs, parse them, write a result.json skeleton.

Usage:
    python run_case.py [--case /input/case.json] [--task /input/task.md]
                       [--out /app/output] [--work /app/work] [--pdf extra.pdf ...] [--preview]

After it finishes, fill the answer fields of OUT/result.json using
find_evidence.py, then run verify_result.py. Nothing in the skeleton is an
answer: every contract field starts as null and must be filled from the PDF.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parse_pdf  # noqa: E402

CONTRACT_KEYS = ("output_contract", "outputs_contract", "output_schema", "required_outputs", "output_fields", "outputs")


def find_contract(obj):
    if isinstance(obj, dict):
        for k in CONTRACT_KEYS:
            if k in obj:
                return obj[k]
        for v in obj.values():
            r = find_contract(v)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_contract(v)
            if r is not None:
                return r
    return None


def contract_fields(contract):
    """Normalise many contract shapes into a list of field names."""
    if contract is None:
        return []
    if isinstance(contract, dict):
        for k in ("fields", "required", "required_fields", "properties"):
            if k in contract:
                return contract_fields(contract[k])
        return list(contract.keys())
    if isinstance(contract, list):
        out = []
        for it in contract:
            if isinstance(it, str):
                out.append(it)
            elif isinstance(it, dict):
                n = it.get("name") or it.get("field") or it.get("key") or it.get("id")
                if n:
                    out.append(n)
        return out
    if isinstance(contract, str):
        return [contract]
    return []


def find_pdf_strings(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values():
            find_pdf_strings(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_pdf_strings(v, acc)
    elif isinstance(obj, str) and obj.lower().endswith(".pdf"):
        acc.append(obj)
    return acc


def resolve(p, bases):
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for b in bases:
        q = os.path.join(b, p)
        if os.path.exists(q):
            return q
    hits = []
    for b in bases:
        hits += glob.glob(os.path.join(b, "**", os.path.basename(p)), recursive=True)
    return hits[0] if hits else None


def sha256(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="/input/case.json")
    ap.add_argument("--task", default=None)
    ap.add_argument("--out", default="/app/output")
    ap.add_argument("--work", default="/app/work")
    ap.add_argument("--pdf", nargs="*", default=[])
    ap.add_argument("--password", default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--pages", default=None, help="parse only these pages, e.g. 1-3,8 (unread pages are listed)")
    ap.add_argument("--cache", default=None, help="parse cache dir (default WORK/parse_cache); 'off' disables")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    case, case_err = {}, None
    if os.path.exists(a.case):
        try:
            with open(a.case, encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception as e:
            case_err = f"case.json unreadable: {e}"
    else:
        case_err = f"{a.case} not found"
    case_dir = os.path.dirname(os.path.abspath(a.case))
    task_path = a.task or next((p for p in (os.path.join(case_dir, "task.md"), "/input/task.md", "task.md") if os.path.exists(p)), None)
    bases = [case_dir, "/input", os.getcwd()]
    pdfs = []
    for s in find_pdf_strings(case, []) + list(a.pdf):
        r = resolve(s, bases)
        if r and r not in pdfs:
            pdfs.append(r)
    if not pdfs:
        for b in (case_dir, "/input"):
            for p in sorted(glob.glob(os.path.join(b, "**", "*.pdf"), recursive=True)):
                if p not in pdfs:
                    pdfs.append(p)
    sources, change_log = [], []
    for p in pdfs:
        stem = os.path.splitext(os.path.basename(p))[0]
        pdir = os.path.join(a.out, "parsed", stem)
        cache = None if a.cache == "off" else (a.cache or os.path.join(a.work, "parse_cache"))
        r = parse_pdf.parse(p, pdir, password=a.password, pages=parse_pdf.parse_pages(a.pages), preview=a.preview,
                            cache_dir=cache)
        sources.append({"file": p, "sha256": sha256(p), "parsed_dir": pdir, "parse_status": r["status"],
                        "parse_error": r.get("error"), "page_count": r.get("meta", {}).get("page_count"),
                        "unresolved_pages": r.get("unresolved_pages", []), "ocr_pages": r.get("ocr_pages", []),
                        "image_only_pages": [pg["page"] for pg in r.get("pages", []) if pg.get("page_image")],
                        "pages_parsed": r.get("pages_parsed"), "pages_not_parsed": r.get("pages_not_parsed"),
                        "cache": (r.get("meta") or {}).get("cache"),
                        "warnings": r.get("warnings", [])[:20], "n_warnings": len(r.get("warnings", [])),
                        "stats": r.get("stats", {})})
        change_log.append({"step": "parse", "file": p, "status": r["status"], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    contract = find_contract(case)
    fields = contract_fields(contract)
    case_id = case.get("case_id") or case.get("id") or os.environ.get("CASE_ID")
    result = {"case_id": case_id}
    for f in fields:
        if f not in result:
            result[f] = None
    meta = {
        "status": "pending_agent_fill",
        "sources": sources,
        "evidence": [],
        "uncertainty": [],
        "unreported": [],
        "conflicts": [],
        "field_status": {f: None for f in fields},
        "run_status": {"parse_seconds": round(time.time() - t0, 2), "case_error": case_err,
                       "n_pdfs": len(pdfs), "task_file": task_path},
        "change_log": change_log,
        "_contract_fields": fields,
        "_contract_raw": contract,
        "_fill_instructions": "Fill every contract field from the parsed PDF. Evidence items carry 'field': "
                              "table values -> {field, type:'table_cell', table, page, row{KEY:VALUE}, column:'A/B', value}; "
                              "text -> {field, page, block_id|table_id, quote}; page images -> {field, type:'image_region', page, bbox, crop, transcription}. "
                              "field_status per field: text_layer | image_read | unreadable. "
                              "Use null + an 'unreported' entry when the PDF does not report it or it cannot be read.",
    }
    for k, v in meta.items():
        if k not in result:
            result[k] = v
    if not pdfs:
        result["status"] = "failed"
        result["run_status"]["reason"] = "no PDF found in case.json or the input directory"
    elif all(s["parse_status"] == "failed" for s in sources):
        result["status"] = "failed"
        result["run_status"]["reason"] = "; ".join(f"{s['file']}: {s['parse_error']}" for s in sources)
    with open(os.path.join(a.out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    summary = {"case_id": case_id, "pdfs": pdfs, "contract_fields": fields, "result": os.path.join(a.out, "result.json"),
               "parsed": [s["parsed_dir"] for s in sources], "parse_status": [s["parse_status"] for s in sources],
               "task_file": task_path, "case_error": case_err}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
