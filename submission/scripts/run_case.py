#!/usr/bin/env python3
"""One-shot case runner: locate the case's PDFs, parse them, write a result.json skeleton.

Usage:
    python run_case.py [--case /input/case.json] [--task /input/task.md]
                       [--out /app/output] [--work /app/work] [--pdf extra.pdf ...] [--preview]

After it finishes, fill the answer fields with fill_result.py (put / unreported /
finalize) - never by editing result.json by hand. Nothing in the skeleton is an
answer: every contract field starts as null and must be filled from the PDF.
Per PDF it writes parsed/<stem>/outline.md (one compact map of the paper: sections
with pages, tables with column paths and row keys, equations, figures, pages that
need visual reading) - read it first instead of the whole document.md.
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

META_FIELDS = {"case_id", "status", "sources", "evidence", "uncertainty", "unreported", "conflicts", "run_status",
               "change_log", "field_status", "warnings"}
CONTRACT_PRIORITY = ("output_contract", "outputs_contract", "output_schema", "schema",
                     "required_outputs", "output_fields", "outputs", "deliverables")
CONTRACT_KEYS = set(CONTRACT_PRIORITY)
TYPE_ALIASES = {"int": "integer", "integer": "integer", "float": "number", "number": "number",
                "str": "string", "string": "string", "text": "string", "bool": "boolean", "boolean": "boolean",
                "list": "array", "array": "array", "dict": "object", "object": "object"}
SCHEMA_META_KEYS = {"type", "title", "description", "$schema", "additionalProperties", "required", "properties", "items", "enum"}
PRIMARY_ROLES = {"primary", "main", "source", "target", "paper", "manuscript"}
ATTACHMENT_ROLES = {"attachment", "supplement", "supplementary", "appendix", "supporting", "secondary", "reference"}
PRIMARY_CONTAINERS = {"primary", "primary_pdf", "main", "main_pdf", "paper", "document", "target"}
ATTACHMENT_CONTAINERS = {"attachment", "attachments", "supplement", "supplements", "supplementary", "appendix", "appendices", "supporting"}
NON_PDF_EXTS = (".md", ".markdown", ".txt", ".rst", ".tex", ".docx", ".doc", ".html", ".htm")


def _collect_contracts(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in CONTRACT_KEYS and v is not None:
                out.append((k, v))
            else:
                _collect_contracts(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_contracts(v, out)


def find_contract(obj):
    found = []
    _collect_contracts(obj, found)
    if not found:
        return None
    rank = {k: i for i, k in enumerate(CONTRACT_PRIORITY)}
    found.sort(key=lambda kv: rank.get(kv[0], 99))
    return found[0][1] if len(found) == 1 else {k: v for k, v in found}


def _norm_schema_node(node):
    if isinstance(node, str):
        t = TYPE_ALIASES.get(node.strip().lower())
        return {"type": t} if t else {}
    if isinstance(node, dict):
        out = dict(node)
        if isinstance(out.get("type"), str):
            t = TYPE_ALIASES.get(out["type"].strip().lower())
            if t:
                out["type"] = t
        elif "type" not in out:
            if isinstance(out.get("properties"), dict):
                out["type"] = "object"
            elif "items" in out:
                out["type"] = "array"
        if isinstance(out.get("properties"), dict):
            out["properties"] = {pk: _norm_schema_node(pv) for pk, pv in out["properties"].items()}
        if isinstance(out.get("items"), dict):
            out["items"] = _norm_schema_node(out["items"])
        elif isinstance(out.get("items"), list):
            out["items"] = [_norm_schema_node(x) for x in out["items"]]
        return out
    return {}


def _parse_single_contract(contract):
    """Return (fields_order, required_set, schemas_dict) for one contract node."""
    fields, req, schemas = [], set(), {}

    def add_field(name, schema, is_required):
        if not isinstance(name, str) or not name:
            return
        if name not in schemas:
            fields.append(name)
            schemas[name] = schema
        elif schema and not schemas[name]:
            schemas[name] = schema
        if is_required:
            req.add(name)

    if contract is None:
        return fields, req, schemas
    if isinstance(contract, str):
        add_field(contract, {}, True)
        return fields, req, schemas
    if isinstance(contract, list):
        for it in contract:
            if isinstance(it, str):
                add_field(it, {}, True)
            elif isinstance(it, dict):
                n = it.get("name") or it.get("field") or it.get("key") or it.get("id")
                if n:
                    sub = it.get("schema") if isinstance(it.get("schema"), dict) else (
                        it if any(k in it for k in ("type", "properties", "items", "enum")) else {})
                    is_req = it.get("required", True)
                    if not isinstance(is_req, bool):
                        is_req = True
                    add_field(n, _norm_schema_node(sub), is_req)
        return fields, req, schemas
    if isinstance(contract, dict):
        has_props = isinstance(contract.get("properties"), dict)
        has_fields = "fields" in contract
        has_req = isinstance(contract.get("required"), list)
        has_req_fields = "required_fields" in contract
        has_opt = isinstance(contract.get("optional"), list) or isinstance(contract.get("optional_fields"), list)
        if has_props or has_fields or has_req or has_req_fields or has_opt:
            if has_props:
                props = contract["properties"]
                req_list = contract.get("required") if has_req else None
                for pk, pv in props.items():
                    ps = _norm_schema_node(pv)
                    if req_list is not None:
                        is_req = pk in req_list
                    else:
                        is_req = pv.get("required", True) if isinstance(pv, dict) and isinstance(pv.get("required"), bool) else True
                    add_field(pk, ps, is_req)
                if req_list:
                    for rn in req_list:
                        if isinstance(rn, str):
                            add_field(rn, {}, True)
            if has_fields:
                f2, r2, s2 = _parse_single_contract(contract["fields"])
                req_override = set(contract["required"]) if has_req else None
                for fn in f2:
                    add_field(fn, s2.get(fn, {}), (fn in req_override) if req_override is not None else (fn in r2))
            if has_req_fields:
                f2, r2, s2 = _parse_single_contract(contract["required_fields"])
                for fn in f2:
                    add_field(fn, s2.get(fn, {}), fn in r2)
            elif has_req and not has_props and not has_fields:
                f2, r2, s2 = _parse_single_contract(contract["required"])
                for fn in f2:
                    add_field(fn, s2.get(fn, {}), True)
            for opt_k in ("optional", "optional_fields"):
                if isinstance(contract.get(opt_k), list):
                    f2, _, s2 = _parse_single_contract(contract[opt_k])
                    for fn in f2:
                        add_field(fn, s2.get(fn, {}), False)
            return fields, req, schemas
        # empty JSON schema wrapper like {"type": "object"}
        if contract.get("type") == "object" and set(contract.keys()) <= SCHEMA_META_KEYS:
            return fields, req, schemas
        # flat dict {field_name: type_or_schema}
        for k, v in contract.items():
            if isinstance(v, dict):
                is_req = v.get("required", True) if isinstance(v.get("required"), bool) else True
                add_field(k, _norm_schema_node(v), is_req)
            elif isinstance(v, str):
                add_field(k, _norm_schema_node(v), True)
            else:
                add_field(k, {}, True)
    return fields, req, schemas


def normalize_contracts(case_obj):
    """Collect and merge all contract wrappers in case_obj -> (fields, required, schemas, raw, conflicts)."""
    found = []
    _collect_contracts(case_obj, found)
    if not found:
        return [], [], {}, None, []
    rank = {k: i for i, k in enumerate(CONTRACT_PRIORITY)}
    found.sort(key=lambda kv: rank.get(kv[0], 99))
    fields, req_set, schemas, conflicts = [], set(), {}, []
    for cname, cval in found:
        f2, r2, s2 = _parse_single_contract(cval)
        for fn in f2:
            ns = s2.get(fn, {})
            if fn not in schemas:
                fields.append(fn)
                schemas[fn] = ns
            else:
                old_t = schemas[fn].get("type")
                new_t = ns.get("type")
                if old_t and new_t and old_t != new_t:
                    conflicts.append({"field": fn, "kept_from": "higher_priority", "kept_type": old_t,
                                      "ignored_from": cname, "ignored_type": new_t})
                elif not schemas[fn] and ns:
                    schemas[fn] = ns
            if fn in r2:
                req_set.add(fn)
    req_list = [f for f in fields if f in req_set]
    raw = found[0][1] if len(found) == 1 else {k: v for k, v in found}
    return fields, req_list, schemas, raw, conflicts


def contract_fields(contract):
    """Normalise many contract shapes into a list of field names."""
    fields, _, _ = _parse_single_contract(contract)
    return fields


def _extract_case_id(case):
    if isinstance(case, dict):
        for k in ("case_id", "id"):
            if k in case and case[k] not in (None, ""):
                return case[k]
        for subk in ("case", "meta", "metadata", "task"):
            sub = case.get(subk)
            if isinstance(sub, dict):
                for k in ("case_id", "id"):
                    if k in sub and sub[k] not in (None, ""):
                        return sub[k]
    env_id = os.environ.get("CASE_ID")
    return env_id if env_id not in (None, "") else None


def _role_from_dict(obj, inherited):
    for k in ("role", "kind", "type"):
        v = obj.get(k)
        if isinstance(v, str):
            vl = v.strip().lower()
            if vl in PRIMARY_ROLES:
                return "primary"
            if vl in ATTACHMENT_ROLES:
                return "attachment"
    for k in ("primary", "is_primary", "main"):
        if obj.get(k) is True:
            return "primary"
    return inherited


def collect_declared_sources(obj, key_ctx="", role_ctx="unspecified", acc=None):
    """Walk case.json keeping source role (primary vs attachment) and format."""
    if acc is None:
        acc = []
    if isinstance(obj, dict):
        role = _role_from_dict(obj, role_ctx)
        for k, v in obj.items():
            kl = k.lower()
            if kl in CONTRACT_KEYS:
                continue
            next_role = role
            if kl in PRIMARY_CONTAINERS:
                next_role = "primary"
            elif kl in ATTACHMENT_CONTAINERS:
                next_role = "attachment"
            collect_declared_sources(v, kl, next_role, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect_declared_sources(v, key_ctx, role_ctx, acc)
    elif isinstance(obj, str):
        low = obj.strip().lower()
        if low.endswith(".pdf"):
            acc.append({"raw": obj.strip(), "role": role_ctx, "format": "pdf"})
        elif any(low.endswith(ext) for ext in NON_PDF_EXTS) and key_ctx not in ("task_file", "task", "output", "out"):
            acc.append({"raw": obj.strip(), "role": role_ctx, "format": "non_pdf"})
    return acc


def find_pdf_strings(obj, acc):
    for item in collect_declared_sources(obj):
        if item["format"] == "pdf":
            acc.append(item["raw"])
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


def _short(t, n=90):
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[:n - 1] + "…"


def build_structure(doc, pdir):
    """structure_summary.json: document skeleton for structure tasks and for navigating long papers."""
    blocks = doc.get("blocks", [])
    secs = doc.get("sections", [])
    pages = [p["page"] for p in doc.get("pages", [])]
    last_page = max(pages) if pages else None
    out_secs = []
    for i, s in enumerate(secs):
        nxt = next((x for x in secs[i + 1:] if x.get("level", 1) <= s.get("level", 1)), None)
        out_secs.append({"id": s.get("id"), "number": s.get("number"), "title": s.get("title"), "level": s.get("level"),
                         "page": s.get("page"), "end_page": nxt["page"] if nxt else last_page, "block_id": s.get("block_id"),
                         "parent": s.get("parent"), "kind": s.get("kind")})
    from table_query import columns_of
    tabs = []
    for t in doc.get("tables", []):
        nh = (t.get("header") or {}).get("n_header_rows", 1)
        keys = t.get("key_columns", [0])
        tabs.append({"id": t.get("id"), "page": t.get("page"), "caption": t.get("caption"), "n_rows": len(t.get("rows") or []),
                     "n_body_rows": max(0, len(t.get("rows") or []) - nh),
                     "n_cols": t.get("n_cols"), "continued_from": t.get("continued_from"),
                     "column_paths": ["/".join(c["path"]) or c["name"] for c in columns_of(t)],
                     "row_keys": [" | ".join(str(t["rows"][r][k]) for k in keys if k < len(t["rows"][r]))
                                  for r in range(nh, len(t.get("rows") or []))][:40]})
    abstract = [b for b in blocks if b.get("role") == "abstract"]
    if not abstract:
        hd = next((b for b in blocks if b.get("role") == "heading" and b["text"].strip().lower().startswith("abstract")), None)
        if hd:
            k = blocks.index(hd)
            abstract = [b for b in blocks[k + 1:k + 3] if b.get("role") == "body"][:1]
    page_map = {}
    for b in blocks:
        if b.get("role") in ("title", "heading", "caption", "abstract"):
            page_map.setdefault(b["page"], []).append(_short(b["text"], 50))
    return {
        "title": [b["text"] for b in blocks if b.get("role") == "title"],
        "authors_affiliations": [{"block_id": b["id"], "text": _short(b["text"], 300)} for b in blocks if b.get("role") == "author_affiliation"][:12],
        "abstract": [{"block_id": b["id"], "page": b["page"], "text": b["text"]} for b in abstract][:3],
        "sections": out_secs,
        "tables": tabs,
        "tables_as_images": [{k: t.get(k) for k in ("id", "page", "caption", "bbox", "crop")} for t in doc.get("tables_as_images", []) or []],
        "figures": [{"id": f.get("id"), "page": f.get("page"), "kind": f.get("kind"), "caption": _short(f.get("caption"), 160)}
                    for f in doc.get("figures", []) if f.get("caption")],
        "equations": [{"number": e.get("number"), "page": e.get("page"), "status": e.get("transcription_status"),
                       "latex": e.get("expr_latex") or e.get("latex"), "crop": e.get("crop")} for e in doc.get("equations", [])],
        "footnotes": len(doc.get("footnotes", [])),
        "n_references": len(doc.get("references", [])),
        "references_sample": [_short(r.get("text"), 120) for r in doc.get("references", [])[:3]],
        "page_map": {str(k): v[:8] for k, v in sorted(page_map.items())},
        "pages_without_text_layer": sorted(doc.get("unresolved_pages", [])),
        "pages_with_pictures_without_text": sorted(p["page"] for p in doc.get("pages", []) if p.get("raster_regions")),
        "warning_codes": sorted({w.get("code") for w in doc.get("warnings", [])}),
        "document_markdown": os.path.join(pdir, "document.md"),
    }


def write_outline(doc, st, pdir, skill_scripts):
    """outline.md: one small file that tells the agent where everything is (budget: read this, then query)."""
    meta = doc.get("meta", {})
    L = [f"# Outline: {_short(' '.join(st['title']) or os.path.basename(meta.get('source_file', '')), 140)}",
         f"pages {meta.get('page_count')} | parse status {doc.get('status')} | parsed pages "
         f"{len(doc.get('pages_parsed') or [])}" + (f" | NOT parsed: {doc.get('pages_not_parsed')}" if doc.get("pages_not_parsed") else ""),
         f"parsed dir: {pdir}"]
    if st["warning_codes"]:
        L.append("warnings: " + ", ".join(st["warning_codes"]) + "  (details: qa_report.json)")
    if st["pages_without_text_layer"]:
        L.append(f"SCANNED pages (no text layer, no OCR): {st['pages_without_text_layer']} -> pages[].page_image + image_regions; "
                 "read with crop_region.py, cite as image_region")
    if st["pages_with_pictures_without_text"]:
        L.append(f"pages with pictures whose content is not in the text layer: {st['pages_with_pictures_without_text']} "
                 "(figures/tables as images; see pages[].raster_regions in doc.json)")
    if st["abstract"]:
        a = st["abstract"][0]
        L.append(f"abstract: {a['block_id']} p{a['page']}: {_short(a['text'], 220)}")
    L.append("\n## Sections (number title | pages | block)")
    for s in st["sections"]:
        pad = "  " * max(0, (s.get("level") or 1) - 1)
        rng = f"p{s['page']}" + (f"-{s['end_page']}" if s.get("end_page") and s["end_page"] != s["page"] else "")
        L.append(f"{pad}- {(s.get('number') or '').strip()} {_short(s.get('title'), 70)} | {rng} | {s.get('block_id')}")
    if st["tables"]:
        L.append('\n## Tables (query: table_query.py PARSED --table ID|"Table N" --row KEY=VALUE --col PATH)')
        for t in st["tables"]:
            cont = f" (continues {t['continued_from']})" if t.get("continued_from") else ""
            L.append(f"- {t['id']} p{t['page']}{cont}: {_short(t.get('caption'), 90)}")
            L.append(f"    cols: {_short(' ; '.join(t['column_paths']), 300)}")
            if t["row_keys"]:
                n_show = min(12, len(t["row_keys"]))
                hdr = (f"body rows (first {n_show}/{t['n_body_rows']}; full: table_query.py --table {t['id']} --list)"
                       if t["n_body_rows"] > 12 else f"body rows ({t['n_body_rows']})")
                L.append(f"    {hdr}: {_short(' ; '.join(t['row_keys'][:12]), 300)}")
    if st["tables_as_images"]:
        L.append("\n## Tables that are images (read the crop; cite as image_region)")
        for t in st["tables_as_images"]:
            L.append(f"- {t['id']} p{t['page']} bbox {t['bbox']}: {_short(t.get('caption'), 90)} -> {t.get('crop')}")
    if st["equations"]:
        L.append("\n## Numbered equations (find_evidence.py PARSED --equation N)")
        L.append(", ".join(f"({e['number']}) p{e['page']} {e['status']}" for e in st["equations"] if e.get("number"))[:1500] or "(none numbered)")
    if st["figures"]:
        L.append("\n## Figures")
        for f in st["figures"][:40]:
            L.append(f"- {f['id']} p{f['page']} [{f['kind']}]: {_short(f['caption'], 100)}")
    L.append(f"\nreferences: {st['n_references']} entries | footnotes: {st['footnotes']}")
    L.append('full text in reading order with anchors: document.md ; text search: find_evidence.py PARSED "words" --top 5')
    with open(os.path.join(pdir, "outline.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return os.path.join(pdir, "outline.md")


def sha256(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.flush()
    with open(tmp, encoding="utf-8") as fh:
        json.load(fh)
    os.replace(tmp, path)


def _has_existing_work(res):
    if not isinstance(res, dict):
        return False
    if res.get("status") in ("success", "partial", "refused"):
        return True
    if res.get("evidence") or res.get("unreported"):
        return True
    fields = res.get("_contract_fields") or [k for k in res if k not in META_FIELDS and not k.startswith("_")]
    for f in fields:
        if f not in META_FIELDS and res.get(f) not in (None, "", [], {}):
            return True
    return False


def _load_existing_result(rpath):
    if not os.path.exists(rpath):
        return None
    try:
        with open(rpath, encoding="utf-8-sig") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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
    ap.add_argument("--budget", type=float, default=600.0, help="cumulative OCR budget in seconds (default 600)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing filled result.json")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    rpath = os.path.join(a.out, "result.json")
    existing = _load_existing_result(rpath)
    preserve_existing = (not a.force) and _has_existing_work(existing)

    def write_skeleton(obj):
        if preserve_existing:
            return
        atomic_write_json(rpath, obj)
        for sk in (os.path.join(a.out, ".result.skeleton.json"), os.path.join(a.work, "result.skeleton.json")):
            try:
                atomic_write_json(sk, obj)
            except Exception:
                pass

    # Validate --pages and --budget before touching anything
    pages_arg = None
    early_err = None
    if a.pages is not None:
        try:
            pages_arg = parse_pdf.parse_pages(a.pages)
        except ValueError as ex:
            early_err = str(ex)
    if a.budget is not None and a.budget <= 0:
        early_err = early_err or f"invalid --budget: {a.budget} (must be > 0)"

    case, case_err, case_fatal = {}, None, False
    if os.path.exists(a.case):
        try:
            with open(a.case, encoding="utf-8-sig") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                case_err = f"case.json top level must be a JSON object, got {type(loaded).__name__}"
                case_fatal = True
            else:
                case = loaded
        except Exception as e:
            case_err = f"case.json unreadable: {e}"
            case_fatal = True
    else:
        case_err = f"{a.case} not found"

    case_dir = os.path.dirname(os.path.abspath(a.case))
    task_path = a.task or next((p for p in (os.path.join(case_dir, "task.md"), "/input/task.md", "task.md") if os.path.exists(p)), None)
    bases = [case_dir, "/input", os.getcwd()]

    fields, required_fields, schema_map, contract_raw, contract_conflicts = normalize_contracts(case)
    case_id = _extract_case_id(case)

    if early_err:
        if not preserve_existing:
            fail_res = {
                "case_id": case_id, "status": "failed", "sources": [], "evidence": [], "uncertainty": [],
                "unreported": [], "conflicts": [], "field_status": {f: None for f in fields},
                "run_status": {"parse_seconds": round(time.time() - t0, 2), "case_error": case_err,
                               "reason": early_err, "n_pdfs": 0, "task_file": task_path},
                "change_log": [{"step": "run_case", "status": "failed", "error": early_err,
                                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
                "_contract_fields": fields, "_contract_required": required_fields, "_contract_schema": schema_map,
            }
            for f in fields:
                if f not in fail_res:
                    fail_res[f] = None
            write_skeleton(fail_res)
        print(json.dumps({"ok": False, "status": "failed", "error": early_err, "preserved_existing_result": preserve_existing},
                         ensure_ascii=False, indent=1))
        sys.exit(1)

    # Collect declared sources with role priority (primary < unspecified < attachment)
    role_rank = {"primary": 0, "unspecified": 1, "attachment": 2}
    declared = collect_declared_sources(case) if not case_err else []
    cli_items = [{"raw": s, "role": "unspecified", "format": "pdf", "cli": True} for s in a.pdf]
    missing_explicit = []
    missing_cli = []
    pdf_entries = {}  # abs_path -> {path, role, rank, order, raw}
    non_pdf_sources = []
    order_seq = 0
    for item in declared + cli_items:
        order_seq += 1
        r = resolve(item["raw"], bases)
        if not r:
            if item.get("cli"):
                missing_cli.append(item["raw"])
                missing_explicit.append(item["raw"])
            elif item["format"] == "pdf":
                missing_explicit.append(item["raw"])
            continue
        ap_r = os.path.abspath(r)
        rk = role_rank.get(item["role"], 1)
        if item["format"] == "pdf":
            prev = pdf_entries.get(ap_r)
            if prev is None or rk < prev["rank"]:
                pdf_entries[ap_r] = {"path": r, "role": item["role"], "rank": rk,
                                     "order": prev["order"] if prev else order_seq, "raw": item["raw"]}
        else:
            if not any(s.get("file") == r for s in non_pdf_sources):
                non_pdf_sources.append({
                    "file": r, "sha256": sha256(r), "role": item["role"], "format": "non_pdf",
                    "parsed_dir": None, "parse_status": "non_pdf_source",
                })

    if missing_cli:
        err_msg = f"explicit --pdf not found: {', '.join(missing_cli)}"
        if not preserve_existing:
            fail_res = {
                "case_id": case_id, "status": "failed", "sources": non_pdf_sources, "evidence": [], "uncertainty": [],
                "unreported": [], "conflicts": [], "field_status": {f: None for f in fields},
                "run_status": {"parse_seconds": round(time.time() - t0, 2), "case_error": case_err,
                               "reason": err_msg, "missing_pdfs": missing_explicit, "n_pdfs": 0, "task_file": task_path},
                "change_log": [{"step": "run_case", "status": "failed", "error": err_msg,
                                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}],
                "_contract_fields": fields, "_contract_required": required_fields, "_contract_schema": schema_map,
            }
            for f in fields:
                if f not in fail_res:
                    fail_res[f] = None
            write_skeleton(fail_res)
        print(json.dumps({"ok": False, "status": "failed", "error": err_msg, "preserved_existing_result": preserve_existing},
                         ensure_ascii=False, indent=1))
        sys.exit(1)

    declared_pdfs_count = sum(1 for item in declared if item["format"] == "pdf")
    if not pdf_entries and not declared_pdfs_count and not a.pdf and not case_err:
        for b in (case_dir, "/input"):
            for p in sorted(glob.glob(os.path.join(b, "**", "*.pdf"), recursive=True)):
                ap_p = os.path.abspath(p)
                if ap_p not in pdf_entries:
                    order_seq += 1
                    pdf_entries[ap_p] = {"path": p, "role": "unspecified", "rank": 1, "order": order_seq, "raw": p}

    ordered_pdfs = sorted(pdf_entries.values(), key=lambda x: (x["rank"], x["order"]))
    pdfs = [x["path"] for x in ordered_pdfs]

    # Disambiguate parsed/<source_id> when different PDFs share a filename stem
    pdf_shas = [sha256(p) for p in pdfs]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in pdfs]
    stem_counts = {}
    for st_name in stems:
        stem_counts[st_name] = stem_counts.get(st_name, 0) + 1

    sources, change_log, outlines = [], [], []
    budget_deadline = t0 + a.budget if a.budget is not None else None
    for idx, entry in enumerate(ordered_pdfs):
        p = entry["path"]
        file_sha = pdf_shas[idx]
        stem = stems[idx]
        source_id = f"{stem}_{file_sha[:8]}" if stem_counts[stem] > 1 else stem
        pdir = os.path.normpath(os.path.join(a.out, "parsed", source_id))
        cache = None if a.cache == "off" else (a.cache or os.path.join(a.work, "parse_cache"))
        r = parse_pdf.parse(p, pdir, password=a.password, pages=pages_arg, preview=a.preview,
                            cache_dir=cache, budget_deadline=budget_deadline)
        outline = None
        if r.get("status") != "failed" and os.path.exists(os.path.join(pdir, "doc.json")):
            try:
                with open(os.path.join(pdir, "doc.json"), encoding="utf-8") as dfh:
                    doc = json.load(dfh)
                st = build_structure(doc, pdir)
                atomic_write_json(os.path.join(pdir, "structure_summary.json"), st)
                outline = write_outline(doc, st, pdir, HERE)
            except Exception as ex:
                change_log.append({"step": "structure_summary", "file": p, "error": str(ex)})
        outlines.append(outline)
        sources.append({"file": p, "sha256": file_sha, "source_id": source_id, "role": entry["role"], "format": "pdf",
                        "parsed_dir": pdir, "parse_status": r["status"],
                        "parse_error": r.get("error"), "page_count": r.get("meta", {}).get("page_count"),
                        "unresolved_pages": r.get("unresolved_pages", []), "ocr_pages": r.get("ocr_pages", []),
                        "image_only_pages": [pg["page"] for pg in r.get("pages", []) if pg.get("page_image")],
                        "pages_parsed": r.get("pages_parsed"), "pages_not_parsed": r.get("pages_not_parsed"),
                        "cache": (r.get("meta") or {}).get("cache"),
                        "warnings": r.get("warnings", [])[:20], "n_warnings": len(r.get("warnings", [])),
                        "stats": r.get("stats", {})})
        change_log.append({"step": "parse", "file": p, "source_id": source_id, "role": entry["role"],
                           "status": r["status"], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    # Keep non-PDF primary/supplementary sources ordered by role alongside PDF sources
    all_sources = sorted(sources + non_pdf_sources, key=lambda s: role_rank.get(s.get("role", "unspecified"), 1))
    result = {"case_id": case_id}
    meta = {
        "status": "pending_agent_fill",
        "sources": all_sources,
        "evidence": [],
        "uncertainty": [],
        "unreported": [],
        "conflicts": [],
        "field_status": {f: None for f in fields},
        "run_status": {"parse_seconds": round(time.time() - t0, 2), "case_error": case_err,
                       "n_pdfs": len(pdfs), "task_file": task_path,
                       "missing_pdfs": missing_explicit or None,
                       "contract_conflicts": contract_conflicts or None,
                       "has_non_pdf_primary": any(s.get("role") == "primary" for s in non_pdf_sources)},
        "change_log": change_log,
        "_contract_fields": fields,
        "_contract_required": required_fields,
        "_contract_schema": schema_map,
        "_contract_raw": contract_raw,
        "_fill_instructions": "Fill every contract field from the parsed PDF. Evidence items carry 'field': "
                              "table values -> {field, type:'table_cell', table, page, row{KEY:VALUE}, column:'A/B', value}; "
                              "text -> {field, page, block_id|table_id, quote}; page images -> {field, type:'image_region', page, bbox, crop, transcription}. "
                              "field_status per field: text_layer | image_read | unreadable. "
                              "Use null + an 'unreported' entry when the PDF does not report it or it cannot be read.",
    }
    for k, v in meta.items():
        if k not in result:
            result[k] = v
    for f in fields:  # contract fields; a contract key equal to a meta key (evidence, sources...) keeps the meta value
        if f not in result:
            result[f] = None
    if case_fatal or early_err:
        preserve_existing = preserve_existing or _has_existing_work(existing)
    if case_fatal:
        result["status"] = "failed"
        result["run_status"]["reason"] = case_err
    elif not pdfs:
        result["status"] = "failed"
        result["run_status"]["reason"] = (f"declared PDF(s) not found: {', '.join(missing_explicit)}"
                                          if missing_explicit else "no PDF found in case.json or the input directory")
    elif all(s["parse_status"] == "failed" for s in sources):
        result["status"] = "failed"
        result["run_status"]["reason"] = "; ".join(f"{s['file']}: {s.get('parse_error') or 'parse failed'}" for s in sources)

    if preserve_existing:
        summary_status = existing.get("status")
    else:
        write_skeleton(result)
        summary_status = result["status"]

    py = os.path.basename(sys.executable) if sys.executable else "python3"
    S = HERE
    parsed0 = sources[0]["parsed_dir"] if sources else "PARSED"
    f0 = next((f for f in fields if f not in META_FIELDS), "FIELD")
    summary = {"case_id": existing.get("case_id") if preserve_existing else case_id,
               "pdfs": pdfs, "contract_fields": fields, "required_fields": required_fields,
               "result": rpath, "preserved_existing_result": preserve_existing,
               "parsed": [s["parsed_dir"] for s in sources], "parse_status": [s["parse_status"] for s in sources],
               "page_count": [s.get("page_count") for s in sources],
               "scanned_pages": [s.get("unresolved_pages") or [] for s in sources],
               "outline": outlines, "task_file": task_path, "case_error": case_err,
               "status": summary_status,
               "next_steps": [
                   f"read {outlines[0] if outlines and outlines[0] else parsed0 + '/outline.md'} (small map of the paper), then the task",
                   f"\"{py}\" \"{S}/table_query.py\" \"{parsed0}\" --table \"Table N\" --row \"KEY=VALUE\" --col \"Group/Leaf\"",
                   f"\"{py}\" \"{S}/find_evidence.py\" \"{parsed0}\" \"words from the question\" --top 5   |  --equation N",
                   f"\"{py}\" \"{S}/fill_result.py\" put {f0} --table \"Table N\" --row \"KEY=VALUE\" --col \"Leaf\"   (or --quote/--equation/--crop)",
                   f"\"{py}\" \"{S}/fill_result.py\" unreported FIELD --reason \"...\"",
                   f"\"{py}\" \"{S}/fill_result.py\" finalize",
               ]}
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
