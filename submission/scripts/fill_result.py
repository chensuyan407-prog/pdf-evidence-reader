#!/usr/bin/env python3
"""Safe, command-style writer for result.json: the agent never edits the JSON by hand.

Every command loads result.json, checks the request against the parsed PDF, and writes
atomically (temp file + rename, previous version kept as result.json.bak). Protected keys
(case_id, status, sources, evidence, run_status, change_log, ...) cannot be set with `put`.

Usage (RESULT defaults to /app/output/result.json, or $RESULT_JSON):
  fill_result.py show                                       # contract fields, values, evidence, what is missing
  fill_result.py put FIELD [VALUE] --table "Table 2" --row "Model=Transformer (big)" --col "BLEU/EN-DE"
  fill_result.py put FIELD [VALUE] --equation 8 [--page 4]
  fill_result.py put FIELD VALUE --quote "verbatim words" [--page 3] [--block p3-b14]
  fill_result.py put FIELD [VALUE] --crop parsed/x/crops/p2_....png --transcription "what you read" [--legibility partial]
  fill_result.py unreported FIELD --reason "the paper does not report it" [--unreadable]
  fill_result.py note --uncertainty "text" [--field F]      # or --conflict "text"
  fill_result.py finalize                                   # fill gaps honestly, run verify_result.py, set status
  fill_result.py repair                                     # restore a result.json broken by a manual edit

VALUE omitted: the table cell / equation transcription / image transcription is used as the answer.
VALUE is a string; add --json to pass a number, list or object (e.g. --json '[1, 2]').
--add keeps the evidence already recorded for FIELD (several quotes for one answer); --append
adds VALUE to a list-valued FIELD. A dotted FIELD (answers.q1) writes inside an object field.
Exit code: 0 = written / finalize without blocking issues; 1 = refused (nothing written) or blocking issues.
"""
import argparse
import copy
import difflib
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import table_query  # noqa: E402
import verify_result as V  # noqa: E402

PROTECTED = set(V.META_KEYS) | {"verification", "status_reason"}
LIST_META = ("evidence", "uncertainty", "unreported", "conflicts", "change_log", "sources")
HINTS = {
    "value_mismatch": "the answer differs from the cited cell/equation: omit VALUE to take the cell, or fix --row/--col",
    "column_mismatch": "the answer sits in another column of that row: fix --col",
    "row_mismatch": "the answer sits in another row of that column: fix --row",
    "evidence_unsupported": "the cited place does not hold that content: re-run table_query/find_evidence and put again",
    "parsed_cell_incomplete": "the parsed cell is truncated: crop review_location with crop_region.py, read it, put --crop",
    "value_not_in_evidence": "the answer is not in the quote: quote the sentence that states it (or several with --add)",
    "no_evidence": "put the field again with --table/--equation/--quote/--crop",
    "null_without_reason": "unreported FIELD --reason '...'",
    "image_provenance_mismatch": "redo the crop with crop_region.py and put --crop again",
    "answer_incomplete": "answer looks truncated: give the complete value",
    "equation_needs_visual_check": "transcription partial: read the equation crop; if it differs, crop_region.py --equation N and put --crop",
    "string_match_only": "ok if the quote really states it (meaning is not machine-checked)",
    "image_read_unverified": "kept as image reading; stays in review",
    "contract_type_mismatch": "value violates the output_contract schema (check integer vs number, nested items/properties, or pass --json)",
    "not_determined": "required field still missing: fill with put or record unreported --reason ...",
}


def die(msg, **extra):
    print(json.dumps(dict({"ok": False, "error": msg}, **extra), ensure_ascii=False, indent=1))
    sys.exit(1)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_result():
    for p in (os.environ.get("RESULT_JSON"), "/app/output/result.json", os.path.join(os.getcwd(), "output", "result.json"),
              os.path.join(os.getcwd(), "result.json")):
        if p and os.path.exists(p):
            return p
    return os.environ.get("RESULT_JSON") or "/app/output/result.json"


def skeleton_paths(rpath):
    rdir = os.path.dirname(os.path.abspath(rpath))
    return [os.path.join(rdir, ".result.skeleton.json"),
            os.path.join(os.path.dirname(rdir), "work", "result.skeleton.json"),
            "/app/work/result.skeleton.json"]


def load(rpath, allow_repair=True):
    try:
        with open(rpath, encoding="utf-8-sig") as fh:
            res = json.load(fh)
        if not isinstance(res, dict):
            raise ValueError("top level is not an object")
        return res, None
    except FileNotFoundError:
        die(f"{rpath} not found: run run_case.py first (it writes the result skeleton)")
    except Exception as e:
        if not allow_repair:
            die(f"{rpath} is not valid JSON ({e}); run: fill_result.py repair")
        for b in [rpath + ".bak"] + skeleton_paths(rpath):
            if os.path.exists(b):
                try:
                    with open(b, encoding="utf-8") as fh:
                        res = json.load(fh)
                    return res, f"result.json was not valid JSON ({e}); restored from {os.path.basename(b)}"
                except Exception:
                    continue
        die(f"{rpath} is not valid JSON ({e}) and no backup/skeleton exists")


def save(rpath, res):
    rdir = os.path.dirname(os.path.abspath(rpath))
    base = os.path.basename(rpath)
    if os.path.exists(rpath):
        try:
            with open(rpath, encoding="utf-8-sig") as fh:
                cur = json.load(fh)
            bak_tmp = os.path.join(rdir, f".{base}.{os.getpid()}.bak.tmp")
            with open(bak_tmp, "w", encoding="utf-8") as bfh:
                json.dump(cur, bfh, ensure_ascii=False, indent=1)
                bfh.flush()
            os.replace(bak_tmp, rpath + ".bak")
        except Exception:
            pass
    tmp = os.path.join(rdir, f".{base}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
        fh.flush()
    with open(tmp, encoding="utf-8") as fh:
        json.load(fh)  # never replace a good file with a broken one
    os.replace(tmp, rpath)


def guard_core(res, rpath, log):
    """Restore keys the agent must not lose or overwrite (case_id, contract fields, meta lists)."""
    sk = None
    for p in skeleton_paths(rpath):
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as fh:
                    sk = json.load(fh)
                break
            except Exception:
                pass
    if sk:
        if "case_id" in sk and sk["case_id"] not in (None, "") and res.get("case_id") != sk["case_id"]:
            log.append(f"case_id {res.get('case_id')!r} restored to {sk['case_id']!r} (from the run_case skeleton)")
            res["case_id"] = sk["case_id"]
        for k in ("_contract_fields", "sources"):
            if not res.get(k) and sk.get(k):
                res[k] = sk[k]
        for k in ("_contract_required", "_contract_schema"):
            if k not in res and k in sk:
                res[k] = sk[k]
        for f in sk.get("_contract_fields") or []:
            if f not in res:
                res[f] = copy.deepcopy(sk.get(f))
                log.append(f"contract field {f!r} was missing; restored")
    for k in LIST_META:
        if k in fields_of(res) and not isinstance(res.get(k), list):
            continue  # the contract itself defines this key; leave its shape alone
        if not isinstance(res.get(k), list):
            if res.get(k) not in (None, [], {}):
                log.append(f"{k} was not a list; wrapped")
                res[k] = [res[k]]
            else:
                res[k] = []
    if not isinstance(res.get("field_status"), dict):
        res["field_status"] = {f: None for f in fields_of(res)}
    if not isinstance(res.get("run_status"), dict):
        res["run_status"] = {}


def fields_of(res):
    return list(res.get("_contract_fields") or [k for k in res if k not in PROTECTED])


def resolve_field(res, name):
    fields = fields_of(res)
    top = name.split(".")[0]
    if top in PROTECTED and top not in fields:
        die(f"{top!r} is maintained by the scripts and cannot be set with put",
            contract_fields=fields)
    if top in fields:
        return name, top
    low = {f.lower(): f for f in fields}
    if top.lower() in low:
        return low[top.lower()] + name[len(top):], low[top.lower()]
    die(f"{top!r} is not an output_contract field", contract_fields=fields,
        did_you_mean=difflib.get_close_matches(top, fields, n=3, cutoff=0.4))


def set_value(res, path, value, append=False):
    parts = path.split(".")
    obj = res
    for p in parts[:-1]:
        if not isinstance(obj.get(p), dict):
            obj[p] = {}
        obj = obj[p]
    k = parts[-1]
    if append:
        cur = obj.get(k)
        cur = [] if cur in (None, "") else (cur if isinstance(cur, list) else [cur])
        cur.append(value)
        obj[k] = cur
    else:
        obj[k] = value


def parsed_docs(res, rpath, source_filter=None):
    rdir = os.path.dirname(os.path.abspath(rpath))
    dirs = [s.get("parsed_dir") for s in res.get("sources", []) if isinstance(s, dict) and s.get("parsed_dir")]
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "doc.json"))]
    if not dirs:
        dirs = [os.path.dirname(p) for p in glob.glob(os.path.join(rdir, "parsed", "*", "doc.json"))]
    out = []
    for d in dirs:
        with open(os.path.join(d, "doc.json"), encoding="utf-8") as fh:
            doc = json.load(fh)
        if source_filter:
            sha = (doc.get("meta") or {}).get("sha256") or ""
            bname = os.path.basename(os.path.normpath(d))
            if not (source_filter == bname or sha.startswith(source_filter) or source_filter in d):
                continue
        out.append((d, doc))
    return out


def ev_table(a, docs):
    rows = list(a.row or [])
    for r in rows:
        if "=" not in r:
            die(f"--row needs KEY=VALUE, got {r!r}")
    last = None
    for d, doc in docs:
        q = table_query.query(doc, a.table, rows, a.col or "")
        if q["status"] == "ok":
            main_id = (q.get("table_parts") or [q["table_id"]])[0]
            ev = {"type": "table_cell", "table": a.table, "table_id": main_id, "page": q["page"],
                  "row": dict(r.split("=", 1) for r in rows), "column": a.col, "value": q["value"]}
            sha = (doc.get("meta") or {}).get("sha256")
            if sha:
                ev["source_sha256"] = sha
            if q.get("part") and q["part"] != main_id:
                ev["table_part"] = q["part"]
            pretty = q.get("value_pretty") or q["value"]
            default = pretty if V.same_value(pretty, q["value"], {"value": q.get("value_numeric")}) else q["value"]
            return ev, default, "text_layer", {k: q.get(k) for k in ("value", "value_pretty", "page", "column_path", "cell_bbox",
                                                                     "shared_across", "footnote_marker") if q.get(k) is not None}
        last = q
    if last and last["status"] == "table_is_image":
        die("this table is an image without text layer: read its crop and use put --crop ... --transcription ...",
            lookup=last, parsed_dir=docs[0][0] if docs else None)
    die(f"table lookup failed: {last['status'] if last else 'no parsed document'}", lookup=last)


def ev_equation(a, docs):
    for d, doc in docs:
        eqs = [e for e in doc.get("equations", []) if str(e.get("number")) == str(a.equation)
               and (a.page is None or e["page"] == a.page)]
        if eqs:
            e = eqs[0]
            lx = e.get("expr_latex") or e.get("latex")
            info = {"page": e["page"], "transcription_status": e.get("transcription_status"), "crop": e.get("crop") and
                    os.path.join(d, e["crop"]), "expr_unicode": e.get("expr_unicode") or e.get("unicode_math")}
            if e.get("transcription_status") != "auto":
                info["warning"] = ("transcription is partial: read the crop; if your reading differs use "
                                   f"crop_region.py {d} --equation {a.equation} and put --crop ... --transcription ...")
            ev = {"type": "equation", "equation": str(a.equation), "page": e["page"]}
            sha = (doc.get("meta") or {}).get("sha256")
            if sha:
                ev["source_sha256"] = sha
            return ev, lx, "text_layer", info
    nums = sorted({str(e.get("number")) for _, doc in docs for e in doc.get("equations", []) if e.get("number")})
    die(f"equation ({a.equation}) not found", available_numbers=nums[:60])


def ev_quote(a, docs):
    qn = V.norm(a.quote)
    if not qn:
        die("empty --quote")
    hits = []
    for d, doc in docs:
        sha = (doc.get("meta") or {}).get("sha256")
        for b in doc.get("blocks", []):
            if qn in V.norm(b["text"]) or qn in V.norm(b.get("marked") or ""):
                hits.append((b, "block", sha))
        for t in doc.get("tables", []):
            txt = " ".join(" ".join(c or "" for c in r) for r in t["rows"]) + " " + (t.get("caption") or "")
            if qn in V.norm(txt):
                hits.append(({"id": t["id"], "page": t["page"], "role": "table"}, "table", sha))
    if a.block:
        hits = [h for h in hits if h[0]["id"] == a.block] or die(
            f"quote not found in block {a.block}", found_in=[h[0]["id"] for h in hits][:8])
    if a.page is not None:
        on_page = [h for h in hits if h[0]["page"] == a.page]
        if not on_page and hits:
            die(f"quote is not on page {a.page}", found_on=[(h[0]["id"], h[0]["page"]) for h in hits][:8])
        hits = on_page
    if not hits:
        sugg = []
        try:
            import find_evidence
            for d, doc in docs:
                sugg += [{"block_id": r.get("block_id") or r.get("id"), "page": r.get("page"), "text": (r.get("text") or "")[:160]}
                         for r in find_evidence.search(doc, a.quote, top=3)]
        except Exception:
            pass
        die("quote not found verbatim in the parsed PDF (copy it from document.md / find_evidence output)", closest=sugg[:3])
    body = [h for h in hits if h[0].get("role") not in ("header", "footer", "margin_note", "figure_text")]
    b, kind, sha = (body or hits)[0]
    ev = {"type": "text", "page": b["page"], "quote": a.quote}
    if sha:
        ev["source_sha256"] = sha
    ev["table_id" if kind == "table" else "block_id"] = b["id"]
    info = {"found_in": b["id"], "role": b.get("role")}
    if len(hits) > 1:
        info["also_found_in"] = [h[0]["id"] for h in hits[1:6]]
    if b.get("role") in ("header", "footer", "margin_note", "figure_text"):
        info["warning"] = f"quote comes from {b['role']} text, not the body"
    return ev, None, "text_layer", info


def ev_crop(a, rpath, docs):
    rdir = os.path.dirname(os.path.abspath(rpath))
    bases = [rdir] + [d for d, _ in docs]
    crop = V.resolve_path(a.crop, bases)
    if not crop:
        die(f"crop {a.crop!r} not found (paths are tried relative to result.json and the parsed folders)")
    import provenance
    rec = provenance.load_sidecar(crop)
    if not rec:
        die("crop has no provenance sidecar (<crop>.prov.json): make it with crop_region.py, not by hand")
    if not a.transcription:
        die("--transcription is required: write exactly what you read in the crop")
    ev = {"type": "image_region", "page": rec["page"], "bbox": rec["bbox"],
          "crop": os.path.relpath(crop, rdir).replace("\\", "/"), "provenance": rec,
          "transcription": a.transcription, "read_by": "agent_vision", "legibility": a.legibility}
    return ev, a.transcription, "image_read", {"crop": ev["crop"], "page": rec["page"], "bbox": rec["bbox"]}


def cmd_put(a, rpath):
    res, note = load(rpath)
    log = [note] if note else []
    guard_core(res, rpath, log)
    path, top = resolve_field(res, a.field)
    docs = parsed_docs(res, rpath, source_filter=getattr(a, "source", None))
    kinds = [k for k in ("table", "equation", "quote", "crop") if getattr(a, k)]
    if len(kinds) != 1:
        die("give exactly one evidence source: --table/--row/--col, --equation, --quote or --crop "
            "(no evidence -> use 'unreported FIELD --reason ...')")
    if not docs and kinds[0] != "crop":
        die("no parsed document found (sources[].parsed_dir); run run_case.py first")
    if kinds[0] == "table":
        ev, default, fstatus, info = ev_table(a, docs)
    elif kinds[0] == "equation":
        ev, default, fstatus, info = ev_equation(a, docs)
    elif kinds[0] == "quote":
        ev, default, fstatus, info = ev_quote(a, docs)
    else:
        ev, default, fstatus, info = ev_crop(a, rpath, docs)
    sub_schema = V.subschema_for_path(res.get("_contract_schema") or {}, path)
    if a.value is None:
        if default is None:
            die("VALUE is required with --quote (the answer the quote supports)")
        value = default
        if isinstance(value, str) and isinstance(sub_schema, dict):
            st = sub_schema.get("type")
            if st == "integer" and re.fullmatch(r"[-+]?\d+", value.strip()):
                value = int(value.strip())
            elif st == "number":
                try:
                    value = float(value.strip())
                except ValueError:
                    pass
    elif a.json:
        try:
            value = json.loads(a.value)
        except Exception as e:
            die(f"--json VALUE is not valid JSON: {e}")
    else:
        value = a.value
    # validate against contract type/nested schema before touching result.json
    if sub_schema and not a.append:
        serrs = V.validate_value(value, sub_schema, path)
        if serrs:
            hint = "pass --json for numbers, booleans, lists, or objects" if isinstance(value, str) else "fix value type/structure to match output_contract"
            die("value violates output_contract schema: " + "; ".join(serrs), hint=hint)
    if a.append and sub_schema and isinstance(sub_schema.get("items"), dict):
        serrs = V.validate_value(value, sub_schema["items"], f"{path}[]")
        if serrs:
            die("appended item violates output_contract schema: " + "; ".join(serrs))
    # pre-checks with the verifier's own rules: refuse what finalize would reject
    if kinds[0] == "table" and not isinstance(value, (dict, list)) and not V.same_value(str(value), ev["value"]):
        die(f"VALUE {value!r} differs from the cell {ev['value']!r}; omit VALUE to use the cell, or fix --row/--col",
            cell=info)
    if kinds[0] == "table" and isinstance(value, (dict, list)) and path == top:
        die("table_cell evidence supports a scalar or a dotted subfield (e.g. answers.bleu), not a composite value")
    if kinds[0] == "equation" and isinstance(value, str):
        rec = [V.norm(default or ""), V.norm(info.get("expr_unicode") or "")]
        rec_eq = [V.norm_eq(default or ""), V.norm_eq(info.get("expr_unicode") or "")]
        if not (any(r and r in V.norm(value) for r in rec) or any(r and r in V.norm_eq(value) for r in rec_eq)):
            die("VALUE does not contain the parser's transcription of this equation; omit VALUE, or read the crop "
                "and use put --crop (crop_region.py --equation N) for your own reading", equation=info)
    if kinds[0] == "quote" and not a.force:
        leaves = [str(x) for x in (value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value])]
        joined = " ".join([a.quote] + [e.get("quote", "") for e in res["evidence"] if a.add and e.get("field") == top])
        nums = V.values_in(" ".join(leaves))
        ok = (all(any(V.close(x, y) for y in V.values_in(joined)) for x in nums) if nums
              else all(V.norm(lf) in V.norm(joined) for lf in leaves if V.norm(lf)))
        if not ok:
            die("VALUE does not occur in the quote, so verify_result.py would reject it (value_not_in_evidence). "
                "Quote the words that state it, add more quotes with --add, or pass --force to keep a paraphrase "
                "(it will be reported as unsupported).", quote=a.quote, value=value)
    ev = dict({"field": top}, **ev)
    if path != top:
        ev["subfield"] = path[len(top) + 1:]
    if a.note:
        ev["note"] = a.note
    if not a.add:
        res["evidence"] = [e for e in res["evidence"] if not (isinstance(e, dict) and (e.get("field") or e.get("claim")) == top
                                                               and e.get("subfield") == ev.get("subfield"))]
    res["evidence"].append(ev)
    set_value(res, path, value, append=a.append)
    res["unreported"] = [u for u in res["unreported"] if not (isinstance(u, dict) and u.get("field") == top)]
    res["field_status"][top] = fstatus
    if a.note:
        res["uncertainty"].append({"field": top, "note": a.note})
    res["change_log"].append({"step": "put", "field": path, "evidence": ev.get("type"), "at": now()})
    if res.get("status") not in ("pending_agent_fill", None, "failed"):
        res["status"] = "pending_agent_fill"  # a later finalize recomputes it
    save(rpath, res)
    print(json.dumps({"ok": True, "field": path, "value": value, "field_status": fstatus, "evidence": ev.get("type"),
                      "checked": info, "notes": log or None, "next": "more put/unreported, then finalize"},
                     ensure_ascii=False, indent=1))


def cmd_unreported(a, rpath):
    res, note = load(rpath)
    log = [note] if note else []
    guard_core(res, rpath, log)
    path, top = resolve_field(res, a.field)
    if not a.reason or len(a.reason.strip()) < 8:
        die("--reason must say why (e.g. 'Table 3 lists no value for this setting', 'page 4 is an unreadable scan')")
    set_value(res, path, None)
    res["evidence"] = [e for e in res["evidence"] if not (isinstance(e, dict) and e.get("field") == top)]
    res["unreported"] = [u for u in res["unreported"] if not (isinstance(u, dict) and u.get("field") == top)]
    kind = "unreadable" if a.unreadable else "not_reported"
    item = {"field": top, "reason": a.reason, "kind": kind}
    if a.page is not None:
        item["page"] = a.page
    res["unreported"].append(item)
    res["field_status"][top] = kind
    res["change_log"].append({"step": "unreported", "field": top, "kind": kind, "at": now()})
    if res.get("status") not in ("pending_agent_fill", None, "failed"):
        res["status"] = "pending_agent_fill"
    save(rpath, res)
    print(json.dumps({"ok": True, "field": top, "value": None, "unreported": item, "notes": log or None}, ensure_ascii=False, indent=1))


def cmd_note(a, rpath):
    res, note = load(rpath)
    guard_core(res, rpath, [])
    if not (a.uncertainty or a.conflict):
        die("give --uncertainty TEXT or --conflict TEXT")
    item = {"note": a.uncertainty or a.conflict}
    if a.field:
        item["field"] = resolve_field(res, a.field)[1]
    if a.page is not None:
        item["page"] = a.page
    res["uncertainty" if a.uncertainty else "conflicts"].append(item)
    res["change_log"].append({"step": "note", "kind": "uncertainty" if a.uncertainty else "conflict", "at": now()})
    save(rpath, res)
    print(json.dumps({"ok": True, "added": item}, ensure_ascii=False))


def cmd_show(a, rpath):
    res, note = load(rpath)
    guard_core(res, rpath, [])
    ver = (res.get("verification") or {}).get("field_verdicts") or {}
    rows = []
    for f in fields_of(res):
        v = res.get(f)
        evs = [e for e in res.get("evidence", []) if isinstance(e, dict) and (e.get("field") or e.get("claim")) == f]
        un = next((u for u in res.get("unreported", []) if isinstance(u, dict) and u.get("field") == f), None)
        sv = v if isinstance(v, str) else (json.dumps(v, ensure_ascii=False) if v is not None else None)
        rows.append({"field": f, "value": (sv[:120] + "…") if sv and len(sv) > 120 else sv,
                     "field_status": res.get("field_status", {}).get(f), "evidence": [e.get("type") or "text" for e in evs],
                     "unreported": un and un.get("reason"), "last_verdict": ver.get(f)})
    todo = [r["field"] for r in rows if r["value"] is None and not r["unreported"]]
    print(json.dumps({"result": rpath, "case_id": res.get("case_id"), "status": res.get("status"), "notes": note,
                      "parsed": [d for d, _ in parsed_docs(res, rpath)], "fields": rows, "todo": todo},
                     ensure_ascii=False, indent=1))


def cmd_repair(a, rpath):
    res, note = load(rpath)
    log = [note] if note else []
    guard_core(res, rpath, log)
    res.setdefault("change_log", []).append({"step": "repair", "notes": log, "at": now()})
    save(rpath, res)
    print(json.dumps({"ok": True, "repaired": log or ["nothing to repair"]}, ensure_ascii=False, indent=1))


def cmd_finalize(a, rpath):
    res, note = load(rpath)
    log = [note] if note else []
    guard_core(res, rpath, log)
    fields = fields_of(res)
    req_fields = res["_contract_required"] if isinstance(res.get("_contract_required"), list) else [f for f in fields if f not in PROTECTED]
    req_set = set(req_fields)
    for f in fields:
        if f in PROTECTED:
            continue
        if f in req_set and res.get(f) in (None, "", [], {}) and not any(isinstance(u, dict) and u.get("field") == f for u in res["unreported"]):
            res[f] = None
            res["unreported"].append({"field": f, "reason": "not determined: no supporting evidence was located in the PDF "
                                                            "within the run budget", "kind": "not_determined"})
            res["field_status"][f] = "not_determined"
    undetermined = [
        f for f in req_fields if f not in PROTECTED and res.get(f) in (None, "", [], {}) and (
            res.get("field_status", {}).get(f) == "not_determined" or
            any(isinstance(u, dict) and u.get("field") == f and u.get("kind") == "not_determined" for u in res["unreported"])
        )
    ]
    if res.get("status") not in ("failed", "refused"):
        res["status"] = "pending_agent_fill"
    save(rpath, res)
    cmd = [sys.executable, os.path.join(HERE, "verify_result.py"), "--result", rpath] + (["--pdf"] + a.pdf if a.pdf else [])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    vpath = os.path.join(os.path.dirname(os.path.abspath(rpath)), "verification.json")
    try:
        with open(vpath, encoding="utf-8") as fh:
            summ = json.load(fh)["summary"]
    except Exception:
        summ = None
    res, _ = load(rpath)
    blocking = p.returncode != 0
    verdicts = (summ or {}).get("field_verdicts", {})
    pdf_sources = [s for s in (res.get("sources") or []) if isinstance(s, dict) and s.get("format", "pdf") == "pdf"]
    no_sources = not pdf_sources
    parse_failed = no_sources or all(s.get("parse_status") == "failed" for s in pdf_sources)
    case_fatal = bool((res.get("run_status") or {}).get("case_error")) and res.get("status") == "failed"
    reasons = []
    if summ is None:
        reasons.append("verify_result.py did not run: " + (p.stderr or "")[-300:])
    if case_fatal:
        status = "failed"
        reasons.append(str((res.get("run_status") or {}).get("reason") or (res.get("run_status") or {}).get("case_error")))
    elif parse_failed:
        status = "failed"
        if no_sources:
            reasons.append(str((res.get("run_status") or {}).get("reason") or "no PDF source was parsed"))
        else:
            reasons.append("no PDF could be parsed: " + "; ".join(str(s.get("parse_error")) for s in pdf_sources))
    elif res.get("status") == "refused":
        status = "refused"
    else:
        bad = {f: v.get("verdict") for f, v in verdicts.items() if v.get("verdict") in V.FAIL_VERDICTS}
        unread = [f for f in fields if res.get("field_status", {}).get(f) == "unreadable"]
        img = [f for f, v in verdicts.items() if v.get("verdict") in ("image_read_unverified", "image_read_unbound")]
        kw_bad = {f: v.get("reasons", []) for f, v in verdicts.items() if (v.get("checks") or {}).get("quote_keyword_mismatch")}
        answered = [f for f in fields if f not in PROTECTED and res.get(f) not in (None, "", [], {})]
        if bad:
            reasons.append("fields failing verification: " + ", ".join(f"{f} ({v})" for f, v in bad.items()))
        if undetermined:
            reasons.append("not determined: " + ", ".join(undetermined))
        if unread:
            reasons.append("unreadable in the PDF: " + ", ".join(unread))
        if img:
            reasons.append("read from page images (not machine-verified): " + ", ".join(img))
        if kw_bad:
            reasons.append("quote/metric mismatch needs review: " + "; ".join(f"{f}: {' '.join(rs)}" for f, rs in kw_bad.items()))
        conflicts_list = [c for c in (res.get("conflicts") or []) if isinstance(c, dict)]
        uncertainty_list = [u for u in (res.get("uncertainty") or []) if isinstance(u, dict)]
        if conflicts_list:
            c_desc = "; ".join(f"{c.get('field') or 'general'}: {c.get('note') or c.get('conflict') or ''}".strip(": ") for c in conflicts_list)
            reasons.append("material conflicts recorded (not hidden): " + c_desc)
        if uncertainty_list:
            u_desc = "; ".join(f"{u.get('field') or 'general'}: {u.get('note') or u.get('uncertainty') or ''}".strip(": ") for u in uncertainty_list)
            reasons.append("uncertainties recorded: " + u_desc)
        if not answered and any(f not in PROTECTED for f in req_fields):
            status = "failed" if (undetermined or unread) else "success"
            if status == "failed":
                reasons.append("no contract field could be answered")
        elif blocking or bad or undetermined or unread or img or kw_bad or summ is None:
            status = "partial"
        else:
            status = "success"
            rev = (summ or {}).get("needs_review") or []
            if rev:
                reasons.append("answered from verified evidence (needs review: " + ", ".join(rev) + ")")
    res["status"] = status
    res["status_reason"] = reasons or ["every contract field is answered from verified evidence or reported as absent"]
    res.setdefault("run_status", {})["finalized_at"] = now()
    res["change_log"].append({"step": "finalize", "status": status, "at": now(), "notes": log or None})
    save(rpath, res)
    out = {"status": status, "status_reason": res["status_reason"], "blocking_issues": blocking,
           "contract_gaps": (summ or {}).get("contract_gaps") or [],
           "conflicts": res.get("conflicts") or [],
           "uncertainty": res.get("uncertainty") or [],
           "fields": {f: {"verdict": (verdicts.get(f) or {}).get("verdict"),
                          "hint": HINTS.get((verdicts.get(f) or {}).get("verdict"))} for f in fields},
           "needs_review": (summ or {}).get("needs_review"), "verification": vpath, "notes": log or None}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(1 if blocking else 0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    sub.add_parser("repair")
    pf = sub.add_parser("finalize")
    pf.add_argument("--pdf", nargs="*", default=None)
    pp = sub.add_parser("put")
    pp.add_argument("field")
    pp.add_argument("value", nargs="?", default=None)
    pp.add_argument("--json", action="store_true")
    pp.add_argument("--table")
    pp.add_argument("--row", action="append")
    pp.add_argument("--col")
    pp.add_argument("--equation")
    pp.add_argument("--quote")
    pp.add_argument("--block")
    pp.add_argument("--page", type=int)
    pp.add_argument("--crop")
    pp.add_argument("--transcription")
    pp.add_argument("--legibility", default="clear", choices=["clear", "partial", "illegible"])
    pp.add_argument("--note")
    pp.add_argument("--add", action="store_true", help="keep evidence already recorded for this field")
    pp.add_argument("--append", action="store_true", help="append VALUE to a list-valued field")
    pp.add_argument("--force", action="store_true", help="keep a VALUE that the quote does not contain (reported as unsupported)")
    pp.add_argument("--source", default=None, help="restrict lookup to a specific parsed source_id or sha256 prefix")
    pu = sub.add_parser("unreported")
    pu.add_argument("field")
    pu.add_argument("--reason", required=True)
    pu.add_argument("--unreadable", action="store_true", help="the value is in the PDF but cannot be read (vs. not reported)")
    pu.add_argument("--page", type=int)
    pn = sub.add_parser("note")
    pn.add_argument("--uncertainty")
    pn.add_argument("--conflict")
    pn.add_argument("--field")
    pn.add_argument("--page", type=int)
    a = ap.parse_args()
    rpath = a.result or default_result()
    {"show": cmd_show, "put": cmd_put, "unreported": cmd_unreported, "note": cmd_note,
     "finalize": cmd_finalize, "repair": cmd_repair}[a.cmd](a, rpath)


if __name__ == "__main__":
    main()
