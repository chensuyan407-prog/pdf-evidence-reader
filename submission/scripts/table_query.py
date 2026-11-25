#!/usr/bin/env python3
"""Look up one table cell by row labels and column header path.

Usage:
    python table_query.py PARSED_DIR --table "Table H.1" --row Name=HellaSwag --row Metric=acc --row Split=dev --col "Zero-Shot/13B"
    python table_query.py PARSED_DIR --table T26 --list          # columns + row keys

--table accepts a table id (T26), a caption prefix ("Table H.1") or "page:63".
--row KEY=VALUE matches a body row whose cell in column KEY equals VALUE
(KEY matches a column's leaf header or full name, case-insensitive).
--col is a header path, parts separated by "/"; each part must equal one
header element of the column, in order (e.g. "Zero-Shot/13B" or "13B").

    python table_query.py PARSED_DIR --table "Table 2" --markdown     # whole table as Markdown (+ continuation parts)

Prints JSON: status (ok | table_not_found | table_is_image | row_not_found | row_ambiguous |
col_not_found | col_ambiguous), value, page, table_id, row, col,
column_path, row_key, cell_bbox, caption. Exit code 0 only when status=ok.
A table continued on the next page ("Table 2 (continued)") is searched as one table;
`part` names the piece the row was found in. On a miss, `did_you_mean` lists the closest
row labels / column names. table_is_image: the table is a picture without text layer - read
the returned crop (it has a provenance sidecar) and cite it as image_region evidence.
"""
import difflib
import argparse
import json
import os
import re
import sys


def _n(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def load(parsed):
    p = parsed if parsed.endswith(".json") else os.path.join(parsed, "doc.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def find_table(doc, spec):
    tabs = doc.get("tables", [])
    if spec is None:
        return None
    for t in tabs:
        if t.get("id") == spec:
            return t
    if spec.startswith("page:"):
        pg = int(spec.split(":")[1])
        c = [t for t in tabs if t["page"] == pg]
        return c[0] if len(c) == 1 else None
    key = _n(spec)
    c = [t for t in tabs if _n(t.get("caption")).startswith(key) and not t.get("continued_from")
         and not _n(t.get("caption")).endswith("(continued)") and _caption_label_ok(t.get("caption"), key)]
    return c[0] if c else None


def _caption_label_ok(caption, key):
    """'Table 1' must not match 'Table 12: ...'."""
    rest = _n(caption)[len(key):]
    return not (key[-1:].isdigit() and rest[:1].isdigit())


def parts_of(doc, t):
    """The table followed by its continuation pieces on later pages."""
    out, cur = [t], t
    while True:
        nxt = next((x for x in doc.get("tables", []) if x.get("continued_from") == cur["id"]), None)
        if nxt is None or nxt in out:
            return out
        out.append(nxt)
        cur = nxt


def image_table(doc, spec):
    key = _n(spec or "")
    for t in doc.get("tables_as_images", []) or []:
        if t.get("id") == spec or (key and _n(t.get("caption")).startswith(key) and _caption_label_ok(t.get("caption"), key)):
            return t
    return None


def _close(word, options, n=5):
    opts = [o for o in options if o]
    hits = difflib.get_close_matches(word, opts, n=n, cutoff=0.5)
    hits += [o for o in opts if _n(word) and _n(word) in _n(o) and o not in hits]
    return hits[:n]


def columns_of(t):
    if t.get("columns"):
        return t["columns"]
    # tables from older parser versions: single header row
    names = (t.get("header") or {}).get("names") or (t["rows"][0] if t["rows"] else [])
    return [{"index": i, "path": [n] if n else [], "name": n or f"col{i}"} for i, n in enumerate(names)]


def col_matches_key(col, key):
    k = _n(key)
    return (col["path"] and _n(col["path"][-1]) == k) or _n(col["name"]) == k or k in [_n(p) for p in col["path"]]


def match_col_path(cols, spec):
    parts = [_n(p) for p in spec.split("/") if p.strip()]
    hits = []
    for c in cols:
        path = [_n(p) for p in c["path"]]
        i = 0
        for p in path:
            if i < len(parts) and p == parts[i]:
                i += 1
        if i == len(parts) and path and path[-1] == parts[-1]:
            hits.append(c)
    if len(hits) > 1:  # prefer exact full-path match
        exact = [c for c in hits if [_n(p) for p in c["path"]] == parts]
        if len(exact) == 1:
            hits = exact
    return hits


def query(doc, table, rows, col):
    t = find_table(doc, table)
    if t is None:
        im = image_table(doc, table)
        if im:
            return {"status": "table_is_image", "table": table, "table_id": im["id"], "page": im["page"],
                    "bbox": im["bbox"], "crop": im.get("crop"), "caption": im.get("caption"), "ocr_text": im.get("ocr_text"),
                    "next_step": "read the crop (parsed dir + crop path) and record image_region evidence with its "
                                 "provenance sidecar; field_status = image_read"}
        return {"status": "table_not_found", "table": table,
                "available": [(x["id"], x["page"], x.get("caption")) for x in doc.get("tables", [])],
                "available_as_images": [(x["id"], x["page"], x.get("caption")) for x in doc.get("tables_as_images", []) or []]}
    parts = parts_of(doc, t)
    if len(parts) == 1:
        return _query_one(t, rows, col)
    results = []
    for k, part in enumerate(parts):
        r = _query_one(part if k == 0 else _with_columns_of(part, t), rows, col)
        r["part"] = part["id"]
        results.append(r)
    ok = [r for r in results if r["status"] == "ok"]
    if len(ok) == 1:
        ok[0]["table_parts"] = [p["id"] for p in parts]
        return ok[0]
    if len(ok) > 1:
        return {"status": "row_ambiguous", "table_id": t["id"], "page": t["page"], "caption": t.get("caption"),
                "conditions": rows, "candidates": [{"part": r["part"], "page": r["page"], "value": r["value"]} for r in ok],
                "reason": "the row matches in several pieces of a continued table"}
    first = results[0]
    first["table_parts"] = [p["id"] for p in parts]
    if first["status"] == "row_not_found":
        labels = [x for r in results for x in (r.get("did_you_mean") or [])]
        first["did_you_mean"] = list(dict.fromkeys(labels))[:8]
    return first


def _with_columns_of(part, main):
    """A continuation piece is read with the main table's columns (its own header row, if repeated, is skipped)."""
    mcols = columns_of(main)
    if part.get("n_cols") != main.get("n_cols") or [c["name"] for c in columns_of(part)] == [c["name"] for c in mcols]:
        return part
    leaf = [_n(c["path"][-1] if c["path"] else "") for c in mcols]
    repeated = bool(part["rows"]) and [_n(c) for c in part["rows"][0]] == leaf
    q = dict(part, columns=mcols, key_columns=main.get("key_columns", part.get("key_columns", [0])))
    q["header"] = dict(part.get("header") or {}, n_header_rows=1 if repeated else 0)
    return q


def _query_one(t, rows, col):
    cols = columns_of(t)
    nh = (t.get("header") or {}).get("n_header_rows", 1)
    body = list(range(nh, len(t["rows"])))
    conds = []
    for r in rows:
        if "=" not in r:
            return {"status": "bad_row_condition", "condition": r}
        k, v = r.split("=", 1)
        kc = [c for c in cols if col_matches_key(c, k)]
        if not kc:
            return {"status": "row_not_found", "reason": f"no column named {k!r}", "columns": [c["name"] for c in cols],
                    "did_you_mean": _close(k, [c["name"] for c in cols]), "table_id": t["id"], "page": t["page"]}
        conds.append((kc[0]["index"], _n(v), k, v))
    def cell_is(text, v):  # a trailing footnote marker ("Baseline^a") does not change the label
        return _n(text) == v or _n(re.sub(r"\^\S+$", "", text or "")) == v
    cand = [ri for ri in body if all(cell_is(t["rows"][ri][ci], v) for ci, v, _, _ in conds)]
    base = {"table_id": t["id"], "page": t["page"], "caption": t.get("caption")}
    if not cand:
        labels = [t["rows"][ri][conds[0][0]] for ri in body] if conds else []
        return dict(base, status="row_not_found", conditions=rows, did_you_mean=_close(conds[0][3], labels) if conds else [])
    if len(cand) > 1:
        return dict(base, status="row_ambiguous", conditions=rows,
                    candidates=[[t["rows"][ri][ci] for ci in t.get("key_columns", [0])] for ri in cand])
    ri = cand[0]
    ch = match_col_path(cols, col)
    via_group = False
    if not ch:
        # a group header path ("Training Cost (FLOPs)") is acceptable when one cell spans all its columns in this row
        parts = [_n(p) for p in col.split("/") if p.strip()]
        grp = [c for c in cols if [_n(p) for p in c["path"]][:len(parts)] == parts]
        starts = {span_start(t, ri, c["index"]) for c in grp}
        if grp and len(starts) == 1:
            ch, via_group = [cols[starts.pop()]], True
        elif grp:
            return dict(base, status="col_ambiguous", column=col, candidates=[c["name"] for c in grp],
                        reason="group header covers several columns that hold different cells in this row")
    if not ch:
        names = [c["name"] for c in cols]
        return dict(base, status="col_not_found", column=col, columns=names,
                    did_you_mean=_close(col, names + ["/".join(c["path"]) for c in cols]))
    if len(ch) > 1:
        return dict(base, status="col_ambiguous", column=col, candidates=[c["name"] for c in ch])
    ci = ch[0]["index"]
    c0 = span_start(t, ri, ci)
    cell = next((c for c in t.get("cells", []) if c["row"] == ri and c["col"] == c0), None) or {}
    span = cell.get("colspan", 1)
    key = {cols[k]["name"]: t["rows"][ri][k] for k in t.get("key_columns", [])}
    norm = cell.get("norm") or {}
    out = dict(base, status="ok", value=t["rows"][ri][c0], row=ri, col=c0, column_path=ch[0]["path"],
               row_key=key, row_values=t["rows"][ri], cell_bbox=cell.get("bbox"),
               value_raw=cell.get("raw"), value_parts=cell.get("parts"),
               value_numeric=norm.get("value"), value_canonical=norm.get("canonical"),
               value_pretty=norm.get("pretty") or t["rows"][ri][c0], value_kind=norm.get("kind"))
    if span > 1:
        out["shared_across"] = [cols[k]["name"] for k in range(c0, c0 + span)]
        out["span_note"] = f"one printed value spans {span} columns; it applies to each of them"
    if via_group:
        out["matched_via"] = "group header"
    if norm.get("kind") == "footnoted":
        out["footnote_marker"] = norm.get("footnote")
    return out


def span_start(t, ri, ci):
    """Column where the cell covering (ri, ci) starts (None in the grid marks covered cells)."""
    k = ci
    while k > 0 and t["rows"][ri][k] is None:
        k -= 1
    return k


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parsed")
    ap.add_argument("--table", required=True)
    ap.add_argument("--row", action="append", default=[])
    ap.add_argument("--col", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--markdown", action="store_true", help="print the whole table (all continuation parts) as Markdown")
    a = ap.parse_args()
    doc = load(a.parsed)
    if a.markdown:
        t = find_table(doc, a.table)
        if t is None:
            print(json.dumps(query(doc, a.table, [], ""), ensure_ascii=False, indent=1))
            sys.exit(1)
        esc = lambda c: (c if c is not None else "").replace("|", "\\|")
        for part in parts_of(doc, t):
            if not part["rows"]:
                continue
            print(f"<!-- {part['id']} page {part['page']} bbox {part.get('bbox')} caption: {part.get('caption')} -->")
            print("| " + " | ".join(esc(c) for c in part["rows"][0]) + " |")
            print("|" + "---|" * len(part["rows"][0]))
            for r in part["rows"][1:]:
                print("| " + " | ".join(esc(c) for c in r) + " |")
            print()
        return
    if a.list:
        t = find_table(doc, a.table)
        if t is None:
            print(json.dumps(query(doc, a.table, [], ""), ensure_ascii=False, indent=1))
            sys.exit(1)
        keys = t.get("key_columns", [0])
        parts = parts_of(doc, t)
        row_keys = []
        for k, part in enumerate(parts):
            q = part if k == 0 else _with_columns_of(part, t)
            nh = (q.get("header") or {}).get("n_header_rows", 1)
            row_keys += [[q["rows"][ri][kk] for kk in keys if kk < len(q["rows"][ri])] for ri in range(nh, len(q["rows"]))]
        cols = columns_of(t)
        print(json.dumps({"table_id": t["id"], "page": t["page"], "caption": t.get("caption"),
                          "parts": [[p["id"], p["page"]] for p in parts],
                          "columns": [c["name"] for c in cols],
                          "column_paths": ["/".join(c["path"]) for c in cols],
                          "row_key_columns": [cols[k]["name"] for k in keys if k < len(cols)],
                          "row_keys": row_keys}, ensure_ascii=False, indent=1))
        return
    r = query(doc, a.table, a.row, a.col or "")
    print(json.dumps(r, ensure_ascii=False, indent=1))
    sys.exit(0 if r["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
