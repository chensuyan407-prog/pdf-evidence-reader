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

Prints JSON: status (ok | table_not_found | row_not_found | row_ambiguous |
col_not_found | col_ambiguous), value, page, table_id, row, col,
column_path, row_key, cell_bbox, caption. Exit code 0 only when status=ok.
"""
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
    c = [t for t in tabs if _n(t.get("caption")).startswith(key) and not _n(t.get("caption")).endswith("(continued)")]
    return c[0] if c else None


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
        return {"status": "table_not_found", "table": table,
                "available": [(x["id"], x["page"], x.get("caption")) for x in doc.get("tables", [])]}
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
                    "table_id": t["id"], "page": t["page"]}
        conds.append((kc[0]["index"], _n(v), k, v))
    def cell_is(text, v):  # a trailing footnote marker ("Baseline^a") does not change the label
        return _n(text) == v or _n(re.sub(r"\^\S+$", "", text or "")) == v
    cand = [ri for ri in body if all(cell_is(t["rows"][ri][ci], v) for ci, v, _, _ in conds)]
    base = {"table_id": t["id"], "page": t["page"], "caption": t.get("caption")}
    if not cand:
        return dict(base, status="row_not_found", conditions=rows)
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
        return dict(base, status="col_not_found", column=col, columns=[c["name"] for c in cols])
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
    a = ap.parse_args()
    doc = load(a.parsed)
    if a.list:
        t = find_table(doc, a.table)
        if t is None:
            print(json.dumps({"status": "table_not_found"}, ensure_ascii=False))
            sys.exit(1)
        nh = (t.get("header") or {}).get("n_header_rows", 1)
        keys = t.get("key_columns", [0])
        print(json.dumps({"table_id": t["id"], "page": t["page"], "caption": t.get("caption"),
                          "columns": [c["name"] for c in columns_of(t)],
                          "row_keys": [[t["rows"][ri][k] for k in keys] for ri in range(nh, len(t["rows"]))]},
                         ensure_ascii=False, indent=1))
        return
    r = query(doc, a.table, a.row, a.col or "")
    print(json.dumps(r, ensure_ascii=False, indent=1))
    sys.exit(0 if r["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
