#!/usr/bin/env python3
"""Search a parsed document (doc.json from parse_pdf.py) for evidence.

Usage:
    python find_evidence.py PARSED_DIR "query words" [--top 8] [--role body,caption]
    python find_evidence.py PARSED_DIR --regex "Table\\s*2" [--top 8]
    python find_evidence.py PARSED_DIR --section "2.1"        # dump one section
    python find_evidence.py PARSED_DIR --table T2             # dump one table
    python find_evidence.py PARSED_DIR --page 3               # dump one page in reading order
    python find_evidence.py PARSED_DIR --equation 8           # one equation by its number

Prints JSON hits: block_id, page, role, section_path, bbox, score and the full
block text, so a quote can be copied verbatim into result.json evidence.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]+|\d+(?:\.\d+)?%?|[一-鿿]")
STOP = set("the a an of and or to in on for with by is are was were be this that these those from as at it its we our".split())


def tokens(s):
    return [t.lower() for t in TOKEN.findall(s or "") if t.lower() not in STOP]


def load(parsed_dir):
    p = parsed_dir if parsed_dir.endswith(".json") else os.path.join(parsed_dir, "doc.json")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def units(doc):
    """Searchable units: blocks + table rows + references."""
    out = []
    for b in doc.get("blocks", []):
        if b["role"] == "table":
            continue
        out.append({"kind": "block", "block_id": b["id"], "page": b["page"], "role": b["role"],
                    "section_path": b.get("section_path"), "bbox": b["bbox"], "text": b["text"],
                    "marked": b.get("marked")})
    for t in doc.get("tables", []):
        hdr = t["rows"][0] if t["rows"] else []
        for ri, row in enumerate(t["rows"]):
            txt = " | ".join(c or "" for c in row)
            out.append({"kind": "table_row", "table_id": t["id"], "row": ri, "page": t["page"], "role": "table",
                        "caption": t.get("caption"), "bbox": t["bbox"], "header": hdr, "text": txt})
    return out


def search(doc, query, top=8, roles=None, regex=None):
    us = units(doc)
    if roles:
        us = [u for u in us if u["role"] in roles]
    if regex:
        rx = re.compile(regex, re.I)
        hits = [dict(u, score=1.0) for u in us if rx.search(u["text"])]
        return hits[:top]
    q = tokens(query)
    if not q:
        return []
    df = Counter()
    toks = []
    for u in us:
        t = tokens(u["text"] + " " + (u.get("caption") or ""))
        toks.append(t)
        for w in set(t):
            df[w] += 1
    N = len(us) or 1
    avg = sum(len(t) for t in toks) / N
    hits = []
    for u, t in zip(us, toks):
        tf = Counter(t)
        s = 0.0
        for w in q:
            if tf[w]:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * 2.2 / (tf[w] + 1.2 * (0.25 + 0.75 * len(t) / (avg or 1)))
        phrase = query.lower().strip()
        if len(phrase) > 6 and phrase in u["text"].lower():
            s += 5
        if s > 0:
            hits.append(dict(u, score=round(s, 3)))
    hits.sort(key=lambda h: -h["score"])
    return hits[:top]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parsed_dir")
    ap.add_argument("query", nargs="?", default="")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--role", default=None)
    ap.add_argument("--regex", default=None)
    ap.add_argument("--section", default=None, help="section number or title substring")
    ap.add_argument("--table", default=None)
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--equation", default=None, help="equation number, e.g. 8")
    a = ap.parse_args()
    doc = load(a.parsed_dir)
    if a.equation:
        eqs = [e for e in doc.get("equations", []) if str(e.get("number")) == str(a.equation)]
        if not eqs:
            print(json.dumps({"status": "not_found", "equation": a.equation,
                              "numbers": [e.get("number") for e in doc.get("equations", []) if e.get("number")]}, ensure_ascii=False))
            sys.exit(1)
        print(json.dumps(eqs[0] if len(eqs) == 1 else eqs, ensure_ascii=False, indent=1))
        return
    if a.table:
        t = next((t for t in doc["tables"] if t["id"] == a.table), None)
        print(json.dumps(t, ensure_ascii=False, indent=1))
        return
    if a.section:
        sec = next((s for s in doc["sections"] if s["number"] == a.section or a.section.lower() in s["title"].lower()), None)
        if not sec:
            print(json.dumps({"error": "section not found", "sections": [(s["number"], s["title"]) for s in doc["sections"]]}, ensure_ascii=False))
            return
        ids = {sec["id"]} | {s["id"] for s in doc["sections"] if s["parent"] == sec["id"]}
        blocks = [b for b in doc["blocks"] if b.get("section_id") in ids or b["id"] == sec["block_id"]]
        print(json.dumps({"section": sec, "blocks": [{k: b.get(k) for k in ("id", "page", "role", "bbox", "text")} for b in blocks]}, ensure_ascii=False, indent=1))
        return
    if a.page:
        blocks = [b for b in doc["blocks"] if b["page"] == a.page]
        print(json.dumps([{k: b.get(k) for k in ("id", "order", "role", "column", "bbox", "text")} for b in blocks], ensure_ascii=False, indent=1))
        return
    roles = set(a.role.split(",")) if a.role else None
    hits = search(doc, a.query, a.top, roles, a.regex)
    print(json.dumps(hits, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
