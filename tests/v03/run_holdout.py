#!/usr/bin/env python3
"""Held-out evaluation on papers never opened during development (see tests/HOLDOUT_v0.3.md).

Usage (from v0.3/): python tests/v03/run_holdout.py --skill submission/scripts --label v03
Ground truth: tests/holdout_truth_v0.3.json (written from page images before this ran).
"""
import argparse
import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)


def load_mod(S, name):
    spec = importlib.util.spec_from_file_location(f"{name}_{abs(hash(S))}", os.path.join(S, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, S)
    spec.loader.exec_module(m)
    sys.path.remove(S)
    return m


def nz(s):
    return re.sub(r"\s+", "", s or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    S = os.path.abspath(a.skill)
    TQ = load_mod(S, "table_query")
    # scoring helpers come from the harness's own copy, so every tested version is scored the same way
    V = load_mod(os.path.join(os.path.dirname(TESTS), "submission", "scripts"), "verify_result")
    truth = json.load(open(os.path.join(TESTS, "holdout_truth_v0.3.json"), encoding="utf-8"))
    out = os.path.join(HERE, "out", "holdout_" + a.label)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    checks = []

    def add(cid, desc, ok, detail):
        checks.append({"id": cid, "check": desc, "pass": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + cid.ljust(30) + desc[:100], flush=True)

    docs, secs = {}, {}
    for key in ("kaplan_2001.08361", "bn_1502.03167"):
        pdf = os.path.join(TESTS, truth[key]["pdf"])
        t = time.time()
        subprocess.run([sys.executable, os.path.join(S, "parse_pdf.py"), pdf, "--out", os.path.join(out, key)], capture_output=True)
        secs[key] = round(time.time() - t, 2)
        docs[key] = json.load(open(os.path.join(out, key, "doc.json"), encoding="utf-8"))

    def find_table(doc, page, prefix):
        c = [t for t in doc["tables"] if t["page"] == page and (t.get("caption") or "").startswith(prefix)]
        if not c:
            c = [t for t in doc["tables"] if t["page"] == page]
        return c

    # ---- Kaplan tables 5 and 6
    K = docs["kaplan_2001.08361"]
    for tk in ("table5_p20", "table6_p20"):
        tt = truth["kaplan_2001.08361"][tk]
        cand = find_table(K, tt["page"], tt["caption_prefix"])
        t = next((x for x in cand if (x.get("caption") or "").startswith(tt["caption_prefix"])), None)
        add(f"K-{tk}-found", f"Kaplan {tt['caption_prefix']} detected and linked to its caption", t is not None,
            {"tables_on_page": [(x["id"], (x.get("caption") or "")[:20], x["n_cols"]) for x in cand]})
        if not t:
            for r in tt["rows"]:
                add(f"K-{tk}-{r['row_contains']}", f"{tt['caption_prefix']} row {r['row_contains']} value {r['value_numeric']:g}", False, "table not found")
            continue
        nh = (t.get("header") or {}).get("n_header_rows", 1)
        cols = TQ.columns_of(t)
        vc = next((c for c in cols if tt["value_column"].lower() in c["name"].lower()), None)
        for r in tt["rows"]:
            rows = [ri for ri in range(nh, len(t["rows"])) if any(r["row_contains"] in (t["rows"][ri][k] or "") for k in range(len(cols)) if vc is None or k != vc["index"])]
            if len(rows) != 1 or vc is None:
                add(f"K-{tk}-{r['row_contains']}", f"{tt['caption_prefix']} row {r['row_contains']} -> {r['value_numeric']:g}", False,
                    {"rows_matched": len(rows), "value_column": vc and vc["name"], "columns": [c["name"] for c in cols]})
                continue
            ri = rows[0]
            k0 = TQ.span_start(t, ri, vc["index"]) if hasattr(TQ, "span_start") else vc["index"]
            cell = t["rows"][ri][k0] or ""
            vals = V.values_in(cell)
            num_ok = any(abs(x - r["value_numeric"]) <= 1e-9 * r["value_numeric"] for x in vals)
            txt_ok = all(w in cell for w in r["value_text_contains"])
            add(f"K-{tk}-{r['row_contains']}", f"{tt['caption_prefix']} row '{r['row_contains']}' -> {r['value_numeric']:g} {r['value_text_contains']}",
                num_ok and txt_ok, {"cell": cell, "parsed_values": vals, "row": t["rows"][ri]})
    # ---- Kaplan equation (1.4)
    et = truth["kaplan_2001.08361"]["equation_1_4_p5"]
    e = next((x for x in K.get("equations", []) if str(x.get("number")) == et["number"] and x["page"] == et["page"]), None)
    add("K-eq1.4-found", "Kaplan eq. (1.4) returned by number with bbox, crop and status", e is not None and e.get("crop"),
        {k: (e or {}).get(k) for k in ("page", "bbox", "crop", "transcription_status")})
    lx = nz((e or {}).get("expr_latex") or (e or {}).get("latex") or "")
    sym_ok = {}
    for name, pats in et["symbols"].items():
        ok = all(nz(p) in lx for p in pats)
        sym_ok[name] = ok
        add(f"K-eq1.4-{nz(name)[:16]}", f"eq. (1.4) symbol: {name}", ok, {"latex": lx[:240], "want": pats})
    status = (e or {}).get("transcription_status")
    all_sym = all(sym_ok.values())
    add("K-eq1.4-honest-status", "eq. (1.4) status is 'auto' only if every symbol checked is right (no false 'auto')",
        not (status == "auto" and not all_sym), {"status": status, "all_symbols_right": all_sym, "uncertain": (e or {}).get("uncertain_symbols")})
    # ---- BN table captioned as Figure 4
    B = docs["bn_1502.03167"]
    bt = truth["bn_1502.03167"]["table_p8_captioned_figure4"]
    cand = [t for t in B["tables"] if t["page"] == bt["page"]]
    add("B-table-found", "BN p8 results table (captioned 'Figure 4') detected", bool(cand),
        {"tables": [(t["id"], t.get("caption"), t["n_cols"]) for t in cand]})
    for c in bt["cells"]:
        ok, detail = False, "table not found"
        if cand:
            t = cand[0]
            q = TQ.query(B, t["id"], [f"{bt['row_key_column']}={c['row']}"], c["col"])
            vals = V.values_in(q.get("value") or "")
            ok = q.get("status") == "ok" and any(abs(x - c["value_numeric"]) < 1e-9 * max(1, c["value_numeric"]) for x in vals)
            detail = {k: q.get(k) for k in ("status", "value", "value_numeric")}
        add(f"B-{nz(c['row'])[:14]}-{nz(c['col'])[:8]}", f"BN {c['row']} x {c['col']} == {c['value_numeric']:g}", ok, detail)
    # ---- Chinchilla (spare, used as third held-out)
    ck = "chinchilla_2203.15556"
    if ck in truth:
        pdf = os.path.join(TESTS, truth[ck]["pdf"])
        t0 = time.time()
        subprocess.run([sys.executable, os.path.join(S, "parse_pdf.py"), pdf, "--out", os.path.join(out, ck)], capture_output=True)
        secs[ck] = round(time.time() - t0, 2)
        C = json.load(open(os.path.join(out, ck, "doc.json"), encoding="utf-8"))
        for tk in ("table3_p8", "table2_p8"):
            tt = truth[ck][tk]
            t = next((x for x in C["tables"] if (x.get("caption") or "").startswith(tt["caption_prefix"])), None)
            add(f"C-{tk}-found", f"Chinchilla {tt['caption_prefix']} detected and linked to its caption", t is not None,
                {"tables_on_page": [(x["id"], (x.get("caption") or "")[:20], x["n_cols"]) for x in C["tables"] if x["page"] == tt["page"]]})
            for c in tt["cells"]:
                ok, detail = False, "table not found"
                if t is not None:
                    cols = TQ.columns_of(t)
                    kc = next((x["name"] for x in cols if x["name"].lower().startswith(tt["row_key_column"].lower())), cols[0]["name"])
                    col = c.get("col") or next((x["name"] for x in cols if c["col_contains"].lower() in x["name"].lower()), "?")
                    q = TQ.query(C, t["id"], [f"{kc}={c['row']}"], col)
                    if "value_numeric" in c:
                        vals = V.values_in(q.get("value") or "")
                        ok = q.get("status") == "ok" and any(abs(x - c["value_numeric"]) <= 1e-9 * abs(c["value_numeric"]) for x in vals)
                    else:
                        ok = q.get("status") == "ok" and nz(q.get("value")) == nz(c["value_text"])
                    detail = {"status": q.get("status"), "value": q.get("value"), "column": col, "key": kc}
                add(f"C-{tk}-{nz(c['row'])[:12]}-{nz(c.get('col') or c['col_contains'])[:10]}",
                    f"{tt['caption_prefix']} {c['row']} x {c.get('col') or c['col_contains']} == {c.get('value_numeric', c.get('value_text'))}", ok, detail)
        for ek in ("equation_2_p6", "equation_3_p6"):
            et = truth[ck][ek]
            e = next((x for x in C.get("equations", []) if str(x.get("number")) == et["number"] and x["page"] == et["page"]), None)
            add(f"C-eq{et['number']}-found", f"Chinchilla eq. ({et['number']}) returned by number with bbox, crop and status",
                e is not None and e.get("crop"), {k: (e or {}).get(k) for k in ("page", "bbox", "crop", "transcription_status")})
            lx = nz((e or {}).get("expr_latex") or (e or {}).get("latex") or "")
            sym = {}
            for name, pats in et["symbols"].items():
                ok = all(any(nz(alt) in lx for alt in p.split("|")) for p in pats)
                sym[name] = ok
                add(f"C-eq{et['number']}-{nz(name)[:14]}", f"Chinchilla eq. ({et['number']}) symbol: {name}", ok, {"latex": lx[:240], "want": pats})
            st = (e or {}).get("transcription_status")
            add(f"C-eq{et['number']}-honest-status", f"eq. ({et['number']}) status 'auto' only if every checked symbol is right",
                not (st == "auto" and not all(sym.values())), {"status": st, "all_symbols_right": all(sym.values()),
                                                                "uncertain": (e or {}).get("uncertain_symbols")})
    # ---- verifier on held-out data: correct table answer + constructed truncation
    t5 = next((x for x in K["tables"] if (x.get("caption") or "").startswith("Table 5")), None)
    if t5 is not None:
        nh = (t5.get("header") or {}).get("n_header_rows", 1)
        cols = TQ.columns_of(t5)
        kc = cols[0]["name"]
        vc = next((c for c in cols if "scale" in c["name"].lower()), None)
        ri = next((ri for ri in range(nh, len(t5["rows"])) if "0.076" in (t5["rows"][ri][0] or "")), None)
        if vc is not None and ri is not None:
            label = t5["rows"][ri][0]
            for name, mutate in (("correct", None), ("truncated", "trunc")):
                d = os.path.join(out, "verify_" + name)
                pd = os.path.join(d, "parsed", "kaplan")
                shutil.copytree(os.path.join(out, "kaplan_2001.08361"), pd)
                val = t5["rows"][ri][vc["index"]]
                ans = "8.8 × 10¹³ params (non-embedding)"
                if mutate:
                    dj = json.load(open(os.path.join(pd, "doc.json"), encoding="utf-8"))
                    tt = next(x for x in dj["tables"] if x["id"] == t5["id"])
                    cell = next(c for c in tt["cells"] if c["row"] == ri and c["col"] == vc["index"])
                    parts = cell.get("parts") or []
                    cut = next((i for i, p in enumerate(parts) if p["text"] in ("×", "·")), None)
                    if cut is not None:
                        keep = parts[:cut + 1]
                        cell["text"] = " ".join(p["text"] for p in keep)
                        cell["bbox"] = [min(p["bbox"][0] for p in keep), min(p["bbox"][1] for p in keep),
                                        max(p["bbox"][2] for p in keep), max(p["bbox"][3] for p in keep)]
                        cell.pop("norm", None)
                        tt["rows"][ri][vc["index"]] = cell["text"]
                        val = cell["text"]
                        ans = re.sub(r"^.*?=\s*", "", cell["text"])
                    json.dump(dj, open(os.path.join(pd, "doc.json"), "w", encoding="utf-8"), ensure_ascii=False)
                res = {"case_id": "kaplan", "_contract_fields": ["nc"], "nc": ans, "unreported": [], "uncertainty": [],
                       "conflicts": [], "field_status": {"nc": "text_layer"}, "status": "success",
                       "evidence": [{"field": "nc", "type": "table_cell", "table": "Table 5", "page": 20,
                                     "row": {kc: label}, "column": vc["name"], "value": val}]}
                os.makedirs(d, exist_ok=True)
                json.dump(res, open(os.path.join(d, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                p = subprocess.run([sys.executable, os.path.join(S, "verify_result.py"), "--result", os.path.join(d, "result.json")],
                                   capture_output=True, text=True, encoding="utf-8")
                fv = json.load(open(os.path.join(d, "verification.json"), encoding="utf-8"))["summary"]["field_verdicts"]["nc"]
                want = ("supported_by_table_cell", False) if name == "correct" else ("parsed_cell_incomplete", True)
                add(f"K-verify-{name}", f"verifier on held-out Table 5 answer ({name}) -> {want}",
                    (fv.get("verdict"), bool(fv.get("needs_review"))) == want,
                    {"answer": ans, "cited_value": val, "verdict": fv.get("verdict"), "needs_review": fv.get("needs_review"),
                     "reasons": fv.get("reasons")})
    n = sum(c["pass"] for c in checks)
    rep = {"label": a.label, "skill": S, "parse_seconds": secs, "passed": f"{n}/{len(checks)}", "checks": checks}
    json.dump(rep, open(os.path.join(out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"passed": rep["passed"], "parse_seconds": secs}))


if __name__ == "__main__":
    main()
