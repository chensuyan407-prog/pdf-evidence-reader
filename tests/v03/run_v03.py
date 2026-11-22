#!/usr/bin/env python3
"""v0.3 checks on the v2-review defects, runnable against any Skill version.

Usage (from v0.3/):
    python tests/v03/run_v03.py --skill submission/scripts --label v03
    python tests/v03/run_v03.py --skill <copy of v0.2 scripts> --label v02

Groups: table (Attention p8 Table 2 + synthetic number forms), equation
(GloVe p4 eq. 8 symbol by symbol), cases (answers + verification),
controls (verifier positive/negative controls, verified after moving),
efficiency (cold / second query / cached / page range). Everything is located
by content and coordinates, never by block id. Writes tests/v03/out/<label>/report.json.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PDFS = os.path.join(ROOT, "tests", "pdfs")
sys.path.insert(0, HERE)
import agent_fill_v03 as AF  # noqa: E402


def load_mod(S, name):
    spec = importlib.util.spec_from_file_location(f"{name}_{abs(hash(S))}", os.path.join(S, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, S)
    spec.loader.exec_module(m)
    sys.path.remove(S)
    return m


def run(S, script, *args, timeout=600):
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(S, script), *args], capture_output=True, text=True,
                       encoding="utf-8", timeout=timeout)
    return p, round(time.time() - t, 3)


def nz(s):
    return re.sub(r"\s+", "", s or "")


class Report:
    def __init__(self):
        self.checks = []

    def add(self, cid, group, desc, ok, detail=None, evidence=None):
        self.checks.append({"id": cid, "group": group, "check": desc, "pass": bool(ok), "detail": detail, "evidence": evidence})
        print(("PASS " if ok else "FAIL ") + cid.ljust(34) + desc[:100], flush=True)


def table_checks(R, S, out):
    TQ = load_mod(S, "table_query")
    pd = os.path.join(out, "parsed", "attention")
    d = json.load(open(os.path.join(pd, "doc.json"), encoding="utf-8"))
    t = next((t for t in d["tables"] if t["page"] == 8), None)
    R.add("TAB-ATT-cols", "table", "Attention p8 Table 2 has 5 logical columns (Model, 2 BLEU, 2 Training Cost)",
          t is not None and t["n_cols"] == 5, {"n_cols": t and t["n_cols"], "columns": t and [c.get("name") for c in t.get("columns", [])]})
    names = ["ByteNet [18]", "Deep-Att + PosUnk [39]", "GNMT + RL [38]", "ConvS2S [9]", "MoE [32]",
             "Deep-Att + PosUnk Ensemble [39]", "GNMT + RL Ensemble [38]", "ConvS2S Ensemble [9]",
             "Transformer (base model)", "Transformer (big)"]
    got = [r[0] for r in t["rows"][(t.get("header") or {}).get("n_header_rows", 1):]] if t else []
    R.add("TAB-ATT-rownames", "table", "model names kept whole (no split at spaces/citations)", got == names, {"got": got})

    def q(row, col):
        return TQ.query(d, "Table 2", [f"Model={row}"], col)
    a = q("Transformer (base model)", "BLEU/EN-FR")
    R.add("TAB-ATT-38.1", "table", "Transformer (base model) x BLEU/EN-FR == 38.1", a.get("status") == "ok" and a.get("value") == "38.1",
          {k: a.get(k) for k in ("status", "value", "cell_bbox")})
    for row, mant, exp, val in (("Transformer (base model)", "3.3", "18", 3.3e18), ("Transformer (big)", "2.3", "19", 2.3e19)):
        a = q(row, "Training Cost (FLOPs)")
        raw_ok = a.get("value") == f"{mant} · 10^{exp}"
        norm_ok = a.get("value_numeric") is not None and abs(a["value_numeric"] - val) < 1e-6 * val and \
            a.get("value_pretty") == f"{mant} × 10" + exp.translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
        parts = a.get("value_parts") or []
        bb = a.get("cell_bbox")
        cover = bool(parts) and bb is not None and bb[0] <= min(p["bbox"][0] for p in parts) + 0.01 and \
            bb[2] >= max(p["bbox"][2] for p in parts) - 0.01 and any(p["script"] == "sup" and p["text"] == exp for p in parts)
        shared = a.get("shared_across") == ["Training Cost (FLOPs) / EN-DE", "Training Cost (FLOPs) / EN-FR"]
        tag = row.split()[1].strip("()")
        R.add(f"TAB-ATT-cost-{tag}-raw", "table", f"{row} cost as printed == '{mant} · 10^{exp}'", raw_ok, {"value": a.get("value"), "status": a.get("status")})
        R.add(f"TAB-ATT-cost-{tag}-norm", "table", f"{row} cost normalised == {val:g} ('{mant} × 10^{exp}')", norm_ok,
              {k: a.get(k) for k in ("value_numeric", "value_pretty", "value_canonical")})
        R.add(f"TAB-ATT-cost-{tag}-span", "table", f"{row} cost is one value shared by EN-DE and EN-FR", shared, {"shared_across": a.get("shared_across")})
        R.add(f"TAB-ATT-cost-{tag}-bbox", "table", f"{row} cost bbox covers mantissa, operator, 10 and raised exponent", cover,
              {"bbox": bb, "parts": parts})
    for row, col, want in (("Transformer (base model)", "BLEU/EN-DE", "27.3"), ("Transformer (big)", "BLEU/EN-DE", "28.4"),
                           ("Transformer (big)", "BLEU/EN-FR", "41.8"), ("GNMT + RL [38]", "Training Cost (FLOPs)/EN-DE", "2.3 · 10^19"),
                           ("GNMT + RL [38]", "Training Cost (FLOPs)/EN-FR", "1.4 · 10^20")):
        a = q(row, col)
        R.add(f"TAB-ATT-adj-{nz(row)[:12]}-{col[-5:]}", "table", f"{row} x {col} == {want} (neighbour columns kept apart)",
              a.get("status") == "ok" and a.get("value") == want, {"value": a.get("value"), "status": a.get("status")})
    a = q("GNMT + RL [38]", "Training Cost (FLOPs)")
    R.add("TAB-ATT-group-ambiguous", "table", "group-level query on a row with two different costs is reported ambiguous",
          a.get("status") == "col_ambiguous", {"status": a.get("status")})
    # synthetic number forms
    pn = os.path.join(out, "parsed", "numeric_table")
    dn = json.load(open(os.path.join(pn, "doc.json"), encoding="utf-8"))
    truth = json.load(open(os.path.join(HERE, "numeric_table.truth.json"), encoding="utf-8"))
    for row, vals in truth["cells"].items():
        for col, want in vals.items():
            if col.endswith("_footnote"):
                continue
            a = TQ.query(dn, "Table 1", [f"Method={row}"], col)
            got = a.get("value_numeric")
            ok = a.get("status") == "ok" and got is not None and abs(got - want) <= 1e-9 * max(1, abs(want))
            kind = "scientific" if isinstance(want, float) and abs(want) >= 1e15 else ("negative" if want < 0 else "number")
            R.add(f"TAB-NUM-{nz(row)[:10]}-{col[:6]}", "table", f"synthetic {kind}: {row} x {col} == {want:g}", ok,
                  {"status": a.get("status"), "value": a.get("value"), "value_numeric": got})
        if "Score_footnote" in vals:
            a = TQ.query(dn, "Table 1", [f"Method={row}"], "Score")
            R.add(f"TAB-NUM-fn-{nz(row)[:10]}", "table", f"synthetic footnote '{vals['Score_footnote']}' after {vals['Score']} kept as a marker, not an exponent",
                  a.get("footnote_marker") == vals["Score_footnote"] and a.get("value_numeric") == vals["Score"],
                  {"value": a.get("value"), "footnote_marker": a.get("footnote_marker"), "value_numeric": a.get("value_numeric")})
    tn = dn["tables"][0] if dn.get("tables") else None
    R.add("TAB-NUM-cols", "table", "synthetic table: 5 columns, citation label 'Prior work [12]' kept whole",
          tn is not None and tn["n_cols"] == 5 and any(r[0] == "Prior work [12]" for r in tn["rows"]),
          {"n_cols": tn and tn["n_cols"], "labels": tn and [r[0] for r in tn["rows"]]})


def equation_checks(R, S, out):
    pd = os.path.join(out, "parsed", "glove")
    d = json.load(open(os.path.join(pd, "doc.json"), encoding="utf-8"))
    e = next((e for e in d.get("equations", []) if str(e.get("number")) == "8"), None)
    R.add("EQ-8-found", "equation", "GloVe eq. (8) returned by number with page 4, bbox, crop, status",
          e is not None and e["page"] == 4 and e.get("bbox") and e.get("crop") and os.path.exists(os.path.join(pd, e["crop"]))
          and e.get("transcription_status") in ("auto", "partial", "crop_only"),
          {k: (e or {}).get(k) for k in ("page", "bbox", "crop", "transcription_status")})
    lx = nz((e or {}).get("expr_latex") or (e or {}).get("latex") or "")
    syms = {
        "sum limits i,j=1 and V": r"\sum_{i,j=1}^{V}",
        "weight f(X_ij)": "f(X_{ij})",
        "transpose of w_i": ("w_{i}^{T}", "w^{T}_{i}"),
        "tilde w_j": r"\tilde{w}_{j}",
        "tilde b_j": r"\tilde{b}_{j}",
        "minus before log": r"-\logX_{ij}",
        "square outside the bracket": ")^{2}",
    }
    for name, pat in syms.items():
        pats = pat if isinstance(pat, tuple) else (pat,)
        R.add(f"EQ-8-{nz(name)[:18]}", "equation", f"eq. (8) symbol: {name}", any(nz(p) in lx for p in pats), {"latex": lx[:200]})
    R.add("EQ-8-auto", "equation", "eq. (8) counted as automatic transcription only if status is 'auto' and no uncertain symbols",
          e is not None and e.get("transcription_status") == "auto" and not e.get("uncertain_symbols"),
          {"status": (e or {}).get("transcription_status"), "uncertain": (e or {}).get("uncertain_symbols")})
    gaps = [w for w in d.get("warnings", []) if w.get("code") == "equation_number_gap"]
    R.add("EQ-no-gap", "equation", "GloVe equation numbers have no gap (7 -> 8 -> 9 ...)", not gaps, {"gaps": gaps})


def case_checks(R, S, out):
    expect = {
        "attention": {"bleu": lambda v: v and nz(v) == "38.1",
                      "cost": lambda v: v and any(abs(x - 2.3e19) < 1e9 for x in vals(v)) and "FLOP" in v},
        "glove": {"objective": lambda v: v and all(nz(p) in nz(v) for p in ("∑_{i,j=1}^{V}", "f(X_{ij})", "w̃_{j}", "b̃_{j}", "−logX_{ij}", ")^{2}"))
                  and ("w_{i}^{T}" in nz(v) or "w^{T}_{i}" in nz(v)),
                  "alpha": lambda v: v and nz(v) == "3/4"},
        "hubble": {"velocity": lambda v: v and "−130" in v.replace("-", "−") and "km" in v,
                   "distance": lambda v: v and "0.214" in v and "10⁶" in v and "pc" in v},
    }
    summary = {}
    for case, fill in (("attention", AF.fill_attention), ("glove", AF.fill_glove), ("hubble", AF.fill_hubble)):
        cdir = os.path.join(out, "cases", case)
        shutil.rmtree(cdir, ignore_errors=True)
        os.makedirs(cdir)
        pd = os.path.join(cdir, "parsed", case)
        shutil.copytree(os.path.join(out, "parsed", case), pd)
        rp = os.path.join(cdir, "result.json")
        t0 = time.time()
        res = fill(S, pd, rp, cdir) if case == "hubble" else fill(S, pd, rp)
        p, vt = run(S, "verify_result.py", "--result", rp)
        try:
            vs = json.load(open(os.path.join(cdir, "verification.json"), encoding="utf-8"))["summary"]
        except Exception:
            vs = {"field_verdicts": {}, "needs_review": []}
        fv = vs.get("field_verdicts", {})
        summary[case] = {"seconds_fill_verify": round(time.time() - t0, 2), "verify_exit": p.returncode, "fields": {}}
        for f, ok in expect[case].items():
            v = res.get(f)
            good = bool(ok(v))
            verdict = fv.get(f, {}).get("verdict") if isinstance(fv.get(f), dict) else fv.get(f)
            review = fv.get(f, {}).get("needs_review") if isinstance(fv.get(f), dict) else None
            if review is None:  # older verifiers only list needs_review in the summary
                review = f in (vs.get("needs_review") or [])
            status = (res.get("field_status") or {}).get(f)
            summary[case]["fields"][f] = {"answer": v, "correct": good, "verdict": verdict, "needs_review": review,
                                          "field_status": status}
            R.add(f"CASE-{case}-{f}", "case", f"{case}.{f} correct vs original page", good,
                  {"answer": v, "verdict": verdict, "needs_review": review, "field_status": status}, rp)
    return summary


def vals(s):
    out = []
    m = re.search(r"([\d.]+)\s*[×·x]\s*10\s*(?:\^?\{?(-?\d+)\}?|([⁰¹²³⁴⁵⁶⁷⁸⁹]+))", s)
    if m:
        e = m.group(2) or m.group(3).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
        out.append(float(m.group(1)) * 10 ** int(e))
    return out


def control_checks(R, S, out):
    FX = os.path.join(HERE, "fixtures", "controls")
    exp = json.load(open(os.path.join(FX, "expected.json"), encoding="utf-8"))
    cdir = os.path.join(out, "controls")
    shutil.rmtree(cdir, ignore_errors=True)
    rows = []
    for name, want in exp.items():
        d = os.path.join(cdir, name)
        shutil.copytree(os.path.join(FX, name), d)  # verified after moving
        p, _ = run(S, "verify_result.py", "--result", os.path.join(d, "result.json"))
        try:
            v = json.load(open(os.path.join(d, "verification.json"), encoding="utf-8"))["summary"]
        except Exception:
            v = {"field_verdicts": {}}
        fv = v.get("field_verdicts", {})
        got = {f: (fv.get(f, {}).get("verdict"),
                   fv.get(f, {}).get("needs_review") if "needs_review" in fv.get(f, {}) else f in (v.get("needs_review") or []))
               for f in want}
        ok = all(got[f][0] == w[0] and bool(got[f][1]) == w[1] for f, w in want.items())
        negative = name not in ("tab_correct", "img_correct", "img_moved_with_pdf", "img_moved_without_pdf")
        # a wrong answer "accepted as supported": verdict starts with supported_ and no review requested
        accepted = negative and any((g[0] or "").startswith("supported_") and not g[1] for g in got.values())
        rows.append({"control": name, "exit": p.returncode, "got": got, "want": want, "wrong_answer_accepted": bool(accepted)})
        R.add(f"CTL-{name}", "control", f"verifier on '{name}' -> {want}", ok, {"exit": p.returncode, "got": got},
              os.path.join(d, "verification.json"))
    return rows


def efficiency(R, S, out):
    """Same question (Attention base EN-FR BLEU), same page range rules for A and B."""
    import pymupdf as fitz
    E = {}
    inp = os.path.join(out, "eff", "input")
    shutil.rmtree(os.path.join(out, "eff"), ignore_errors=True)
    os.makedirs(inp)
    shutil.copy(os.path.join(PDFS, "attention.pdf"), os.path.join(inp, "paper.pdf"))
    json.dump({"case_id": "attention", "materials": ["paper.pdf"], "output_contract": {"bleu": "string"}},
              open(os.path.join(inp, "case.json"), "w", encoding="utf-8"))
    qargs = lambda pd: [pd, "--table", "Table 2", "--row", "Model=Transformer (base model)", "--col", "BLEU/EN-FR"]
    # A: generic extraction of the same page range (whole document vs page 8 only) + page image
    for label, pages in (("A_full", None), ("A_page8", [8])):
        t = time.time()
        d = fitz.open(os.path.join(inp, "paper.pdf"))
        rng = pages or range(1, d.page_count + 1)
        txt = "\n".join(d[i - 1].get_text() for i in rng)
        d[7].get_pixmap(dpi=150).tobytes("png")
        E[label] = {"tool_seconds": round(time.time() - t, 3), "steps": 2, "found_38.1_in_text": "38.1" in txt,
                    "structured_lookup": False, "manual_box_search": "yes (read page image / count columns)"}

    def b_run(label, case_args, outdir):
        steps, secs = [], 0.0
        p, s1 = run(S, "run_case.py", "--case", os.path.join(inp, "case.json"), "--out", outdir, "--work",
                    os.path.join(outdir, "_work"), *case_args)
        steps.append(("run_case", s1, p.returncode))
        pd = os.path.join(outdir, "parsed", "paper")
        p2, s2 = run(S, "table_query.py", *qargs(pd))
        try:
            val = json.loads(p2.stdout).get("value")
        except Exception:
            val = None
        steps.append(("table_query", s2, p2.returncode))
        secs = s1 + s2
        E[label] = {"tool_seconds": round(secs, 3), "steps": len(steps), "detail": steps, "value": val,
                    "manual_box_search": "no" if val == "38.1" else "yes",
                    "run_case_rc": p.returncode, "stderr_tail": p.stderr[-200:] if p.returncode else ""}
        try:
            dj = json.load(open(os.path.join(pd, "doc.json"), encoding="utf-8"))
            E[label]["pages_parsed"] = dj.get("pages_parsed")
            E[label]["pages_not_parsed"] = dj.get("pages_not_parsed")
            E[label]["cache"] = (dj.get("meta") or {}).get("cache")
        except Exception:
            pass
        return pd
    cold = b_run("B_cold_full", ["--cache", "off"], os.path.join(out, "eff", "b_cold"))
    p, s = run(S, "table_query.py", *qargs(cold))
    try:
        v2 = json.loads(p.stdout).get("value")
    except Exception:
        v2 = None
    E["B_second_query_same_parse"] = {"tool_seconds": s, "steps": 1, "value": v2, "manual_box_search": "no" if v2 == "38.1" else "yes"}
    cache = os.path.join(out, "eff", "cache")
    b_run("B_cache_fill", ["--cache", cache], os.path.join(out, "eff", "b_cache1"))
    b_run("B_cache_hit_new_output", ["--cache", cache], os.path.join(out, "eff", "b_cache2"))
    b_run("B_page8_only", ["--cache", "off", "--pages", "8"], os.path.join(out, "eff", "b_p8"))
    R.add("EFF-second-query", "efficiency", "second query on an existing parse needs no re-parse and returns 38.1",
          E["B_second_query_same_parse"]["value"] == "38.1", E["B_second_query_same_parse"])
    hit = (E.get("B_cache_hit_new_output", {}).get("cache") or {}).get("hit")
    R.add("EFF-cache-hit", "efficiency", "re-running the case on the same PDF hits the parse cache", hit is True,
          {k: E.get("B_cache_hit_new_output", {}).get(k) for k in ("tool_seconds", "cache")})
    p8 = E.get("B_page8_only", {})
    R.add("EFF-page-range", "efficiency", "page-range parse answers from page 8 and lists unread pages",
          p8.get("value") == "38.1" and p8.get("pages_parsed") == [8] and len(p8.get("pages_not_parsed") or []) == 14,
          {k: p8.get(k) for k in ("value", "pages_parsed", "pages_not_parsed", "tool_seconds")})
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True)
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    S = os.path.abspath(a.skill)
    out = os.path.join(HERE, "out", a.label)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "parsed"))
    R = Report()
    timing = {}
    for name, pdf in (("attention", "attention.pdf"), ("glove", "glove.pdf"), ("hubble", "hubble.pdf")):
        p, s = run(S, "parse_pdf.py", os.path.join(PDFS, pdf), "--out", os.path.join(out, "parsed", name))
        timing[name] = s
    subprocess.run([sys.executable, os.path.join(HERE, "make_numeric_table.py"), os.path.join(out, "numeric_table.pdf")],
                   capture_output=True)
    p, s = run(S, "parse_pdf.py", os.path.join(out, "numeric_table.pdf"), "--out", os.path.join(out, "parsed", "numeric_table"))
    timing["numeric_table"] = s
    for fn in (table_checks, equation_checks):
        try:
            fn(R, S, out)
        except Exception as ex:
            R.add(f"{fn.__name__}-crashed", "harness", f"{fn.__name__} could not run on this version", False, repr(ex)[:300])
    cases = case_checks(R, S, out)
    controls = control_checks(R, S, out)
    eff = efficiency(R, S, out)
    # four headline metrics
    fields = [(c, f, x) for c, v in cases.items() for f, x in v["fields"].items()]
    metrics = {
        "final_answers_correct": [sum(1 for *_, x in fields if x["correct"]), len(fields)],
        "structured_correct_without_visual_reading": [
            sum(1 for *_, x in fields if x["correct"] and x["field_status"] == "text_layer"
                and (x["verdict"] or "").startswith("supported_by_")), len(fields)],
        "wrong_answers_accepted_as_supported": [sum(1 for r in controls if r["wrong_answer_accepted"]),
                                                sum(1 for r in controls if r["control"] not in
                                                    ("tab_correct", "img_correct", "img_moved_with_pdf", "img_moved_without_pdf"))],
        "fields_requiring_human_review": [sum(1 for *_, x in fields if x["needs_review"]), len(fields)],
    }
    summ = {}
    for c in R.checks:
        s = summ.setdefault(c["group"], [0, 0])
        s[0] += c["pass"]
        s[1] += 1
    report = {"label": a.label, "skill": S, "parse_seconds": timing, "summary": {k: f"{v[0]}/{v[1]}" for k, v in summ.items()},
              "metrics": metrics, "cases": cases, "controls": controls, "efficiency": eff, "checks": R.checks}
    json.dump(report, open(os.path.join(out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"summary": report["summary"], "metrics": metrics, "parse_seconds": timing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
