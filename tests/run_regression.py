#!/usr/bin/env python3
"""Regression checks on the real-paper grading cases, runnable against any Skill version.

Usage (from v0.2/):
    python tests/run_regression.py --skill submission/scripts --label new
    python tests/run_regression.py --skill ../bundle/submission/scripts --label old --parse-only

Checks are located by content and coordinates, never by block id. Results go
to tests/out/<label>/report.json. --parse-only skips the steps that need the
v0.2 tools (agent fill, image crops) so the v0.1 parser can be measured with
the same checks.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PDFS = os.path.join(HERE, "pdfs")
NEW_SCRIPTS = os.path.join(os.path.dirname(HERE), "submission", "scripts")
sys.path.insert(0, NEW_SCRIPTS)
import table_query  # noqa: E402  (query logic is shared so old/new parse outputs are read the same way)

sys.path.insert(0, HERE)
import agent_fill  # noqa: E402


def parse(S, pdf, out, extra=()):
    shutil.rmtree(out, ignore_errors=True)
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(S, "parse_pdf.py"), pdf, "--out", out, *extra],
                       capture_output=True, text=True, encoding="utf-8")
    secs = round(time.time() - t, 2)
    doc = None
    if os.path.exists(os.path.join(out, "doc.json")):
        doc = json.load(open(os.path.join(out, "doc.json"), encoding="utf-8"))
    return doc, secs, p.returncode, p.stderr[-400:]


def nz(s):
    return re.sub(r"\s+", "", s or "")


def find_block(doc, needle, roles=None, include_aside=True):
    pool = doc["blocks"] + (doc.get("headers_footers", []) if include_aside else [])
    for b in pool:
        if needle in b["text"] and (roles is None or b["role"] in roles):
            return b
    return None


def check(results, cid, group, desc, ok, detail, evidence=None):
    results.append({"id": cid, "group": group, "check": desc, "pass": bool(ok), "detail": detail, "evidence": evidence})


def table_checks(R, doc, gdir):
    asks = [
        ("T1", ["Name=HellaSwag", "Metric=acc", "Split=dev"], "Zero-Shot/13B", "70.9"),
        ("T2", ["Name=LAMBADA", "Metric=ppl", "Split=test"], "One-Shot/6.7B", "4.61"),
        ("T3a", ["Name=HellaSwag", "Metric=acc", "Split=dev"], "Zero-Shot/175B", "78.9"),
        ("T3b", ["Name=HellaSwag", "Metric=acc", "Split=dev"], "Zero-Shot/6.7B", "67.4"),
        ("T4", ["Name=LAMBADA", "Metric=acc", "Split=test"], "One-Shot/6.7B", "65.4"),
    ]
    for cid, rows, col, want in asks:
        q = table_query.query(doc, "Table H.1", rows, col)
        if q["status"] == "table_not_found":
            q = table_query.query(doc, "page:63", rows, col)
        got = q.get("value")
        check(R, cid, "structured_table", f"GPT-3 p63 Table H.1 {'/'.join(r.split('=')[1] for r in rows)} x {col} == {want}",
              q["status"] == "ok" and got == want, {"status": q["status"], "value": got, "page": q.get("page"),
                                                    "column_path": q.get("column_path"), "cell_bbox": q.get("cell_bbox")},
              os.path.join(gdir, "doc.json"))
    q = table_query.query(doc, "Table 3.4", ["Setting=GPT-3 Few-Shot"], "De→En")
    check(R, "T5", "structured_table", "GPT-3 p15 Table 3.4 (2nd dev table) GPT-3 Few-Shot x De→En == 40.6",
          q["status"] == "ok" and q.get("value") == "40.6", {"status": q["status"], "value": q.get("value")}, os.path.join(gdir, "doc.json"))
    hb = [b for b in doc["blocks"] if b["page"] == 63 and "HellaSwag" in b["text"]]
    bad = [b["id"] for b in hb if (b.get("section_path") or [""])[-1:] == ["Acknowledgements"]]
    check(R, "L5", "layout", "GPT-3 p63 Table H.1 data not attached to the Acknowledgements section",
          hb and not bad, {"blocks": [(b["id"], b["role"], b.get("section_path")) for b in hb]}, os.path.join(gdir, "doc.json"))


def resnet_layout_checks(R, doc, rdir):
    ev = os.path.join(rdir, "doc.json")
    a1 = find_block(doc, "Deeper neural networks are more difficult to train", include_aside=False)
    a2 = find_block(doc, "The depth of representations is of central importance", include_aside=False)
    if a1 and a2:
        i1, i2 = doc["blocks"].index(a1), doc["blocks"].index(a2)
        between = doc["blocks"][i1 + 1:i2] if i2 > i1 else ["(order reversed)"]
        intr = [(b["id"], b["role"], b["text"][:40]) for b in between if isinstance(b, dict)]
        check(R, "L1", "layout", "ResNet p1 abstract paragraphs 1 and 2 consecutive in reading order",
              i2 == i1 + 1, {"para1": a1["id"], "para2": a2["id"], "in_between": intr}, ev)
        right = [x for x in between if isinstance(x, dict) and x["bbox"][0] > 300]
        check(R, "L2", "layout", "ResNet p1 no right-column caption/body inserted inside the abstract",
              i2 > i1 and not right, {"right_column_blocks_between": [(b["id"], b["text"][:40]) for b in right]}, ev)
    else:
        check(R, "L1", "layout", "ResNet p1 abstract paragraphs found", False, {"a1": bool(a1), "a2": bool(a2)}, ev)
    for cid, t in (("L3a", "training error (%)"), ("L3b", "test error (%)")):
        b = find_block(doc, t)
        check(R, cid, "layout", f"ResNet p1 axis title '{t}' assigned to the figure (not margin/body)",
              b is not None and b["role"] == "figure_text", {"role": b and b["role"], "id": b and b["id"]}, ev)
    b = find_block(doc, "arXiv:1512.03385")
    check(R, "L4a", "layout", "ResNet p1 arXiv stamp removed from body (margin_note)",
          b is not None and b["role"] == "margin_note", {"role": b and b["role"]}, ev)
    pn = [x for x in doc.get("headers_footers", []) if x["page"] == 1 and x["text"].strip() == "1"]
    inbody = [x for x in doc["blocks"] if x["page"] == 1 and x["text"].strip() == "1"]
    check(R, "L4b", "layout", "ResNet p1 page number removed from body", pn and not inbody,
          {"footer": [x["role"] for x in pn], "in_body": len(inbody)}, ev)


def formula_checks(R, doc, rdir):
    want = {"1": "𝐲=ℱ(𝐱,{W_{i}})+𝐱.(1)", "2": "𝐲=ℱ(𝐱,{W_{i}})+W_{s}𝐱.(2)"}
    for n, w in want.items():
        e = next((e for e in doc.get("equations", []) if e.get("page") == 3 and e.get("number") == n), None)
        got = e and (e.get("unicode_math") or e.get("marked") or e.get("text"))
        crop_ok = bool(e and e.get("crop") and os.path.exists(os.path.join(rdir, e["crop"])))
        check(R, f"F{n}", "formula", f"ResNet p3 eq ({n}) symbol-level: bold y/x, calligraphic F, subscripts, number",
              e is not None and nz(got) == w, {"got": got, "want": w, "latex": e and e.get("latex")}, os.path.join(rdir, "doc.json"))
        check(R, f"F{n}c", "formula", f"ResNet p3 eq ({n}) crop image saved", crop_ok, {"crop": e and e.get("crop")})


def scanned_checks(R, doc, bdir, S, parse_only):
    p1 = doc["pages"][0] if doc else {}
    img = p1.get("page_image")
    check(R, "S1", "scanned", "BERT scan p1: page image + layout regions provided when OCR is unavailable",
          bool(img) and os.path.exists(os.path.join(bdir, img)) and len(p1.get("image_regions") or []) > 0,
          {"page_image": img, "n_regions": len(p1.get("image_regions") or []), "status": doc and doc["status"]})


def verifier_checks(R, S, parsed_gpt3, out):
    fx = os.path.join(HERE, "fixtures", "verifier")
    expected = json.load(open(os.path.join(fx, "expected.json"), encoding="utf-8"))
    for name, exp in expected.items():
        src = os.path.join(fx, name, "result.json")
        if not os.path.exists(src):
            continue
        d = os.path.join(out, "verifier", name)
        os.makedirs(d, exist_ok=True)
        shutil.copy(src, os.path.join(d, "result.json"))
        p = subprocess.run([sys.executable, os.path.join(S, "verify_result.py"), "--result", os.path.join(d, "result.json"),
                            "--parsed", parsed_gpt3], capture_output=True, text=True, encoding="utf-8")
        try:
            summ = json.loads(p.stdout)
        except Exception:
            summ = {"raw": p.stdout[-300:]}
        fv = {f: v.get("verdict") for f, v in (summ.get("field_verdicts") or {}).items()}
        should_fail = name != "correct"
        caught = (p.returncode != 0) if should_fail else (p.returncode == 0)
        verdict_ok = all(fv.get(f) == v for f, v in exp.items()) if fv else None
        check(R, f"V-{name}", "verifier", f"verifier on '{name}' -> expect {exp}",
              caught and (verdict_ok is not False), {"exit": p.returncode, "field_verdicts": fv or "(not reported by this version)",
                                                     "issues": [i.get("code") for i in summ.get("issues", [])][:6]},
              os.path.join(d, "verification.json"))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # GBK consoles cannot print the math glyphs in check names
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default=NEW_SCRIPTS)
    ap.add_argument("--label", default="new")
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    S = os.path.abspath(a.skill)
    out = os.path.join(HERE, "out", a.label)
    os.makedirs(out, exist_ok=True)
    R, timing = [], {}
    docs = {}
    for name in ("resnet", "gpt3", "bert", "bert_masked", "encrypted_resnet"):
        pdf = os.path.join(PDFS, name + ".pdf")
        if not os.path.exists(pdf):
            continue
        docs[name], timing[name], rc, err = parse(S, pdf, os.path.join(out, name))
        if docs[name] is None:
            check(R, f"P-{name}", "robustness", f"parse {name}", False, {"rc": rc, "stderr": err})
    g = os.path.join(out, "gpt3")
    r = os.path.join(out, "resnet")
    b = os.path.join(out, "bert")
    table_checks(R, docs["gpt3"], g)
    resnet_layout_checks(R, docs["resnet"], r)
    formula_checks(R, docs["resnet"], r)
    scanned_checks(R, docs["bert"], b, S, a.parse_only)
    e = docs.get("encrypted_resnet")
    check(R, "X1", "robustness", "encrypted PDF without password -> failed/encrypted",
          e is not None and e["status"] == "failed" and e.get("error") == "encrypted", {"status": e and e["status"], "error": e and e.get("error")})
    d13, s13, rc13, err13 = parse(S, os.path.join(PDFS, "resnet.pdf"), os.path.join(out, "resnet_p1_3"), ("--pages", "1,3"))
    check(R, "X2", "robustness", "--pages 1,3 (non-contiguous) parses", d13 is not None and d13["status"] == "success",
          {"rc": rc13, "stderr_tail": err13[-200:] if d13 is None else ""})
    verifier_checks(R, S, g, out)
    answers = {}
    if not a.parse_only:
        cases = os.path.join(out, "cases")
        os.makedirs(cases, exist_ok=True)
        expect = json.load(open(os.path.join(HERE, "fixtures", "expected_answers.json"), encoding="utf-8"))
        fills = {"gpt3": lambda p: agent_fill.fill_gpt3(S, g, p), "resnet": lambda p: agent_fill.fill_resnet(S, r, p),
                 "bert": lambda p: agent_fill.fill_bert(S, b, p), "bert_masked": lambda p: agent_fill.fill_bert(S, os.path.join(out, "bert_masked"), p, masked=True)}
        for case, fn in fills.items():
            rp = os.path.join(cases, case, "result.json")
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            t = time.time()
            res = fn(rp)
            pdir = {"gpt3": g, "resnet": r, "bert": b, "bert_masked": os.path.join(out, "bert_masked")}[case]
            p = subprocess.run([sys.executable, os.path.join(S, "verify_result.py"), "--result", rp, "--parsed", pdir],
                               capture_output=True, text=True, encoding="utf-8")
            vs = json.loads(p.stdout)
            secs = round(time.time() - t, 2)
            for f, rule in expect[case].items():
                got = res.get(f)
                if rule.get("null"):
                    ok = got is None
                elif "contains_nospace" in rule:
                    ok = got is not None and all(nz(x) in nz(got) for x in rule["contains_nospace"])
                else:
                    ok = got == rule["equals"]
                verdict = (vs.get("field_verdicts") or {}).get(f, {}).get("verdict")
                check(R, f"A-{case}-{f}", "final_answer", f"{case}.{f} correct vs original page ({rule})", ok,
                      {"answer": got, "verifier_verdict": verdict, "field_status": (res.get("field_status") or {}).get(f)}, rp)
            answers[case] = {"seconds_fill_and_verify": secs, "verify_exit": p.returncode,
                             "field_verdicts": {f: v.get("verdict") for f, v in vs.get("field_verdicts", {}).items()}}
    summary = {}
    for x in R:
        s = summary.setdefault(x["group"], [0, 0])
        s[0] += x["pass"]
        s[1] += 1
    report = {"label": a.label, "skill": S, "parse_seconds": timing, "summary": {k: f"{v[0]}/{v[1]}" for k, v in summary.items()},
              "answers": answers, "checks": R}
    json.dump(report, open(os.path.join(out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for x in R:
        print(("PASS " if x["pass"] else "FAIL ") + x["id"].ljust(22) + x["check"][:95])
    print(json.dumps({"summary": report["summary"], "parse_seconds": timing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
