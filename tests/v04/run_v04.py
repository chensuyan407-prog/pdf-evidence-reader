#!/usr/bin/env python3
"""v0.4 checks: new parsing capabilities, the fill_result.py workflow, contract shapes, package limits.

Usage (from the repository root):
    python tests/v04/run_v04.py [--skill submission/scripts] [--label v04]

Groups
  synthetic  generated PDFs with exact truth (tests/v04/make_samples_v04.py): picture-only table,
             3-column + cross-page paragraph, unnumbered/ALL-CAPS/appendix headings, continued table
  real       real papers from tests/pdfs and tests/holdout_pdfs (skipped when absent): section
             recovery (BN, GloVe, Chinchilla), equation fixes (Attention eq. 1/3, Chinchilla eq. 2)
  workflow   run_case + fill_result.py on Attention: refusals, finalize statuses, repair, tamper guard,
             contract shapes (flat, JSON Schema, list, list of dicts, nested output_fields, deliverables,
             contract naming meta keys), encrypted input
  package    submission/ limits (<= 50 UTF-8 text files, <= 500 KB each, <= 2 MB, no __pycache__/symlinks)
Writes tests/v04/out/<label>/report.json. Located by content and coordinates, never by block id.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PDFS = os.path.join(ROOT, "tests", "pdfs")
HOLD = os.path.join(ROOT, "tests", "holdout_pdfs")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")


class Report:
    def __init__(self):
        self.checks = []

    def add(self, cid, group, desc, ok, detail=None):
        self.checks.append({"id": cid, "group": group, "check": desc, "pass": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + cid.ljust(30) + desc[:100], flush=True)


def run(S, script, *args):
    p = subprocess.run([sys.executable, os.path.join(S, script), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=ENV, timeout=600)
    try:
        out = json.loads(p.stdout) if p.stdout.strip().startswith(("{", "[")) else p.stdout
    except Exception:
        out = p.stdout
    return p.returncode, out, p.stderr[-500:]


def parse(S, pdf, out):
    shutil.rmtree(out, ignore_errors=True)
    run(S, "parse_pdf.py", pdf, "--out", out)
    with open(os.path.join(out, "doc.json"), encoding="utf-8") as fh:
        return json.load(fh)


def secs(doc):
    return [(s.get("level"), s.get("number"), s.get("title")) for s in doc["sections"]]


def synthetic(R, S, out):
    sdir = os.path.join(out, "samples")
    subprocess.run([sys.executable, os.path.join(HERE, "make_samples_v04.py"), sdir], capture_output=True, env=ENV)
    truth = json.load(open(os.path.join(sdir, "truth_v04.json"), encoding="utf-8"))
    # picture-only table
    d = parse(S, os.path.join(sdir, "mixed_raster.pdf"), os.path.join(out, "p_mixed"))
    ti = d.get("tables_as_images") or []
    R.add("S-raster-detected", "synthetic", "embedded-image table detected and linked to its 'Table 1' caption",
          len(ti) == 1 and (ti[0].get("caption") or "").startswith("Table 1"), ti)
    crop = ti[0].get("crop") if ti else None
    R.add("S-raster-crop-prov", "synthetic", "image table has a crop with a provenance sidecar",
          bool(crop) and os.path.exists(os.path.join(out, "p_mixed", crop + ".prov.json")), crop)
    R.add("S-raster-warning", "synthetic", "warning raster_content_not_in_text_layer raised",
          any(w["code"] == "raster_content_not_in_text_layer" for w in d["warnings"]), [w["code"] for w in d["warnings"]])
    rc, q, _ = run(S, "table_query.py", os.path.join(out, "p_mixed"), "--table", "Table 1", "--row", "Method=Ours", "--col", "Acc")
    R.add("S-raster-query", "synthetic", "table_query on the image table returns table_is_image + crop (not table_not_found)",
          isinstance(q, dict) and q.get("status") == "table_is_image" and q.get("crop"), q if isinstance(q, dict) else str(q)[:300])
    # three columns
    d = parse(S, os.path.join(sdir, "three_column.pdf"), os.path.join(out, "p_3col"))
    tags = []
    for b in d["blocks"]:
        t = b["text"]
        if t.startswith("[P"):
            tags.append(t.split("]")[0] + "]")
        elif t.startswith("ends on the next page"):
            tags.append("[CONT]")
    R.add("S-3col-gutters", "synthetic", "two gutters found on both 3-column pages",
          all(len(p["column_gutters"]) == 2 for p in d["pages"]), [p["column_gutters"] for p in d["pages"]])
    R.add("S-3col-order", "synthetic", "3-column reading order equals the truth (12 paragraphs, 2 pages)",
          tags == truth["three_column"]["order"], {"got": tags, "want": truth["three_column"]["order"]})
    R.add("S-3col-header-removed", "synthetic", "running header and page numbers not in body blocks",
          not any(truth["three_column"]["header"] in b["text"] for b in d["blocks"]) and
          not any(b["text"].strip() in ("1", "2") for b in d["blocks"]))
    R.add("S-3col-crosspage-join", "synthetic", "paragraph continued from p1 col 3 to p2 col 1 is joined into one paragraph",
          any(truth["three_column"]["continued_tail"] in p["text"] and len(p["sources"]) == 2 for p in d["paragraphs"]))
    # headings
    d = parse(S, os.path.join(sdir, "headings.pdf"), os.path.join(out, "p_head"))
    got = secs(d)
    for h in truth["headings"]["headings"]:
        ok = any(t == h["title"] and l == h["level"] and (h.get("number") is None or n == h["number"]) for l, n, t in got)
        R.add(f"S-head-{h['title'][:16]}", "synthetic", f"heading '{h['title']}' level {h['level']} number {h.get('number')}",
              ok, got)
    R.add("S-head-title-not-section", "synthetic", "document title is not a section",
          not any(t in truth["headings"]["not_headings"] for _, _, t in got))
    # continued table
    pdir = os.path.join(out, "p_cont")
    d = parse(S, os.path.join(sdir, "continued_table.pdf"), pdir)
    tc = truth["continued_table"]
    R.add("S-cont-linked", "synthetic", "'Table 2 (continued)' on page 2 linked to Table 2 (continued_from)",
          any(t.get("continued_from") and t["page"] == 2 for t in d["tables"]), [(t["id"], t["page"], t.get("continued_from")) for t in d["tables"]])
    for c in tc["cells"]:
        rc, q, _ = run(S, "table_query.py", pdir, "--table", tc["caption_prefix"], "--row", f"{tc['row_key']}={c['row']}", "--col", c["col"])
        R.add(f"S-cont-{c['row']}", "synthetic", f"Table 2 {c['row']} x {c['col']} == {c['value']} (page {c['page']}) via the caption",
              isinstance(q, dict) and q.get("status") == "ok" and q.get("value") == c["value"] and q.get("page") == c["page"],
              {k: q.get(k) for k in ("status", "value", "page", "part")} if isinstance(q, dict) else q)
    rc, q, _ = run(S, "table_query.py", pdir, "--table", "Table 2", "--row", "Material=Zink", "--col", "Density")
    R.add("S-cont-didyoumean", "synthetic", "row typo 'Zink' -> row_not_found with did_you_mean ['Zinc']",
          isinstance(q, dict) and q.get("status") == "row_not_found" and "Zinc" in (q.get("did_you_mean") or []), q)
    rc, md, _ = run(S, "table_query.py", pdir, "--table", "Table 2", "--markdown")
    R.add("S-cont-markdown", "synthetic", "--markdown prints both parts of the continued table",
          isinstance(md, str) and "Steel" in md and "Zinc" in md and md.count("<!--") == 2, str(md)[:400])


def real(R, S, out):
    sys.path.insert(0, S)
    import mathlayout
    cases = [(r"\hat{𝐿}(𝑁,𝐷)≜𝐸+\frac{𝐴}{𝑁^{𝛼}}", r"\hat{L}(N,D)≜E+\frac{A}{N^{\alpha}}"), ("α𝐴", r"\alpha A"),
             ("𝐱", r"\mathbf{x}"), (r"\alpha𝐵", r"\alpha B"), (r"a\cdotmin(b)", r"a\cdot min(b)"), (r"\infty+\int", r"\infty+\int"),
             ("ϵ𝜀", r"\epsilon\varepsilon"), ("∂𝑓", r"\partial f")]
    for i, (a, want) in enumerate(cases):
        got = getattr(mathlayout, "latex_plain", lambda s: s)(a)  # absent before v0.4
        R.add(f"R-latexplain-{i}", "real", f"latex_plain({a!r}) == {want!r}", got == want, got)
    have = lambda p: os.path.exists(p)
    bn = os.path.join(HOLD, "1502.03167.pdf")
    if have(bn):
        g = secs(parse(S, bn, os.path.join(out, "r_bn")))
        for num, lvl in (("2", 1), ("3", 1), ("3.1", 2), ("3.2", 2), ("3.3", 2), ("3.4", 2)):
            R.add(f"R-bn-sec-{num}", "real", f"BN section {num} recovered (one printed line split into several MuPDF lines)",
                  any(n == num and l == lvl for l, n, _ in g), g)
        R.add("R-bn-no-1N", "real", "BN: footnote-like '1 N' is not a heading", not any(t == "N" for _, _, t in g))
        R.add("R-bn-hyphen", "real", "BN: 'Batch-' + 'Normalized' at a line end keeps the compound",
              any(t == "Training and Inference with Batch-Normalized Networks" for _, _, t in g))
    gl = os.path.join(PDFS, "glove.pdf")
    if have(gl):
        g = secs(parse(S, gl, os.path.join(out, "r_glove")))
        R.add("R-glove-4.4", "real", "GloVe section 4.4 recovered", any(n == "4.4" for _, n, _ in g), g)
    ch = os.path.join(HOLD, "2203.15556.pdf")
    if have(ch):
        cd = parse(S, ch, os.path.join(out, "r_chin"))
        g = secs(cd)
        for num in ("D.1", "D.2", "H.1", "H.4"):
            R.add(f"R-chin-{num}", "real", f"Chinchilla appendix section {num} ('D.2. Title' style) recovered at level 2",
                  any(n == num and l == 2 for l, n, _ in g), [x for x in g if x[1] and x[1][0] in "DHJ"])
        e2 = next((e for e in cd["equations"] if e.get("number") == "2"), None)
        lx = (e2 or {}).get("expr_latex") or ""
        R.add("R-chin-eq2-ascii", "real", "Chinchilla eq. (2) LaTeX uses plain letters and \\alpha/\\beta (no math-italic code points)",
              bool(lx) and not any(0x1D400 <= ord(c) <= 0x1D7FF for c in lx) and "\\alpha" in lx and "\\frac{A}{N^{\\alpha}}" in lx, lx)
    at = os.path.join(PDFS, "attention.pdf")
    if have(at):
        ad = parse(S, at, os.path.join(out, "r_att"))
        e1 = next((e for e in ad["equations"] if e.get("number") == "1"), {})
        R.add("R-att-eq1-frac-sqrt", "real", "Attention eq. (1): radical inside a denominator -> softmax(\\frac{QK^{T}}{\\sqrt{d_{k}}})V",
              "\\frac{QK^{T}}{\\sqrt{d_{k}}}" in (e1.get("expr_latex") or "").replace(" ", ""), e1.get("expr_latex"))
        R.add("R-att-eq1-status", "real", "Attention eq. (1) status auto only with the right transcription (was a false auto in v0.3)",
              e1.get("transcription_status") != "auto" or "\\frac{QK^{T}}{\\sqrt{d_{k}}}" in (e1.get("expr_latex") or "").replace(" ", ""),
              e1.get("transcription_status"))
        e3 = next((e for e in ad["equations"] if e.get("number") == "3"), {})
        R.add("R-att-eq3-cmd-space", "real", "Attention eq. (3): '\\cdot min' not the undefined command '\\cdotmin'",
              "\\cdotmin" not in (e3.get("expr_latex") or "") and "\\cdot min" in (e3.get("expr_latex") or ""), e3.get("expr_latex"))


def workflow(R, S, out):
    at = os.path.join(PDFS, "attention.pdf")
    if not os.path.exists(at):
        R.add("W-skip", "workflow", "attention.pdf missing: workflow checks skipped", False)
        return
    if not os.path.exists(os.path.join(S, "fill_result.py")):
        R.add("W-fill-tool", "workflow", "fill_result.py present (safe structured result writer)", False)
        return
    base = os.path.join(out, "wf")
    shutil.rmtree(base, ignore_errors=True)
    inp = os.path.join(base, "input")
    os.makedirs(inp)
    shutil.copy(at, inp)
    case = {"case_id": "wf-1", "material": {"pdf": "attention.pdf"},
            "output_contract": {"type": "object", "required": ["case_id", "bleu", "eq1", "warmup", "gpus", "missing", "evidence"],
                                "properties": {k: {} for k in ["case_id", "bleu", "eq1", "warmup", "gpus", "missing", "evidence"]}}}
    json.dump(case, open(os.path.join(inp, "case.json"), "w", encoding="utf-8"))
    o, w = os.path.join(base, "out"), os.path.join(base, "work")
    rc, summ, err = run(S, "run_case.py", "--case", os.path.join(inp, "case.json"), "--out", o, "--work", w)
    rp = os.path.join(o, "result.json")
    R.add("W-runcase", "workflow", "run_case: contract fields from JSON Schema, outline.md written, next_steps printed",
          isinstance(summ, dict) and summ.get("contract_fields") == case["output_contract"]["required"] and
          summ.get("outline") and os.path.exists(summ["outline"][0]) and summ.get("next_steps"), summ if isinstance(summ, dict) else err)
    res = json.load(open(rp, encoding="utf-8"))
    R.add("W-meta-contract", "workflow", "contract field named 'evidence' keeps the evidence list (not null)",
          isinstance(res.get("evidence"), list), type(res.get("evidence")).__name__)
    F = lambda *a: run(S, "fill_result.py", "--result", rp, *a)
    rc, q, _ = F("put", "bleu", "--table", "Table 2", "--row", "Model=Transformer (base model)", "--col", "BLEU/EN-FR")
    R.add("W-put-table", "workflow", "put from table cell (VALUE omitted) -> 38.1", rc == 0 and q.get("value") == "38.1", q)
    rc, q, _ = F("put", "bleu", "41.0", "--table", "Table 2", "--row", "Model=Transformer (base model)", "--col", "BLEU/EN-FR")
    R.add("W-refuse-wrong-value", "workflow", "put with a value that differs from the cell is refused", rc == 1 and not q.get("ok"), q)
    rc, q, _ = F("put", "bleu", "--table", "Table 2", "--row", "Model=Transformer (base model)", "--col", "BLEU/EN-FX")
    R.add("W-refuse-bad-col", "workflow", "put with an unknown column is refused with did_you_mean",
          rc == 1 and (q.get("lookup") or {}).get("did_you_mean"), q)
    rc, q, _ = F("put", "eq1", "--equation", "1")
    R.add("W-put-eq", "workflow", "put from equation record", rc == 0 and "\\frac" in (q.get("value") or ""), q)
    rc, q, _ = F("put", "warmup", "four thousand", "--quote", "warmup_steps = 4000")
    R.add("W-refuse-paraphrase", "workflow", "paraphrased value not in the quote is refused", rc == 1, q)
    rc, q, _ = F("put", "warmup", "4000", "--quote", "warmup_steps = 4000 and")
    R.add("W-refuse-fake-quote", "workflow", "quote not verbatim in the PDF is refused", rc == 1, q)
    rc, q, _ = F("put", "warmup", "4000", "--quote", "We used warmup_steps = 4000")
    R.add("W-put-quote", "workflow", "verbatim quote located automatically (block id filled)",
          rc == 0 and (q.get("checked") or {}).get("found_in", "").startswith("p7-"), q)
    rc, q, _ = F("put", "status", "success", "--quote", "x")
    R.add("W-refuse-protected", "workflow", "put on a protected key (status) is refused", rc == 1, q)
    rc, q, _ = F("put", "bleuu", "--table", "Table 2", "--row", "Model=Transformer (big)", "--col", "BLEU/EN-FR")
    R.add("W-refuse-unknown-field", "workflow", "unknown field refused with did_you_mean", rc == 1 and "bleu" in (q.get("did_you_mean") or []), q)
    rc, q, _ = F("put", "gpus", "8", "--quote", "8 NVIDIA P100 GPUs")
    rc, q, _ = F("unreported", "missing", "--reason", "the paper does not report this quantity")
    rc, fin, _ = F("finalize")
    res = json.load(open(rp, encoding="utf-8"))
    R.add("W-finalize-success", "workflow", "finalize: all answered or reported absent -> status success, exit 0",
          rc == 0 and res["status"] == "success" and fin.get("fields", {}).get("evidence", {}).get("verdict") == "meta_field",
          {"rc": rc, "status": res["status"], "reason": res.get("status_reason")})
    # broken JSON from a manual edit
    s = open(rp, encoding="utf-8").read()
    open(rp, "w", encoding="utf-8").write(s[:-50])
    rc, q, _ = F("show")
    R.add("W-recover-corrupt", "workflow", "result.json broken by a manual edit is restored from the backup",
          rc == 0 and isinstance(q, dict) and q.get("notes") and "restored" in q["notes"], q.get("notes") if isinstance(q, dict) else q)
    rc, q, _ = F("repair")
    # tampered case_id + deleted field
    res = json.load(open(rp, encoding="utf-8"))
    res["case_id"] = "WRONG"
    del res["gpus"]
    json.dump(res, open(rp, "w", encoding="utf-8"))
    rc, fin, _ = F("finalize")
    res = json.load(open(rp, encoding="utf-8"))
    R.add("W-guard-caseid", "workflow", "tampered case_id restored from the skeleton", res.get("case_id") == "wf-1", res.get("case_id"))
    R.add("W-guard-field", "workflow", "deleted contract field restored; reported not_determined -> status partial",
          "gpus" in res and res["status"] == "partial" and any("gpus" in r for r in res.get("status_reason", [])),
          {"status": res["status"], "reason": res.get("status_reason")})
    # contract shapes
    shapes = {
        "flat": {"case_id": "c1", "pdf": "attention.pdf", "output_contract": {"a": "string", "b": "number"}},
        "list": {"case_id": "c2", "pdf": "attention.pdf", "output_contract": ["a", "b"]},
        "list_dicts": {"case_id": "c3", "pdf": "attention.pdf", "output_contract": [{"name": "a"}, {"field": "b"}]},
        "nested_fields": {"case": {"case_id": "c4"}, "task": {"pdf": "attention.pdf", "output_fields": ["a", "b"]}},
        "deliverables": {"id": "c5", "inputs": ["attention.pdf"], "deliverables": {"fields": ["a", "b"]}},
    }
    for name, c in shapes.items():
        cdir = os.path.join(base, "shape_" + name)
        os.makedirs(cdir, exist_ok=True)
        shutil.copy(at, cdir)
        json.dump(c, open(os.path.join(cdir, "case.json"), "w", encoding="utf-8"))
        rc, sm, err = run(S, "run_case.py", "--case", os.path.join(cdir, "case.json"), "--out", os.path.join(cdir, "out"),
                          "--work", os.path.join(cdir, "work"))
        r = json.load(open(os.path.join(cdir, "out", "result.json"), encoding="utf-8"))
        R.add(f"W-shape-{name}", "workflow", f"contract shape '{name}': fields a,b present, case_id set",
              r.get("a", 1) is None and r.get("b", 1) is None and r.get("case_id") and r.get("_contract_fields") == ["a", "b"],
              {"fields": r.get("_contract_fields"), "case_id": r.get("case_id")})
    # encrypted
    enc = os.path.join(PDFS, "encrypted_resnet.pdf")
    if os.path.exists(enc):
        cdir = os.path.join(base, "enc")
        os.makedirs(cdir, exist_ok=True)
        shutil.copy(enc, cdir)
        json.dump({"case_id": "enc", "pdf": "encrypted_resnet.pdf", "output_contract": ["x"]}, open(os.path.join(cdir, "case.json"), "w"))
        run(S, "run_case.py", "--case", os.path.join(cdir, "case.json"), "--out", os.path.join(cdir, "out"), "--work", os.path.join(cdir, "w"))
        rc, fin, _ = run(S, "fill_result.py", "--result", os.path.join(cdir, "out", "result.json"), "finalize")
        r = json.load(open(os.path.join(cdir, "out", "result.json"), encoding="utf-8"))
        R.add("W-encrypted-failed", "workflow", "encrypted PDF -> finalize status failed with the reason",
              r["status"] == "failed" and any("encrypted" in x for x in r.get("status_reason", [])), r.get("status_reason"))


def package(R):
    sub = os.path.join(ROOT, "submission")
    files, bad = [], []
    for dp, dn, fn in os.walk(sub):
        if "__pycache__" in dn:
            bad.append(os.path.join(dp, "__pycache__"))
        for f in fn:
            p = os.path.join(dp, f)
            if os.path.islink(p):
                bad.append(p)
            files.append(p)
    sizes = [os.path.getsize(p) for p in files]
    utf8 = True
    for p in files:
        try:
            open(p, encoding="utf-8").read()
        except Exception:
            utf8 = False
            bad.append(p)
    R.add("P-limits", "package", "submission/: <= 50 UTF-8 text files, <= 500 KB each, <= 2 MB total, no __pycache__/symlinks",
          len(files) <= 50 and max(sizes) <= 500 * 1024 and sum(sizes) <= 2 * 1024 * 1024 and utf8 and not bad,
          {"files": len(files), "max_kb": round(max(sizes) / 1024, 1), "total_kb": round(sum(sizes) / 1024, 1), "bad": bad})


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default=os.path.join(ROOT, "submission", "scripts"))
    ap.add_argument("--label", default="v04")
    a = ap.parse_args()
    S = os.path.abspath(a.skill)
    out = os.path.join(HERE, "out", a.label)
    os.makedirs(out, exist_ok=True)
    R = Report()
    synthetic(R, S, out)
    real(R, S, out)
    workflow(R, S, out)
    package(R)
    groups = {}
    for c in R.checks:
        g = groups.setdefault(c["group"], [0, 0])
        g[0] += c["pass"]
        g[1] += 1
    summary = {k: f"{v[0]}/{v[1]}" for k, v in groups.items()}
    json.dump({"summary": summary, "checks": R.checks}, open(os.path.join(out, "report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps({"summary": summary}))
    sys.exit(0 if all(c["pass"] for c in R.checks) else 1)


if __name__ == "__main__":
    main()
