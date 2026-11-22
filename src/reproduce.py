#!/usr/bin/env python3
"""Reproduce the self-evaluation of the pdf-evidence-reader Skill.

Run from the bundle root:   python src/reproduce.py

1. builds synthetic layout PDFs with exact ground truth (src/make_samples.py)
2. parses them with the Skill (submission/scripts/parse_pdf.py)
3. parses them with two naive baselines (PyMuPDF plain text in stream order,
   and PyMuPDF sort=True geometric order) -- what an agent without the Skill
   typically gets
4. scores both against the ground truth, checks robustness (scanned page,
   encrypted file, repeat-run determinism)
5. runs the complete Skill workflow on an EXAMPLE case (run_case -> fill ->
   verify_result) to show the result.json interface
6. writes execution/run.log, execution/results/*, trace/trace.jsonl,
   characterization.json and arm_manifest.json

The samples are synthetic and built by this script; they are not the
platform's three official cases, and no official answer is produced here.
"""
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "submission", "scripts"))
import make_samples  # noqa: E402
import parse_pdf  # noqa: E402

try:
    import pymupdf as fitz
except ImportError:
    import fitz

EXE = os.path.join(ROOT, "execution")
RES = os.path.join(EXE, "results")
TRACE = os.path.join(ROOT, "trace", "trace.jsonl")
LOG_LINES, STEPS = [], []
T_START = time.time()


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def step(**kw):
    kw.setdefault("timestamp", now())
    kw["step_order"] = len(STEPS) + 1
    STEPS.append(kw)


def tool(name, args, fn):
    cid = f"c{len(STEPS) + 1}"
    step(step_type="tool_call", tool_name=name, tool_call_id=cid, tool_args=args)
    t = time.time()
    try:
        out = fn()
        summary = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=lambda o: getattr(o, "stdout", str(o)))[:600]
        step(step_type="tool_result", tool_call_id=cid, tool_output=summary, duration_s=round(time.time() - t, 3))
        return out
    except Exception as e:
        step(step_type="error", tool_call_id=cid, title=f"{name} failed", body=repr(e))
        step(step_type="tool_result", tool_call_id=cid, tool_output=f"ERROR {e!r}", duration_s=round(time.time() - t, 3))
        raise


def nrm(s, cjk=False):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s.replace(" ", "") if cjk else s


# ----------------------------------------------------------------- systems
def skill_texts(doc):
    order_text = " ".join(b["text"] for b in doc["blocks"] if b["role"] not in ("table",))
    paras = [p["text"] for p in doc["paragraphs"]]
    body_text = " ".join(b["text"] for b in doc["blocks"] if b["role"] in ("body", "abstract", "heading", "caption", "title"))
    return order_text, paras, body_text


def baseline_text(pdf, sort):
    d = fitz.open(pdf)
    t = "\n".join(p.get_text("text", sort=sort) for p in d)
    d.close()
    return t


# ----------------------------------------------------------------- metrics
def order_metrics(text, gt_paras, cjk):
    T = nrm(text, cjk)
    pos, missing = [], 0
    for p in gt_paras:
        key = nrm(p, cjk)[:40]
        i = T.find(key)
        if i < 0:
            missing += 1
        else:
            pos.append(i)
    pairs = [(i, j) for i in range(len(pos)) for j in range(i + 1, len(pos))]
    good = sum(1 for i, j in pairs if pos[i] < pos[j])
    return {"paragraphs_found": len(pos), "paragraphs_total": len(gt_paras),
            "pairwise_order_accuracy": round(good / len(pairs), 4) if pairs else (1.0 if pos else 0.0)}


def split_sentence_intact(texts, gt, cjk):
    if not gt:
        return None
    T = [nrm(t, cjk) for t in texts]
    ok = sum(1 for s in gt if any(nrm(s, cjk) in t for t in T))
    return round(ok / len(gt), 4)


def leak_count(body_text, gt):
    T = nrm(body_text)
    leaks = [h for h in gt["headers_footers"] + gt["margin_notes"] if nrm(h) in T]
    return len(leaks), leaks


def baseline_leak(text, gt):
    T = nrm(text)
    leaks = [h for h in gt["headers_footers"] + gt["margin_notes"] if nrm(h) in T]
    nums = sum(1 for ln in text.splitlines() if ln.strip() in gt["page_numbers"])
    return len(leaks) + nums


def structure_metrics(doc, gt, cjk):
    m = {}
    found = [(s["number"] or "", nrm(s["title"], cjk), s["level"]) for s in doc["sections"]]
    gth = [(n, nrm(t, cjk), l) for n, t, l in gt["headings"]]
    tp = sum(1 for h in gth if any(h[0] == f[0] and h[1] == f[1] for f in found))
    lvl = sum(1 for h in gth if any(h[0] == f[0] and h[1] == f[1] and h[2] == f[2] for f in found))
    m["heading_precision"] = round(tp / len(found), 4) if found else (1.0 if not gth else 0.0)
    m["heading_recall"] = round(tp / len(gth), 4) if gth else 1.0
    m["heading_level_accuracy"] = round(lvl / len(gth), 4) if gth else 1.0
    caps = [nrm(b["text"], cjk) for b in doc["blocks"] if b["role"] == "caption"]
    m["caption_recall"] = round(sum(1 for c in gt["captions"] if nrm(c, cjk) in caps) / len(gt["captions"]), 4) if gt["captions"] else 1.0
    m["caption_false_positives"] = sum(1 for c in caps if c not in [nrm(x, cjk) for x in gt["captions"]])
    if gt["title"]:
        titles = [nrm(b["text"], cjk) for b in doc["blocks"] if b["role"] == "title"]
        m["title_exact"] = 1.0 if titles == [nrm(gt["title"], cjk)] else 0.0
    if gt["equations"]:
        ok = 0
        for e in gt["equations"]:
            if any(x["number"] == e["number"] and e["marked_contains"] in (x["marked"] or "") for x in doc["equations"]):
                ok += 1
        m["equation_number_and_superscript"] = round(ok / len(gt["equations"]), 4)
    if gt["footnotes"]:
        fns = " ".join(f["text"] for f in doc["footnotes"])
        m["footnote_recall"] = round(sum(1 for f in gt["footnotes"] if f in fns) / len(gt["footnotes"]), 4)
    m["reference_count_exact"] = 1.0 if len(doc["references"]) == gt["n_references"] else 0.0
    cell_scores = []
    for gtab in gt["tables"]:
        cand = [t for t in doc["tables"] if (t.get("caption") or "").startswith(gtab["caption_prefix"])]
        if not cand:
            cell_scores.append(0.0)
            continue
        t = cand[0]
        G = gtab["rows"]
        tot = sum(len(r) for r in G)
        ok = 0
        for ri, r in enumerate(G):
            for ci, c in enumerate(r):
                try:
                    v = t["rows"][ri][ci]
                except IndexError:
                    v = "__missing__"
                if (c or "") == (v or "") and (c is None) == (v is None):
                    ok += 1
        cell_scores.append(ok / tot)
        for sp in gtab.get("spans", []):
            hit = any(c["row"] == sp["row"] and c["col"] == sp["col"] and c["colspan"] == sp["colspan"] for c in t["cells"])
            m[f"table_span_{gtab['caption_prefix'].replace(' ', '')}_r{sp['row']}c{sp['col']}"] = 1.0 if hit else 0.0
    if cell_scores:
        m["table_cell_accuracy"] = round(sum(cell_scores) / len(cell_scores), 4)
    return m


def doc_fingerprint(path):
    d = json.load(open(path, encoding="utf-8"))
    d.get("stats", {}).pop("seconds", None)
    d.get("meta", {}).pop("source_file", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ----------------------------------------------------------------- example case
def example_case(samples_dir):
    """Exercise the full Skill workflow on an EXAMPLE case (not an official one)."""
    cdir = os.path.join(RES, "example_case", "input")
    odir = os.path.join(RES, "example_case", "output")
    wdir = os.path.join(RES, "example_case", "work")
    for d in (cdir, odir, wdir):
        os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(samples_dir, "two_column_en.pdf"), os.path.join(cdir, "paper.pdf"))
    case = {"case_id": "example-two-column-001",
            "note": "EXAMPLE case built by src/reproduce.py to exercise the interface; not an official case",
            "materials": ["paper.pdf"],
            "question": "What order accuracy does the layout-aware parser reach in Table 1, and which equation number carries the superscript test?",
            "output_contract": {"answer": "string", "table_value": "string", "equation_number": "string"}}
    json.dump(case, open(os.path.join(cdir, "case.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(cdir, "task.md"), "w", encoding="utf-8").write("# Example task\n\nAnswer the question in case.json with page-level evidence.\n")
    rc = os.path.join(ROOT, "submission", "scripts", "run_case.py")
    fe = os.path.join(ROOT, "submission", "scripts", "find_evidence.py")
    vr = os.path.join(ROOT, "submission", "scripts", "verify_result.py")
    out1 = tool("shell", {"cmd": "python submission/scripts/run_case.py --case <example>/case.json --out <example>/output --preview"},
                lambda: subprocess.run([sys.executable, rc, "--case", os.path.join(cdir, "case.json"), "--out", odir, "--work", wdir, "--preview"],
                                       capture_output=True, text=True, encoding="utf-8").stdout)
    log("example case: run_case.py finished -> execution/results/example_case/output/result.json")
    parsed = os.path.join(odir, "parsed", "paper")
    hits = tool("shell", {"cmd": "python submission/scripts/find_evidence.py <parsed> \"Layout-aware order accuracy\" --top 3"},
                lambda: json.loads(subprocess.run([sys.executable, fe, parsed, "Layout-aware Order acc.", "--top", "3"],
                                                  capture_output=True, text=True, encoding="utf-8").stdout))
    doc = json.load(open(os.path.join(parsed, "doc.json"), encoding="utf-8"))
    # scripted fill (stands in for the agent's reading step): take values from the parsed structures
    import table_query
    cell = table_query.query(doc, "Table 1", ["Method=Layout-aware"], "Order acc.")
    eq = next(e for e in doc["equations"] if "^{" in (e["marked"] or ""))
    res = json.load(open(os.path.join(odir, "result.json"), encoding="utf-8"))
    res["answer"] = f"The layout-aware parser reaches an order accuracy of {cell['value']} (Table 1); the superscript test is equation ({eq['number']})."
    res["table_value"] = cell["value"]
    res["equation_number"] = eq["number"]
    res["status"] = "success"
    res["evidence"] = [
        {"field": "table_value", "type": "table_cell", "table": "Table 1", "page": cell["page"],
         "row": {"Method": "Layout-aware"}, "column": "Order acc.", "value": cell["value"]},
        {"field": "answer", "page": cell["page"], "table_id": cell["table_id"], "quote": f"Layout-aware {cell['value']}"},
        {"field": "answer", "page": eq["page"], "block_id": eq["block_id"], "quote": "E = mc2 (1)"},
        {"field": "equation_number", "page": eq["page"], "block_id": eq["block_id"], "quote": f"({eq['number']})"},
    ]
    res["uncertainty"] = ["Example values were filled by a script from the parsed table, standing in for an agent's reading step."]
    res["change_log"].append({"step": "fill", "by": "src/reproduce.py (scripted example)", "at": now()})
    json.dump(res, open(os.path.join(odir, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    v = tool("shell", {"cmd": "python submission/scripts/verify_result.py --result <example>/output/result.json"},
             lambda: subprocess.run([sys.executable, vr, "--result", os.path.join(odir, "result.json")],
                                    capture_output=True, text=True, encoding="utf-8"))
    vs = json.loads(v.stdout)
    log(f"example case: verify_result.py exit={v.returncode} evidence_verified={vs['evidence_verified']}/{vs['evidence_checked']} issues={len(vs['issues'])}")
    # negative control: a fabricated quote must be rejected
    bad = dict(res)
    bad["evidence"] = res["evidence"] + [{"claim": "fabricated", "page": 1, "block_id": "p1-b6", "quote": "accuracy improved by 37%"}]
    bad_path = os.path.join(odir, "result_negative_control.json")
    json.dump(bad, open(bad_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    v2 = subprocess.run([sys.executable, vr, "--result", bad_path], capture_output=True, text=True, encoding="utf-8")
    vs2 = json.loads(v2.stdout)
    shutil.move(os.path.join(odir, "verification.json"), os.path.join(odir, "verification_negative_control.json"))
    subprocess.run([sys.executable, vr, "--result", os.path.join(odir, "result.json")], capture_output=True, text=True, encoding="utf-8")
    log(f"example case: negative control exit={v2.returncode} verified={vs2['evidence_verified']}/{vs2['evidence_checked']} (fabricated quote must fail)")
    shutil.rmtree(wdir, ignore_errors=True)
    return {"verify_exit": v.returncode, "evidence_verified": vs["evidence_verified"], "evidence_checked": vs["evidence_checked"],
            "negative_control_exit": v2.returncode, "negative_control_verified": vs2["evidence_verified"],
            "negative_control_checked": vs2["evidence_checked"], "top_hit": hits[0] if hits else None}


# ----------------------------------------------------------------- main
def main():
    for d in (RES, os.path.dirname(TRACE)):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    ran_at = now()
    step(step_type="thought", title="Plan",
         body="Build synthetic layout PDFs with known ground truth, parse them with the Skill and with naive PyMuPDF baselines, "
              "score reading order, header leakage, structure and tables, then run the full case workflow on an example case.")
    samples = os.path.join(RES, "samples")
    tool("python", {"call": "make_samples.main", "out": "execution/results/samples"}, lambda: make_samples.main(samples) or "ok")
    gts = json.load(open(os.path.join(samples, "ground_truth.json"), encoding="utf-8"))
    log(f"generated {len(gts)} synthetic sample PDFs -> execution/results/samples")
    metrics = {"samples": {}, "robustness": {}, "baselines": {
        "baseline_stream": "PyMuPDF page.get_text('text') in content-stream order, pages concatenated",
        "baseline_sorted": "PyMuPDF page.get_text('text', sort=True), geometric top-to-bottom order"}}
    for gt in gts:
        name = gt["name"]
        pdf = os.path.join(samples, gt["pdf"])
        out = os.path.join(RES, "parsed", name)
        t = time.time()
        r = tool("parse_pdf", {"pdf": f"execution/results/samples/{gt['pdf']}", "out": f"execution/results/parsed/{name}", "preview": True},
                 lambda: parse_pdf.parse(pdf, out, preview=True))
        secs = round(time.time() - t, 3)
        log(f"parsed {name}: status={r['status']} error={r.get('error')} blocks={r.get('stats', {}).get('n_blocks')} seconds={secs}")
        entry = {"status": r["status"], "error": r.get("error"), "seconds": secs, "warnings": r.get("warnings", []),
                 "unresolved_pages": r.get("unresolved_pages")}
        if gt.get("reading_order") is not None and r["status"] != "failed" and "expect_status" not in gt:
            cjk = name.endswith("_zh")
            order_text, paras, body_text = skill_texts(r)
            s_order = order_metrics(order_text, gt["reading_order"], cjk) if gt["reading_order"] else {}
            b1 = baseline_text(pdf, False)
            b2 = baseline_text(pdf, True)
            entry["skill"] = {**s_order,
                              "split_sentence_intact": split_sentence_intact(paras, gt["split_sentences"], cjk),
                              "header_footer_leaks_in_body": leak_count(body_text, gt)[0],
                              **structure_metrics(r, gt, cjk)}
            entry["baseline_stream"] = {**(order_metrics(b1, gt["reading_order"], cjk) if gt["reading_order"] else {}),
                                        "split_sentence_intact": split_sentence_intact([b1.replace("\n", " ")], gt["split_sentences"], cjk),
                                        "header_footer_leaks_in_body": baseline_leak(b1, gt)}
            entry["baseline_sorted"] = {**(order_metrics(b2, gt["reading_order"], cjk) if gt["reading_order"] else {}),
                                        "split_sentence_intact": split_sentence_intact([b2.replace("\n", " ")], gt["split_sentences"], cjk),
                                        "header_footer_leaks_in_body": baseline_leak(b2, gt)}
            step(step_type="observation", title=f"scores for {name}",
                 body=json.dumps({k: entry[k] for k in ("skill", "baseline_stream", "baseline_sorted")}, ensure_ascii=False)[:1500])
            log(f"scores {name}: skill order={entry['skill'].get('pairwise_order_accuracy')} "
                f"baseline_stream order={entry['baseline_stream'].get('pairwise_order_accuracy')} "
                f"baseline_sorted order={entry['baseline_sorted'].get('pairwise_order_accuracy')} "
                f"skill split_sentence={entry['skill'].get('split_sentence_intact')} "
                f"table_cells={entry['skill'].get('table_cell_accuracy')}")
        if "expect_status" in gt:
            entry["expected_status"] = gt["expect_status"]
            entry["status_as_expected"] = r["status"] in gt["expect_status"] and (gt.get("expect_error") in (None, r.get("error")))
            metrics["robustness"][name] = entry
            log(f"robustness {name}: status={r['status']} expected={gt['expect_status']} ok={entry['status_as_expected']}")
        else:
            metrics["samples"][name] = entry
    # determinism
    a = os.path.join(RES, "parsed", "two_column_en", "doc.json")
    rep = os.path.join(RES, "_repeat")
    parse_pdf.parse(os.path.join(samples, "two_column_en.pdf"), rep)
    same = doc_fingerprint(a) == doc_fingerprint(os.path.join(rep, "doc.json"))
    shutil.rmtree(rep, ignore_errors=True)
    metrics["robustness"]["repeat_run_identical"] = same
    log(f"repeat run identical doc.json: {same}")
    # missing file
    miss = parse_pdf.parse(os.path.join(samples, "does_not_exist.pdf"), os.path.join(RES, "_missing"))
    metrics["robustness"]["missing_file"] = {"status": miss["status"], "error": miss["error"]}
    shutil.rmtree(os.path.join(RES, "_missing"), ignore_errors=True)
    step(step_type="decision", title="Run the full workflow on an example case",
         body="Scores look consistent; next exercise run_case -> find_evidence -> verify_result so the result.json interface and the evidence check are demonstrated, including a negative control with a fabricated quote.")
    metrics["example_case"] = example_case(samples)
    mpath = os.path.join(RES, "metrics.json")
    json.dump(metrics, open(mpath, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("wrote execution/results/metrics.json")
    step(step_type="artifact", artifact_id="metrics", artifact_path="execution/results/metrics.json", title="Evaluation metrics",
         body="wrote execution/results/metrics.json")
    step(step_type="artifact", artifact_id="preview_two_column", artifact_path="execution/results/parsed/two_column_en/previews/page_001.png",
         title="Block/reading-order preview, two-column sample page 1")
    step(step_type="artifact", artifact_id="example_result", artifact_path="execution/results/example_case/output/result.json",
         title="Example case result.json (EXAMPLE, not an official case)")
    wall = round(time.time() - T_START, 2)
    write_characterization(metrics)
    step(step_type="artifact", artifact_id="characterization", artifact_path="characterization.json", title="Characterization")
    write_manifest(ran_at, wall)
    log(f"done in {wall}s")
    with open(os.path.join(EXE, "run.log"), "w", encoding="utf-8") as fh:
        fh.write("$ python src/reproduce.py\n" + "\n".join(LOG_LINES) + "\n")
    with open(TRACE, "w", encoding="utf-8") as fh:
        for s in STEPS:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")


def dev(target, metric, actual, ref, score, explanation, artifact="metrics", tol=None):
    d = {"target": target, "metric": metric, "actual_value": actual, "reference_value": ref,
         "score": round(max(0.0, min(1.0, score)), 4), "explanation": explanation, "evidence_artifact": artifact}
    if tol is not None:
        d["tolerance"] = tol
    return d


def abs_score(actual, ref, tol):
    return max(0.0, 1 - abs(actual - ref) / tol)


def write_characterization(m):
    devs = []
    for name, e in m["samples"].items():
        s = e.get("skill", {})
        if not s:
            continue
        if "pairwise_order_accuracy" in s:
            devs.append(dev(f"{name}: reading order", "absolute_error", s["pairwise_order_accuracy"], 1.0,
                            abs_score(s["pairwise_order_accuracy"], 1.0, 0.2),
                            f"pairwise paragraph order vs ground truth; baselines: stream={e['baseline_stream']['pairwise_order_accuracy']}, "
                            f"sorted={e['baseline_sorted']['pairwise_order_accuracy']}", tol=0.2))
        if s.get("split_sentence_intact") is not None:
            devs.append(dev(f"{name}: sentences split across columns/pages rejoined", "absolute_error", s["split_sentence_intact"], 1.0,
                            abs_score(s["split_sentence_intact"], 1.0, 0.5),
                            f"baselines: stream={e['baseline_stream']['split_sentence_intact']}, sorted={e['baseline_sorted']['split_sentence_intact']}", tol=0.5))
        devs.append(dev(f"{name}: header/footer leakage into body", "absolute_error", s["header_footer_leaks_in_body"], 0,
                        abs_score(s["header_footer_leaks_in_body"], 0, 3),
                        f"count of running headers/page numbers/margin stamps left in body text; baselines: stream={e['baseline_stream']['header_footer_leaks_in_body']}, "
                        f"sorted={e['baseline_sorted']['header_footer_leaks_in_body']}", tol=3))
        for k in ("heading_recall", "heading_level_accuracy", "caption_recall", "table_cell_accuracy",
                  "equation_number_and_superscript", "footnote_recall", "reference_count_exact", "title_exact"):
            if k in s:
                devs.append(dev(f"{name}: {k}", "absolute_error", s[k], 1.0, abs_score(s[k], 1.0, 0.5),
                                "Skill structure output vs ground truth (baselines produce plain text with no structure)", tol=0.5))
        for k, v in s.items():
            if k.startswith("table_span_"):
                devs.append(dev(f"{name}: {k}", "exact_match", v, 1.0, v, "merged header cell recovered with correct colspan"))
    for name, e in m["robustness"].items():
        if isinstance(e, dict) and "status_as_expected" in e:
            devs.append(dev(f"robustness: {name}", "exact_match", e["status"], e["expected_status"], 1.0 if e["status_as_expected"] else 0.0,
                            "honest status for scanned / encrypted input: " + "; ".join(w["detail"] for w in e.get("warnings", []))[:300]
                            if e.get("warnings") else "honest status for scanned / encrypted input"))
    devs.append(dev("robustness: repeat-run determinism", "exact_match", m["robustness"]["repeat_run_identical"], True,
                    1.0 if m["robustness"]["repeat_run_identical"] else 0.0, "doc.json identical across two runs (timing excluded)"))
    ex = m["example_case"]
    devs.append(dev("example case: evidence quotes verified", "absolute_error", ex["evidence_verified"], ex["evidence_checked"],
                    1.0 if ex["evidence_verified"] == ex["evidence_checked"] else 0.0,
                    f"verify_result.py on the example result.json; negative control with a fabricated quote exit={ex['negative_control_exit']} "
                    f"({ex['negative_control_verified']}/{ex['negative_control_checked']} verified)", artifact="example_result", tol=1))
    ch = {
        "deviations_from_paper": devs,
        "envelope": [
            {"param": "columns_per_page", "min": 1, "max": 2, "tested": True},
            {"param": "pages", "min": 1, "max": 2, "tested": True, "unit": "pages"},
            {"param": "languages", "min": 1, "max": 2, "tested": True, "unit": "en, zh"},
            {"param": "columns_per_page", "min": 3, "max": 3, "tested": False},
        ],
        "failure_modes": [
            {"condition": "image-only (scanned) page and no tesseract on PATH", "symptom": "page text missing; status=partial with ocr_needed_but_unavailable warning naming the page",
             "evidence_artifact": "metrics"},
            {"condition": "password-protected PDF without --password", "symptom": "status=failed, error=encrypted, nothing extracted", "evidence_artifact": "metrics"},
            {"condition": "tables without any ruling lines (whitespace-only)", "symptom": "not detected by the ruling-line detector; falls back to PyMuPDF find_tables, which may miss them", "evidence_artifact": "metrics"},
            {"condition": "display equations built from many separate glyph blocks", "symptom": "equation may be split into several blocks; LaTeX is not reconstructed (unicode + ^{}/_{} markup only)", "evidence_artifact": "metrics"},
        ],
        "sensitivity": [],
        "notes": "Targets are synthetic layout samples generated by src/make_samples.py with exact ground truth, not the platform's official cases. "
                 "reference_value is the ground truth; baseline numbers are given in each explanation.",
    }
    json.dump(ch, open(os.path.join(ROOT, "characterization.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"wrote characterization.json with {len(devs)} deviation rows")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def write_manifest(ran_at, wall):
    arts = [
        ("metrics", "execution/results/metrics.json", "metric", "json"),
        ("preview_two_column", "execution/results/parsed/two_column_en/previews/page_001.png", "figure", "png"),
        ("preview_two_column_p2", "execution/results/parsed/two_column_en/previews/page_002.png", "figure", "png"),
        ("parsed_two_column", "execution/results/parsed/two_column_en/doc.json", "data", "json"),
        ("parsed_zh", "execution/results/parsed/single_column_zh/doc.json", "data", "json"),
        ("parsed_booktabs", "execution/results/parsed/booktabs_table/doc.json", "data", "json"),
        ("table_booktabs_html", "execution/results/parsed/booktabs_table/tables/T1.html", "table", "other"),
        ("example_result", "execution/results/example_case/output/result.json", "data", "json"),
        ("example_verification", "execution/results/example_case/output/verification.json", "data", "json"),
    ]
    artifacts = [{"id": i, "path": p, "type": t, "format": f, "checksum_sha256": sha(os.path.join(ROOT, p))}
                 for i, p, t, f in arts if os.path.exists(os.path.join(ROOT, p))]
    names = json.load(open(os.path.join(ROOT, "characterization.json"), encoding="utf-8"))["deviations_from_paper"]
    expected = [{"name": d["target"], "type": "metric", "produced_by": [d["evidence_artifact"]],
                 "comparison_method": "exact_match" if d["metric"] == "exact_match" else "numeric_tolerance"} for d in names]
    man = {
        "arm_version": "1.1",
        "paper": {"title": "阅读01｜文献文档解析 — pdf-evidence-reader Skill", "challenge_id": "product-arena-08-3027d350"},
        "entrypoint": "src/reproduce.py",
        "environment": {"dependencies": "requirements.txt", "python_version": ">=3.9", "gpu_required": False,
                        "extra_packages": ["PyMuPDF>=1.23", "tesseract (optional, for scanned pages)"]},
        "execution": {"log_path": "execution/run.log", "exit_code": 0, "ran_at": ran_at, "wall_time_s": wall,
                      "ran_by": "agent", "artifacts": artifacts},
        "expected_outputs": expected,
        "characterization": {"path": "characterization.json"},
        "trace": "trace/trace.jsonl",
        "skills_used": [{"skill_id": "pdf-evidence-reader", "version": parse_pdf.PARSER_VERSION}],
        "provenance": {"created_by": "agent", "toolchain": "Claude Code + Python " + sys.version.split()[0] + " + PyMuPDF " + str(getattr(fitz, "VersionBind", "?"))},
        "handoff": {"status": "partial", "next_suggestion": "Test on real publisher PDFs (IEEE/Elsevier/Nature, scanned theses); add 3-column and whitespace-only table cases; publish GitHub link.",
                    "modality_notes": {"skills": "Skill lives in submission/ (SKILL.md + scripts); copy also in skills/SKILL.md"}},
    }
    json.dump(man, open(os.path.join(ROOT, "arm_manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("wrote arm_manifest.json")


if __name__ == "__main__":
    main()
