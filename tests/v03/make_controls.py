#!/usr/bin/env python3
"""Build verifier control inputs (run once; the same inputs are then given to every verifier version).

Each control is a folder with result.json and the parsed output it cites, so it
can be verified in place or after moving. Table controls cite Attention p8
Table 2; image controls cite Hubble p2 Table 1. Every negative control changes
exactly one thing relative to its correct counterpart.

Usage: python tests/v03/make_controls.py <v0.3 scripts dir> <old grading-v2 dir>
"""
import copy
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PDFS = os.path.join(ROOT, "tests", "pdfs")
OUT = os.path.join(HERE, "fixtures", "controls")

EXPECT = {
    # name: {field: (verdict, needs_review)}
    "tab_correct": {"bleu": ("supported_by_table_cell", False), "cost": ("supported_by_table_cell", False)},
    "tab_value_999": {"bleu": ("value_mismatch", False)},
    "tab_adjacent_27_3": {"bleu": ("column_mismatch", False)},
    "tab_wrong_row": {"bleu": ("evidence_unsupported", False)},
    "tab_wrong_page": {"bleu": ("evidence_unsupported", False)},
    "tab_truncated_old_parse": {"cost": ("parsed_cell_incomplete", True)},
    "tab_truncated_constructed_op": {"cost": ("parsed_cell_incomplete", True)},
    "tab_truncated_constructed_noop": {"cost": ("parsed_cell_incomplete", True)},
    "tab_missing_exponent_constructed": {"cost": ("parsed_cell_incomplete", True)},
    "txt_unrelated_dropout": {"bleu": ("string_match_only", True)},
    # added after the Chinchilla held-out probe (now regression): a unit-only cell from a split parse
    "tab_unit_split_old_parse": {"tokens": ("parsed_cell_incomplete", True)},
    "img_correct": {"velocity": ("image_read_unverified", True), "distance": ("image_read_unverified", True)},
    "img_wrong_page": {"velocity": ("image_provenance_mismatch", False)},
    "img_wrong_document": {"velocity": ("image_provenance_mismatch", False)},
    "img_replaced_crop": {"velocity": ("image_provenance_mismatch", False)},
    "img_out_of_bounds": {"velocity": ("image_provenance_mismatch", False)},
    "img_legacy_no_provenance": {"velocity": ("image_read_unbound", True)},
    "img_moved_with_pdf": {"velocity": ("image_read_unverified", True)},
    "img_moved_without_pdf": {"velocity": ("image_read_unverified", True)},
}


def sh(S, script, *args):
    p = subprocess.run([sys.executable, os.path.join(S, script), *args], capture_output=True, text=True, encoding="utf-8")
    return json.loads(p.stdout) if p.stdout.strip().startswith(("{", "[")) else {"stderr": p.stderr}


def write(d, res):
    os.makedirs(d, exist_ok=True)
    json.dump(res, open(os.path.join(d, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def main(S, old_grading):
    keep = os.path.join(OUT, "tab_unit_split_old_parse")  # built from the Freeze-1 Chinchilla parse; not regenerated
    stash = os.path.join(HERE, "_stash_unit_split")
    if os.path.exists(keep):
        shutil.rmtree(stash, ignore_errors=True)
        shutil.move(keep, stash)
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    if os.path.exists(stash):
        shutil.move(stash, keep)
    # ---------------- table controls (Attention p8 Table 2)
    base_parsed = os.path.join(OUT, "_attention_parsed")
    subprocess.run([sys.executable, os.path.join(S, "parse_pdf.py"), os.path.join(PDFS, "attention.pdf"), "--out", base_parsed],
                   capture_output=True)
    def tab_case(name, mutate=None, doc_mutate=None):
        d = os.path.join(OUT, name)
        pd = os.path.join(d, "parsed", "attention")
        shutil.copytree(base_parsed, pd)
        if doc_mutate:
            dj = json.load(open(os.path.join(pd, "doc.json"), encoding="utf-8"))
            doc_mutate(dj)
            json.dump(dj, open(os.path.join(pd, "doc.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        res = {"case_id": "attention", "_contract_fields": ["bleu", "cost"], "bleu": "38.1", "cost": "2.3 × 10¹⁹ FLOPs",
               "status": "success", "unreported": [], "uncertainty": [], "conflicts": [],
               "field_status": {"bleu": "text_layer", "cost": "text_layer"},
               "evidence": [
                   {"field": "bleu", "type": "table_cell", "table": "Table 2", "page": 8,
                    "row": {"Model": "Transformer (base model)"}, "column": "BLEU/EN-FR", "value": "38.1"},
                   {"field": "cost", "type": "table_cell", "table": "Table 2", "page": 8,
                    "row": {"Model": "Transformer (big)"}, "column": "Training Cost (FLOPs)", "value": "2.3 · 10^19"}]}
        if mutate:
            mutate(res)
        write(d, res)

    tab_case("tab_correct")
    tab_case("tab_value_999", lambda r: r.update(bleu="999"))
    tab_case("tab_adjacent_27_3", lambda r: r.update(bleu="27.3"))
    tab_case("tab_wrong_row", lambda r: r["evidence"][0].update(row={"Model": "Transformer (big)"}))
    tab_case("tab_wrong_page", lambda r: r["evidence"][0].update(page=7))

    def big_cost_cell(dj):
        t = next(t for t in dj["tables"] if t["page"] == 8)
        ri = next(i for i, r in enumerate(t["rows"]) if r[0] == "Transformer (big)")
        ci = next(c for c in t["cells"] if c["row"] == ri and c.get("norm", {}).get("kind") == "scientific")
        return t, ri, ci

    def truncate(dj, text, upto):
        t, ri, c = big_cost_cell(dj)
        parts = c["parts"][:upto]
        c["text"] = text
        c["bbox"] = [min(p["bbox"][0] for p in parts), min(p["bbox"][1] for p in parts),
                     max(p["bbox"][2] for p in parts), max(p["bbox"][3] for p in parts)]
        c.pop("norm", None)
        c["parts"] = parts
        t["rows"][ri][c["col"]] = text

    tab_case("tab_truncated_constructed_op", lambda r: r.update(cost="2.3 ·") or r["evidence"][1].update(value="2.3 ·"),
             lambda dj: truncate(dj, "2.3 ·", 2))
    tab_case("tab_truncated_constructed_noop", lambda r: r.update(cost="2.3") or r["evidence"][1].update(value="2.3"),
             lambda dj: truncate(dj, "2.3", 1))

    def drop_exp(dj):
        # constructed: the cell keeps "2.3 · 10" and loses the raised exponent
        truncate(dj, "2.3 · 10", 3)
    tab_case("tab_missing_exponent_constructed", lambda r: r.update(cost="2.3 · 10") or r["evidence"][1].update(value="2.3 · 10"),
             drop_exp)

    d = os.path.join(OUT, "txt_unrelated_dropout")
    shutil.copytree(base_parsed, os.path.join(d, "parsed", "attention"))
    dj = json.load(open(os.path.join(base_parsed, "doc.json"), encoding="utf-8"))
    blk = next(b for b in dj["blocks"] if "used dropout rate" in b["text"])
    q = blk["text"][blk["text"].index("The Transformer (big) model trained"):]
    q = q[:q.index("0.3.") + 4]
    write(d, {"case_id": "attention", "_contract_fields": ["bleu"], "bleu": "0.1", "status": "success",
              "unreported": [], "uncertainty": [], "conflicts": [], "field_status": {"bleu": "text_layer"},
              "evidence": [{"field": "bleu", "page": blk["page"], "block_id": blk["id"], "quote": q}]})

    # the review's truncated control against the review's own v0.2 parse (copied, paths made local)
    d = os.path.join(OUT, "tab_truncated_old_parse")
    src = os.path.join(old_grading, "runs", "attention", "B", "parsed", "paper")
    shutil.copytree(src, os.path.join(d, "parsed", "paper"), ignore=shutil.ignore_patterns("previews", "page_images"))
    old = json.load(open(os.path.join(old_grading, "controls", "truncated_cell", "result.json"), encoding="utf-8"))
    for k in ("sources", "verification"):
        old.pop(k, None)
    write(d, old)

    # ---------------- image controls (Hubble p2, scanned)
    hub_parsed = os.path.join(OUT, "_hubble_parsed")
    subprocess.run([sys.executable, os.path.join(S, "parse_pdf.py"), os.path.join(PDFS, "hubble.pdf"), "--out", hub_parsed],
                   capture_output=True)

    def img_case(name, mutate=None, after=None, page=2, bbox="48,128,386,474", src_pdf=None):
        d = os.path.join(OUT, name)
        pd = os.path.join(d, "parsed", "hubble")
        shutil.copytree(hub_parsed, pd, ignore=shutil.ignore_patterns("page_images"))
        args = [pd, "--page", str(page), "--bbox", bbox, "--dpi", "150"] + (["--pdf", src_pdf] if src_pdf else [])
        c = sh(S, "crop_region.py", *args)
        crop_rel = os.path.relpath(c["crop"], d).replace("\\", "/")
        res = {"case_id": "hubble", "_contract_fields": ["velocity", "distance"], "velocity": "−130 km/s",
               "distance": "0.214 × 10⁶ pc", "status": "success", "unreported": [], "uncertainty": [], "conflicts": [],
               "field_status": {"velocity": "image_read", "distance": "image_read"},
               "evidence": [{"field": f, "type": "image_region", "page": 2, "bbox": c["bbox"], "crop": crop_rel,
                             "provenance": c["provenance"], "transcription": t, "read_by": "agent_vision", "legibility": "clear"}
                            for f, t in (("velocity", "N. G. C. 6822 v = − 130 (km./sec.)"),
                                         ("distance", "N. G. C. 6822 r = 0.214 (10⁶ parsecs)"))]}
        if mutate:
            mutate(res, d, c)
        write(d, res)
        if after:
            after(d, c)
        return d

    img_case("img_correct")
    img_case("img_wrong_page", lambda r, d, c: r["evidence"][0].update(page=5))

    def wrong_doc(r, d, c):
        # crop cut from page 2 of a different PDF (Attention), cited as Hubble page 2
        pd = os.path.join(OUT, "_attention_parsed")
        c2 = sh(S, "crop_region.py", pd, "--page", "2", "--bbox", "48,128,386,474", "--dpi", "150")
        dst = os.path.join(d, "parsed", "hubble", "crops", "other_doc.png")
        shutil.copy(c2["crop"], dst)
        r["evidence"][0].update(crop=os.path.relpath(dst, d).replace("\\", "/"), provenance=c2["provenance"])
    img_case("img_wrong_document", wrong_doc)

    def replace_crop(d, c):
        # overwrite the crop with page 3's pixels, keep the record
        pd = os.path.join(d, "parsed", "hubble")
        c3 = sh(S, "crop_region.py", pd, "--page", "3", "--bbox", "48,128,386,474", "--dpi", "150")
        shutil.copy(c3["crop"], c["crop"])
    img_case("img_replaced_crop", after=replace_crop)

    def oob(r, d, c):
        for e in r["evidence"][:1]:
            e["bbox"] = [48.0, 128.0, 900.0, 474.0]
            e["provenance"] = dict(e["provenance"], bbox=[48.0, 128.0, 900.0, 474.0])
    img_case("img_out_of_bounds", oob)

    def legacy(r, d, c):
        for e in r["evidence"]:
            e.pop("provenance", None)
        os.remove(c["crop"] + ".prov.json")
    img_case("img_legacy_no_provenance", legacy)

    # moved folders: copy a correct case elsewhere; point doc.json at a path that no longer exists
    for name, with_pdf in (("img_moved_with_pdf", True), ("img_moved_without_pdf", False)):
        d = img_case(name)
        dj_path = os.path.join(d, "parsed", "hubble", "doc.json")
        dj = json.load(open(dj_path, encoding="utf-8"))
        dj["meta"]["source_file"] = "Z:/nowhere/paper.pdf"
        json.dump(dj, open(dj_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if with_pdf:
            os.makedirs(os.path.join(d, "input"), exist_ok=True)
            shutil.copy(os.path.join(PDFS, "hubble.pdf"), os.path.join(d, "input", "paper.pdf"))
    json.dump({k: {f: list(v) for f, v in x.items()} for k, x in EXPECT.items()},
              open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8"), indent=1)
    print("controls written to", OUT)


if __name__ == "__main__":
    main(*sys.argv[1:])
