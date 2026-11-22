#!/usr/bin/env python3
"""Scripted stand-in for the agent's reading step on the three grading cases.

This is NOT part of the Skill. It does what SKILL.md tells an agent to do -
query tables by names, search blocks, read equations, crop page images - and
writes result.json per case. Question-specific search terms live here (as
they would in an agent's head), never in the parser.

BERT (scanned, no OCR): the transcriptions below were read by eye from the
crop produced by crop_region.py (tests/out/<run>/bert/crops/...). They are
image readings, recorded as such, and the verifier labels them
image_read_unverified. The occluded variant gets null + a reason when the crop
is blank.
"""
import json
import os
import re
import subprocess
import sys

# read from the clean crop of page 1, abstract (bbox below), by eye
BERT_IMAGE_READING = {
    "bbox": [85, 240, 280, 552],
    "expansion": "Bidirectional Encoder Representations from Transformers",
    "expansion_transcription": "BERT, which stands for Bidirectional Encoder Representations from Transformers",
    "squad": "83.1",
    "squad_transcription": "SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)",
}


def run(cmd):
    p = subprocess.run([sys.executable] + cmd, capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout


def sentence_with(text, pattern):
    for s in re.split(r"(?<=[.!?])\s+", text):
        if re.search(pattern, s):
            return s.strip()
    return None


def fill_gpt3(S, parsed, out):
    res = {"case_id": "gpt3", "_contract_fields": ["zero_acc", "one_ppl"], "evidence": [], "unreported": [],
           "uncertainty": [], "conflicts": [], "status": "success"}
    asks = {"zero_acc": (["Name=HellaSwag", "Metric=acc", "Split=dev"], "Zero-Shot/13B"),
            "one_ppl": (["Name=LAMBADA", "Metric=ppl", "Split=test"], "One-Shot/6.7B")}
    for field, (rows, col) in asks.items():
        args = [os.path.join(S, "table_query.py"), parsed, "--table", "Table H.1", "--col", col]
        for r in rows:
            args += ["--row", r]
        code, outp = run(args)
        q = json.loads(outp)
        if q.get("status") == "ok":
            res[field] = q["value"]
            res["evidence"].append({"field": field, "type": "table_cell", "table": "Table H.1", "page": q["page"],
                                    "row": dict(r.split("=", 1) for r in rows), "column": col, "value": q["value"],
                                    "cell_bbox": q.get("cell_bbox")})
        else:
            res[field] = None
            res["unreported"].append({"field": field, "reason": f"table lookup: {q.get('status')}"})
            res["status"] = "partial"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def fill_resnet(S, parsed, out):
    res = {"case_id": "resnet", "_contract_fields": ["ensemble_error", "projection"], "evidence": [], "unreported": [],
           "uncertainty": [], "conflicts": [], "status": "success"}
    doc = json.load(open(os.path.join(parsed, "doc.json"), encoding="utf-8"))
    code, outp = run([os.path.join(S, "find_evidence.py"), parsed, "ensemble residual nets error ImageNet test set", "--top", "5"])
    hit = next((h for h in json.loads(outp) if re.search(r"\d+(?:\.\d+)?%\s*error on the ImageNet", h["text"])), None)
    if hit:
        sent = sentence_with(hit["text"], r"error on the ImageNet")
        res["ensemble_error"] = re.search(r"(\d+(?:\.\d+)?%)\s*error on the ImageNet", sent).group(1)
        res["evidence"].append({"field": "ensemble_error", "page": hit["page"], "block_id": hit["block_id"], "quote": sent})
    else:
        res["ensemble_error"] = None
        res["unreported"].append({"field": "ensemble_error", "reason": "not found"})
    eq = next((e for e in doc.get("equations", []) if e.get("number") == "2"), None)
    code, outp = run([os.path.join(S, "find_evidence.py"), parsed, "linear projection shortcut connections match the dimensions", "--top", "5"])
    hit = next((h for h in json.loads(outp) if re.search(r"linear projection", h["text"])), None)
    if eq and hit:
        expl = sentence_with(hit["text"], r"linear projection")
        formula = eq.get("unicode_math") or eq.get("marked") or eq["text"]
        res["projection"] = (f"{formula}  —— 公式 (2)。捷径项 W_s𝐱 对 𝐱 做线性投影 W_s，使其维度与 ℱ(𝐱, {{W_i}}) 一致，"
                             f"以便二者逐元素相加（原文：{expl}）")
        res["evidence"].append({"field": "projection", "page": eq["page"], "block_id": eq["block_id"], "quote": eq["text"],
                                "equation_crop": eq.get("crop"), "latex": eq.get("latex")})
        res["evidence"].append({"field": "projection", "page": hit["page"], "block_id": hit["block_id"], "quote": expl})
        res["uncertainty"].append("Glyph styles (bold vectors, calligraphic F) come from font names; checked against the equation crop.")
    else:
        res["projection"] = None
        res["unreported"].append({"field": "projection", "reason": "equation (2) or its explanation not found"})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def fill_bert(S, parsed, out, masked=False):
    res = {"case_id": "bert" + ("_masked" if masked else ""), "_contract_fields": ["expansion", "squad"], "evidence": [],
           "unreported": [], "uncertainty": [], "conflicts": [], "field_status": {}, "status": "partial"}
    doc = json.load(open(os.path.join(parsed, "doc.json"), encoding="utf-8"))
    p1 = doc["pages"][0]
    bb = BERT_IMAGE_READING["bbox"]
    code, outp = run([os.path.join(S, "crop_region.py"), parsed, "--page", "1", "--bbox", ",".join(map(str, bb)), "--dpi", "170"])
    crop = json.loads(outp)
    res["uncertainty"].append(f"page 1 has no text layer (source={p1.get('source')}); no OCR on this machine")
    if crop["blank"]:
        for f in ("expansion", "squad"):
            res[f] = None
            res["field_status"][f] = "unreadable"
            res["unreported"].append({"field": f, "reason": f"abstract region page 1 bbox {crop['bbox']} is blank in the image "
                                                            f"(ink_ratio={crop['ink_ratio']}); cannot be read, not guessed"})
        res["uncertainty"].append("The right-column body text on page 1 may state the BERT expansion, but the question asks for the abstract; not used.")
    else:
        for f, tf in (("expansion", "expansion_transcription"), ("squad", "squad_transcription")):
            res[f] = BERT_IMAGE_READING[f]
            res["field_status"][f] = "image_read"
            res["evidence"].append({"field": f, "type": "image_region", "page": 1, "bbox": crop["bbox"], "crop": crop["crop"],
                                    "transcription": BERT_IMAGE_READING[tf], "read_by": "agent_vision", "legibility": "clear"})
        res["uncertainty"].append("expansion and squad were read from the page image, not from a text layer.")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res
