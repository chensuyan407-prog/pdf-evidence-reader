#!/usr/bin/env python3
"""Scripted stand-in for the agent's reading step on the three v2-review cases.

NOT part of the Skill. It follows SKILL.md: table values by row/column names,
equations by number, text by search, scanned pages by crop + visual reading.
Question-specific search terms live here, never in the parser.

HUBBLE_IMAGE_READING was read by eye (by the developer, acting as the agent
with vision) from the crop that crop_region.py makes of page 2, bbox
48,128,386,474 at 150 dpi. It is recorded as an image reading and the verifier
must label it image_read_unverified.
"""
import json
import os
import re
import subprocess
import sys

HUBBLE_IMAGE_READING = {
    "page": 2, "bbox": [48, 128, 386, 474], "dpi": 150,
    "velocity": "−130 km/s",
    "velocity_transcription": "N. G. C. 6822 ... v = − 130; footnote: v = measured velocities in km./sec.",
    "distance": "0.214 × 10⁶ pc",
    "distance_transcription": "N. G. C. 6822 ... r = 0.214; footnote: r = distance in units of 10⁶ parsecs.",
}


def run(S, script, *args):
    p = subprocess.run([sys.executable, os.path.join(S, script), *args], capture_output=True, text=True, encoding="utf-8")
    try:
        return p.returncode, json.loads(p.stdout)
    except Exception:
        return p.returncode, {"status": "error", "stdout": p.stdout[-300:], "stderr": p.stderr[-300:]}


def base(case_id, fields, parsed_rel):
    return {"case_id": case_id, "_contract_fields": fields, "evidence": [], "unreported": [], "uncertainty": [],
            "conflicts": [], "field_status": {}, "status": "success",
            "sources": [{"parsed_dir": parsed_rel}]}


def fill_attention(S, parsed, out):
    res = base("attention", ["bleu", "cost"], parsed)
    asks = {"bleu": (["Model=Transformer (base model)"], "BLEU/EN-FR", ""),
            "cost": (["Model=Transformer (big)"], "Training Cost (FLOPs)", " FLOPs")}
    for f, (rows, col, unit) in asks.items():
        args = [parsed, "--table", "Table 2", "--col", col]
        for r in rows:
            args += ["--row", r]
        code, q = run(S, "table_query.py", *args)
        if q.get("status") == "ok":
            res[f] = (q.get("value_pretty") or q["value"]) + unit
            res["field_status"][f] = "text_layer"
            res["evidence"].append({"field": f, "type": "table_cell", "table": "Table 2", "page": q["page"],
                                    "row": dict(r.split("=", 1) for r in rows), "column": col, "value": q["value"]})
            if q.get("shared_across"):
                res["uncertainty"].append(f"{f}: one printed value spans {q['shared_across']}")
        else:
            res[f] = None
            res["field_status"][f] = "unreadable"
            res["unreported"].append({"field": f, "reason": f"table lookup: {q.get('status')}"})
            res["status"] = "partial"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def fill_glove(S, parsed, out):
    res = base("glove", ["objective", "alpha"], parsed)
    code, e = run(S, "find_evidence.py", parsed, "--equation", "8")
    if code == 0 and isinstance(e, dict) and e.get("number") == "8":
        if e.get("transcription_status") == "auto":
            res["objective"] = e.get("expr_unicode") + "  (8)"
            res["field_status"]["objective"] = "text_layer"
            res["evidence"].append({"field": "objective", "type": "equation", "page": e["page"], "equation": "8",
                                    "block_id": e.get("block_id"), "quote": "(8)", "latex": e.get("expr_latex"),
                                    "crop": e.get("crop")})
        else:
            res["objective"] = None
            res["field_status"]["objective"] = "unreadable"
            res["unreported"].append({"field": "objective", "reason": f"equation (8) transcription {e.get('transcription_status')}; "
                                                                      f"crop {e.get('crop')} needs visual reading"})
    else:
        res["objective"] = None
        res["field_status"]["objective"] = "unreadable"
        res["unreported"].append({"field": "objective", "reason": "equation (8) not found by number"})
    doc = json.load(open(os.path.join(parsed, "doc.json"), encoding="utf-8"))
    hit = None
    for b in doc["blocks"]:
        for s in re.split(r"(?<=[.!?])\s+", b["text"]):
            if re.search(r"α\s*=\s*[\d/.]+", s) and re.search(r"experiment|found|use|fix|set", s):
                hit = (b, s)
                break
        if hit:
            break
    if hit:
        b, s = hit
        res["alpha"] = re.search(r"α\s*=\s*([\d/.]+)", s).group(1)
        res["field_status"]["alpha"] = "text_layer"
        res["evidence"].append({"field": "alpha", "page": b["page"], "block_id": b["id"], "quote": s})
    else:
        res["alpha"] = None
        res["unreported"].append({"field": "alpha", "reason": "not found"})
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res


def fill_hubble(S, parsed, out, result_dir):
    res = base("hubble", ["velocity", "distance"], parsed)
    R = HUBBLE_IMAGE_READING
    code, c = run(S, "crop_region.py", parsed, "--page", str(R["page"]), "--bbox", ",".join(map(str, R["bbox"])),
                  "--dpi", str(R["dpi"]))
    if code != 0 or c.get("blank"):
        for f in ("velocity", "distance"):
            res[f] = None
            res["field_status"][f] = "unreadable"
            res["unreported"].append({"field": f, "reason": f"crop unavailable or blank: {c}"})
    else:
        crop_rel = os.path.relpath(c["crop"], result_dir).replace("\\", "/")
        for f in ("velocity", "distance"):
            res[f] = R[f]
            res["field_status"][f] = "image_read"
            res["evidence"].append({"field": f, "type": "image_region", "page": R["page"], "bbox": c["bbox"],
                                    "crop": crop_rel, "provenance": c.get("provenance"),
                                    "transcription": R[f + "_transcription"], "read_by": "agent_vision", "legibility": "clear"})
        res["uncertainty"].append("velocity and distance were read from the page image; no text layer")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return res
