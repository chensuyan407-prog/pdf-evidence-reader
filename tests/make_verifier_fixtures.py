#!/usr/bin/env python3
"""Build the verifier control inputs from one correct GPT-3 result.

Each variant changes exactly one condition of the correct result:
  correct            70.9 from HellaSwag/acc/dev x Zero-Shot/13B          -> must pass
  answer_999         answer 999, evidence untouched                          -> value_mismatch
  answer_789_13b     answer 78.9 (real value of Zero-Shot/175B, same row),
                     evidence still claims Zero-Shot/13B                    -> column_mismatch
  wrong_row          correct answer, evidence row = StoryCloze               -> evidence_unsupported
  wrong_page         correct answer, evidence page = 62                      -> evidence_unsupported
The grader's old-format control (controls/wrong-cell, answer 999 with true
text quotes) is copied unchanged as legacy_wrong_cell.
"""
import copy
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fixtures", "verifier")

CORRECT = {
    "case_id": "gpt3",
    "zero_acc": "70.9",
    "one_ppl": "4.61",
    "status": "success",
    "_contract_fields": ["zero_acc", "one_ppl"],
    "evidence": [
        {"field": "zero_acc", "type": "table_cell", "table": "Table H.1", "page": 63,
         "row": {"Name": "HellaSwag", "Metric": "acc", "Split": "dev"}, "column": "Zero-Shot/13B", "value": "70.9"},
        {"field": "one_ppl", "type": "table_cell", "table": "Table H.1", "page": 63,
         "row": {"Name": "LAMBADA", "Metric": "ppl", "Split": "test"}, "column": "One-Shot/6.7B", "value": "4.61"},
    ],
    "unreported": [], "uncertainty": [], "conflicts": [],
}


def variants():
    out = {"correct": copy.deepcopy(CORRECT)}
    v = copy.deepcopy(CORRECT)
    v["zero_acc"] = "999"
    out["answer_999"] = v
    v = copy.deepcopy(CORRECT)
    v["zero_acc"] = "78.9"
    out["answer_789_13b"] = v
    v = copy.deepcopy(CORRECT)
    v["evidence"][0]["row"] = {"Name": "StoryCloze", "Metric": "acc", "Split": "test"}
    out["wrong_row"] = v
    v = copy.deepcopy(CORRECT)
    v["evidence"][0]["page"] = 62
    out["wrong_page"] = v
    return out


EXPECT = {
    "correct": {"zero_acc": "supported_by_table_cell", "one_ppl": "supported_by_table_cell"},
    "answer_999": {"zero_acc": "value_mismatch"},
    "answer_789_13b": {"zero_acc": "column_mismatch"},
    "wrong_row": {"zero_acc": "evidence_unsupported"},
    "wrong_page": {"zero_acc": "evidence_unsupported"},
    "legacy_wrong_cell": {"zero_acc": "value_not_in_evidence"},
}


def main(legacy_src=None):
    os.makedirs(OUT, exist_ok=True)
    for name, res in variants().items():
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        json.dump(res, open(os.path.join(d, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if legacy_src and os.path.exists(legacy_src):
        d = os.path.join(OUT, "legacy_wrong_cell")
        os.makedirs(d, exist_ok=True)
        shutil.copy(legacy_src, os.path.join(d, "result.json"))
    json.dump(EXPECT, open(os.path.join(OUT, "expected.json"), "w", encoding="utf-8"), indent=1)
    print("fixtures written to", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
