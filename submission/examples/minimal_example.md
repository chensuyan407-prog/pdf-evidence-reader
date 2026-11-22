# Minimal call example

> **示例 / EXAMPLE ONLY.** The input below is a synthetic two-column PDF built by
> `src/make_samples.py`; the output is what this Skill produced on it. It is not
> an official case and not an official answer.

## Input

`/input/case.json`
```json
{
 "case_id": "example-two-column-001",
 "materials": ["paper.pdf"],
 "question": "What order accuracy does the layout-aware parser reach in Table 1, and which equation number carries the superscript test?",
 "output_contract": {"answer": "string", "table_value": "string", "equation_number": "string"}
}
```

## Calls

```bash
python $SKILL_DIR/scripts/run_case.py --case /input/case.json --out /app/output --work /app/work --preview
# -> {"case_id": "example-two-column-001", "pdfs": ["/input/paper.pdf"],
#     "contract_fields": ["answer", "table_value", "equation_number"], "parse_status": ["success"], ...}

python $SKILL_DIR/scripts/find_evidence.py /app/output/parsed/paper "Layout-aware Order acc." --top 3
# -> [{"kind": "table_row", "table_id": "T1", "row": 0, "page": 2,
#      "caption": "Table 1: Results on the synthetic benchmark.",
#      "header": ["Method", "Order acc.", "Header leak"], ...}, ...]

python $SKILL_DIR/scripts/table_query.py /app/output/parsed/paper --table "Table 1" --row Method=Layout-aware --col "Order acc."
```

## Agent fills `/app/output/result.json` (excerpt, example output)

```json
{
 "case_id": "example-two-column-001",
 "answer": "The layout-aware parser reaches an order accuracy of 1.00 (Table 1); the superscript test is equation (1).",
 "table_value": "1.00",
 "equation_number": "1",
 "status": "success",
 "evidence": [
  {"field": "table_value", "type": "table_cell", "table": "Table 1", "page": 2,
   "row": {"Method": "Layout-aware"}, "column": "Order acc.", "value": "1.00"},
  {"field": "answer", "page": 2, "table_id": "T1", "quote": "Layout-aware 1.00"},
  {"field": "equation_number", "page": 1, "block_id": "p1-b14", "quote": "(1)"}
 ],
 "field_status": {"answer": "text_layer", "table_value": "text_layer", "equation_number": "text_layer"},
 "unreported": [],
 "conflicts": [],
 "uncertainty": [],
 "sources": [{"file": "/input/paper.pdf", "sha256": "…", "parsed_dir": "/app/output/parsed/paper", "parse_status": "success"}],
 "run_status": {"...": "..."},
 "change_log": [{"step": "parse", "...": "..."}, {"step": "fill", "...": "..."}]
}
```

## Verify

```bash
python $SKILL_DIR/scripts/verify_result.py --result /app/output/result.json
# -> field_verdicts: table_value=supported_by_table_cell, answer/equation_number=string_match_only (string match, not semantic)
```

A fabricated quote (e.g. `"accuracy improved by 37%"` cited to `p1-b6`) is
reported as `evidence_not_verifiable` and the script exits with code 1.
