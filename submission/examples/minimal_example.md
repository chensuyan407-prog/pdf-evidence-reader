# Minimal call example

> **示例 / EXAMPLE ONLY.** The input below is a synthetic two-column PDF built by
> `src/make_samples.py`; the output is what this Skill (v0.4.0) produced on it. It is not
> an official case and not an official answer. Paths are shortened.

## Input

`/input/case.json`
```json
{
 "case_id": "example-two-column-001",
 "materials": ["paper.pdf"],
 "question": "What order accuracy does the layout-aware parser reach in Table 1, and which equation number carries the superscript test?",
 "output_contract": {"table_value": "string", "equation_number": "string", "dataset_size": "string"}
}
```

## Calls (5 shell commands)

```bash
for d in "${SKILL_DIR:-}" /skills/weekly-skill /app/submission /app/skills /app/skills/* /skills/* "$(pwd)"; do [ -f "$d/scripts/run_case.py" ] && SKILL_DIR="$d" && break; done
PY=$(command -v python3 || command -v python || true)
[ -n "${SKILL_DIR:-}" ] && [ -n "${PY:-}" ] || { echo "ERROR: Skill directory or Python 3 interpreter not found" >&2; exit 1; }
S="$SKILL_DIR/scripts"

"$PY" "$S/run_case.py" --case /input/case.json --out /app/output --work /app/work
# -> {"case_id": "example-two-column-001", "contract_fields": ["table_value", "equation_number", "dataset_size"],
#     "parse_status": ["success"], "outline": [".../parsed/paper/outline.md"], "status": "pending_agent_fill", "next_steps": [...]}

cat /app/output/parsed/paper/outline.md
# -> ## Sections: 1 Introduction | p1 ... 3 Results | p2 ...
#    ## Tables: T1 p2: Table 1: Results on the synthetic benchmark.
#         cols: Method ; Order acc. ; Header leak     body rows (3): Baseline ; Layout-aware ; Human
#    ## Numbered equations: (1) p1 auto

"$PY" "$S/fill_result.py" put table_value --table "Table 1" --row Method=Layout-aware --col "Order acc." ; \
"$PY" "$S/fill_result.py" put equation_number 1 --quote "E = mc2 (1)" ; \
"$PY" "$S/fill_result.py" unreported dataset_size --reason "the paper does not state the benchmark size"
# -> {"ok": true, "field": "table_value", "value": "1.00", "field_status": "text_layer", "evidence": "table_cell", ...}
#    {"ok": true, "field": "equation_number", "value": "1", "checked": {"found_in": "p1-b14", "role": "equation"}, ...}
#    {"ok": true, "field": "dataset_size", "value": null, "unreported": {"kind": "not_reported", ...}}

"$PY" "$S/fill_result.py" finalize
# -> {"status": "success",
#     "fields": {"table_value": {"verdict": "supported_by_table_cell"},
#                "equation_number": {"verdict": "string_match_only", "hint": "ok if the quote really states it ..."},
#                "dataset_size": {"verdict": "unreported_ok"}},
#     "needs_review": ["equation_number"], ...}
```

## Resulting `/app/output/result.json` (excerpt, example output)

```json
{
 "case_id": "example-two-column-001",
 "table_value": "1.00",
 "equation_number": "1",
 "dataset_size": null,
 "status": "success",
 "status_reason": ["every contract field is answered from verified evidence or reported as absent"],
 "evidence": [
  {"field": "table_value", "type": "table_cell", "table": "Table 1", "table_id": "T1", "page": 2,
   "row": {"Method": "Layout-aware"}, "column": "Order acc.", "value": "1.00"},
  {"field": "equation_number", "type": "text", "page": 1, "quote": "E = mc2 (1)", "block_id": "p1-b14"}
 ],
 "field_status": {"table_value": "text_layer", "equation_number": "text_layer", "dataset_size": "not_reported"},
 "unreported": [{"field": "dataset_size", "reason": "the paper does not state the benchmark size", "kind": "not_reported"}],
 "uncertainty": [], "conflicts": [],
 "sources": [{"file": "/input/paper.pdf", "sha256": "…", "parsed_dir": "/app/output/parsed/paper", "parse_status": "success"}],
 "verification": {"field_verdicts": {"table_value": "supported_by_table_cell", "...": "..."}, "report": "verification.json"},
 "change_log": [{"step": "parse"}, {"step": "put", "field": "table_value"}, "...", {"step": "finalize", "status": "success"}]
}
```

## What is refused (nothing is written, exit code 1)

```bash
$PY $S/fill_result.py put table_value 0.95 --table "Table 1" --row Method=Layout-aware --col "Order acc."
# -> "VALUE '0.95' differs from the cell '1.00'; omit VALUE to use the cell, or fix --row/--col"
$PY $S/fill_result.py put table_value --table "Table 1" --row Method=Layout-awre --col "Order acc."
# -> "table lookup failed: row_not_found", "did_you_mean": ["Layout-aware"]
$PY $S/fill_result.py put status success --quote "x"
# -> "'status' is maintained by the scripts and cannot be set with put"
```

The older flow (edit `result.json`, then `verify_result.py --result /app/output/result.json`) still works;
after a manual edit run `fill_result.py repair` and `fill_result.py finalize`.
