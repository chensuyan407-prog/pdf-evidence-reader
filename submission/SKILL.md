---
name: pdf-evidence-reader
description: Layout-aware parsing of scientific PDFs with page/block provenance, for research reading tasks (question answering with evidence, extraction, translation, table/formula/section structuring). Use when a task gives paper PDFs and asks for answers or structured content that must point back to page, block, table cell or equation. Restores multi-column reading order across pages, removes running headers/footers, rebuilds the section tree, ruled/booktabs tables (grouped headers, spanning cells, tables continued on the next page) queryable by row/column name, numbered display equations as LaTeX with an honest auto/partial status, and flags scanned pages / image tables with provenance-bound crops. Answers are written with a command-line filler and verified against the cited cells, quotes and the original page.
---

# pdf-evidence-reader (v0.5.0)

One parse → read a one-page outline → look values up by name → `fill_result.py put` → `finalize`. Never edit
`result.json` by hand. Python 3 + PyMuPDF, no network. Target: **3–6 requests**, batch commands with `;` (not `&&`).

## 1. Setup + parse (one request)
```bash
for d in "${SKILL_DIR:-}" /skills/weekly-skill /app/submission /app/skills /app/skills/* /skills/* "$(pwd)"; do [ -f "$d/scripts/run_case.py" ] && SKILL_DIR="$d" && break; done; PY=$(command -v python3 || command -v python); S="$SKILL_DIR/scripts"
"$PY" "$S/run_case.py" --case /input/case.json --out /app/output --work /app/work; cat /input/task.md 2>/dev/null | head -80
```
Writes a valid skeleton `/app/output/result.json` (`case_id`, all contract fields `null`, `evidence`, `unreported`, `sources`,
`run_status`, `change_log`) and `parsed/<stem>/` (`outline.md`, `structure_summary.json`, `doc.json`, `tables/`, `equations/`, `crops/`).
It prints `contract_fields` and **absolute `next_steps` commands** (shell variables may not survive between requests: copy them).
The primary PDF comes from `case.json` (role `primary`/`main` first); a filled `result.json` is never overwritten (`--force`).

## 2. Read the outline, then query (one or two requests)
`cat PARSED/outline.md` = sections with pages and block ids, every table with column paths and row keys, image tables,
numbered equations with status, figures, scanned pages, warnings. Do not `cat document.md`; query instead:
```bash
"$PY" "$S/table_query.py" PARSED --table "Table 2" --list        # columns, column paths, row keys (all parts)
"$PY" "$S/table_query.py" PARSED --table "Table 2" --row "Model=Transformer (big)" --col "BLEU/EN-DE"
"$PY" "$S/find_evidence.py" PARSED "words from the question" --top 5      # also --page N, --section 3.2, --equation 8
```
Structure tasks (title, authors, abstract, section levels, references): `PARSED/structure_summary.json`. Row label = full
printed label, `--col` = header path; misses return `did_you_mean`; `row_ambiguous` → add keys, never guess. Continued
tables are searched as one (`part`); `table_is_image` → use its `crop` with `put --crop`.

## 3. Write answers and finalize (one or two requests)
```bash
F="$PY $S/fill_result.py"
$F put bleu_big --table "Table 2" --row "Model=Transformer (big)" --col "BLEU/EN-DE"   # VALUE omitted = the cell
$F put eq8 --equation 8                                                              # VALUE omitted = expr_latex
$F put dropout 0.1 --quote "we use a rate of P_drop = 0.1"                           # verbatim words from the PDF
$F put n_layers 6 --json --quote "composed of a stack of N = 6 identical layers"     # --json for numbers/lists/objects
$F put scan_val "12.3" --crop parsed/x/crops/p2_....png --transcription "12.3"       # crop from crop_region.py
$F unreported gpu_type --reason "the paper does not name the GPU model"              # --unreadable if illegible
$F note --uncertainty "Table 3 says 40 epochs, text says 50" --field epochs          # or --conflict
$F finalize                                                                          # always the last command
```
- `put` checks evidence with the verifier's rules first; a refusal writes nothing and says why (wrong cell, quote not
  verbatim, value not in quote, field not in contract → `did_you_mean`). Fix that one put; the others in the batch still apply.
- Dotted `answers.q1` writes inside an object field; `--append` adds to a list field; `--add` keeps earlier evidence.
- `finalize` fills untouched fields with an honest `not_determined` entry, verifies, and sets `status`:
  `success` (answers verified; open items listed in `warnings`) | `partial` (an answer contradicts the PDF, the primary PDF
  could not be parsed, or most required fields are undetermined) | `failed` (no PDF / nothing answerable).
  Exit 0 in every case except an answer contradicting the PDF (exit 1: re-`put` those fields once, finalize again).
  Broken hand-edited JSON → `$F repair`, then `finalize`.

## Scanned pages, image tables, partial equations
`outline.md` lists SCANNED pages (no text layer) and image regions. Crop, read it yourself, then `put --crop`:
```bash
"$PY" "$S/crop_region.py" PARSED --page 2 --bbox x0,y0,x1,y1      # or --region p2-r5, --equation 8
```
Crops get a `.prov.json` (PDF SHA-256, page, bbox, hash); `blank: true` → `unreported FIELD --unreadable`. Equation
`auto` = rebuilt with nothing uncertain; `partial` = read its crop (uncertain symbols listed).

## Honesty rules (never relaxed)
- No value without evidence in a parsed text/table/equation record or a crop you actually read; otherwise `unreported`
  with a reason. Never invent numbers, tables, formulas or pages. Never answer with a fragment of a number (`2.3 ·`).
- Margin/figure text (`p3-m2`, `p3-g4`) is not body evidence; conflicts → `note --conflict`; heuristic structure is
  labelled (`role_rule`, `detector`, `transcription_status`) — say so when you rely on it.

## Budget and failures (64 requests / 900 s): if time runs short, `unreported` the rest and `finalize`

| Situation | What happens | You |
|---|---|---|
| Encrypted PDF | `parse_status=failed, error=encrypted` | `run_case.py --password` only if given; else `finalize` |
| No text layer, no OCR | page image + regions in outline | crop and read, or `unreported --unreadable` |
| Table not detected | `table_not_found` + `available` | `find_evidence.py --page N`, crop, `put --crop` |
| Bad arguments / missing `--pdf` | exit 1, existing `result.json` kept | fix the argument |

Block ids `p{page}-b{n}` body, `-m` margin, `-g` figure text, `-x` picture; PDF points, top-left origin, 1-based pages.
Garbled fonts → warning `possible_garbled_text`, prefer crops. Methods, limits, examples: `README.md`, `examples/`.
