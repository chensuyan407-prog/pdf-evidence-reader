---
name: pdf-evidence-reader
description: Layout-aware parsing of scientific PDFs with page/block provenance, for research reading tasks (question answering with evidence, extraction, translation, table/formula/section structuring). Use when a task gives paper PDFs and asks for answers or structured content that must point back to page, block, table cell or equation. Restores multi-column reading order; rebuilds ruled tables with grouped headers, cells spanning columns, scientific notation (2.3 × 10¹⁹), footnote markers and signs, queryable by row and column names; finds numbered display equations by their number (pieces gathered across blocks, TeX math glyphs decoded, limits/accents/scripts/fractions rebuilt) with crop and transcription status. Verifies answers against cited cells, re-reading the original PDF region so truncated values are not reported as supported; binds page-image crops to the PDF by hash, page and bbox. Flags scanned, encrypted or garbled pages instead of guessing.
---

# pdf-evidence-reader (v0.3.0)

Parse the PDF once into a structured, citable form, answer from that form, and
check every answer against its evidence (and the original page) before handing in.
Plain Python 3 on PyMuPDF (+ numpy; fontTools optional for CFF/TrueType glyph names). No network.

`SKILL_DIR` = the directory containing this file (at evaluation time `/skills/weekly-skill`).

## Workflow (≈6–12 model requests)

1. **Parse** (seconds; cached by PDF content + parser version + options):
   ```bash
   python $SKILL_DIR/scripts/run_case.py --case /input/case.json --out /app/output --work /app/work
   # long paper and you know where to look:  --pages 7-9   (unread pages are listed in sources[].pages_not_parsed)
   ```
   Writes `/app/output/parsed/<pdf-stem>/…` and a `/app/output/result.json` skeleton
   (`case_id`, every `output_contract` field = `null`, `field_status`, `evidence`, `unreported`,
   `uncertainty`, `conflicts`, `sources`, `run_status`, `change_log`). Save early: it is a valid partial result.
2. **Read the task** (`task.md`, `case.json`).
3. **Read the parsed paper**: `qa_report.json` first if status ≠ success; `document.md` (reading order,
   `<!-- p3-b12 role … -->` anchors); `doc.json` (`sections`, `paragraphs`, `tables`, `equations`, `figures`, …).
4. **Get values by name, not by counting**
   - Tables:
     ```bash
     python $SKILL_DIR/scripts/table_query.py PARSED --table "Table 2" --list
     python $SKILL_DIR/scripts/table_query.py PARSED --table "Table 2" --row "Model=Transformer (base model)" --col "BLEU/EN-FR"
     python $SKILL_DIR/scripts/table_query.py PARSED --table "Table 2" --row "Model=Transformer (big)" --col "Training Cost (FLOPs)"
     ```
     Use the **full row label** as printed (citations included, e.g. `GNMT + RL [38]`; a trailing footnote mark may be omitted).
     `--col` is the header path (group/leaf). Output: `value` as printed (`2.3 · 10^19`), `value_pretty` (`2.3 × 10¹⁹`),
     `value_numeric` (2.3e19), `value_parts` (every original glyph with its box), `cell_bbox`,
     `shared_across` (one printed value spanning several columns), `footnote_marker` (`35.0^b` → `b`, not an exponent).
     `row_ambiguous` / `col_ambiguous`: add keys; never pick one yourself.
   - Equations by number:
     ```bash
     python $SKILL_DIR/scripts/find_evidence.py PARSED --equation 8
     ```
     Returns page, bbox, `expr_latex`, `expr_unicode`, `crop` (PNG), `transcription_status`, `uncertain_symbols`.
     `auto` = rebuilt automatically with nothing uncertain. `partial` = some pieces/glyphs uncertain (listed) →
     read the crop and transcribe yourself. Only `auto` counts as an automatic transcription.
   - Text: `python $SKILL_DIR/scripts/find_evidence.py PARSED "query words" --top 5` (`--regex`, `--section`, `--table`, `--page`).
5. **Fill `result.json`** (keep every key). Evidence items carry the contract field in `field`:
   - table: `{"field": F, "type": "table_cell", "table": "Table 2", "page": 8, "row": {"Model": "Transformer (big)"}, "column": "Training Cost (FLOPs)", "value": "2.3 · 10^19"}`
   - equation: `{"field": F, "type": "equation", "equation": "8", "page": 4}` (answer = its `expr_unicode` or `expr_latex`)
   - text: `{"field": F, "page": 3, "block_id": "p3-b14", "quote": "verbatim text from that block"}`
   - page image: `{"field": F, "type": "image_region", "page": 1, "bbox": [...], "crop": "<path relative to result.json>", "provenance": <object printed by crop_region.py>, "transcription": "exactly what you read", "read_by": "agent_vision", "legibility": "clear|partial"}`
   Set `field_status[F]` = `text_layer` | `image_read` | `unreadable`; `unreported` (with reason) for values not given
   or not readable; `conflicts`; `uncertainty`; `status` = `success` | `partial` | `failed` | `refused`.
   Answer with the complete value as printed or normalised (`2.3 × 10¹⁹ FLOPs`), never a fragment.
6. **Verify, then fix**
   ```bash
   python $SKILL_DIR/scripts/verify_result.py --result /app/output/result.json
   ```
   For each field it reports `checks.value_equals_parsed_cell`, `checks.source_region_complete` (the original
   PDF, found by SHA-256, is re-read around the cell: value continuing past the cell edge, trailing operator, missing
   exponent, unit left outside, digits missing) and `checks.needs_visual_check`, plus `verdict`, `needs_review`, `reasons`,
   `review_location` (page + bbox to crop).
   | verdict | meaning | action |
   |---|---|---|
   | `supported_by_table_cell` | answer = cell, original region complete | done |
   | `supported_by_equation_record` | answer = automatic transcription (status auto) | done (crop available) |
   | `parsed_cell_incomplete` | the parsed cell is truncated / the page shows more | crop `review_location`, read, answer from the image (`image_region`) |
   | `supported_by_parsed_cell_region_unchecked` | PDF not found to re-check | pass `--pdf` or review |
   | `value_mismatch` / `column_mismatch` / `row_mismatch` / `evidence_unsupported` | answer not at the cited place | fix |
   | `string_match_only` | answer string found in a verified quote — **not a semantic check** | make sure the quote means it |
   | `equation_needs_visual_check` | equation transcription partial | read the crop |
   | `image_read_unverified` | crop tied to the PDF (hash, page, bbox, pixels); reading not machine-checked | keep, it stays in review |
   | `image_read_unbound` | crop has no provenance record | redo the crop with crop_region.py |
   | `image_provenance_mismatch` | wrong page / other document / replaced crop / bbox off the page | redo |
   | `value_not_in_evidence` / `no_evidence` / `null_without_reason` | | add evidence or an `unreported` reason |

## Scanned pages (no text layer)

Without OCR the page is in `unresolved_pages`; `pages[i]` has `page_image` and `image_regions` (bbox in PDF points).
1. look at `page_image`, choose the region;
2. `python $SKILL_DIR/scripts/crop_region.py PARSED --page 2 --bbox x0,y0,x1,y1 [--dpi 220]` (or `--region p2-r5`, `--equation 8`)
   → crop PNG + `<crop>.prov.json` (source PDF SHA-256, physical page, page size, bbox, dpi, crop SHA-256) + `blank`;
3. read the crop; record an `image_region` item **with the printed `provenance`**, `field_status = image_read`.
`blank: true` = nothing to read: leave `null`, `field_status = unreadable`, give the reason.
Provenance proves *where the image came from*, not that your reading is right: such fields always stay in `needs_review`.

## Honesty rules

- No value without evidence in the parsed text/table/equation record or an image you actually read. Otherwise `null` + `unreported`.
- Never answer with a fragment of a number (`2.3 ·`); if the verifier says `parsed_cell_incomplete`, go to the image.
- A located quote is not proof of meaning; numbers in a quote with several numbers are not attributed.
- Heuristic structure is labelled (`detector`, `span_rule`, `transcription_status`, `uncertain_symbols`); mention it when you rely on it.
- Do not cite removed headers/footers/figure text as evidence for body claims.

## Failure handling

| Situation | Scripts | You |
|---|---|---|
| Encrypted | `status=failed`, `error=encrypted` | `--password` only if given; else report |
| No text layer, no OCR | `partial`, page image + regions | read images (above) or mark unread |
| `row_ambiguous` (e.g. a table that repeats its header block mid-table) | candidates listed | add keys or read the page; say so |
| Equation `partial` / missing number | `uncertain_symbols`, crop, `equation_number_gap` warning | read the crop |
| Long document | — | `run_case.py --pages A-B` (unread pages listed) |
| Console shows encoding errors | scripts print UTF-8 | read the JSON files instead |

Block ids: `p{page}-b{n}` body, `p{page}-m{n}` removed margin text, `p{page}-g{n}` figure text. Coordinates: PDF points, origin top-left.
See `README.md` for methods and limits.
