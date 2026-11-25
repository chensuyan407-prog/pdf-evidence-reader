---
name: pdf-evidence-reader
description: Layout-aware parsing of scientific PDFs with page/block provenance, for research reading tasks (question answering with evidence, extraction, translation, table/formula/section structuring). Use when a task gives paper PDFs and asks for answers or structured content that must point back to page, block, table cell or equation. Restores multi-column (2-3 column) reading order across pages, removes running headers/footers, rebuilds the section tree (numbered, unnumbered bold, ALL-CAPS and appendix headings); rebuilds ruled/booktabs tables with grouped headers, spanning cells, continued-on-next-page parts and scientific notation, queryable by row and column names; finds numbered display equations by number with LaTeX (fractions, radicals, limits, accents, scripts, Greek) and an honest auto/partial status; flags tables and figures that are embedded images or scanned pages and gives provenance-bound crops. Answers are written with a command-line filler (no manual JSON editing) and verified against the cited cells, quotes and the original page.
---

# pdf-evidence-reader (v0.4.0)

Parse the PDF once into a structured, citable form; navigate it with a one-page outline; take values by name;
write answers with `fill_result.py` (never edit `result.json` by hand); `finalize` verifies everything and sets the status.
Python 3 + PyMuPDF (+ numpy; fontTools optional). No network.

## 0. Locate the Skill and Python (start of every shell command)

```bash
for d in "${SKILL_DIR:-}" /skills/weekly-skill /app/submission /app/skills /app/skills/* /skills/* "$(pwd)"; do [ -f "$d/scripts/run_case.py" ] && SKILL_DIR="$d" && break; done
PY=$(command -v python3 || command -v python || true)
[ -n "${SKILL_DIR:-}" ] && [ -n "${PY:-}" ] || { echo "ERROR: Skill directory or Python 3 interpreter not found" >&2; exit 1; }
S="$SKILL_DIR/scripts"
```
Shell variables may not survive between tool calls: either repeat this line, or copy the absolute commands that
`run_case.py` prints in `next_steps`.

## 1. Parse (1 request, seconds)

```bash
"$PY" "$S/run_case.py" --case /input/case.json --out /app/output --work /app/work
```
Prints `contract_fields`, `parsed` dir, `outline`, `scanned_pages`, `status` and ready-to-run `next_steps`.
Writes `/app/output/result.json` (valid partial result: `case_id`, every contract field `null`, `field_status`,
`evidence`, `unreported`, `uncertainty`, `conflicts`, `sources`, `run_status`, `change_log`) and, per PDF,
`/app/output/parsed/<stem>/`: `outline.md`, `structure_summary.json`, `document.md`, `doc.json`, `tables/`, `equations/`, `crops/`.
Options: `--preview` (page PNGs with boxes), `--pages 7-9` (only those pages; the rest are listed as not parsed).

## 2. Read (1–2 requests)

`task.md` / `case.json`, then **`outline.md`** (sections with page ranges and block ids, every table with column paths
and row keys, tables that are images, numbered equations with status, figures, scanned pages, warnings).
Do not `cat` `document.md` of a long paper; open only what you need:
`find_evidence.py PARSED "words" --top 5` (also `--page N`, `--section 3.2`, `--regex`, `--table`),
or `grep -n "p7-b" PARSED/document.md`. Structure tasks: `structure_summary.json` (title, authors, abstract,
sections with levels/pages, tables, figures, equations, references, page_map).

## 3. Look values up by name (batch several lookups in one command with `;`)

```bash
"$PY" "$S/table_query.py" PARSED --table "Table 2" --list                  # columns, column paths, row keys (all parts)
"$PY" "$S/table_query.py" PARSED --table "Table 2" --row "Model=Transformer (big)" --col "BLEU/EN-DE"
"$PY" "$S/table_query.py" PARSED --table "Table 2" --markdown              # whole table (continuation parts included)
"$PY" "$S/find_evidence.py" PARSED --equation 8                            # page, bbox, expr_latex, crop, transcription_status
```
- Row label = full printed label (citations included, `GNMT + RL [38]`; a trailing footnote mark may be omitted). `--col` = header path.
- `value` as printed (`2.3 · 10^19`), `value_pretty` (`2.3 × 10¹⁹`), `value_numeric`, `cell_bbox`, `shared_across` (one value spanning columns), `footnote_marker` (`35.0^b` → `b`).
- Misses return `did_you_mean`. `row_ambiguous`/`col_ambiguous`: add keys; never pick one yourself.
- A table continued on the next page is searched as one table (`part` tells which piece).
- `table_is_image`: the table is a picture without text layer → read its `crop` (provenance sidecar already made) → step 4 `--crop`.
- Equations: `auto` = rebuilt with nothing uncertain; `partial` = read the crop yourself (uncertain symbols listed).

## 4. Write answers with fill_result.py (never edit result.json by hand)

Each `put` re-checks the evidence with the verifier's own rules and **refuses** (exit 1, nothing written) what would fail.
```bash
fill() { "$PY" "$S/fill_result.py" "$@"; }
fill put bleu_big --table "Table 2" --row "Model=Transformer (big)" --col "BLEU/EN-DE"     # VALUE omitted = the cell
fill put eq8 --equation 8                                                                  # VALUE omitted = expr_latex
fill put dropout 0.1 --quote "we use a rate of P_drop = 0.1" [--page 8]                    # block found automatically
fill put scan_value "12.3 ± 0.4" --crop parsed/x/crops/p2_....png --transcription "12.3 ± 0.4" [--legibility partial]
fill unreported gpu_type --reason "the paper does not name the GPU model"                  # or --unreadable (in PDF, illegible)
fill note --uncertainty "Table 3 and the text give different epochs (40 vs 50)" --field epochs   # or --conflict
fill show                                                                                  # what is filled / still missing
fill finalize                                                                              # always the last step
```
- `--json` for numbers/lists/objects (`--json '[1,2]'`); `--add` keeps earlier evidence (several quotes); `--append` for list fields;
  dotted `answers.q1` writes inside an object field. Protected keys (`case_id`, `status`, `sources`, `evidence`, …) cannot be put.
- A put is refused when: the value differs from the cell (wrong row/col), the quote is not verbatim in the PDF or not at `--page`,
  the value is not in the quote (paraphrase; `--force` keeps it but it is reported unsupported), the crop has no provenance,
  the field is not in the contract (`did_you_mean` given).
- `finalize`: fields still `null` without reason get an honest `not_determined` entry; runs `verify_result.py`; sets `status`
  = `success` (all answered from verified evidence or reported absent) | `partial` (failing / undetermined / unreadable /
  image-read fields, listed in `status_reason`) | `failed` (no PDF parsed or nothing answerable); prints per-field verdict + hint.
  Exit 1 = blocking issues: fix the listed fields (put again) and finalize again (≤ 2 rounds).
- If you did edit `result.json` by hand: `$F repair` (restores broken JSON from the backup, `case_id`/contract keys from the skeleton), then `finalize`.

| verdict | meaning | action |
|---|---|---|
| `supported_by_table_cell` / `supported_by_equation_record` / `unreported_ok` / `meta_field` | verified | done |
| `string_match_only` | answer occurs in a verified quote; meaning not machine-checked | make sure the quote states it |
| `parsed_cell_incomplete` | cell truncated vs the original page | crop `review_location`, read it, `put --crop` |
| `equation_needs_visual_check` | transcription partial | read the crop; your own reading → `crop_region.py --equation N` + `put --crop` |
| `image_read_unverified` | crop bound to the PDF page; reading not machine-checked | keep (status stays partial) |
| `value_mismatch` / `column_mismatch` / `row_mismatch` / `evidence_unsupported` / `value_not_in_evidence` | wrong place or value | put again |

## Scanned pages and pictures without text layer

`outline.md` lists SCANNED pages (no text layer, OCR unavailable: `pages[i].page_image` + `image_regions`) and pages whose
pictures hold content not in the text layer (`pages[i].raster_regions`, `tables_as_images`; OCR text only if tesseract exists, unverified).
```bash
"$PY" "$S/crop_region.py" PARSED --page 2 --bbox x0,y0,x1,y1 [--dpi 220]     # or --region p2-r5, --equation 8
```
→ crop PNG + `.prov.json` (PDF SHA-256, page, bbox, dpi, crop hash) + `blank`. Read the crop; `put --crop … --transcription …`.
`blank: true` → `unreported FIELD --unreadable --reason …`. Provenance proves where the image came from, not that the reading is right.

## Budget (64 requests / 900 s)

Typical case: discover+parse 1, task+outline 1–2, lookups 2–4 (batched), puts 1–3 (batched with `;`), finalize 1, fixes ≤ 2 → 8–13 requests.
Long papers (30–80 pages) parse in a few seconds; the cost is reading, so read `outline.md` and query, never page through `document.md`.
If time is short: `unreported` the rest with a reason and `finalize` — a finalized partial result beats an unfinished one.

## Honesty rules

- No value without evidence in a parsed text/table/equation record or an image you actually read; otherwise `unreported` with the reason.
- Never answer with a fragment of a number (`2.3 ·`); `parsed_cell_incomplete` means go to the image.
- Heuristic structure is labelled (`role_rule`, `detector`, `transcription_status`, `uncertain_symbols`, section `kind`); say so when you rely on it.
- Removed headers/footers/figure text (`p3-m2`, `p3-g4`) are not evidence for body claims. Material conflicts go to `note --conflict`.

## Failure handling

| Situation | Scripts | You |
|---|---|---|
| Encrypted | `parse_status=failed`, `error=encrypted` | `run_case.py --password` only if given; else `finalize` (status failed, reason recorded) |
| No text layer, no OCR | page image + regions, listed in outline | read crops, or `unreported --unreadable` |
| Garbled font encoding | warning `possible_garbled_text`, status partial | prefer crops for affected values |
| Table detected nowhere | `table_not_found` + `available` list | `find_evidence.py --page N`, crop the region, `put --crop` |
| `row_ambiguous` (header repeated mid-table, continued table) | candidates listed | add keys / read the page |
| Equation missing / numbering gap | `equation_number_gap` warning | crop the area |
| Broken result.json | `fill_result.py` restores from `.bak` automatically | `repair`, then `finalize` |
| Console encoding errors | scripts print UTF-8 | read the JSON files |

Block ids: `p{page}-b{n}` body, `p{page}-m{n}` removed margin text, `p{page}-g{n}` figure text, `p{page}-x{n}` picture
without text layer. Coordinates: PDF points, origin top-left, physical page numbers (1-based). Methods and limits: `README.md`.
