# pdf-evidence-reader

An Agent Skill for reading scientific PDFs with evidence: layout-aware parsing with page/block provenance,
tables you can query by row and column names, numbered equations you can fetch by number, and a verifier that
checks every answer against the parsed structure **and the original PDF region** before it is handed in.

Built for the Bohrium Playground challenge “阅读01｜文献文档解析” (`product-arena-08-3027d350`). Current version: **v0.3.0**.

## What it does

| Capability | How |
|---|---|
| Reading order | column gutters from block occupancy; full-width blocks cut pages into bands; headers/footers/margin stamps and figure text kept out of the body |
| Tables | ruled tables (booktabs or grid); columns from character-level token alignment; grouped headers via `\cmidrule`; cells spanning columns; scientific notation (`2.3 · 10^19` → 2.3e19), footnote marks (`35.0^b`), signs and citations kept apart |
| Table lookup | `table_query.py PARSED --table "Table 2" --row "Model=Transformer (big)" --col "Training Cost (FLOPs)"` → value as printed, normalised value, cell box, every glyph with its box, shared-across columns |
| Equations | number-anchored: pieces gathered across text blocks, TeX math glyphs decoded from the embedded font, limits / accents / scripts / fractions / roots rebuilt; status `auto` only when nothing is uncertain, otherwise `partial` + crop |
| Verification | per field: answer = parsed cell? original PDF region complete (no value continuing past the cell edge, no missing exponent or unit)? visual check needed? Text quotes are reported as *string match only*, never as semantic proof |
| Scanned pages | page images + layout regions without OCR; crops bound to the PDF by SHA-256, page, bbox and pixel re-render; visual readings stay "needs review" |
| Speed | parse cache keyed by PDF content + parser version + options; page-range parsing lists unread pages |

## Install and use

```bash
pip install -r submission/requirements.txt      # pymupdf, numpy, fonttools (optional)

python submission/scripts/run_case.py --case /input/case.json --out /app/output --work /app/work
python submission/scripts/table_query.py /app/output/parsed/<stem> --table "Table 2" --list
python submission/scripts/find_evidence.py /app/output/parsed/<stem> --equation 8
python submission/scripts/verify_result.py --result /app/output/result.json
```

The Skill itself is the `submission/` folder (`SKILL.md` + `scripts/`); see [`submission/SKILL.md`](submission/SKILL.md)
for the agent workflow, evidence formats and verdicts, and [`submission/README.md`](submission/README.md) for methods and limits.

## Evaluation

All numbers below come from local runs; see the reports for exact inputs, evidence paths and failures.

- [`ITERATION_REPORT_v0.3.md`](ITERATION_REPORT_v0.3.md) — v0.2 → v0.3 on the second review's cases (Attention Table 2, GloVe eq. (8), scanned Hubble 1929):
  final answers 3/6 → 6/6; wrong answers accepted as supported in 15 negative controls 2 → 0;
  held-out papers chosen and hashed before coding (Kaplan et al. 2020, Batch Norm) and Chinchilla as regression, with failures kept as measured.
- [`ITERATION_REPORT_v0.2.md`](ITERATION_REPORT_v0.2.md) — v0.1 → v0.2 (ResNet, GPT-3, scanned BERT).
- `tests/` — regression and held-out harnesses (`tests/v03/run_v03.py`, `tests/v03/run_holdout.py`, `tests/run_regression.py`),
  ground truth written from page images, and the verifier control definitions. Test PDFs are public arXiv papers and are not
  redistributed here; sources and SHA-256 are listed in [`tests/HOLDOUT_v0.3.md`](tests/HOLDOUT_v0.3.md).
- `src/reproduce.py` — self-test on synthetic layout PDFs with exact ground truth.

## Known limits (v0.3)

Tables without any ruling lines are not detected; right-aligned multi-word headers can lose their first word to the
left column; nested scripts inside exponents and Unicode math-italic letters are not normalised; stacked tables return
`row_ambiguous`; no OCR quality testing (tesseract not installed). Details in `submission/README.md`.

## License

MIT — see [LICENSE](LICENSE).
