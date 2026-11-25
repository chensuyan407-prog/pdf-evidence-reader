# Tests for pdf-evidence-reader (not part of the Skill resources)

Run from `v0.2/`:

```bash
python tests/make_masked_pdf.py                       # builds tests/pdfs/bert_masked.pdf (abstract painted white)
python tests/make_verifier_fixtures.py <grading>/controls/wrong-cell/result.json   # verifier control inputs
python tests/run_regression.py --skill submission/scripts --label new               # v0.2, all checks
python tests/run_regression.py --skill <copy of v0.1 scripts> --label old --parse-only   # v0.1, same checks where possible
python src/reproduce.py                                # author self-test on synthetic samples
```

- `pdfs/`: the grading cases (ResNet 1512.03385, GPT-3 2005.14165, BERT 1810.04805 scanned at 150 dpi, AES-256 encrypted ResNet), copied unchanged from the grading folder, plus the generated masked BERT.
- `fixtures/verifier/`: one correct GPT-3 result and variants that each change one condition; `expected.json` lists the required verdicts.
- `fixtures/expected_answers.json`: answers read from the original pages.
- `agent_fill.py`: scripted stand-in for the agent's reading step (question-specific search terms live here, never in the parser). BERT values are what was read by eye from the crop image.
- `HOLDOUT.md`, `out/holdout_result.json`: the table kept out of development and its final check.
- `out/<label>/report.json`: every check with pass/fail, details and the evidence file path.

Checks locate content by text and coordinates, not by block ids.

## v0.3 additions

```bash
python tests/v03/make_controls.py submission/scripts <grading-v2 dir>   # verifier controls (same inputs for every version)
python tests/v03/run_v03.py --skill submission/scripts --label v03        # tables / equation (8) / 3 cases / controls / efficiency
python tests/v03/run_holdout.py --skill submission/scripts --label v03    # held-out papers (tests/holdout_pdfs, see HOLDOUT_v0.3.md)
```

- `v03/agent_fill_v03.py`: agent-side answering for Attention, GloVe, Hubble (Hubble values are the developer's visual reading of the crop).
- `v03/make_numeric_table.py`: synthetic table with integers, negatives, footnote superscripts and scientific notation (+ truth JSON).
- `v03/fixtures/controls/`: 19 verifier controls; each folder is self-contained and is verified after being copied elsewhere.
- `holdout_truth_v0.3.json`: ground truth for held-out pages, written from page images before the frozen code ran.
- PDFs (`pdfs/`, `holdout_pdfs/`) are not shipped in the package; sources and SHA-256 are in HOLDOUT_v0.3.md and the review folders.

## v0.4 additions

```bash
python tests/v04/make_samples_v04.py tests/v04/samples     # optional: the runner regenerates them under out/<label>/samples
python tests/v04/run_v04.py --skill submission/scripts --label v04
```

- `v04/make_samples_v04.py`: synthetic PDFs with exact truth - a "Table 1" that is an embedded PNG on a text-layer page,
  a 3-column 2-page layout (content stream reversed, running header, a paragraph continuing from p1 col 3 to p2 col 1),
  unnumbered / ALL-CAPS / "Appendix A:" / "A.1" headings, and a ruled table continued on the next page.
- `v04/run_v04.py` groups: `synthetic`, `real` (BN / GloVe / Chinchilla sections, Attention eq. 1 and 3, Chinchilla eq. 2,
  `latex_plain` unit cases; skipped when the PDFs are absent), `workflow` (run_case + fill_result.py on Attention: refusals,
  finalize statuses, broken-JSON recovery, tampered case_id, deleted field, 5 contract shapes, contract naming `evidence`,
  encrypted input), `package` (submission/ size and file limits).
- The same runner on the v0.3.1 scripts (`git archive dea5dc8 submission/scripts`) is the before/after comparison.
