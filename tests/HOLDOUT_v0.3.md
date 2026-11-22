# v0.3 held-out papers (selected before any v0.3 code change)

Downloaded 2026-11-21 from arXiv, stored unchanged in `tests/holdout_pdfs/`.
No page of these files was opened, rendered or parsed while v0.3 was being developed.
Target pages/tables/equations are chosen only after the implementation is frozen
(recorded below at that time). If a held-out result is used to tune code, that
paper becomes a regression case and the spare paper takes its place.

| file | source | bytes | pages | SHA-256 | role |
|---|---|---:|---:|---|---|
| 2001.08361.pdf | https://arxiv.org/pdf/2001.08361 (Kaplan et al. 2020, Scaling Laws for Neural Language Models) | 2,493,004 | 30 | a41bd7877fd1a6bcbba096b2619618bd2f90e02488f2365644903eb2f7c6a494 | held-out 1 |
| 1502.03167.pdf | https://arxiv.org/pdf/1502.03167 (Ioffe & Szegedy 2015, Batch Normalization) | 173,548 | 11 | bdc69e0b568f41d8d3181cc9077a461f58fcdd1c70c9f49efe60a05ee6109655 | held-out 2 |
| 2203.15556.pdf | https://arxiv.org/pdf/2203.15556 (Hoffmann et al. 2022, Chinchilla) | 6,004,349 | 36 | 3fd3632a8ef48171bd25282990221d49535d75356192f068b3b2ebe08f2aedd4 | spare |

## Freeze log
(filled after implementation freeze)

### Freeze 1 — 2026-11-21T15:44:46Z

Implementation frozen before any held-out page was opened. SHA-256 of submission/scripts:

```
6045418a7417b4cc55b0f57ba9d5129183909553fcebeeba1065679f9495829a *crop_region.py
537561a2fbf6019892f0f05cd2a311775189244305582ad811596a303ca4fbf7 *find_evidence.py
cb6182c860f0b5f58714765809d91d2c43481d9c3f931540600fb84c03edca77 *mathlayout.py
ee798476cede4e608ab6a0e05579fed098e3a875fce17f21e6e6dc0ee58bb84c *parse_pdf.py
5e817c30d0c3bc596de7d533ea1c703b0a08116673ce2711d2b7466c25b5a6a5 *provenance.py
1134b2b027dd0c282839185c138a6eb5663d202001491aadd306b550d2496ee1 *run_case.py
fa6630f70b1e8ce587f151ef0691a51f8cd7c55e7101a698f3994e178f169709 *table_query.py
9ec502d6dd15133bc19c916dc0dd53cbf699c9b82243772ecc266e0ae003ae5e *verify_result.py
```

### Held-out results at Freeze 1 (before any change below)

`tests/v03/out/holdout_v03_freeze1/report.json` — 24/55 checks. Kaplan tables 10/10, Kaplan eq. (1.4) 3/5 symbols
(status partial), BN figure-captioned table 0/7 (not detected: no ruling lines), Chinchilla tables 0/11 lookups
(row_not_found: labels/values split into columns), Chinchilla eq. (2) auto but written with Unicode math italic
(pre-registered ASCII checks fail), eq. (3) partial. Verifier on Kaplan: correct -> supported, truncated -> rejected.
Extra probe on Chinchilla Table 3: a cell holding only "Billion" (its number in the next column) was accepted as
supported_by_table_cell without review - a failure of the primary acceptance goal.

### Change after held-out -> Chinchilla converted to regression

Because the Chinchilla probe exposed a verifier hole, two general rules were added after Freeze 1
(word-space adjacency in the verifier's region check; word-space column merging in the parser).
Chinchilla (2203.15556) is therefore a regression case from here on. No unused paper remains, so these two
rules are validated only on regression data; the Kaplan/BN results above remain the held-out evidence for Freeze 1.

### Freeze 2 — 2026-11-21T15:57:59Z (final v0.3.0 code)

```
6045418a7417b4cc55b0f57ba9d5129183909553fcebeeba1065679f9495829a *crop_region.py
537561a2fbf6019892f0f05cd2a311775189244305582ad811596a303ca4fbf7 *find_evidence.py
cb6182c860f0b5f58714765809d91d2c43481d9c3f931540600fb84c03edca77 *mathlayout.py
4809063aea1d5eea1081df065e64fdfe79ce0390270af97969579abc7c3a9750 *parse_pdf.py
5e817c30d0c3bc596de7d533ea1c703b0a08116673ce2711d2b7466c25b5a6a5 *provenance.py
1134b2b027dd0c282839185c138a6eb5663d202001491aadd306b550d2496ee1 *run_case.py
fa6630f70b1e8ce587f151ef0691a51f8cd7c55e7101a698f3994e178f169709 *table_query.py
405885ba2df897f4a01a56ac5ca2a2d9a32926ecb4490b4920d92843f8b3b762 *verify_result.py
```

Held-out re-run at Freeze 2: tests/v03/out/holdout_v03_freeze2/report.json (26/55; Kaplan+BN unchanged from Freeze 1, Chinchilla now regression).
