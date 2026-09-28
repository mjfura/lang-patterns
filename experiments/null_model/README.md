# Null model: each corpus against its own word-shuffled version

`experiments/clean_domains` found that neither single-corpus reading takes one value for clean
text of every source. `null_model.py` builds each corpus's null: the same documents with their
words shuffled inside each document (vocabulary, word frequencies and document lengths kept, order
destroyed), and reads each measure as the corpus's distance from its null:

- **order distance**: green over `context_support` at 4000 merges, corpus minus null;
- **redundancy distance**: the red gap (red over `count` minus over `context_support`) at 16000
  merges, corpus minus null.

Only the nulls are scored here (`<target>_null_results.json`); the corpora's own readings come from
`experiments/clean_domains` (sample 0) and `experiments/contamination` (seed 0), whose documents
this script rebuilds. `null_model.py clean <source>`, `null_model.py contamination <kind> <dose>`,
then `null_model.py summarize` prints the table below. One sample or seed per corpus.

## Result (2026-09-27)

| Corpus | Order distance | Redundancy distance |
|---|---|---|
| Wikipedia | 0.131 | 0.008 |
| FineWeb-Edu | 0.118 | 0.009 |
| arXiv | 0.107 | 0.009 |
| PubMed | 0.099 | 0.009 |
| C4 | 0.089 | 0.011 |
| FineWeb (contamination base) | 0.088 | 0.009 |
| Case law | 0.032 | 0.011 |
| MultiNLI | 0.054 | −0.010 |

FineWeb base plus 10% / 30% / 50%:

| Added | Order distance | Redundancy distance |
|---|---|---|
| control | 0.087 / 0.084 / 0.084 | 0.009 / 0.010 / 0.010 |
| duplicate | 0.084 / 0.077 / 0.073 | 0.018 / 0.031 / 0.041 |
| python | 0.086 / 0.084 / 0.080 | 0.010 / 0.021 / 0.022 |
| shuffled | 0.057 / 0.040 / 0.037 | 0.009 / 0.008 / 0.007 |

- **The null narrows the redundancy reading across domains.** Clean long-document sources span
  0.008–0.011 as a distance, against 0.005–0.013 as a raw gap (arXiv's null gap is −0.004, case
  law's +0.002: the null absorbs part of each source's own offset). Duplicates now stand out at
  10% (0.018). Python does not at 10% (0.010, inside the clean range; the raw gap had put it at
  0.015–0.016) and does from 30% (0.021).
- **MultiNLI stays an outlier** (−0.010), although its null has the same document lengths. The
  hypothesis from `clean_domains` — that short documents set the gap's baseline — does not
  explain it, and it remains unexplained.
- **The order distance separates shuffled text within FineWeb but stays domain-dependent.** It
  falls with the shuffled dose (0.087 → 0.037) and barely moves with the other kinds. But clean
  sources span 0.032–0.131: clean case law (0.032) and MultiNLI (0.054) read below FineWeb with
  30% shuffled text (0.040). Distance to the null spreads the sources as widely as the raw level
  did (0.097 raw, 0.099 as a distance).

Consequence: a corpus's own null is enough to normalize the redundancy reading for long-document
text, not the order reading. Different sources genuinely differ in how much sequential structure
the order reading measures, and a corpus-only computation cannot say whether case law's 0.032 is
too little for a model to generalize. That needs the calibration against training this project
planned: models trained on corpora whose readings differ, measured on held-out loss and syntactic
validity.
