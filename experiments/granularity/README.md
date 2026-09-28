# Granularity: comparing corpora at equal tokens per word

Every cross-domain comparison before this one (`experiments/clean_domains`,
`experiments/null_model`) scored all corpora at the same merge budgets, 4000 and 16000, carried
over from FineWeb. The same budget does not give equally word-like tokens in every corpus, and the
curves depend directly on how word-like the tokens are, so those comparisons mixed domain with
granularity.

`granularity.py score <source> [null]` sweeps each clean source's first sample (6.5 MB,
`experiments/clean_domains`) and its word-shuffled null (`experiments/null_model`) over 250 to
32000 merges, recording `tokens_per_word` (tokens over whitespace-separated words, so punctuation
attached to a word keeps it above 1.0) and `whole_word_fraction` (of the merges learned since the
previous budget, the share that spell a whole alphabetic word of the sample) next to the readings.
BPE merges are a prefix of a longer run on the same corpus (checked: same merges and ids), so each
version learns 32000 merges once and is cut to each budget. `granularity.py summarize` interpolates
every reading at a target `tokens_per_word` between the two budgets that bracket it (linear in the
reading, log in merges). One sample per source.

## Result (2026-09-28)

Tokens per word at each budget:

| Source | 1000 | 4000 | 16000 | 32000 |
|---|---|---|---|---|
| arXiv | 1.93 | 1.41 | 1.20 | 1.17 |
| PubMed | 2.07 | 1.49 | 1.20 | 1.14 |
| MultiNLI | 2.06 | 1.57 | 1.28 | 1.21 |
| C4 | 2.23 | 1.71 | 1.38 | 1.28 |
| FineWeb | 2.29 | 1.77 | 1.43 | 1.34 |
| FineWeb-Edu | 2.34 | 1.78 | 1.41 | 1.31 |
| Wikipedia | 2.46 | 1.86 | 1.46 | 1.35 |
| Case law | 2.54 | 1.95 | 1.64 | 1.56 |

At 1.5 tokens per word (order distance = green over `context_support` minus the null's;
redundancy distance = red count/context gap minus the null's):

| Source | Merges needed | Order distance | Redundancy distance |
|---|---|---|---|
| arXiv | 3,001 | 0.090 | 0.029 |
| PubMed | 3,931 | 0.084 | 0.026 |
| MultiNLI | 5,291 | 0.056 | −0.038 |
| C4 | 8,696 | 0.064 | 0.013 |
| FineWeb-Edu | 10,802 | 0.063 | 0.010 |
| FineWeb | 11,563 | 0.046 | 0.009 |
| Wikipedia | 13,674 | 0.091 | 0.008 |
| Case law | not reached (1.56 at 32000) | — | — |

- **The merges needed for the same granularity differ more than fourfold** between sources (3,001
  for arXiv, 13,674 for Wikipedia) and exceed 32000 for case law. At the fixed 4000 merges of the
  earlier comparisons, arXiv was at 1.41 tokens per word and case law at 1.95: a comparison of
  near-words against word pieces.
- **The redundancy reading is not domain-free at equal granularity.** At a fixed 16000 merges it
  spanned only 0.008–0.011 across long-document sources, which `null_model` read as close to one
  threshold. At 1.5 tokens per word, web and encyclopedic text read 0.008–0.013 but PubMed and
  arXiv 0.026 and 0.029; the reading falls as merges grow (PubMed 0.026 at 1.5, 0.016 at 1.3, 0.010
  at 1.2). The earlier narrow range came from comparing each source at a different granularity.
- **The order distance narrows but stays domain-dependent**: 0.046–0.091 across the seven sources
  that reach 1.5 tokens per word, against 0.054–0.131 for the same seven at a fixed 4000 merges.
- **MultiNLI remains an outlier** on redundancy (−0.038).
- `whole_word_fraction` is high at the first budget (the first merges build short common words
  such as " the"), dips, then rises again, so it does not order the budgets monotonically;
  `tokens_per_word` does, and is the axis used above.

Consequence: the conclusions of `clean_domains` and `null_model` that compared sources at fixed
budgets are withdrawn, including the reading of case law's low order distance as formulaic text,
which was measured at a much coarser granularity than the other sources. Neither reading has a
domain-free value at equal granularity either, which leaves the calibration against training as
the way to set what a reading means for a given corpus.
