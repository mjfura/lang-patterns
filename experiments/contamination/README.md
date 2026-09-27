# Contamination: does the indicator tell useful data from volume?

`contamination.py <base|control|duplicate|python|shuffled>` takes the first 2000 FineWeb
documents (6.5 MB) as a base and adds 10% / 30% / 50% of its bytes of one kind of data. `control`
is further FineWeb documents; `duplicate` is exact copies of base documents; `python` is Python
source (`codeparrot/codeparrot-clean-valid`); `shuffled` is the `control` documents with their
words shuffled (same vocabulary and word frequencies, order destroyed). Every kind adds the same
bytes at a given dose (within 0.7%), so variants are compared at equal size. Each variant learns
its own tokenizer at every budget of `experiments/merge_sweep` and is scored on its order-1
`<bos>`/`<eos>`-framed graph. Results: `<kind>_results.json`. The first run of `python` downloads
800 files and caches them in `python_documents.json` (gitignored); the FineWeb documents come from
`experiments/merge_sweep`'s cache. One seed.

A useful indicator should rise with `control` and not with the other three.

## Result (2026-09-27; log AUC at a 50% dose, base in the first row)

| Kind | red 250 | red 4000 | red 16000 | document-red 16000 | blue 16000 | green 4000 | green 16000 |
|---|---|---|---|---|---|---|---|
| base | 0.625 | 0.399 | 0.221 | 0.247 | 0.410 | 0.718 | 0.469 |
| control | 0.634 | 0.421 | 0.248 | 0.280 | 0.433 | 0.729 | 0.491 |
| duplicate | 0.642 | 0.424 | 0.254 | 0.291 | 0.433 | 0.731 | 0.494 |
| python | 0.673 | 0.444 | 0.268 | 0.263 | 0.445 | 0.627 | 0.420 |
| shuffled | 0.654 | 0.428 | 0.245 | 0.279 | 0.451 | 0.679 | 0.459 |

Green at 4000 merges, by dose (base 0.718):

| Kind | 10% | 30% | 50% |
|---|---|---|---|
| control | 0.720 | 0.726 | 0.729 |
| duplicate | 0.721 | 0.727 | 0.731 |
| python | 0.695 | 0.653 | 0.627 |
| shuffled | 0.697 | 0.682 | 0.679 |

- **Red, blue and document-red fail: every kind of contamination raises them, most by more than
  the control does.** Python raises red the most (0.673 vs 0.634 at 250 merges): code repeats its
  indentation and keywords so often that it adds heavy, well-connected edges. Shuffled text raises
  red and blue too: random word order puts frequent words next to each other in new pairs, which
  connects the graph further. Red and blue measure connectivity and repetition, and noise adds
  both — the user's "if everything is a pattern, nothing is".
- **Duplicates fool every curve**, document-red included (0.291 vs control's 0.280 at 16000
  merges): an exact copy is a separate document, so it reads as independent support. As expected.
- **Green (flow persistence) is the only curve that separates Python and shuffled text from the
  control.** From 4000 merges on, both lower it below the base while the control raises it, and
  the drop grows with the dose (Python 0.695 → 0.627, shuffled 0.697 → 0.679 at 4000 merges).
  At 250 merges or fewer it rises with every kind: the separation exists only once nodes approach
  words.
- Python also lowers bytes per token (4.13 → 3.79 at 16000 merges), since the tokenizer spends
  merges on code; the other kinds leave it unchanged.

Consequence: as they stand, red and blue cannot be the quality score, because contamination
raises them. Green at high merge budgets is the only candidate so far, and it does not detect
duplicates, which needs a measure that credits a transition by the distinct contexts supporting it
rather than by its occurrences. One seed only; the subsampling experiment's seed spread (≤ 0.01)
is below the green gaps (0.05–0.09) but above some red gaps.
