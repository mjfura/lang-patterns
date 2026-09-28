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
`experiments/merge_sweep`'s cache. `contamination.py <kind> <seed>` with a seed other than 0 shuffles
the FineWeb documents before the base/rest split, and the Python files, so both the base and what
is added change; its results go to `<kind>_seed<N>_results.json`. Seeds 0, 1 and 2 were run.

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

## Three seeds

Change against the same seed's base, mean over seeds 0–2 with [min, max]:

| Curve | Kind | 10% | 30% | 50% |
|---|---|---|---|---|
| green, 4000 merges | control | +0.003 [+0.002, +0.003] | +0.010 [+0.009, +0.013] | +0.013 [+0.011, +0.017] |
| | duplicate | +0.003 [+0.002, +0.003] | +0.009 [+0.008, +0.010] | +0.015 [+0.013, +0.017] |
| | python | +0.004 [−0.022, +0.019] | −0.047 [−0.065, −0.025] | −0.073 [−0.091, −0.045] |
| | shuffled | −0.021 [−0.026, −0.016] | −0.034 [−0.037, −0.029] | −0.035 [−0.039, −0.031] |
| green, 16000 merges | control | +0.007 [+0.003, +0.010] | +0.019 [+0.015, +0.021] | +0.026 [+0.022, +0.030] |
| | python | +0.009 [−0.015, +0.022] | −0.018 [−0.035, −0.001] | −0.032 [−0.049, −0.010] |
| | shuffled | −0.006 [−0.012, −0.002] | −0.010 [−0.016, −0.005] | −0.005 [−0.010, −0.001] |
| red, 250 merges | control | +0.003 [+0.002, +0.004] | +0.008 [+0.008, +0.009] | +0.011 [+0.009, +0.012] |
| | python | +0.010 [+0.008, +0.013] | +0.032 [+0.022, +0.040] | +0.038 [+0.024, +0.048] |
| | shuffled | +0.007 [+0.006, +0.008] | +0.021 [+0.020, +0.023] | +0.031 [+0.029, +0.033] |

- **The green result holds, best at 4000 merges.** Shuffled text lowers green at every dose and
  every seed, with no overlap with the control; Python does from 30%, while at 10% its range
  straddles zero. At 16000 merges the shuffled drop shrinks to −0.005, near the seed spread, and at
  1000 merges nothing separates (every kind within −0.022 to +0.013).
- **Red and blue rise with every kind on every seed**, contamination by more than the control
  (red at 250 merges: Python +0.038 and shuffled +0.031 against the control's +0.011).
- **Duplicates stay indistinguishable from the control** on every curve.

Consequence: as they stand, red and blue cannot be the quality score, because contamination
raises them. Green at around 4000 merges is the only candidate so far, and it does not detect
duplicates, which needs a measure that credits a transition by the distinct contexts supporting it
rather than by its occurrences (`docs/2026-09-27-context-support-design.md`).

## Context support

The same three seeds rerun with red, blue and green also read over `context_support` (the distinct
token-before/token-after contexts of a transition; `docs/2026-09-27-context-support-design.md`),
as `context_<curve>_log`. The earlier columns came out identical. Change against the base, mean
over seeds [min, max], 50% dose:

| Curve | control | duplicate | python | shuffled |
|---|---|---|---|---|
| red, 16000 merges (count) | +0.026 [+0.026, +0.027] | +0.033 [+0.032, +0.034] | +0.048 [+0.047, +0.051] | +0.023 [+0.023, +0.023] |
| red, 16000 merges (context) | +0.026 [+0.025, +0.026] | −0.000 [−0.000, −0.000] | +0.031 [+0.030, +0.032] | +0.025 [+0.025, +0.026] |
| green, 4000 merges (count) | +0.013 [+0.011, +0.017] | +0.015 [+0.013, +0.017] | −0.073 [−0.091, −0.045] | −0.035 [−0.039, −0.031] |
| green, 4000 merges (context) | +0.015 [+0.014, +0.016] | −0.001 [−0.001, −0.000] | +0.031 [+0.027, +0.034] | −0.032 [−0.034, −0.030] |

- **Duplicates no longer move any curve** read over `context_support` (every budget, dose and
  seed within ±0.003), while the control still raises them all: duplicates are now told apart.
- **Green over `context_support` loses Python**: code raises it (+0.031 at 4000 merges), where
  green over `count` fell by 0.073. Shuffled text still lowers it at 4000 and 16000 merges.
- **No single curve separates all three kinds** from the control. Green over `count` catches
  Python and shuffled text; any curve over `context_support` catches duplicates.
- **The gap between the two readings flags redundancy**: at red, 16000 merges, count minus
  context is 0.000 for the control, +0.033 for duplicates, +0.017 for Python and −0.002 for
  shuffled text.

The spec's success criterion is met for duplicates and shuffled text, not for Python. These are
changes against a known base; on a single corpus, the analogue to test would be the level of each
reading and the count/context gap, not their change.
