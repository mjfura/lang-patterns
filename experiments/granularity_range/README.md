# Over which tokens-per-word range do the verdicts hold?

Step 1 of the settings-from-data path. ["Step 1: the tokens-per-word range"](../../docs/2026-10-11-settings-from-data-design.md#step-1-the-tokens-per-word-range)
sets out the design and predictions, written before running. Corpus side only: no model is trained.

One tokenizer is learned on F (the FineWeb pool, 14.3 MB) to 60,000 merges; each granularity
is a cut of it. At each, F is read once with `prepare_reference`, and the 22 labeled cuts of
`experiments/class_patterns` are diagnosed against it, four in parallel.

    poetry run python experiments/granularity_range/granularity_range.py budgets
    poetry run python experiments/granularity_range/granularity_range.py diagnose <target>
    poetry run python experiments/granularity_range/granularity_range.py summarize
    poetry run python experiments/granularity_range/granularity_range.py diagnose 1.50 --min-trunk 5
    poetry run python experiments/granularity_range/granularity_range.py summarize-min-trunk
    poetry run python experiments/granularity_range/granularity_range.py exchangeable 48 [--vary-draws]
    poetry run python experiments/granularity_range/draw_count.py pool 200
    poetry run python experiments/granularity_range/draw_count.py rates

The merges go to `runs/granularity_range/merges.json` (gitignored), results to `results/`.

## Result with 101 draws and alpha 0.06 (2026-10-12)

The bar is now the 6th largest draw maximum of 101 draws, so that a random corpus's chance of a
flag stays near alpha = 0.06 for any one set of draws.

- `exchangeable 48`: random whole-document cuts of F against one fixed draw set. With 16 draws, 6
  of 48 were flagged (0 with fresh draws per cut). With 101 draws, 2 of 48 were.
- `draw_count.py`: a pool of 200 draws shows the conditional rate's spread across draw sets at
  16–128 draws, as Beta(k, n + 1 − k) predicts.
- Sweep: all five classes at every granularity, 0–3 head flags on 12 controls (4 of 72 in all),
  every dropped word found. Cuts take 4–9 minutes.

`results/draws_16/` and `results/exchangeable/draws_16/` hold the 16-draw runs.

## Result with the trunk from the draws (2026-10-12)

The least trunk read is now set by the draws (Tarone's criterion: read only patterns a corpus
could flag), and the rounding floor uses the tested value's own step.

- All five removed classes are found at every granularity from 1.35 to 1.80; no control's class
  row is flagged. The class bar falls from 39–94 to 10–29.
- Four to six times as many head patterns are read. Head flags on the 12 controls are 0–5 per
  granularity (9 of 72, against 7 of 72 with trunk 20), and `absent` still finds all 8 dropped words.
- Under-shown and within-noise swap in about 0.5 of 10,000 patterns between adjacent granularities.

`results/min_trunk/` holds the fixed-trunk sweep at 1.50 (`summarize-min-trunk`), and
`results/fixed_min_trunk_20/` the sweep below; this sweep's targets are in `results/draws_16/`.

## Result with trunk 20 (2026-10-11)

Class rows are read at every node of the head tree; class rows, head rows and `absent` are all
judged against the draws' own bar (`docs/2026-10-11-settings-from-data-design.md`).

- The five removed classes (pronouns, modals, possessives, numbers, base verbs) are found at
  every granularity from 1.40 to 1.80 tokens per word. At 1.35 two are missed.
- No control's class row is flagged at any granularity. Head rows flag 0–2 of the 12 controls,
  the rate the bar is built for; `absent` finds all 8 dropped words everywhere.
- Among patterns read at two adjacent granularities, under-shown and within-noise swap in
  fewer than 3 of 10,000. What moves is whether a pattern is read at all (`min_trunk`).

The verdicts set a lower edge near 1.40 tokens per word and no upper edge up to 1.80. 1.30 does
not fit in 16 GB.

`results/cut_at_64_head_classes/` holds the first sweep (64 head classes, `class_floor` 0.5);
`results/fixed_low_and_sigmas/` the second (every node, but head rows with `low` 0.5 and 3 sd).
