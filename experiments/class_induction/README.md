# Class induction: do its constants change the verdicts?

Step 2 of `docs/2026-10-11-settings-from-data-design.md`. The class-induction constants of
`analysis/syntactic.py` are varied one at a time on F at 1.50 tokens per word. The 22 labeled
cuts of `experiments/granularity_range` are diagnosed against each reading, with 101 draws, alpha
0.06 and the derived trunk. Only the corpus side is measured; no model is trained.

    poetry run python experiments/class_induction/class_induction.py context <n>
    poetry run python experiments/class_induction/class_induction.py summarize

`results/context_<n>.json` holds one reading per value. `results/summary.json` compares each
reading with 1,000, which is `experiments/granularity_range/results/target_1.50.json`.

## `CONTEXT_TOKENS` (2026-10-12)

The neighbor features use the n most frequent tokens; the remaining tokens share one column.

| n | running tokens covered | classes | 5 classes found | class flags on controls | head flags on controls | absent on controls | dropped words found |
|---|---|---|---|---|---|---|---|
| 250 | 51% | 17 | yes | 0 | 2 (r25c, r25e) | r25c, r25e | 8/8 |
| 500 | 59% | 11 | yes | 0 | 1 (control) | r25c, r25e | 8/8 |
| 1,000 | 68% | 12 | yes | 0 | 0 | r25c, r25e | 8/8 |
| 2,000 | 77% | 18 | yes | 0 | 1 (ctrl_d) | r25c, r25e | 8/8 |
| 4,000 | 87% | 12 | yes | 0 | 0 | r25c, r25e | 8/8 |

- **The verdicts hold from 250 to 4,000.** All five removed classes are found at every value.
  No control has a class flag. Control head flags stay at the design rate, at most 2 of 12
  controls. The two absent rows on controls are the same at every value (` luck` in r25c,
  `Maybe` in r25e), and all 8 dropped words are found.
- The bars barely move: class 12.6–15.9, head 4.7–5.0, absent 3.80–3.88.
- **What does change is the partition.** The coarse cut picks 11 to 18 classes, with no trend in
  n. As a result the pattern keys themselves shift.
  - Patterns read at both values agree on 75–86% of verdicts.
  - Nearly all disagreements are read against unread, because a class of a different size
    changes a pattern's trunk.
  - Swaps between a flag and within-noise number 33 (500) to about 120 (250, 2,000). That is
    more than across granularities (26–37). Most fall on the class and drop cuts, where flags
    are expected.
- Detection is weaker at some values: at 2,000 the capitalized pronoun node (`I We You They`) is
  not flagged, and at 250 and 2,000 the possessive node keeps only two heads. In both cases the
  class itself is still found.

**Reading.** `CONTEXT_TOKENS` does not change the cut-level verdicts across a 16x range, so it
is shown not to matter and stays at 1,000. The pattern-level instability comes from the number
of classes the cut picks, which is governed by `min_classes`, `max_classes` and `NOISE_FLOOR`.
Those are the next settings to measure.

## The coarse cut's class count (2026-10-13)

`min_classes = 8`, `max_classes = 64` and `NOISE_FLOOR = 0.05` only decide where the Ward tree is
cut into filler classes. By default the cut lands at 12 classes. Here the cut is forced to
k = 8, 16, 24, 32 and 64 instead (`classes <k>`, through `CLASS_CUT`), which covers every cut the
three settings can choose. The context is the default 1,000 tokens.

| k | 5 classes found | class flags on controls | head flags on controls | absent on controls | dropped words found | agreement with 12 | flags kept |
|---|---|---|---|---|---|---|---|
| 8 | yes | 0 | 0 | r25c, r25e, ctrl_d | 8/8 | 97% | 91/112 |
| 12 (default) | yes | 0 | 0 | r25c, r25e | 8/8 | — | — |
| 16 | yes | 0 | 0 | r25c, r25e | 8/8 | 92% | 122/166 |
| 24 | yes | 0 | 0 | r25c | 8/8 | 79% | 126/236 |
| 32 | yes | 0 | 0 | r25c, ctrl_c | 8/8 | 73% | 126/289 |
| 64 | yes | 0 | 1 (control) | r25c | 8/8 | 63% | 120/409 |

- **Class rows do not depend on k.** At every k they flag the same nodes with the same t: modals
  319.4, pronouns 253.5, capitalized pronouns 42.5, possessives 164.9, numbers 103.5, verbs
  100.5. The class bar is 13.92 at every k. Class rows are read on the whole tree and its filler
  tokens, so the coarse cut never enters them.
- **Cut-level head and absent verdicts hold.**
  - Head flags on the class and head cuts stay at 33–43 and 10–14.
  - There is no head flag on a control, except a single row at k = 64 (`!`).
  - Each control has at most one absent row. The new rows (` Death` in ctrl_d at k = 8, ` For`
    in ctrl_c at k = 32) are single rows, within the family's alpha.
  - All 8 dropped words are found.
  - The head and absent bars move within 4.77–4.98 and 3.73–3.88.
- **Pattern-level agreement falls as k moves away from 12**, from 97% at 8 to 63% at 64. Absent
  rows on the drop cuts grow from 22 to 39. This is expected: a head row asks about the head
  followed by one filler class, so a finer cut asks a finer question.

**Reading.** The class count does not change the cut-level verdicts anywhere from 8 to 64, so
`min_classes`, `max_classes` and `NOISE_FLOOR` are shown not to matter for them. The defaults
stay. With `CONTEXT_TOKENS` this closes step 2.
