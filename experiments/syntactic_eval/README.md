# A syntactic evaluation on natural text

["A syntactic evaluation on natural text"](../../docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md#a-syntactic-evaluation-on-natural-text)
sets out the design and six predictions, written before any model was scored. No training: the
models are those of `dose_response` (plantings, 6) and of `subset_selection` and `cut_choice`
(cuts, 36).

After a real evaluation prefix ending in head h, each model is asked whether it gives a word of
h's slot class that never follows h in the reference more probability than a word of another
open class, matched in reference frequency. Truth does not enter. Classes come from spaCy's
`en_core_web_sm`, never from the indicator:

- a word's class is its majority tag (≥ 90%, ≥ 20 occurrences, NOUN/VERB/ADJ/ADV);
- a head's slot class is the most frequent open-class tag after it, if it holds ≥ 20% (lowered
  from 30% before scoring, from tag counts alone; see the doc).

G is the share of the 20 pairs × 20 contexts that the valid word wins. Chance is 0.5.

    uv run python experiments/syntactic_eval/syntactic_eval.py prepare <plantings|cuts>
    uv run python experiments/syntactic_eval/syntactic_eval.py score <plantings|cuts>
    uv run python experiments/syntactic_eval/syntactic_eval.py summarize

`prepare` tags the reference (1–3 minutes) and writes `results/items_<family>.json`. `score`
writes `results/scores_<family>.json` (a few minutes), and the summary goes to
`results/summary.json`.

## Result (2026-10-04)

All six predictions failed.

- **Controls:** mean G 0.724 / 0.680 (bar 0.70). Strong slots read 0.75–0.94, minority slots
  near chance.
- **Plantings:** the strongest remove class knowledge on seed 0 (`even` 0.76 → 0.34, `now`
  0.78 → 0.36), less on seed 1. There is no dose order: Spearman(D, delta) −0.40 / +0.01.
- **Cuts:** G follows size (`full` 0.748, 25% cuts 0.700–0.736, `r10` 0.69) and document count
  (+0.66 / +0.56), not the score (−0.24 / −0.41). Per head there is nothing (−0.14 to +0.13).
  G per head reproduces across seeds (Pearson 0.87–0.91), so this is not noise.

The drop (probability of the true word) reads which filler. It moves where E says a cut moved
the slot's fillers. G reads which class, and the model keeps that unless a slot collapses. The
natural filters tested here never collapse a slot. On natural text, the indicator predicts a
cut's filler-level cost, and class-level failure only for collapsed slots, as in the recipe
grammar.
