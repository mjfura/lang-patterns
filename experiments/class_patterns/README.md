# Head patterns and class patterns: at which level does a model learn syntax?

["Head patterns and class patterns"](../../docs/2026-09-28-syntactic-sufficiency-design.md#head-patterns-and-class-patterns-at-which-level-does-a-model-learn-syntax)
sets out the design and four predictions, written before running.

From the documents of `r50` (7.2 MB), only sentences are removed. A target sentence has a
pronoun (`I you he she we they`) or a modal (`can could will would should may might must`)
immediately followed by a token spaCy tags VERB.

- `class_cut`: every target sentence is removed (18,126 of 62,025).
- `head_cut`: only those of `she`, `they`, `can` and `will` (5,730), plus random non-target
  sentences to the same bytes.
- `control`: random non-target sentences to the same bytes.

All three hold 4.67 MB and 2,220–2,225 documents. The small model is trained on seeds 0 and 1.
The readings are:

- D per head and per class (the class's heads pooled) against F, minus the control;
- G per head, by `experiments/syntactic_eval`, with the slot class fixed to VERB.

    poetry run python experiments/class_patterns/class_patterns.py prepare
    THREADS=3 poetry run python experiments/class_patterns/class_patterns.py train <name> <seed>
    poetry run python experiments/class_patterns/class_patterns.py items
    poetry run python experiments/class_patterns/class_patterns.py summarize

Corpora go to `runs/class_patterns/corpora/` (gitignored, rebuilt by `prepare`), results to
`results/`.

## Result (2026-10-04)

| Prediction | Result |
|---|---|
| 1. the cuts do what they say (corpus side) | HELD |
| 2. `class_cut` mean ΔG ≤ −0.10 over 14 heads | FAILED: −0.086 (pronouns −0.195, modals −0.004) |
| 3. removed heads lose < half in `head_cut` than in `class_cut` | HELD: −0.031 vs −0.078 (pronouns −0.044 vs −0.177) |
| 4. other heads mean \|ΔG\| ≤ 0.03 | FAILED: 0.052 / 0.040, the noise between models |

For pronouns, `she` and `they` keep their syntax when only they lose the verb slot, and lose it
(about −0.2 G, like every pronoun) when the whole class does. Head-level D said they lost it
equally in both cuts; class-level D said `head_cut` barely touched the pattern, and the model
agrees with the class level. For modals the test is not valid: spaCy's VERB pools all forms,
and only base forms follow a modal (`can kept` is among the "valid" words).
