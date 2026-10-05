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

## Replication on modals and possessives (2026-10-04)

`replication.py` rereads the modals on the models above with a valid test: base-form verbs (VB)
against nouns and adjectives. It also adds possessives (`my your his our their its` → noun,
against adverbs and VBD/VBZ verbs), with `poss_class_cut`, `poss_head_cut` (`his`, `their`) and
`poss_control` from `r50`, 5.59 MB each, seeds 0 and 1.

    poetry run python experiments/class_patterns/replication.py prepare
    THREADS=3 poetry run python experiments/class_patterns/replication.py train <name> <seed>
    poetry run python experiments/class_patterns/replication.py items
    poetry run python experiments/class_patterns/replication.py summarize
    poetry run python experiments/class_patterns/replication.py diagnose <name>
    poetry run python experiments/class_patterns/replication.py summarize-tool

| Class | Class cut ΔG | Removed heads: head cut / class cut | Siblings, head cut |
|---|---|---|---|
| pronouns | −0.195 | −0.044 / −0.177 (25%) | −0.028 |
| modals | −0.126 | −0.040 / −0.081 (49%) | −0.008 |
| possessives | −0.303 | −0.194 / −0.343 (57%) | +0.046 |

Six of seven predictions held. The one that failed is possessive prediction 3: their head cut
cost 57%, against a bar of half.

- Removing a class pattern removes syntax for every member, in all three classes.
- Removing it from two heads costs them a quarter to a little over half as much, and leaves
  their siblings intact.
- Syntax is learned at both levels, mostly at the class.

**The tool's class rows** (`corpus-diagnose`, induced head classes at 64):

- They find the modal class in `class_cut`.
- They miss the pronouns (excess +0.47, floor 0.5) and the possessives, which share an induced
  class with `the` and `a`.
- They raise no false flag in the other cuts.

## The tool, second pass (2026-10-04)

`tool_check.py` builds fresh class and head cuts for two classes never used before. The numbers
are cut after a NOUN, with `five` and `six` as the subgroup. The base verbs are cut after a DET,
with `make` and `get`. It then runs `corpus-diagnose` on those 6 cuts and the 6 earlier ones:

    poetry run python experiments/class_patterns/tool_check.py prepare
    poetry run python experiments/class_patterns/tool_check.py diagnose <name>
    poetry run python experiments/class_patterns/tool_check.py summarize

The class rows found both fresh class cuts (numbers +2.62, base verbs +0.53) and flagged
nothing in the head cuts or the controls. The short heads showed the possessives and `five` and
`six`, but not `make` and `get`: coarse filler classes pool determiners with adjectives. The
new `absent` head rows fire 79–114 times in random controls (127–151 in class cuts), so they
need a noise-aware rule. 2 of 4 predictions held.
