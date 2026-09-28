# Syntactic sufficiency: does a domain's corpus show each of its syntactic patterns with enough different fillers?

**Goal:** the paper's central indicator, agreed on 2026-09-28. It replaces the idea that the
intensity curves (red, blue) score a corpus: `experiments/contamination` showed they rise with
duplicates, shuffled text and code, because they measure only how travelled a transition is. This
design measures what the paper claims matters for generalization: that each syntactic pattern of
the knowledge domain is seen with many different fillers, so a model learns the pattern rather
than memorizing it with one word.

## Conceptual frame

1. **Knowledge domain.** A bounded subset of the syntactic and semantic patterns of human language
   (recipes: mostly imperative actions on ingredients). This paper measures only the syntactic
   ones. Syntax and semantics intersect; the measure is built so that semantics does not leak in
   (see "Classes").
2. **Syntax** is the frame of a sentence regardless of its truth: "the dog has 5 legs" is
   syntactically correct, "the dog has 4 legs" is syntactically and semantically correct, "the dog
   has blue legs" is neither. "Cut the milk" is syntactically correct; that it makes no sense is
   semantics, out of scope.
3. **No semantics without syntax, in generation.** A model that lacks the domain's frames cannot
   produce well-formed sentences, so it cannot produce domain-correct ones. Syntax is the base
   layer the paper diagnoses first.
4. **Syntax is hierarchical and mostly local.** Pronoun + to be is a pattern; third-person
   agreement (he/she/it + is) is a finer one inside it. Patterns are small subgraphs over
   word-like nodes; some dependencies reach further, but less far than semantic ones.
5. **A syntactic pattern in the graph** is a travelled segment that forks: many routes pass
   through a head (I → am, cut) and then split into many different stations of one grammatical
   class (adjectives, -ing verbs; ingredients). Three ingredients: an intense **trunk**, **diverse
   branches**, **coherent branches** (one class). Exact copies have a trunk but one branch
   (memorization); shuffled text has branches of any class (noise).
6. **A generalizable corpus** shows every pattern of its domain with enough distinct fillers that
   the model learns it as independent of the filler word. Patterns outside the domain being rare
   or absent is not a defect: the measure is relative to the domain.
7. **Outliers** — patterns carried by very few documents, text foreign to the domain — contaminate
   the domain's patterns and must show in the metrics.
8. **Domain-relative, single-corpus.** The pattern inventory is each corpus's own; no reading is
   compared across corpora as if domains shared one. What is shared is the procedure, and any
   threshold is set on quantities normalized within the domain (a probability, a fraction of the
   domain's own classes), never on raw counts.

## Definitions

The corpus is tokenized with the model's own BPE, at a merge budget chosen for about 1.5 tokens
per word (`experiments/granularity`), so nodes are word-like and suffixes such as `ing` are nodes
of their own. The transition graph is `build_transition_graph`'s, and every count below is taken
over **distinct contexts** (`context_support`), so exact copies add nothing.

### Classes

Each token of the corpus that occurs in at least `min_documents` documents is described by the
distribution of its left and right neighbors. Tokens are clustered hierarchically on those
distributions, and the tree is cut at **`n_classes` coarse classes** (default 16, the order of a
universal part-of-speech tag set). Only the coarse cut is used: finer cuts start grouping by
meaning (things you pour against things you chop), which would bring semantics back in.

### Patterns

A pattern is `(h, Y)`: a head `h` (one token; two tokens where the finer level is wanted, e.g.
`he` + `is`) followed by a token of class `Y`. For each pattern:

- `trunk`: occurrences of `h` followed by a class-`Y` token, in distinct contexts;
- `fillers`: distinct class-`Y` tokens seen after `h`;
- `coverage = fillers / |Y|`, where `|Y|` is the class's size in this corpus;
- `missing_mass = f1 / trunk`, with `f1` the fillers seen in only one context after `h`
  (Good–Turing): the probability that the next use of the pattern brings a filler not yet seen. Near
  0, the pattern is saturated; near 1, every new use still brings a new filler;
- `documents`: documents the pattern occurs in.

A pattern belongs to the **domain** when `documents >= min_pattern_documents` (default 3) and
`trunk` is at least a small fraction of its head's occurrences (the branch is not a stray). The rest
are **outlier patterns**; their share of all occurrences is reported, and they are listed.

### Corpus readings

- **Sufficiency curve:** for a required coverage `c` swept from 0 to 1, the fraction of domain
  patterns (weighted by `trunk`) with `coverage >= c`, and its area, which equals the
  trunk-weighted mean coverage. It keeps the red curve's form — a threshold swept, a coverage read
  — on the quantity the paper cares about. (First planned on `1 − missing_mass`; see "Measured on
  the recipe grammar" for why coverage replaced it.)
- **Hierarchy:** the same readings for one-token heads and for two-token heads, so a corpus can be
  sufficient at the coarse level and insufficient at the finer one ("third-person agreement is
  seen with few verbs").
- **Diagnosis:** the domain patterns with the highest missing mass, and the outlier patterns — where
  the corpus needs more text, and what does not belong.

## Predictions, to be tested on a synthetic grammar

A probabilistic grammar with known categories (determiners, nouns, adjectives, imperative verbs,
pronouns with agreement) generates corpora of equal size in which only one thing changes. The true
patterns are known, so both the measure and a trained model can be checked against them.

| Variant | Measure predicted | Model predicted |
|---|---|---|
| **Diverse**: each verb seen with most nouns | high sufficiency | generalizes to unseen verb–noun pairs |
| **Restricted**: each verb seen with 1–2 nouns | low sufficiency on verb patterns | does not |
| **Duplicated**: diverse + exact copies | unchanged from diverse | unchanged |
| **Shuffled**: diverse, words shuffled | classes and patterns collapse | no syntax learned |
| **Outliers**: diverse + a few documents from another grammar | outlier share flags them | — |
| **Size sweep**: diverse at growing size | sufficiency rises, then saturates | held-out loss saturates at the same size |

### The recipe grammar — `src/corpus_generalization/synthetic/recipes.py`

Recipes, the example the frame is written in. English, lowercase, words separated by spaces, a
sentence ending in ` .`, a document of 4–10 sentences. The lexicon is closed: imperative verbs,
countable ingredients with a singular and a plural form, adjectives, adverbs, and containers and
utensils each with its preposition. Four sentence rules:

| Rule | Example | Pattern it carries |
|---|---|---|
| action | `cut the ripe tomatoes .` | verb → determiner → (adjective) → ingredient |
| action + manner | `stir the onion gently .` | the same, then an adverb |
| action + place | `fry the potatoes in the pan .` | the same, then preposition → tool |
| state | `the onions are soft .` / `the onion is soft .` | agreement: singular → `is`, plural → `are` |

Two knobs change what the corpus shows, and nothing else:

- **`fillers_per_verb`**: how many ingredients each verb is ever seen with (all of them for the
  diverse variant, 1–2 for the restricted one). Every ingredient still occurs in state sentences,
  so the model always meets it as a noun.
- **Reserved combinations**, drawn from the seed before anything else, so every variant of a seed
  holds out the same ones: for each verb, `reserved_per_verb` ingredients it is never seen with;
  for each ingredient, one adjective it is never described with in a state sentence.

**Minimal pairs** are built only from reserved combinations, so they test generalization, not
recall:

- verb pattern: `cut the tomato .` against `cut the slowly .` (an adverb in the noun's slot);
- agreement: `the tomatoes are ripe .` against `the tomatoes is ripe .`, and the singular
  counterpart.

**Foreign documents** for the outlier variant come from a small finance grammar (`the bank raises
the rate .`) that shares no content word with the recipe lexicon. Copies and word shuffling are
applied to the generated documents by the experiment, as `experiments/contamination` does.

The model is the fixed GPT of `docs/2026-09-28-training-design.md`. Its syntactic generalization is
scored with **minimal pairs on combinations absent from training**: the model should give "cut the
onion" (never seen, class-valid) a higher probability than "cut the blue" (class-violating). The
red and blue curves and the held-out n-gram predictability are read on the same corpora: the
paper's claim holds if sufficiency tracks the minimal-pair accuracy where those do not.

## Measured on the recipe grammar (2026-09-28, 2000 documents, seed 0)

`analysis/syntactic.py` implements the classes, the patterns, the domain split and the curve;
`tests/test_syntactic_classes.py` and `tests/test_syntactic_patterns.py` pin what follows.

- **Classes.** Left and right neighbor distributions (square-rooted, so Euclidean distance is
  Hellinger distance), Ward clustering. On the diverse variant the merge heights jump from 0.41
  to 1.42 at 11 classes, and the cut there is exactly the grammar's categories with singular and
  plural nouns apart and `is` and `are` apart — the agreement level of the hierarchy. Near the
  leaves heights approach zero and their ratios are noise (the most restricted variant was cut at
  64 classes); merges below 5% of the root's height are floored. In the restricted variants the
  ingredients split into those some verb uses and those seen only in state sentences (13 classes):
  a real distributional difference, and the verbs' coverage stays low against the smaller class.
- **Coverage separates the variants, missing mass does not.** The verbs' noun slot (`<verb> the`
  → singular ingredients) covers 0.93 of its class on the diverse variant and 0.09 on the
  restricted ones (2 of 23; 1 of 11). Missing mass reads 0.000 on both: a verb only ever seen with
  one ingredient, many times, looks saturated. From its own pattern alone the corpus cannot tell
  "this verb takes one noun" from "this corpus shows it with one noun"; the class — the same
  ingredients seen elsewhere in the same slot position — can. Missing mass stays as the size
  reading (would more text of the same kind bring new fillers).
- **The corpus-wide area dilutes a local gap.** Restricting only the verbs' noun slots moves the
  area from 0.958 to 0.902: every other pattern is untouched. The indicator's reading is therefore
  per pattern (the list of domain patterns with low coverage), and the paper's test is per
  pattern too: each verb's coverage against the model's minimal-pair accuracy on that verb.

## First trained test: does coverage predict whether the model ties a verb to its nouns?

Written before any training (2026-09-28).

**The minimal pairs above are too easy for the verb pattern.** "cut the slowly ." is rejected by
any model from "the slowly" alone, which occurs nowhere in any variant, so every variant would
score near 100%. They stay as a control. The verb pattern is tested by what the theory is about:
whether the model ties a verb to the fillers it was shown (memorization) or learns the verb + any
ingredient pattern.

**Generalization ratio, per verb.** After `<bos>cut the`, the model's probability of each singular
ingredient. With `U` the ingredients never seen in a training sentence with that verb and `N` all
of them:

    ratio = (P(U) / P(N)) / (|U| / |N|)

1 means the model gives the unseen ingredients the share it gives the seen ones (the pattern is
learned); 0 means only the seen ones (memorized). It is comparable across variants although `U`
differs (2 ingredients in the diverse variant, up to 29 in the most restricted one).

**Design.** `fillers_per_verb` in {1, 2, 4, 8, 16, all}, 2000 documents each, seed 0, the same
reserved combinations. One fixed model for every variant: 2 layers, 4 heads, 128 wide, context
128 (0.5M parameters; the grammar has ~130 word types), 3000 steps, BPE at 1000 merges (every
word a whole token from 290 merges on). Per variant: each verb's coverage (the `<verb> the` →
singular-ingredient pattern), its ratio, the held-out loss, the held-out n-gram predictability
(`analysis.information`), and the minimal-pair accuracy.

**Predictions.**

1. Per verb, coverage and ratio rise together: Spearman above 0.8 across all verbs of all variants.
2. The diverse variant's mean ratio is at least 0.8; the one-filler variant's at most 0.3.
3. The held-out loss and the n-gram predictability do **not** flag the restricted variants: they
   read them as equal or *more* predictable than the diverse one, because held-out text of a
   restricted corpus is restricted in the same way. A held-out measure answers "does it predict
   more text of this corpus", not "has it learned the domain's pattern" — the distinction this
   indicator exists for.
4. Minimal-pair accuracy is near 100% on every variant (the control above).

Where between 1 and 28 fillers the ratio crosses 0.5 is the open question, and the reason the
test is interesting: nouns share every state sentence, so the model could generalize from the
class alone even with few fillers per verb (prediction 1 would then fail at the low end). If it
does, per-head coverage is not what the model needs, and the frame's claim must be restated.

**Outcome (2026-09-28, `experiments/syntactic_sufficiency`).** Predictions 1–3 held (Spearman
0.891; ratio 0.909 diverse, 0.011 one filler; held-out loss and n-gram predictability rank the
variants in reverse). Prediction 4 failed: verb minimal pairs track the ratio (0.25 with one
filler) instead of sitting near 100%. The ratio rises late: 0.17 at half coverage, 0.91 near full.
One seed and one model size; see that README for the limits.

## Tests (before the code)

1. **Classes:** on the synthetic grammar, the coarse cut recovers its categories (V-measure above
   a set bar).
2. **Diverse vs restricted:** the verb patterns' missing mass is low on the first and high on the
   second.
3. **Copies:** appending exact copies leaves every pattern reading unchanged.
4. **Outliers:** patterns seen in fewer than `min_pattern_documents` documents are outliers, not
   domain patterns.
5. **Bounds:** `0 <= coverage <= 1`, `0 <= missing_mass <= 1`.

## Open questions

- `n_classes` and the clustering method (hierarchical on neighbor distributions, or Brown
  clustering, which builds the tree directly); both are checked first on the synthetic grammar.
- Dependencies longer than two tokens (agreement across a relative clause) are out of this step.
- How the natural-language corpora enter once the synthetic results are in.
