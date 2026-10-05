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
  64 classes); merges below 5% of the root's height are floored. At the other end, on small corpora
  the jump to 2 classes can win (500 documents; see "Corpus size and training dynamics"), so only
  cuts of at least 8 classes are searched — a floor on the coarse categories any language has. In the restricted variants the
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

## Replication: seeds and model sizes

Written before running (2026-09-28). The first test used one seed and one model size, and found
the ratio rising late (0.17 at half coverage, 0.91 near full). Before reading that as where
sufficiency lies, it has to survive other seeds and a larger model.

**Design.** `fillers_per_verb` in {1, 2, 4, 8, 16, 20, 24, all} — 20 and 24 locate the rise the
first test left between 16 and 28 — times seeds {0, 1, 2} (each seed draws its own reserved
combinations, fillers, documents and initialization) times two sizes of the same family: `small`
(2 layers, 128 wide, 0.5M parameters, as before) and `medium` (4 layers, 256 wide, 3.3M). Every
other setting as in the first test.

**Predictions.**

1. Per-verb Spearman(coverage, ratio) above 0.8 for every seed and both sizes.
2. The rise stays late for `small`: ratio below 0.5 at 16 fillers on every seed.
3. `medium` memorizes more easily, so its ratio at intermediate coverage (8–24 fillers) is at or
   below `small`'s. This is the least certain: a larger model could instead share more between
   ingredients through their embeddings and generalize earlier.
4. The held-out loss ranks the variants in reverse of the ratio for every seed and size.

**Outcome (2026-09-28, `experiments/syntactic_sufficiency`).** All four held. Per-verb Spearman
0.940–0.975; small's ratio at 16 fillers 0.171–0.194; medium generalizes later (0.554 against
0.901 at 24 fillers); the held-out loss is reversed on every seed and size. The ratio crosses 0.5
at coverage 0.67–0.80 for small and about 0.80 for medium.

## Corpus size and training dynamics

Written before running (2026-09-28). Both earlier tests fixed the corpus at 2000 documents and read
the model at the checkpoint with the lowest held-out loss — the loss that ranks the variants in
reverse of generalization. Two questions follow.

- **Corpus size at fixed coverage.** More documents with the same fillers per verb add repetitions
  of the combinations already seen, not new ones: coverage does not move. If the ratio moves, then
  coverage alone is not the sufficiency reading, and how often each seen filler repeats must enter
  it.
- **When in training.** A model may treat unseen fillers as valid early and learn the restriction
  later (generalize, then memorize). Then the checkpoint choice decides the ratio, and the
  indicator's threshold holds only for a training budget.

**Design.** The small model; seeds {0, 1, 2}; `fillers_per_verb` in {1, 4, 8, 16, 20, 24, all};
documents in {500, 2000, 8000}; 3000 steps as before. `corpus-train` keeps a checkpoint at steps
{250, 500, 1000, 2000, 3000} beside the best one, and the ratio and the minimal pairs are read at
each. Coverage is read on each corpus as before.

**Predictions.**

1. At fixed fillers, coverage is the same at every corpus size (within 0.02).
2. For 16–24 fillers the ratio at the best checkpoint *falls* as documents grow from 500 to 8000:
   repetition of the seen combinations strengthens the restriction. With all fillers it stays
   above 0.8.
3. For 16–24 fillers the ratio is higher at step 250 than at step 3000 on every seed (generalize,
   then memorize); with all fillers it rises and stays.
4. Per-verb Spearman(coverage, ratio) above 0.8 at every corpus size, at the best checkpoint.

**Outcome (2026-09-28, `experiments/syntactic_sufficiency`).** Predictions 1, 2 and 4 failed, 3
held for 16–24 fillers. On 500 documents a verb is not seen with every allowed filler, so coverage
is lower (the measure was right, the prediction wrong), and the correlation drops to 0.77–0.82; on
2000 and 8000 it is 0.955–0.974. Every model first treats all ingredients alike and then learns
the restriction; the best held-out checkpoint catches that at different points (step ~470 on 500
documents, ~3000 on 8000), which is why the ratio is not monotone in corpus size. Coverage predicts
the end point of training; sufficiency is relative to the model and its training budget. On the
way, the automatic class cut was found to fall to 2 classes on small corpora; `induce_classes`
now searches cuts of at least 8 classes (`min_classes`).

## First reading on natural text

Written before running (2026-09-28). The eight clean sources of `experiments/clean_domains`
(arXiv, PubMed, Wikipedia, C4, FineWeb, FineWeb-Edu, case law, MultiNLI), first 6.5 MB sample of
each, each tokenized at the merge budget that gives it about 1.5 tokens per word
(`experiments/granularity`; case law, which never reaches it, at 32,000). One-token heads only:
two-token heads multiply the patterns past what one process holds on this machine. No trained
model and no ground truth: this reading is exploratory, and it is judged by two things that would
make it fail.

1. **The coarse classes must look grammatical.** Each class is listed with its most frequent
   tokens. The reading fails if the cut is one giant class plus crumbs, or if the classes group by
   topic (every chemistry word together) rather than by category (determiners, prepositions, nouns,
   verbs, adjectives, punctuation).
2. **The patterns flagged must be interpretable.** For each source: the most travelled domain
   patterns with their coverage, and the frequent heads with the lowest coverage (heads tied to few
   fillers of their class). The reading fails if the low-coverage list is dominated by tokenization
   debris (word pieces, markup) rather than constructions a reader can name.

Only the tokens of the 4,000 most frequent are classified (`MAX_CLASSIFIED_TOKENS`); rarer fillers
are left out of coverage. Readings are per source and are not compared across sources.

**Outcome (2026-09-28, `experiments/natural_sufficiency`).** Criterion 1 partly met: no class
groups by topic and several are clean categories (Wikipedia's determiners, singular and plural
nouns; participles), but punctuation is lumped with function words, word pieces and markup fill
whole classes, and arXiv and PubMed were cut at the 8-class floor. Criterion 2 failed as written:
debris dominates the low-coverage list, and its word-like entries are fixed constructions ("due
to", "able to", "pursuant to", "didn't"), which coverage cannot tell from under-covered patterns.
Raw coverage is also not comparable between heads (median 0.008–0.023 in every source).

## Word-like tokens, diversity against chance, and a reference corpus

Agreed with the user on 2026-09-28 after the natural-text reading. Three changes.

### 1. Patterns over word-like tokens

The graph keeps the model's BPE tokens as nodes; only *which* tokens a pattern may be made of
changes. A token is **word-like** when, in the corpus:

- its text is a space followed by letters only (`^ [A-Za-z]+$`), so it starts a word and is not
  punctuation, a digit or markup; and
- it ends the word it starts in at least `WORD_FINAL_SHARE` (0.9) of its occurrences: the next
  token begins with a space or a newline, is not a letter or digit, or is `<eos>`.

The second condition separates " model" (a whole word) from " al" or " qu" (word-initial pieces
usually continued by a suffix), which the text alone cannot. Classes are induced over word-like
tokens only (neighbor features still use every token), and a pattern's head and filler are both
word-like. Punctuation and word pieces then leave the classes and the lists; the syntax they carry
("." before a capital) is out of this step.

### 2. Diversity against chance

Coverage divides a head's fillers by its class's size, so a head with 50 occurrences can never
reach what a head with 50,000 does. The reference is what the head would show if it picked its
fillers from the class freely: with `n` the pattern's trunk and `p_y` the share of token `y` among
the occurrences of the class's tokens in the corpus,

    expected_fillers = Σ_y (1 − (1 − p_y) ** n)
    diversity = fillers / expected_fillers

Near 1, the head's fillers are as varied as the class allows for its traffic; far below 1, the
head is tied to few of them. On the recipe grammar this should separate the variants as coverage
did; on natural text it makes heads of different frequency comparable.

### 3. A reference corpus of the same domain

Diversity still cannot tell a fixed construction from an under-shown pattern: "due → to" and "cut →
potato" both read far below 1. What decides is whether the domain uses more fillers in that slot,
which a larger corpus of the same domain can show. With one tokenizer and one set of classes for
both (learned on the reference), each pattern of the target is read in both:

- low diversity in the target, high in the reference: **under-shown** — the domain uses more
  fillers here than this corpus shows (missing data);
- low in both: **fixed** — the domain itself restricts the slot (a construction);
- high in the target: sufficient.

"Low" and "high" are cut at `diversity` 0.5, a first value to be revisited.

**Predictions.**

1. Recipe grammar: the verbs' noun-slot diversity is at least 0.8 on the diverse variant and at
   most 0.2 with one or two fillers per verb.
2. Recipe grammar, restricted variant as target and diverse variant as reference: every verb's
   noun slot is under-shown.
3. Natural text (each source's first sample, 1/8 of it as target and all of it as reference):
   "due", "able", "pursuant", "such" and the contractions ("didn", "doesn", "don") are fixed
   wherever they are frequent enough to be read.
4. Natural text: fixed patterns outnumber under-shown ones among the target's low-diversity heads,
   since most restriction in fluent text is the language's own.

**Outcome (2026-09-29, `experiments/reference_diagnosis`).** All four held (the contractions of
prediction 3 are not read: "'t" is not word-like). Two limits found: the 0.5 cut called the
16-filler recipe verbs sufficient though the model did not generalize on them (it did from 0.82),
so the cut belongs near 0.8; and on natural text the under-shown verdicts are mostly function words
between 0.3 and 0.6 in both corpora, where the cut splits a continuum — they need a margin measured
against sampling noise before they mean missing data. Fixed constructions ("due", "able", "based",
"associated", "such") are identified cleanly, below about 0.15 in both corpora.

## A planted gap in natural text

Written before running (2026-09-29). The reference diagnosis finds the gaps built into the recipe
grammar, but on natural text its "under-shown" verdicts sit in a band where the 0.5 cut splits a
continuum, and nothing says which of them are real. This plants a known gap in real text and asks
whether the indicator finds it, and only it, once the verdict needs a margin over sampling noise.

**Design.** FineWeb's first 6.5 MB sample is the reference; its documents are cut into 8 disjoint
slices, each a target. Tokenizer, word-like tokens and classes are learned on the reference, as
before. Five heads are chosen on the reference automatically: word-like, not among the 100 most
frequent tokens, trunk between 200 and 2,000, diversity at least 0.8. In one slice, every sentence
in which a chosen head is followed by a word other than its two most frequent fillers is removed
— the recipe restriction, planted into real text; everything else in the slice is left as it is.

- **Noise:** each chosen head's diversity over the 8 unplanted slices gives its mean and standard
  deviation; the margin is the slices' mean minus three standard deviations.
- **Verdict with margin:** a pattern is under-shown when its target diversity is below both 0.5 and
  its margin, and its reference diversity is at least 0.5.

**Predictions.**

1. In the unplanted slices every chosen head reads sufficient, with a standard deviation of at most
   0.1.
2. In the planted slice every chosen head falls below its margin and reads under-shown.
3. In the planted slice, of the other patterns read (at least 20 occurrences), at most 5% fall below
   their own margin (the removal of sentences disturbs them little).

**Outcome of the first planting (2026-09-29).** Prediction 1 nearly held (every head sufficient in
every unplanted slice; standard deviations 0.060–0.081 but 0.104 for "But"); 3 held (0 of 320
other patterns below their margin, and 0 of 330 in the same slice unplanted). Prediction 2
failed: 3 of the 5 heads were found under-shown ("people", "she", "only"); "these" and "But" were
not. Removing the sentences also removed most of each head's occurrences (6 left for "these", 23
for "But"), and a head seen 6 times with 2 fillers is not evidence of a restriction — the
diversity measure is right not to call it one. The recipe restriction kept every verb's traffic.

**Second planting, written before running.** Instead of removing the sentence, the word after a
chosen head that is not one of its two kept fillers is **replaced** by the more frequent kept
filler (its trailing punctuation kept): the head keeps its traffic and loses its variety, as the
recipe verbs did. Predictions 1 and 3 as before; prediction 2: every chosen head falls below its
margin and reads under-shown.

**Outcome of the second planting (2026-09-29, `experiments/planted_gap`).** Prediction 2 held: all
five heads fell far below their margins (diversity 0.036–0.197, 93–209 occurrences) and read
under-shown. Prediction 3 held again (0 of 330 other patterns below their margin). The indicator
finds a restriction planted in real text, and only it, when the restriction keeps the head's
traffic; milder restrictions and other sources are untested.

## How strong a gap must be to be found: a sensitivity curve

Written before running (2026-09-29). The planted gap tied five heads to two fillers. A corpus
short of data is rarely that extreme; this measures how the detection falls as the planted
restriction weakens, and whether that holds beyond FineWeb.

**Design.** As in the second planting (replacement, margin of three standard deviations over 8
unplanted slices), on FineWeb, Wikipedia, arXiv and C4. Each chosen head is restricted in slice 0
to its `k` most frequent fillers in the reference, for `k` in {2, 4, 8, 16, 32}: a word after the
head outside those `k` is replaced by the most frequent one. One run per source learns the
tokenizer, the classes and the slices' noise once and plants every `k` in turn.

**Predictions.**

1. Every chosen head is found under-shown at `k` = 2 and 4, in every source.
2. None is found at `k` = 32: a head with 32 of its fillers is not restricted in a slice of about
   250 documents.
3. Detection falls monotonically with `k` in every source.
4. At every `k`, at most 1% of the other patterns read fall below their margin.

**Outcome (2026-09-29, `experiments/planted_gap`).** Heads found, of 20, at `k` = 2, 4, 8, 16, 32:
19, 20, 19, 17, 12. Prediction 1 failed by one head (C4's "who" at `k` = 2, left at 0.51 by
contractions the planting does not replace, in the small class of auxiliaries); 2 failed — the
indicator still finds most heads kept to 32 fillers; 3 held in three sources of four; 4 held, with
no false alarm at any `k`. The limit of detection is the 0.5 cut, not the noise margin.

## Closing the loop on natural text: does the model fail where the indicator flags?

Written before running (2026-09-29). The planted gap is found by the indicator; the question the
paper needs answered is whether a model trained on the gapped corpus then fails, on unseen text of
the domain, exactly at the flagged heads — and nowhere else.

**Design.** FineWeb's first 6.5 MB sample in its 8 slices. Slices 0–6 are the training corpus;
slice 7, never planted, is the evaluation text. The five heads of the planted-gap experiment are
chosen the same way on the whole sample, with the reference's `k` most frequent fillers. Three
training corpora: **control** (slices 0–6 as they are), **k = 2** and **k = 16** (the replacement
restriction planted in every one of slices 0–6). The small model of the recipe tests (2 layers,
128 wide), 3000 steps, learning rate 1e-3, BPE at FineWeb's 11,563 merges learned on each training
corpus, held-out fraction 0.05 for checkpoint selection; training seeds 0 and 1: six runs.

For every occurrence in slice 7 of a chosen head followed by a word, the model's log probability of
that word given up to ~100 tokens of preceding text ending at the head. Read separately for the
words outside the kept fillers (what the gap hid) and inside them. Five comparison heads, never
planted, chosen by the same rule from the next ranks, measure the same thing where nothing was
changed. The indicator reads each training corpus as before.

**Predictions.**

1. The indicator: every planted head reads diversity below 0.5 in the k = 2 and k = 16 corpora and
   at least 0.5 in the control one.
2. With k = 2, the mean log probability of a non-kept word after a planted head is lower than the
   control model's by at least 2 nats, on every head and seed.
3. With k = 16, lower by at least 0.5 nats on at least 4 of the 5 heads, on each seed.
4. At the comparison heads, the planted models stay within 0.3 nats of the control ones.
5. The held-out loss over all of slice 7 differs from the control's by less than 0.02 bits per
   byte: the global loss does not see the gap.

**Outcome (2026-09-29, `experiments/planted_model`).** Predictions 1–3 held: every planted head
is flagged in both gapped corpora, and the models trained on them give the true word after those
heads 2.17–4.60 nats less (`k2`) and 1.12–3.60 nats less (`k16`) than the control, on every head
and seed. Prediction 4 failed once (19 of 20 comparison readings within 0.3 nats; "now" −0.51);
prediction 5 failed (the whole-slice loss rises 0.028–0.035 bits per byte, not below 0.02), though
only a control reveals it and it does not say where.

## A natural gap: a topic-narrow corpus against a random one of the same size

Written before running (2026-09-29). Every gap so far was planted. A domain corpus built by hand
is usually narrow in topic, and narrowness restricts constructions without anyone choosing which:
a natural gap. A random subsample of the same text has no such gap — random subsamples are
exchangeable, which is why the margin raised no false alarm — so it is the right control.

**Design.** FineWeb's 5,000 cached documents. 400, drawn at random, are the evaluation text; the
other 4,600 are the pool and the reference. The pool is clustered by topic (TF-IDF over words,
English stop words removed, k-means with 12 clusters); the **narrow** corpus takes whole clusters,
largest first, until it holds 3.5 MB (revised before any reading: the two clusters with most
documents turned out to be registers, not topics — news, "new, said, year", and conversational
blogs, "just, like, don, really" — so they are left out and the topical clusters are taken by
size in bytes: health, sports, education); the **random** corpus is 3.5 MB of pool documents drawn at
random. The margin comes from 8 further random 3.5 MB draws of the pool. Tokenizer, word-like
tokens and classes are learned on the pool; a pattern of a corpus is flagged under-shown as
before (diversity below 0.5, below its margin, reference at least 0.5; at least 20 occurrences).

The small model is trained on the narrow and on the random corpus (seeds 0 and 1; 3000 steps,
BPE at 11,563 merges learned on each corpus). For each head read in the narrow corpus, and for
each of its occurrences in the evaluation text followed by a word (up to 50), the log probability
of that word: the head's **drop** is the narrow model's mean minus the random model's.

**Predictions.**

1. The indicator flags at least 5 patterns in the narrow corpus and at most 1 in the random one.
2. The mean drop over the flagged heads is lower than over the unflagged heads by at least 0.5
   nats, on each seed.
3. Across all heads read, the head's standardized diversity in the narrow corpus (its diversity
   minus the margin draws' mean, over their standard deviation; the lowest of its patterns) and
   its drop have a Spearman correlation of at least 0.3, on each seed.

**Outcome (2026-09-29, `experiments/natural_gap`).** Prediction 1 held (6 patterns flagged in the
narrow corpus, 0 in the random one). Predictions 2 and 3 failed: the flagged heads drop no more
than the others, and standardized diversity and drop are uncorrelated (Spearman 0.04, 0.05). The
heads with the largest drops ("sales", "desire", "human") have normal diversity in the narrow
corpus: it shows them with as many fillers as a random corpus, but its own topics' fillers. A
topic-narrow corpus keeps its slots open and changes what fills them; diversity measures the
first, and the model's loss comes from the second, which needs a comparison of *which* fillers a
slot shows against a reference of the target domain.

## Reference coverage: does a slot show the fillers the domain uses?

Written before running (2026-09-29), after the natural gap showed that a topic-narrow corpus keeps
its slots open but fills them with its own words. Diversity asks how many different fillers a slot
shows; this asks **which**: how much of the target domain's use of the slot the corpus covers.

**Definition.** For a pattern `(h, Y)` of a corpus C, read against a reference R of the target
domain with the same tokenizer and classes: `p_y` is the share of filler `y` among the fillers of
`(h, Y)` in R (counted in distinct contexts, as the trunk is), `n` is the pattern's trunk in C, and
`S` the fillers C shows in it.

    covered_mass  = Σ_{y in S} p_y
    expected_mass = Σ_y p_y (1 − (1 − p_y) ** n)
    coverage_ratio = covered_mass / expected_mass

`expected_mass` is what `n` draws from the domain's own use would cover. A ratio near 1: the corpus
shows the slot's fillers as the domain would at this traffic. Far below 1: it shows other fillers
(a topic mismatch) or too few (a restriction). A fixed construction ("due → to") is fixed in R too,
so its one filler covers R's mass and the ratio stays near 1: this reads a gap only where the
domain uses more than the corpus shows. It is read per pattern and standardized against the random
draws of the natural-gap experiment, as diversity was.

**Predictions.**

1. Recipe grammar, restricted variant (1 filler per verb) against the diverse one: the verbs'
   noun-slot ratio is at most 0.3; the diverse variant against another seed of itself, at least 0.9.
2. On the natural gap already run (health, sports, education against random), the heads' lowest
   standardized ratio and their drop have a Spearman correlation of at least 0.3 on each seed, and
   the heads flagged by it (ratio below 0.5 and below its margin) drop more than the others by at
   least 0.5 nats. This is the data the idea came from, so it is a check, not a test.
3. **The test**, on a narrow corpus not yet looked at: the conversational-blog cluster (the
   register cluster left out before; "just, like, don, really"), 3.5 MB, against the same random
   corpus and evaluation text, with the small model on seeds 0 and 1. The same two criteria as in
   prediction 2 hold on each seed.

**Outcome (2026-09-30, `experiments/natural_gap`).** Prediction 1 held (restricted 0.067, diverse
1.008). Predictions 2 and 3 failed: coverage flags no slot in either narrow corpus, and correlates
0.09–0.12 with the drop. The loss is not at slots: each narrow model is better than the random
one on documents of its own topics (−0.06, −0.02 bits per byte) and worse on all others (+0.11,
+0.15). A topic-narrow corpus fails a model document-wide, by topic — semantics, outside what a
slot-level syntactic measure reads. The indicator's claim is bounded to syntactic sufficiency,
where the planted experiments show it predicts the model's failure.

## Choosing a training subset: the full corpus as reference

Proposed 2026-09-30; design and predictions written before running. The planted experiments
validate the indicator as a diagnosis. This one validates it as a tool for **choosing data**:
a corpus trains a model well, a cut of it is wanted, to lower the training cost, and the
question is which cut keeps the full corpus's generalization and where the others lose it.

**Data.** FineWeb's 5,000 cached documents, split as in `experiments/natural_gap`. The 400
evaluation documents are never trained on. The other 4,600 (14.4 MB) are the **full corpus F**,
which is also the reference of every diagnosis. The subsets are cut by bytes:

- `r50`, `r25`, `r10`: random 50, 25 and 10% of F (seeded).
- `wiki25`: the 25% of F with the lowest per-word perplexity under a word-bigram model trained
  on the cached Wikipedia documents. This is CCNet's quality filter (Wenzek et al. 2020, used for
  LLaMA), with a bigram standing in for its 5-gram.
- `self25`: the 25% of F with the lowest perplexity under the same bigram model trained on F
  itself. It keeps F's most formulaic documents, the filter most likely to keep slots narrow.

**Diagnosis.** The tool exactly as shipped: `diagnose(subset, reference=F, merge_budget=11563)`,
with its defaults (heads of 1 and 2 tokens, min trunk 20, 8 draws, low 0.5, 3 sigmas).

**Models.** The small model of `experiments/planted_model` (2 layers, 128 wide, 3000 steps,
learning rate 1e-3, BPE at 11,563 merges learned on its own corpus), seeds 0 and 1, trained on
F, `r50`, `r25`, `r10`, `wiki25` and `self25`: 12 runs. 3000 steps of 32 × 128 tokens are about
12M tokens, 3.3 passes over F and 33 over `r10`; held-out early stopping picks the best step.

**Readings.**

- Evaluation bits per byte.
- For every one-token head read in any subset, up to 50 evaluation occurrences of the head
  followed by a word. On each, the model's log probability of that word.
- The **drop** of head h under subset S: the mean over h's occurrences of log p_S − log p_F,
  in nats.
- A head is **flagged** in S when one of its one-token patterns is `under-shown`.
- Its **z** in S is the minimum, over its patterns, of (diversity − draw mean) / draw sd.

**Predictions.**

1. **Random cuts raise no false alarms.** In each of `r50`, `r25` and `r10`, `under-shown` rows
   are at most 1% of the rows read. They are draws from the very distribution that sets the
   margin.
2. **A filter flags what it narrows.** Each filtered subset has at least 10 `under-shown` rows,
   and at least three times as many as `r25`.
3. **Size and filter, globally.** On both seeds the evaluation bits per byte order
   F < `r50` < `r25` < `r10`, and each filtered subset reads above `r25`. The first is a sanity
   check; the second says these filters cost generalization on general text.
4. **The flagged heads are where a filtered model loses (the main test).** This applies to each
   filtered subset with at least 5 flagged heads that have at least 20 evaluation occurrences.
   Let `gap(S)` be the mean drop at the flagged heads minus the mean drop at the unflagged ones,
   for the model of subset S. Then `gap(filtered) − gap(r25) ≤ −0.3` nats, on both seeds. The
   random model of the same size is the control: it separates "the filter hurt these heads" from
   "these heads are hard whenever data is cut", and comparing heads within one model removes a
   document-wide loss (the topic effect of `experiments/natural_gap`).
5. **Sampling noise matters too (exploratory).** In `r10`, over heads with at least 20
   occurrences, Spearman(z, drop) ≥ 0.15 on both seeds: where a random cut happened to show a
   head poorer, its model loses more.

**How to read the outcomes.**

- If 4 holds, a subset's diagnosis against its full corpus says, before training, where a cut
  will cost the model.
- If 2 fails, these filters do not narrow slots: they cut topics or registers, which the
  indicator does not claim to read, and 4 cannot be read.
- If 3 holds and 4 fails, the filter's cost is not at slots.

**Known limits, before running.**

- One source, one model size.
- The tool reads the data, not the model: the diversity a model needs depends on its size (the
  recipe threshold moved with the model).
- A topic cut can pass the diagnosis and still fail the model by topic.

**Outcome (2026-10-01, `experiments/subset_selection`).**

- Predictions 1 and 3 held: random cuts flag 0.15–0.21% of rows, and the losses order
  F < `r50` < `r25` < `r10`, with both filtered cuts above `r25`.
- Prediction 2 held for `self25` (12 rows, exactly the bar) and failed for `wiki25` (9).
- Prediction 4 failed for `self25`: its 7 flagged heads lose less, not more, than its others,
  relative to `r25` (+0.10 / +0.17 nats). It was unreadable for `wiki25` (2 flagged heads).
- Prediction 5 failed: Spearman 0.10 / 0.12, the predicted sign, but no more than log trunk
  alone.

The natural flags are mild (diversity 0.26–0.50; the planted gaps read 0.02–0.16), and at this
model and evaluation size a mild restriction leaves no loss above the per-head noise. A flag
that is significant is not yet large: choosing data needs an effect size on top of the margin,
and the predictive claim stands for strong restrictions only.

## Dose and response: how far must diversity fall before a model pays?

Written before running (2026-10-01). On planted gaps the model lost 1–4.6 nats at heads with
diversity 0.02–0.16 (`experiments/planted_model`). At the mild flags of real cuts (0.26–0.50) it
lost nothing measurable (`experiments/subset_selection`). This experiment measures the curve
between those two points: the model's loss at a head as a function of the diversity the corpus
shows it. The diversity where the loss becomes material is the effect size a flag needs before
it says a model will pay.

**Design.** The corpus and models of `experiments/planted_model`: FineWeb's 6.5 MB sample in 8
slices, where slices 0–6 train and slice 7 evaluates; the small model with seeds 0 and 1. Its
`control` runs are reused (same corpus, settings and seeds), and are rescored on this
experiment's heads.

- **Heads.** 21 heads are chosen by `planted_gap.choose_heads`, the rule used for every planted
  head, plus the next 5 as comparison heads that are never planted (revised below: 14 and 3).
- **Doses.** Each planted head gets a target diversity in {0.1, 0.2, …, 0.7}, three heads per
  level.
  - Its restriction is the replacement planting with two kept fillers. Each occurrence outside
    the kept fillers is replaced with probability q_h, decided by a seeded draw per occurrence.
  - q_h is found by bisection, before any training, so that the head's diversity in the planted
    training corpus (sample tokenizer and classes, as in `planted_model.prepare`) is within 0.03
    of its target. Calibrating on the indicator is part of the construction; it reads nothing
    from a model.
- **Two assignments.** Assignment `A` gives the heads, ranked by trunk, the levels in turn;
  assignment `B` shifts every head by three levels. Each head is therefore read at two doses,
  and a level's effect is not one head's identity.
- **Runs.** `A` and `B` × seeds 0 and 1: 4 new runs, plus the 2 reused controls.

**Reading.** For each head and planted model, the **drop** is the mean, over all its slice-7
occurrences followed by a word (up to 200), of log p_planted − log p_control, in nats. The kept
and the replaced words are not separated: the cost of a corpus is what the model loses on the
domain's real text.

**Predictions.**

1. **Monotone response.** Spearman(planted diversity, drop) over the 14 planted heads is at least
   0.6 in each of the four readings (2 assignments × 2 seeds).
2. **The strong end costs.** The two heads at target 0.1 lose at least 1.0 nat on average, in
   each reading.
3. **The mild end is free.** The six heads at targets 0.5–0.7 lose at most 0.3 nats on average,
   in each reading. This is the regime of the subset cuts' flags.
4. **The threshold lies between.** The per-level mean drops are averaged over the four readings
   and joined by linear interpolation. The first diversity, from the top, at which the mean drop
   reaches −0.3 nats lies in [0.2, 0.5].
5. **Nothing else moves.** Every comparison head's drop is within ±0.3 nats, with at most one
   exception per reading.
6. **Doses, not heads.** At each level, the mean drop under `A` and under `B` differ by at most
   0.3 nats, on each seed.

**Revision before any run (2026-10-01).** The 6.5 MB sample has only 17 heads that pass
`choose_heads`' rule, not 26. The design keeps the rule and the seven levels, with **14 planted
heads (two per level) and 3 comparison heads**; predictions 1–3 and 5 are read on those counts,
the thresholds unchanged. Two heads per level make a level's mean noisier than three would.

**Use.** If 1–4 hold, the interpolated threshold becomes the tool's effect-size cut, for this
model family and size. An `under-shown` row would then also report whether its diversity is
below the diversity where a model of this size pays. That cut must be re-measured for another
model size: the recipe threshold moved with the model.

**Outcome (2026-10-01, `experiments/dose_response`).**

- Predictions 1, 2 and 5 held: the response is monotone (Spearman 0.76–0.92), heads at 0.1 lose
  2.8–3.3 nats, and no comparison head moves.
- Predictions 3, 4 and 6 failed. The mild end is not free: heads at 0.5–0.7 lose 0.38–0.57
  nats. The mean drop crosses −0.3 nats at 0.64, not in [0.2, 0.5]. With two heads per level,
  the assignments differ by up to 0.75 nats.

Against `experiments/subset_selection`, where natural flags at 0.26–0.50 cost nothing, this says
diversity is not the effect size. A planting moves probability mass onto one filler; a natural
cut drops rare fillers. The model's loss follows the first, and diversity counts the second.
The candidate effect size is a divergence between the slot's fillers in the corpus and in the
reference.

## A divergence as the effect size

Written before computing (2026-10-03). Both experiments above are re-read without new training:
the drops are already measured, and only the corpus reading changes.

**Why KL from the reference to the corpus.** A model that learns the corpus's next-word
distribution after a head, S, and is scored on text drawn from the reference distribution, R,
loses on average E_R[log S − log R] = −KL(R ‖ S) nats per occurrence against a model that
learned R. That is exactly how the drop is read. The divergence therefore has the drop's units
and the predicted relation has a slope: −1 for a model that fits the head's conditional and
nothing else.

- Jensen–Shannon would be bounded and symmetric, but it has no such link to the loss.
- The direction matters. KL(R ‖ S) is large when the corpus misses mass the reference uses,
  which is the failure the indicator is after; KL(S ‖ R) is large when the corpus adds mass.

**Definition.**

- The **head distribution** of a one-token head h is over its fillers: the classified word-like
  tokens of classes with two or more members that follow h. These are all of h's patterns
  together, because the drop is read per head.
- Counts are in distinct contexts, as the tool counts them, so that copies add nothing.
- R_h is the reference's maximum-likelihood distribution.
- S_h is the corpus's distribution with Witten–Bell backoff to the corpus's own unigram
  distribution over the same fillers: λ = n / (n + T), where n is h's count and T its distinct
  fillers. This smoothing has no free parameter. A model also gives unseen fillers mass, and an
  unsmoothed KL would be infinite at the first missing filler.
- **D(h) = KL(R_h ‖ S_h)**, in nats.
- Details fixed before computing. The backoff unigram counts every occurrence of each filler in
  the corpus, with one added count per filler of the reference. Without that added count, a
  filler absent from a subset would make D infinite. The fit of prediction 2 reports R² against
  the mean, not against zero. In prediction 5, trunk is the head's count n in the corpus.

**Readings.**

- **Dose and response.** R is the unplanted training corpus (slices 0–6) and S is each
  assignment's planted corpus, with the sample tokenizer and classes of `prepare`. The plantings
  are re-made from `prepare.json` (seeded, deterministic). There are 14 planted heads × 2
  assignments, and the drops are those of the four readings.
- **Subset selection.** R is F and S is each subset, with the tokenizer and classes of that
  subset's diagnosis. The drops are those of the subset models against F, over heads with at
  least 20 evaluation occurrences.

**Predictions.**

1. **D orders the doses at least as well as diversity.** For each seed, pooling `A` and `B`
   gives 28 head readings. On both seeds, |Spearman(D, drop)| is at least 0.05 above
   |Spearman(diversity, drop)|. Calibration already shows why this is expected: at diversity 0.7,
   `these` needed q = 0.11 and `world` q = 0.29.
2. **D has the drop's scale.** A least-squares line through the origin of drop on D over the 56
   planted readings has a slope in [−2, −0.5] and R² ≥ 0.6. The interval is wide on purpose.
   The theory gives −1, but the model reads more context than the head, which pulls the slope
   toward 0. Mild heads lost more than a back-of-envelope D suggests, which pulls it below −1.
3. **The A/B disagreement shrinks.** Prediction 6 of dose and response failed because, at a
   level, the mean drop under `A` and `B` differed by up to 0.75 nats. Replace levels of diversity
   with the fit of prediction 2: the residuals have a standard deviation at most two thirds of
   that of the residuals from a linear fit on diversity.
4. **Natural cuts move little mass.** Let κ be the D at which the fit of prediction 2 reaches
   −0.3 nats. On each seed, at most one of `self25`'s 7 flagged heads has D > κ. This is the
   consistency check with subset selection, where those heads cost nothing.
5. **D ranks natural heads (exploratory).** In `r10` and in `self25`, on both seeds,
   Spearman(D, drop) ≤ −0.2 and is at least 0.05 above |Spearman(log trunk, drop)| in absolute
   value. Without the second condition, D could just be reading trunk: a smoothed estimate from
   fewer occurrences sits further from R.

**How to read the outcomes.**

- If 1–3 hold, D is the effect size on planted gaps, and κ replaces the failed diversity
  threshold as a candidate cut.
- If 4 fails, D predicts costs that the subset models did not pay, and κ is not usable as a cut.
- If 5 fails while 1–4 hold, natural cuts move too little mass for this model and evaluation
  size to resolve, as with diversity. The claim then stays at the scale of plantings.

**Known limits, before computing.** This is a retrospective test: the drops are known, and the
choice of KL was made after seeing the dose-response result. Only the definition and the
thresholds above are fixed in advance. A confirmatory test needs new models, on two plantings
that agree in diversity and disagree in D:

- the replacement planting used so far, which moves mass onto one filler;
- a removal planting that deletes the slot's rare fillers, which lowers diversity and moves
  little mass.

The prediction is that the model pays for the first and not the second.

**Outcome (2026-10-03, `experiments/divergence`).**

- Prediction 2 held: drop ≈ −0.675 · D, R² 0.77 over the 56 planted readings. D has the
  drop's units and roughly its scale.
- Prediction 5 held: in `r10` and `self25`, Spearman(D, drop) is −0.39 to −0.45 over 417 heads
  on both seeds, while log trunk reads +0.13 to +0.17. Diversity's z read only 0.10–0.12 there.
  On natural cuts, D ranks where a model loses, and flags did not.
- Prediction 1 failed: on the planted heads D orders the drops no better than diversity (−0.90
  against 0.90 on seed 0, −0.84 against 0.87 on seed 1). The A/B disagreement is not explained
  by D (prediction 3 failed: residual sd 0.46 against 0.52, ratio 0.88).
- Prediction 4 failed: κ = 0.44, and 5 of `self25`'s 7 flagged heads are above it.

**Why prediction 4 failed: D has a sample-size floor.** The smoothing that keeps D finite also
biases it upward when the corpus has few occurrences. The unplanted comparison heads of dose and
response read 0.21–0.27, not 0, at a few hundred occurrences. In `r10`, the median D is 2.5 for
heads seen fewer than 20 times and 0.39 for heads seen over 1000 times. κ was set on heads with
hundreds of occurrences, so it is not a cut for heads of another size. The flagged heads of
`self25` read about what the same heads read in the random cut of the same size, `r25`: `some`
0.89 against 0.90, `hours` 0.83 against 0.85, `free` 1.64 against 1.40. Their D is the floor of
a 25% cut, not a filter's effect, which agrees with subset selection, where they cost nothing.

**Reading.**

- D is a better per-head predictor of loss than diversity on natural cuts. On plantings, it is
  as good as diversity, not better.
- Part of D's success on natural cuts is that it reads how much of the reference's distribution
  the corpus has sampled. That depends on trunk relative to the reference's spread, which log
  trunk alone does not capture. This is the sufficiency claim itself, but it also means D mixes
  sampling and restriction.
- As with diversity, D needs a size-matched baseline before it can be a cut: the excess D over
  random draws of the corpus's size from the reference. The tool already draws those for
  diversity's margin. The next step is to read that excess against the drops, written before
  computing, and then the confirmatory planting above.

## D against random cuts of the same size, and one score per cut

Written before computing (2026-10-03). The use the tool is after: take F as the reference, cut
it several ways at one size, and keep the cut that best preserves F's patterns. That needs two
things D lacks.

- **A size-matched baseline.** D falls with the head's count, so a smaller cut always reads
  worse. Only cuts of one size are compared, and each against random cuts of that size.
- **One score per cut.**

No training: the subsets, drops and bits per byte are those of `experiments/subset_selection`.

**Definitions.**

- **Draws.** For each size (50, 25 and 10% of F), 8 random cuts of F, seeded 0–7. Documents are
  taken in shuffled order until the cut reaches the bytes of `r50`, `r25` and `r10`
  respectively. The 25% draws serve `r25`, `wiki25` and `self25`, which match in bytes within
  0.1%.
- **Excess.** E_S(h) = D_S(h) − the mean of D(h) over the draws of S's size. D, tokenizer and
  classes are those of `experiments/divergence`.
- **Score.** score(S) = Σ_h w_h E_S(h), where w_h is h's share of F's filler occurrences, over
  all heads read in F. Lower is better.
  - The weights make the score an expected excess per slot occurrence, in D's units. By the
    fit of the previous section, it times −0.68 is an expected loss in nats.
  - For a draw, its own score uses the mean of the other 7 draws. The sd of the 8 draw scores
    gives z(S) = score(S) / sd.

**Predictions.**

1. **Random cuts read as random.** |z| ≤ 2 for `r50`, `r25` and `r10`. They are draws from the
   same process as the baseline.
2. **Filters read above random.** z ≥ 3 for `wiki25` and for `self25`.
3. **The score orders the 25% cuts as the models do.** score(`r25`) < score(`self25`) <
   score(`wiki25`). This matches the bits per byte, 2.13 < 2.15 < 2.18 on both seeds.
   It is retrospective, since the bits per byte are known, and three cuts are too few to
   establish an order: it is a necessary check, not evidence.
4. **The excess predicts where a filter costs (the main test).** For `self25` and for `wiki25`,
   on both seeds, Spearman(E_S(h), drop_S(h) − drop_r25(h)) ≤ −0.2. It is read over the heads
   with at least 20 evaluation occurrences. Subtracting `r25`'s drop removes what a cut of this
   size costs anyway, so only the filter's own cost is left.
5. **Sampling luck (exploratory, no threshold).** In `r10`, Spearman(E, drop) against
   Spearman(D, drop) = −0.44 / −0.45. A drop to near zero means D's success on random cuts was
   sample size alone. A drop that holds means the particular draw's luck at a head costs too.

**How to read the outcomes.**

- If 1, 2 and 4 hold, the score is the tool's way to choose among cuts of one size. The test
  with ~10 cuts and trained models is the next step.
- If 4 fails while 2 holds, the filters change F's slots, but the model does not pay for those
  changes at this size.

**Outcome (2026-10-03, `experiments/divergence`, `baseline`).**

| Cut | Score | z |
|---|---|---|
| `r50` | −0.002 | −0.38 |
| `r25` | −0.003 | −0.42 |
| `r10` | +0.009 | +0.59 |
| `wiki25` | +0.025 | +3.87 |
| `self25` | +0.097 | +15.0 |

- Predictions 1 and 2 held. Random cuts read as random, and both filters read above random.
  `self25`, the filter that keeps formulaic text, reads far above.
- Prediction 4 held, narrowly: Spearman(E, drop − drop_r25) is −0.21 / −0.22 for `self25` and
  −0.28 / −0.21 for `wiki25`, over 417 heads. Within a filtered cut, the heads whose fillers
  moved beyond sampling are those where the model lost more than a random cut of the same size
  costs. This is the first per-head prediction of a natural filter's cost. The effect is modest,
  and three of the four readings are within 0.03 of the bar.
- Prediction 3 failed. The score ranks `self25` worst, but its model reads better than
  `wiki25`'s: 2.15 against 2.18 bits per byte. The score reads slots, and `wiki25` keeps fewer,
  longer documents (856 against 1,235). Its extra cost is most likely topical, which the score
  does not claim to read. In nats, `self25`'s score is about 0.07 per slot occurrence (× 0.68):
  real, but small next to a document-wide loss.
- Prediction 5: in `r10`, E reads 0.02 / −0.04 against D's −0.44 / −0.45. D's success on random
  cuts was sample size alone. A random draw's luck at a head does not show in its model's loss.

**Reading.** The tool can now say, for cuts of one size against F, how far each cut moved F's
slots beyond sampling (the score and its z) and at which heads (E). Both readings are validated
here: the heads it points to are where the model pays. It does not yet say which cut trains the
best model overall, because a filter can also cost by topic. That needs either a topic reading
next to the slot one, or the test with ~10 cuts and trained models, which would show how much
of the bits per byte the score explains.

## Choosing among cuts of one size: does the score order the models?

Written before running (2026-10-03). The tool's use is to cut F several ways at one size and
keep the cut with the lowest score. This tests whether the score orders the models that those
cuts train. With three cuts it did not (`wiki25`). Ten cuts give the order a chance to show,
and a covariate for the topic effect.

**Cuts.** All at 25% of F's bytes, documents taken in order until the cut reaches the bytes.
The three of `subset_selection` (`r25`, `wiki25`, `self25`) are kept with their models. Seven
are new:

- `r25b`, `r25c`: random, seeds 203 and 204.
- `selfhigh25`, `wikihigh25`: the highest perplexity under F's own bigram and under the
  Wikipedia bigram. These are the opposite ends of the two filters, among the documents of at
  least 50 words, as before.
- `short25`, `long25`: the shortest and the longest documents first. These vary the document
  count, which is the suspected topic effect, at one size.
- `mix25`: half the bytes from `self25`'s order, then half at random from the rest. A dose
  between `r25` and `self25`.

**Models.** The small model of `subset_selection`, seeds 0 and 1: 14 new runs, two at a time.
The evaluation occurrences and heads are exactly those of `subset_selection`, so every drop is
against the same `full` models.

**Score.** As in the previous section: E against the 8 random draws at `r25`'s bytes, weighted
by F's filler shares, and z. Nothing is re-tuned.

**Predictions.**

1. **The score orders the models.** Over the 10 cuts, Spearman(score, bits per byte) ≥ 0.5 on
   both seeds.
2. **A dose orders too.** score(`r25`) < score(`mix25`) < score(`self25`), and the bits per
   byte follow the same order on both seeds.
3. **Random cuts agree.** `r25b` and `r25c` read |z| ≤ 2, and the three random cuts' bits per
   byte span at most 0.02 on each seed.
4. **The per-head reading replicates.** On every new non-random cut with z ≥ 3, on both seeds,
   Spearman(E(h), drop(h) − random drop(h)) ≤ −0.2. Here the random drop is the mean over the
   three random cuts.
5. **Topic (exploratory, no threshold).** Spearman(log documents, bits per byte), and the score's
   Spearman with the bits per byte left after a linear fit on log documents.

**How to read the outcomes.**

- If 1 and 2 hold, the score chooses among cuts. The paper can state that the cut with the
  lowest score trains the better model, at this size and model.
- If 1 fails and 5 shows that document count carries the order, the score reads slots
  correctly, but whole-model quality at one size is dominated by how many documents (topics)
  a cut keeps. The tool would then need a topic reading next to the slot one.

**Outcome (2026-10-04, `experiments/cut_choice`).**

| Cut | Score | z | Documents | Bits per byte (seed 0 / 1) |
|---|---|---|---|---|
| `selfhigh25` | −0.018 | −2.80 | 1,428 | 2.131 / 2.134 |
| `r25` | −0.003 | −0.42 | 1,137 | 2.128 / 2.130 |
| `r25c` | +0.003 | +0.46 | 1,196 | 2.119 / 2.116 |
| `r25b` | +0.003 | +0.48 | 1,154 | 2.129 / 2.126 |
| `wiki25` | +0.025 | +3.87 | 856 | 2.181 / 2.181 |
| `short25` | +0.027 | +4.21 | 3,106 | 2.104 / 2.102 |
| `mix25` | +0.051 | +7.97 | 1,179 | 2.136 / 2.133 |
| `long25` | +0.075 | +11.62 | 108 | 2.208 / 2.208 |
| `self25` | +0.097 | +15.01 | 1,235 | 2.146 / 2.145 |
| `wikihigh25` | +0.133 | +20.62 | 1,788 | 2.138 / 2.138 |

- Prediction 1 failed, narrowly: Spearman(score, bits per byte) is 0.55 on seed 0 and 0.45 on
  seed 1, against a bar of 0.5 on both. Over ten cuts neither value is significant on its own.
- Prediction 2 held: `r25` < `mix25` < `self25` in score and in bits per byte, on both seeds.
  When one filter is dosed, the score and the model move together.
- Prediction 3 held: the random cuts read |z| < 0.5, and their bits per byte span 0.010 / 0.014.
- Prediction 4 held on every new filtered cut, on both seeds: Spearman(E, drop − random drop)
  is −0.33 / −0.33 for `wikihigh25`, −0.38 / −0.34 for `long25`, −0.27 / −0.26 for `mix25` and
  −0.25 / −0.23 for `short25`. The per-head reading now replicates on six filters.
- Prediction 5: log documents alone read −0.47 / −0.44 against bits per byte. The score's
  Spearman with what documents leave is 0.50 / 0.49. The two readings are about equally strong
  and largely separate.

**Reading.** The order breaks where document count dominates. `short25` keeps 3,106 short
documents, reads above random in score and trains the best model. `long25` keeps 108
documents and trains the worst. `wikihigh25` reads the worst in score but has many documents,
and its model is mid-table. A cut's whole-model quality at one size has two parts. One is
how far it moves F's slots, which the score reads and whose effect at each head prediction 4
confirms. The other is how many documents, and so topics, it keeps, which the score does not
read. For the paper:

- The score, and its per-head excess, say where and how much a cut changes F's syntactic
  patterns beyond sampling, and the model pays there (six filters, both seeds).
- Along one filter's dose, the score orders the models.
- Across filters of different kinds, the score alone does not choose the best model. It needs
  the document count, a proxy for topical coverage, next to it.

## Two dose ladders: only the slots change

Written before running (2026-10-04). Across kinds of filter, document count confounded the
score. This holds everything else as fixed as a natural cut allows. A filter that moves F's
slots is mixed with random documents at doses of 0, 25, 50, 75 and 100%, and the question is
whether the models order as the doses and the scores do.

**Ladders.** A dose f takes f of the target bytes from the filter's order, then fills the rest
in the random order of `mix25` (seed 205) among the remaining documents. Both ladders are at
25% of F's bytes.

- **`self`:** `r25` (0), `self_m25`, `mix25` (50), `self_m75`, `self25` (100). Documents stay at
  1,137–1,235, so the dose changes little but the slots.
- **`wikihigh`:** `r25` (0), `wikihigh_m25`, `wikihigh_m50`, `wikihigh_m75`, `wikihigh25` (100).
  Documents rise with the dose to 1,788, which helps the model. A loss that still rises with the
  dose would be the stronger evidence.

The endpoints and `mix25` keep their models: 5 new cuts, seeds 0 and 1, 10 runs. Score, z and
drops are read exactly as in the previous section.

**Noise, before running.** The same cut on two seeds differs by at most 0.004 bits per byte.
The three random cuts span 0.010–0.014, a spread that comes from which documents they hold.
The full span of the `self` ladder is 0.018, so neighboring doses may not separate on one seed.

**Predictions.**

1. **The score follows the dose.** In each ladder, the score rises strictly from dose 0 to 100.
2. **The models follow the score.** In each ladder, Spearman(score, bits per byte) ≥ 0.9 on the
   mean of the two seeds, and ≥ 0.7 on each seed.
3. **Not a prediction: the endpoints are known.** Dose 100 reads +0.018 / +0.015 bits per byte
   above dose 0 for `self` and +0.010 / +0.008 for `wikihigh`. The new runs only fill in
   between, and the `wikihigh` ladder's span is near the random cuts' spread.
4. **The per-head reading replicates.** Every new cut with z ≥ 3, on both seeds:
   Spearman(E, drop − random drop) ≤ −0.2, with the random drop of the previous section.

**How to read the outcomes.**

- If 1 and 2 hold, the paper can state that, with other properties of a cut held, the further
  the score says its slots moved from the reference, the worse its model generalizes, in the
  score's order.
- If 2 holds for `self` and fails for `wikihigh`, more documents can outweigh a slot loss of
  that size, and the score must be read next to document count.

**Outcome (2026-10-04, `experiments/cut_choice`, `summarize-ladders`).**

| Dose | `self` cut | Score | Documents | Bits per byte | `wikihigh` cut | Score | Documents | Bits per byte |
|---|---|---|---|---|---|---|---|---|
| 0 | `r25` | −0.003 | 1,137 | 2.128 / 2.130 | `r25` | −0.003 | 1,137 | 2.128 / 2.130 |
| 25 | `self_m25` | +0.020 | 1,128 | 2.124 / 2.125 | `wikihigh_m25` | +0.029 | 1,461 | 2.125 / 2.124 |
| 50 | `mix25` | +0.051 | 1,179 | 2.136 / 2.133 | `wikihigh_m50` | +0.053 | 1,602 | 2.124 / 2.121 |
| 75 | `self_m75` | +0.069 | 1,254 | 2.139 / 2.135 | `wikihigh_m75` | +0.093 | 1,700 | 2.130 / 2.132 |
| 100 | `self25` | +0.097 | 1,235 | 2.146 / 2.145 | `wikihigh25` | +0.133 | 1,788 | 2.138 / 2.138 |

- Prediction 1 held: in both ladders the score rises strictly with the dose.
- Prediction 2 held for `self` and failed for `wikihigh`, the outcome foreseen above.
  - **`self`:** Spearman(score, bits per byte) is 0.90 on the mean and on each seed. The only
    inversion is at dose 25, which reads 0.004 below `r25`, inside the random cuts' spread.
    From there on the loss rises at every step, with the score.
  - **`wikihigh`:** 0.60. The loss first falls (doses 25 and 50 gain up to 0.009 bits per byte
    with 300–460 more documents), then rises (doses 75 and 100). At low doses the added
    documents outweigh the slot loss. From a score of about 0.05 the slot loss dominates.
- Prediction 4 failed. The per-head reading holds at the high doses (`self_m75` −0.24 / −0.20,
  `wikihigh_m75` −0.32 / −0.27, `wikihigh_m50` −0.15 / −0.27). It is near zero at dose 25
  (−0.02 to −0.14). With a score of 0.02–0.03 there is little excess at any head for the model
  to pay for, and the per-head reading is lost in the noise of single heads.

**Reading.** When only the slots change (the `self` ladder, with documents fixed), the score
orders the models: the further a cut's slots move from F's, the worse it generalizes. The
effect is small at this size, 0.017 bits per byte over the whole ladder, but it is monotone
from dose 25 and reproduces on both seeds. When a cut also gains documents (`wikihigh`), the
gain wins at low doses and the slot loss wins at high ones. The score and the document count
are two readings of one cut, as `cut_choice` found. Per head, the reading needs an excess of
about 0.05 in score before single heads separate from noise.

## A syntactic evaluation on natural text

Written before scoring any model (2026-10-04). On natural text every model has so far been
read by the probability of the true next word, the drop. That reading mixes syntax (a noun goes
here) with semantics and topic (this noun is what the text says). The recipe grammar had a
purely syntactic reading: the generalization ratio, with probability given to class-valid
fillers never seen in the slot. Coverage predicted it and the held-out loss ranked it in
reverse. This section brings that reading to FineWeb, so that the claim the paper makes, about
syntactic patterns, is tested by a syntactic measure.

**The question asked of a model.** After a real evaluation prefix that ends in head h, does the
model prefer a word of h's slot class that it never saw after h over a word of another class?
Truth does not enter: after "people", "swim" counts as correct whatever the text goes on to
say.

**Classes come from outside the indicator.** spaCy's `en_core_web_sm` tags the reference
corpus. The indicator's own induced classes are never used here, so the evaluation is not
measured with the same ruler it judges.

- **Word classes.** A lowercase alphabetic word type gets its majority universal tag, if that
  tag covers at least 90% of its tagged occurrences in the reference. The word must occur at
  least 20 times, and the tag must be one of the open classes NOUN, VERB, ADJ and ADV.
- **A head's slot class C(h).** The most frequent open-class tag among the words that follow h
  in the reference, in context. The head is kept if that class covers at least 30% of its
  followers.

**Items.** For each kept head:

- valid words V(h): words of class C(h);
- contrast words I(h): words of the other open classes;
- both sets only take words that never follow h in the reference. They are then never seen after
  h in any training corpus of the family, since every one of them is part of the reference or
  a planting of it. The words must also be a single token in every model's tokenizer of the
  family, so one forward pass per context scores them all.
- **Pairs.** Up to 20 words of V, drawn with seed 0. Each is paired with the I word nearest to
  it in log reference frequency, without replacement, so that frequency does not decide the
  preference.
- **Contexts.** Up to 20 evaluation occurrences of h, the prefix ending at h, built as
  `planted_model.occurrence_scores` builds it.
- **Reading.** G(model, h) is the share of (context, pair) comparisons in which the model gives
  the V word a higher probability than its I word. Chance is 0.5. A model's syntactic score is
  the mean G over the kept heads.

**Two families of models already trained.**

- **Plantings** (`dose_response`): the reference is the unplanted training slices, the
  evaluation slice 7. There are 17 heads, the controls and `A`/`B` on two seeds. Here the gaps
  are known, so this family validates the evaluation itself.
- **Cuts** (`subset_selection`, `cut_choice`): the reference is F, the evaluation its 400
  documents, and the heads those of `subset_selection` with at least 20 evaluation
  occurrences. There are 18 corpora on two seeds: 36 models.

**Predictions.**

1. **The evaluation reads syntax in an unrestricted model.** The controls' mean G over the kept
   heads is at least 0.70, on each seed.
2. **A planted gap removes it.** At the planted heads with target diversity 0.1–0.2 (8 readings
   per seed), the mean of G_planted − G_control is ≤ −0.10. At the comparison heads its mean
   absolute value is ≤ 0.05, on each seed.
3. **Dose.** Over the 28 planted readings per seed, Spearman(D, G_planted − G_control) ≤ −0.5.
4. **The ladder with documents held orders syntax.** Over the `self` ladder's five cuts,
   Spearman(score, mean G over both seeds) ≤ −0.9.
5. **Among the ten cuts, syntax follows the score and not the document count.** On each seed,
   Spearman(score, mean G) ≤ −0.6, and |Spearman(log documents, mean G)| is smaller than
   |Spearman(score, mean G)|. This is where the drop and the bits per byte failed: if the
   topical confound was the semantic and topical part of those readings, a syntactic reading
   should be free of it.
6. **Per head.** In every cut with z ≥ 3 and score ≥ 0.05 (the floor found by the ladders), on
   both seeds: Spearman(E(h), G_S(h) − G_random(h)) ≤ −0.2. G_random is the mean over the three
   random 25% cuts.

**How to read the outcomes.**

- If 1–3 hold, the evaluation measures syntactic generalization on natural text: it sees
  planted gaps, the controls pass it, and it follows the dose.
- If 4–6 also hold, the paper can state its claim in the measure the claim is about: the
  indicator, computed on the corpus alone, predicts how well a model generalizes the domain's
  syntactic patterns, per pattern and per corpus, independently of topic coverage.
- If 1 fails, this model is too small to show syntax in this test on natural text, and the
  test needs a larger model before anything else can be read.

**Construction may be checked, results may not.** The thresholds above (90%, 20 occurrences,
30%, 20 pairs, 20 contexts) may be adjusted only from corpus-side counts, such as how many heads
are kept and how many pairs they get, before any model is scored. Any adjustment is recorded
here.

**Adjustment before scoring (2026-10-04).** At 30%, the plantings kept 6 of 17 heads and the
cuts 151 of 417. Many heads, nouns above all (`team`, `game`, `world`), are mostly followed by
a preposition or an auxiliary. Those are closed classes, with no unseen members to test. For
example, `game` is followed by ADP 0.32 and NOUN 0.20. The slot share is lowered to **20%** for
both families, from these tag counts alone. A head kept at 20–30% is tested on a minority slot
of its own, which makes its reading weaker but not invalid. At 20% the plantings keep 11 heads
(9 planted, 2 comparison) and the cuts 217. Predictions 2 and 3 are read on what is kept: 6
strong-dose readings per seed instead of 8, 18 planted readings instead of 28, and 2 comparison
heads instead of 3.

**Outcome (2026-10-04, `experiments/syntactic_eval`).** All six predictions failed.

- **1, narrowly:** the controls' mean G is 0.724 / 0.680, against a bar of 0.70. The model
  prefers the class-valid unseen word in about 7 of 10 comparisons. Heads with a strong slot
  read 0.75–0.94 (`she`, `then`, `these`, `still`). Heads kept at 20–30% read near chance
  (`world` 0.49, `while` 0.53, `life` 0.54).
- **2 and 3:** the strongest plantings do remove it on seed 0. `even` falls 0.76 → 0.34 and
  `now` 0.78 → 0.36, with a strong-dose mean delta of −0.21 and comparison heads within 0.03.
  On seed 1 the mean delta is −0.09 and the comparison heads move 0.07. Spearman(D, delta) is
  −0.41 / −0.13. Only `now` at dose 0.2 falls on both seeds (0.78 → 0.36, 0.81 → 0.51).
- **4–6:** among cuts, G follows size and document count, not the score.
  - Size: `full` 0.748, the 25% cuts 0.700–0.736, `r10` 0.69.
  - At one size, Spearman(score, mean G) is −0.16 / −0.33, and log documents read +0.66 /
    +0.56. The `self` ladder reads +0.80, the wrong sign.
  - Per head, Spearman(E, G − random G) is −0.14 to +0.12, with no sign.

**Diagnostics, exploratory, run after the outcome.**

- G is not noise: per head, it reproduces across the two seeds of one cut with Pearson
  0.87–0.91.
- Keeping only the 151 heads whose slot class holds at least 30% does not change the reading:
  score −0.02 to +0.08 against G, log documents +0.53 / +0.59.

**Reading.** Two different things were measured, and the indicator predicts only one of them.

- **The drop reads which filler.** It is the probability of the true word. It falls where a
  cut moves the slot's filler distribution, as E predicted on six filters.
- **G reads which class.** It falls only when a slot collapses onto one filler, as in the
  strongest plantings. The natural filters at 25% move mass among fillers that remain, and the
  model still knows what class goes after the head. That knowledge follows how much and how
  varied the text is, size and document count, not how the slot's fillers are distributed.

In the recipe grammar, restrictions went down to 1–24 fillers per verb, the regime where class
knowledge breaks, and coverage predicted it. On natural text, the cuts tested here do not reach
that regime. On natural text, the paper can therefore claim that the indicator predicts the
filler-level cost of a cut, per head, and the class-level failure only for collapsed slots. It
cannot claim that the indicator predicts class-level syntactic generalization among realistic
cuts. Whether that holds for larger models, or for stronger natural restrictions such as
narrow-domain corpora, is open.

## Head patterns and class patterns: at which level does a model learn syntax?

Written before running (2026-10-04), after discussing the syntactic evaluation with the user.

**Hypothesis.** The patterns the indicator reads are lexical-head patterns: `people` → verb,
`she` → verb, each on its own. A syntactic pattern is more abstract: pronoun → verb, with every
pronoun an instance of it. This is point 4 of the frame, that syntax is hierarchical.

- **Class level (the hypothesis):** a model learns class patterns. A head whose own slot is
  poor still gets the class pattern from its siblings, so the model keeps its syntax. Class
  knowledge fails only when the class pattern itself is poor. A single collapsed head fails
  when its own memorized filler overrides the class pattern, as `now` and `even` did.
- **Head level (the alternative):** a model learns each head's slot on its own, and the
  head-level reading predicts class-level syntax per head.

**Design.** Remove a pattern from a whole class, and from part of it, with everything else
held. The heads are two closed classes with a verb slot:

- pronouns `I`, `you`, `he`, `she`, `we`, `they`;
- modals `can`, `could`, `will`, `would`, `should`, `may`, `might`, `must`.

The documents of `r50` are kept (2,234 documents, 7.2 MB). Only sentences are removed, so the
document count and topics stay fixed. A sentence is a **target** when one of its tokens is a
listed head immediately followed by a token spaCy tags VERB. Three corpora:

- **`class_cut`:** every target sentence of both classes is removed. The pronoun → verb and
  modal → verb patterns are gone; pronouns and modals still occur elsewhere, as in "she is",
  "they also" and "can't".
- **`head_cut`:** only the target sentences of `she`, `they`, `can` and `will` are removed. Each
  of those four loses its verb slot, while its siblings keep theirs.
- **`control`:** no target sentence is removed.

In `head_cut` and `control`, non-target sentences are then removed at random (seed 0) until
each corpus has the bytes of `class_cut`. Size is matched, and only which sentences go
differs. The small model is trained on seeds 0 and 1: 6 runs.

**Readings.**

- **Head level, per head:** D(h) against F, as in `experiments/divergence`, minus the same in
  `control`.
- **Class level, per class:** the same D, but with the class's heads pooled as one head, so its
  fillers are those after any of them.
- **Model:** G per head, by the evaluation of the previous section, with items built for these
  14 heads. The change is ΔG = G − G_control, the mean of both seeds.

**Predictions.**

1. **The cuts do what they say (corpus side).** Head-level ΔD is at least 1 nat at the 4 removed
   heads in `head_cut`, and at every listed head in `class_cut`. It is under 0.2 at the 10
   siblings in `head_cut`. Class-level ΔD in `head_cut` is less than half of its value in
   `class_cut`, for both classes.
2. **Removing a class pattern removes class knowledge.** In `class_cut`, the mean ΔG over the 14
   heads is ≤ −0.10.
3. **Removing a head's slot does not, when its class keeps the pattern (the main test).** In
   `head_cut`, the mean ΔG over the 4 removed heads is less than half, in absolute value, of
   their mean ΔG in `class_cut`. Under the head-level alternative, the two would be equal: the
   4 heads lost their slot completely in both.
4. **Nothing else moves.** In both cuts, over the heads of the previous section's cut items that
   are not in the two classes, the mean |ΔG| is ≤ 0.03.

**How to read the outcomes.**

- If 2 and 3 hold, a model learns syntax at the class level. The indicator must read class
  patterns to predict syntax and head patterns to predict which filler. That would make both
  levels part of the paper's claim.
- If 2 holds and 3 fails, a model learns a head's slot on its own. The head-level reading
  predicts syntax wherever a head loses its slot completely, which no realistic cut did.
- If 2 fails, even removing a class pattern does not remove class knowledge at this model size.
  Subjects such as `people` still teach subject → verb, so the class that matters is wider than
  spaCy's tags, and the paper keeps syntax claims to the recipe grammar.

**Outcome (2026-10-04, `experiments/class_patterns`).** Two of four predictions held.

The corpora: `r50` has 62,025 sentences. 18,126 are targets of either class, and 5,730 belong
to the four removed heads. All three corpora hold 4.67 MB and 2,220–2,225 documents.

- **Prediction 1 held.** The cuts do what they say.
  - Head-level ΔD is +1.13 to +1.88 at the four removed heads in `head_cut`, and +1.02 to +2.30
    at every head in `class_cut`. At the ten siblings in `head_cut` it is −0.00 to +0.09.
  - Class-level ΔD is +0.04 (pronouns) and +0.10 (modals) in `head_cut`, against +1.82 and
    +1.82 in `class_cut`.
- **Prediction 2 failed narrowly.** The mean ΔG over the 14 heads in `class_cut` is −0.086,
  against a bar of −0.10. It splits by class:
  - pronouns −0.195 (`I` −0.24, `he` −0.24, `they` −0.25, `you` −0.17, `we` −0.17, `she` −0.10);
  - modals −0.004.
- **Prediction 3 held.** The four removed heads lose −0.031 in `head_cut` against −0.078 in
  `class_cut`. For pronouns the contrast is sharp: `she` and `they` lose −0.044 when only they
  lose the slot, and −0.177 when their class loses it. In `head_cut` the siblings lose about as
  much as the removed heads (−0.028), the share of the class pattern that `head_cut` removed.
- **Prediction 4 failed.** Heads of neither class move by a mean |ΔG| of 0.052 (`class_cut`)
  and 0.040 (`head_cut`), against a bar of 0.03. The bar was set below the per-head difference
  between models trained on different corpora, which was not measured beforehand.

**The modals, explained after the outcome.** The test is not valid for them. After a modal only
a base-form verb is grammatical ("can keep", never "can kept"). spaCy's VERB tag pools every
form, so the valid words for `can` include `kept`, `came`, `involving` and `requires`. That is
why even the controls read `can` 0.64 and `will` 0.62, against 0.82–0.90 for most pronouns.
The modal result says nothing either way. A fine tag (VB) would be needed to test them.

**Reading.** For pronouns, where the test is valid, the outcome is the one the class-level
hypothesis predicted.

- **A head that loses its slot keeps its syntax while its class keeps the pattern.** `she` and
  `they` never appear before a verb in `head_cut`. Yet the model still prefers an unseen verb
  after them almost as much as in the control (−0.04).
- **Removing the class pattern removes it for every member,** about −0.2 in G.
- **The readings agree with the model at the class level only.** Head-level ΔD says `she` and
  `they` lost as much in `head_cut` (+1.46, +1.13) as in `class_cut` (+1.58, +1.20). Class-level
  ΔD says `head_cut` barely touched the pronoun pattern (+0.04 against +1.82), and the model
  agrees.

The indicator therefore has two readings. The head level predicts which filler a model gives,
as the drop experiments showed. The class level predicts whether it keeps the syntax. This is
one class and two seeds, and its effect, about −0.2 in G, is large against the noise of 0.04–0.05.

## Replication on two more classes, with grammatical contrasts

Written before scoring or training (2026-10-04). The class-level result rests on one class,
pronouns. The test was also looser than it should be: its contrast words included adverbs,
and "she quickly" is grammatical. This replication tests two classes whose slot has a clear
grammatical form, with contrast words that are ungrammatical in that slot.

- **Modals → base-form verb.** `can could will would should may might must`.
  - Valid words: VERB words whose majority fine tag is VB (the base form: "can keep").
  - Contrast words: NOUN and ADJ words ("can table", "can happy"). Adverbs are excluded ("can
    quickly run").
  - Read on the existing `class_patterns` models, whose cuts already removed modal + VERB
    sentences, so this needs no training. The items change because the first test was invalid,
    which is recorded above. These predictions are written before the new items are scored.
- **Possessives → noun.** `my your his our their its`.
  - Valid words: NOUN words.
  - Contrast words: ADV words and VERB words whose majority fine tag is VBD or VBZ ("his
    quickly", "his ran", "his runs"). Adjectives ("his old") and gerunds ("his running") are
    excluded.
  - New cuts from `r50`, built as before: a target sentence has a listed possessive immediately
    followed by a NOUN. `poss_class_cut` removes every target sentence. `poss_head_cut` removes
    those of `his` and `their`, so `my`, `your`, `our` and `its` keep the pattern.
    `poss_control` is the random control. Bytes are matched. 6 runs: 3 corpora × seeds 0 and 1.

Words are classed as before: majority coarse tag ≥ 90%, ≥ 20 occurrences. The fine tag is the
majority among the word's VERB occurrences. Pairs are frequency-matched as before.

**Predictions, per class (modals, possessives).**

1. **The cuts do what they say (possessives, corpus side).** Head-level ΔD ≥ 1 at `his` and
   `their` in `poss_head_cut` and at every possessive in `poss_class_cut`. ΔD < 0.2 at the four
   siblings in `poss_head_cut`. Class-level ΔD in `poss_head_cut` is less than half of its value
   in `poss_class_cut`.
2. **Removing the class pattern removes class knowledge.** In the class cut, mean ΔG over the
   class's heads ≤ −0.10.
3. **Removing the slot from some heads does not.** For the heads removed in the head cut
   (`can` and `will`; `his` and `their`), |mean ΔG in the head cut| is less than half of
   |mean ΔG in the class cut|.
4. **No systematic shift elsewhere.** Over the heads of `syntactic_eval`'s cut items that belong
   to neither class, |mean signed ΔG| ≤ 0.02 in every cut. The first test measured the mean
   absolute ΔG of single heads, 0.04–0.05, which is noise between models. A systematic shift
   would show in the signed mean.

**How to read the outcomes.** If 2 and 3 hold for both classes, the class-level result
replicates on three classes, and the paper states it as a finding. If they hold for one, the
paper reports it as class-dependent.

**Outcome (2026-10-04, `experiments/class_patterns/replication.py`).** Six of seven held.

The possessive corpora: 10,315 target sentences, 3,812 of them for `his` and `their`. Each
corpus is 5.59 MB with about 2,230 documents.

- **Prediction 1 held (possessives).** Head-level ΔD is +1.88 / +1.58 at `his` / `their` in
  the head cut, +1.11 to +2.18 at every possessive in the class cut, and −0.00 to +0.03 at the
  siblings. Class-level ΔD is +0.11 (head cut) against +1.94 (class cut).
- **Prediction 2 held for both.**
  - Modals: the control reads G 0.96–1.00 (the corrected test is valid), and the class cut
    −0.126.
  - Possessives: the class cut reads −0.303, and every possessive falls to 0.46–0.61.
- **Prediction 3 held for modals, narrowly, and failed for possessives.**
  - Modals: `can` and `will` lose −0.040 in the head cut against −0.081 in the class cut. The
    bar was half, 0.0405.
  - Possessives: `his` and `their` lose −0.194 against −0.343 (57%). `his` goes 0.79 → 0.60
    → 0.51 and `their` 0.87 → 0.67 → 0.46.
  - Siblings in the head cut: modals −0.008, possessives +0.046.
- **Prediction 4 held for both.** The signed shift elsewhere is +0.006 / +0.006 for modals and
  +0.006 / +0.018 for possessives.

**Reading, over three classes.**

| Class | Class cut ΔG | Removed heads, head cut | Removed heads, class cut | Ratio | Siblings, head cut |
|---|---|---|---|---|---|
| pronouns | −0.195 | −0.044 | −0.177 | 25% | −0.028 |
| modals | −0.126 | −0.040 | −0.081 | 49% | −0.008 |
| possessives | −0.303 | −0.194 | −0.343 | 57% | +0.046 |

- **Removing a class's pattern removes syntax in all three classes,** −0.13 to −0.30 G, every
  member falling.
- **Removing it from two heads costs those heads a quarter to a little over half as much,**
  and leaves their siblings intact.
- So syntax is learned at both levels. The class carries most of it, and a head's own
  evidence adds a part that varies by class. Pronouns lean on their class most and
  possessives least.
- **Class-level D predicts the class cuts** (+1.8 to +1.9 where every class lost its syntax).
  It underestimates what a head cut costs the removed heads: +0.04 to +0.11 class-level D,
  where possessives lost −0.19. The head-level reading flags those heads (+1.6 to +1.9).
- A diagnosis of syntax therefore reads both levels. A class pattern under-shown predicts
  loss for every member. A head under-shown inside a sufficient class predicts a smaller
  loss for that head alone.

## The tool's class rows on the cuts

Written before running (2026-10-04). `corpus-diagnose` now reads class patterns
(`docs/2026-09-30-corpus-diagnose-design.md`). Its head classes are induced, never spaCy's. On
F at 64 classes, they put the six pronouns in one class, the eight modals in one, and the six
possessives in one. Run on the six cuts of this section and the previous one, against F, at
F's merge budget (11,563):

1. In `class_cut`, the head class holding the pronouns and the one holding the modals are
   `under-shown`. In `poss_class_cut`, the one holding the possessives is.
2. In `head_cut`, `poss_head_cut` and both controls, none of those three classes is
   `under-shown`.

**Outcome (2026-10-04, `replication.py diagnose` / `summarize-tool`).** Prediction 2 held and
prediction 1 failed: one class of three was found.

| Cut | Pronoun class: D, excess | Modal class: D, excess | Possessive class: D, excess |
|---|---|---|---|
| `class_cut` | 0.524, +0.466 (within-noise) | 1.439, +1.352 (**under-shown**) | 0.075, −0.023 |
| `head_cut` | 0.071, +0.013 | 0.181, +0.094 | 0.073, −0.024 |
| `control` | 0.049, −0.009 | 0.084, −0.003 | 0.076, −0.022 |
| `poss_class_cut` | 0.050, +0.001 | 0.068, −0.002 | 0.091, +0.013 (within-noise) |
| `poss_head_cut` | 0.044, −0.005 | 0.068, −0.002 | 0.067, −0.011 |
| `poss_control` | 0.041, −0.008 | 0.067, −0.003 | 0.062, −0.016 |

No cut flags any class it should not.

- **Modals are found.** Their induced class is the eight modals plus `cannot` and `dont`.
- **Pronouns are missed, narrowly.** The excess, +0.47, lies just below `class_floor` (0.5).
  Their induced class also holds `it`, `all`, `one`, `there` and indefinites (`something`,
  `nobody`), whose slots the cut did not touch, so the pooled shift is diluted.
- **Possessives are missed.** The 64-class cut puts them with the determiners (`the`, `a`,
  `this`, `an`, …). Pooled counts weight heads by frequency, so `the` and `a` dominate the
  class's fillers, and removing the possessive + noun sentences barely moves them. Yet the model
  lost the possessives' syntax (−0.30 G). The model treats possessives apart from determiners,
  and the neighbor-based clustering does not.

**Reading.** The class rows find a class pattern when the induced class is that pattern's class.
They miss it when a frequent head shares the class. Two changes follow from this run and are
therefore untested:

- weight each head equally in a class's pooled distribution, so `the` cannot hide `his`;
- reconsider the floor, which sits just above the pronouns' excess.

Either needs new cuts to be tested, because these six already decided it.

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
