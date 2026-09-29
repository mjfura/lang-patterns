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
