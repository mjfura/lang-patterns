# Is the Corpus to Blame? Diagnosing the Local Syntactic Sufficiency of a Training Corpus Before Training

*Draft v0.2, 2026-10-15. Working draft. Numbers come from the experiment records in
`docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md`, `docs/superpowers/specs/2026-10-11-settings-from-data-design.md`
and the `experiments/` folders of this repository. Every experiment was preregistered in those
documents before it ran, and failed predictions are reported as failed. v0.2 replaces the
indicator's hand-set thresholds with settings derived from the data, adds a reading without a
reference, and reruns the corpus-side experiments with the new tool. Model-side numbers are
unchanged from v0.1.*

---

## Title (proposals)

1. **Is the Corpus to Blame? Diagnosing the Local Syntactic Sufficiency of a Training Corpus Before Training** (preferred)
2. Served or Missing: A Pre-Training Diagnostic of Syntactic Patterns in Language-Model Corpora
3. Flag the Corpus, Not the Model: Attributing Syntactic Generalization Failures to Training Data

---

## Abstract

When a language model fails to generalize a pattern of its domain, the cause can lie in the
model (its architecture, size or training) or in the corpus (the pattern was never shown with
enough variety to be learned). Existing data-quality measures cannot tell these apart. They give
one score per document or dataset, and the most common of them, held-out perplexity, can rank
corpora in reverse of how well a model trained on them generalizes. We propose a
**corpus-only indicator of local syntactic sufficiency**. It reads every syntactic pattern of a
corpus at two levels: a head followed by a grammatical class (*she* → verb), and a class of heads
followed by its fillers (pronoun → verb), with classes induced from the corpus. Given a reference
corpus of the domain, each pattern is compared with size-matched random samples of the reference
and reported as under-shown, absent or within noise. Without a reference, each head is compared
with its siblings in the corpus's own head tree, and patterns the corpus shows weakly are
reported as weak. **No threshold is set by hand.** The tokenization, the trunk below which a
pattern is not read and the number of random samples all come from the data, and the one
remaining setting is the test's level (alpha = 0.06 per family of verdicts).

We validate the indicator with corpus interventions and trained models, with predictions written
before each experiment:

- On a synthetic recipe grammar, the indicator predicts which verb–noun patterns a model
  generalizes (Spearman 0.94–0.98), while held-out loss ranks the corpora in reverse.
- On natural text (FineWeb), the head-level reading finds planted gaps without false alarms on
  unplanted text. At the planted heads a model loses 1–4.6 nats on the true next word.
- Syntax is learned at the class level. Removing a pattern from two heads costs those heads 25–57%
  as much as removing it from their whole class. Removing it from the class removes the syntax
  of every member (−0.13 to −0.30 in a minimal-pair score) for pronouns, modals and
  possessives.
- **Attribution.** When the indicator flags a class pattern, no model configuration we tried
  recovers it, and larger or longer-trained models do worse. When it flags nothing, a corpus of
  the same size reaches the reference's syntax within 0.02. The modal result holds, and grows,
  at ten times the data (≈10M words, BLiMP 0.63).
- **Settings.** The verdicts do not change with granularity from 1.35 to 1.80 tokens per word,
  with 250 to 4,000 context tokens, or with 8 to 64 filler classes. Random cuts of the reference
  are flagged at the designed rate (2 of 48).

Against one-number corpus metrics (perplexity, unigram KL, MATTR, distinct-2, Vendi score), the
indicator localizes the damaged pattern. It also detects pattern-specific losses of 2,000–3,600
sentences that leave those metrics inside their noise. Its scope is local (adjacency) syntax,
validated with models of up to 6.2M parameters.

---

## 1. Introduction

A practitioner who trains a language model on a domain corpus and finds that it fails to
generalize faces a diagnostic question before any engineering one. Is the corpus missing
something, or is the model wrong for it? The two call for opposite actions. Collecting or
rebalancing data cannot fix a model that is too small or badly trained. Tuning the architecture
cannot fix a corpus that never shows a pattern with enough variety. Today this question is
answered, if at all, by training and inspecting.

We frame the question through the notion of a **knowledge domain**: a bounded subset of the
syntactic and semantic patterns of a language, such as recipes, which are mostly imperative
actions on ingredients. This paper addresses its syntactic layer only. Syntax here is the
well-formed frame of a sentence regardless of its truth. "Cut the milk" is syntactically
correct; that it makes no sense is semantics, and out of scope. The premise follows a long
observation in corpus linguistics (Baayen's productivity measures) and in language-model
learning (Wei et al., 2021; Misra & Mahowald, 2024). A pattern is learned as a pattern, and not
as a memorized word, when the corpus shows its slot with enough different fillers. A corpus is
**syntactically sufficient** for its domain when every local pattern of the domain is shown
that way.

Existing tools do not measure this.

- **Data-quality and selection methods** (perplexity filtering in the CCNet style, DSIR, QuRating,
  the diversity coefficient) give one number per document or dataset, aimed at perplexity or
  downstream accuracy. A single score cannot say which pattern is missing.
- **Held-out loss can mislead.** On a controlled grammar we show that it ranks corpora in reverse
  of syntactic generalization: a restricted corpus is restricted in its held-out split too, so it
  looks more predictable.
- **Filtered-corpus training** (FiCT; Patil et al., 2024) and related studies remove a
  construction and test what the trained model learns. They are post-hoc by design, and their
  main message, that models generalize from indirect evidence, makes the question of which
  removals matter a real one.

A diagnostic of this kind is only as trustworthy as its settings. A threshold tuned on the
corpora where the method was developed can produce, on a new corpus, flags that are artifacts of
the threshold. We therefore require every setting of the indicator either to be derived from the
data or to be shown not to change its verdicts, and we report the measurements (§5.8).

We contribute:

1. **An indicator**, computed on the corpus alone. It reads local syntactic patterns at the head
   level and at every node of an induced tree of heads. With a reference, it compares them with
   size-matched random samples of the reference; without one, with sibling heads in the corpus
   (§3). It is packaged as a command-line tool and a user interface.
2. **Settings from the data.** A measured range of tokenizer granularity, a test level that sets
   the number of random samples, and a trunk read chosen by Tarone's criterion replace every
   hand-set threshold (§3.4, §5.8).
3. **Causal validation** by corpus interventions, in a synthetic grammar and in natural text:
   planted gaps, dose–response, realistic filters, and removal of a pattern from a head versus
   from its whole class (§5).
4. **A finding about the level at which syntax is learned.** A head that loses its slot keeps its
   syntax while its class keeps the pattern. A class that loses the pattern loses the syntax of
   every member (§5.4).
5. **An attribution test.** A flagged class pattern is a corpus fault that no tried model
   configuration fixes. An unflagged corpus lets the reference's syntax be reached (§5.5). This
   is the guarantee a practitioner needs.
6. **Comparisons with one-number corpus metrics and BLiMP**, with an honest account of where they
   do and do not see the same thing (§5.7), and a list of every failed prediction (Appendix A).

---

## 2. Related work

**Data quality and selection for pretraining.**

- Perplexity filtering against a reference language model (the CCNet pipeline) is a standard
  quality filter.
- DSIR (Xie et al., 2023) selects data by importance resampling on hashed n-gram features.
- QuRating (Wettig et al., 2024) uses language-model judgments of writing quality, expertise
  and educational value.
- The diversity coefficient (Miranda et al., 2023) measures a dataset's variability through
  Task2Vec embeddings.
- Training-free predictors of data influence such as DataProphet (2026) rank supervision datasets
  before training.
- Albalak et al. (2024) survey the field.

All of these score documents or datasets as wholes. None reports which syntactic patterns a
corpus under-serves, and none separates a corpus fault from a model fault. We find that
perplexity-based filters are the only cuts of a corpus the indicator flags (§5.3).

**Learning syntax from manipulated corpora.**

- Wei et al. (2021) show that BERT applies subject–verb agreement to pairs never seen together,
  but struggles with infrequent items and strong priors.
- FiCT (Patil et al., 2024) removes a construction from the training corpus and finds that both
  LSTMs and transformers generalize to it from indirect evidence. Luo & Steinert-Threlkeld (2026)
  extend this to unlike coordination.
- Misra & Mahowald (2024) show that a rare construction (AANN, "a beautiful five days") is learned
  from related, more frequent ones, and better with more variability in the input.

Our results agree at the head level: a pronoun that loses its verb slot keeps its syntax through
its siblings. We add the class level, where removal does prevent learning, and an indicator that
says before training which removals matter.

**Productivity and diversity measures.** Baayen's productivity measures use hapax legomena to
estimate how open a morphological category is. The Good–Turing missing mass (singletons over
trunk) applies the same idea to a syntactic slot; we show that on its own it does not flag a
thinned slot (§5.9). Measures of lexical, syntactic and semantic diversity of model outputs
(Guo et al., 2024) assess generated text, not training corpora, and are not tied to what a model
learns.

**Multiple testing with discrete statistics.** The indicator tests tens of thousands of patterns
at once. Its bar is the maximum statistic over random samples (a resampling family-wise bar), and
patterns too rare ever to be flagged are left out of the family, following Tarone (1990).

**Targeted syntactic evaluation.** Minimal pairs (Linzen et al., 2016; Marvin & Linzen, 2018), the
systematic assessment of Hu et al. (2020) and BLiMP (Warstadt et al., 2020) evaluate trained
models. Our model-side reading is a minimal-pair score built to exclude semantics. Unseen
class-valid fillers compete against frequency-matched ungrammatical ones, with classes from an
external tagger (§4.3). We also report BLiMP.

---

## 3. The indicator

### 3.1 Units: tokens, transitions, classes

- **Tokens.** The corpus is tokenized with a byte-level BPE learned on the domain's text (the
  reference's, or the corpus's without one), to the fewest merges that leave 1.50 tokens per
  word. The verdicts were measured not to change from 1.35 to 1.80 (§5.8). On the FineWeb pool
  this is 11,381 merges.
- **Every token takes part.** Every token found in at least two documents can be a head or a
  filler: words, word pieces, punctuation and numbers.
- **Syntactic and lexical transitions.** A transition is the head's last token followed by the
  filler. It is *syntactic* when the filler starts a new pre-token of the tokenizer (a new word,
  punctuation run, number or contraction), and *lexical* when it continues the word of the token
  before it (` model` → `s`). No merge crosses a pre-token boundary, so the rule is exact. The two
  kinds are counted as separate patterns, and syntactic rows are the diagnosis.
- **The class tree.** Each token is described by its left and right neighbor distributions over
  the most frequent 1,000 tokens. The rows are square-rooted, so that Euclidean distance is
  Hellinger distance, and clustered with Ward's method into one tree.
- **Filler classes.** The tree is cut at the largest jump in merge height between 8 and 64
  classes (12 on FineWeb). On the recipe grammar this cut recovers the grammar's categories
  exactly. A finer cut starts grouping fillers by meaning (what can be poured against what can be
  chopped), which would bring semantics into a syntactic measure.
- **Head classes.** Every internal node of the same tree is a class of heads. No cut is chosen:
  class patterns are read at every node (§3.2).

### 3.2 Patterns

- **Head pattern.** A head of one or two tokens followed by a filler of class Y, with its
  transition kind.
- **Occurrences are counted in distinct contexts** (the token before the head, the token after
  the filler), so that exact copies add nothing.
- **Trunk.** A pattern's number of distinct contexts.
- **Class pattern.** A node of the head tree, with the fillers that follow any of its one-token
  heads, pooled.

### 3.3 Readings

- **Diversity:** fillers / expected fillers, where expected fillers = Σ_y [1 − (1 − p_y)^n],
  with n the trunk and p_y the class shares of the corpus. It compares heads of different
  traffic against free draws from the class.
- **Divergence D:** KL(R ‖ S), in nats. R is the reference's filler distribution for the slot.
  S is the corpus's, with Witten–Bell backoff to the corpus's filler unigram, adding one count
  per reference filler. D has the units of the loss: a model that learned S and is scored on
  text from R loses on average KL(R ‖ S) nats per slot occurrence. On planted gaps the drop in
  the model's log probability is −0.68 · D (R² = 0.77).
- **Size-matched draws.** Every reading depends on the corpus's size, so a smaller corpus always
  reads worse. Each corpus is therefore compared with random whole-document samples of the
  reference of the same size in bytes. The reference is tokenized and classified once and reused
  for every corpus diagnosed against it.

### 3.4 Verdicts with a reference

Each pattern's t is how far the corpus falls from the draws' mean, in the draws' spread: diversity
for head rows, trunk for `absent`, and D for class rows. The spread includes the reading's own
rounding (diversity moves in steps of one filler, a trunk in steps of one): sd' = √(sd² + step²/12).

**The bar comes from the draws.** Each draw in turn is left out and read against the others, and
its largest t over all patterns is recorded. A corpus is flagged only where it strays further
than the k-th largest of these maxima: a random sample of the reference then passes the bar with
probability about k / (n + 1).

- **alpha**, the chance that a random sample of the reference is flagged in a family of verdicts,
  is the one setting the user states. The default is 0.06 per family (head rows, `absent`, class
  rows). It is the test's level, a tolerance rather than a property of the data.
- **The number of draws** follows from alpha. The k-th largest of n draw maxima is itself noisy:
  over draw sets, the chance of passing it follows Beta(k, n + 1 − k). The tool uses the fewest
  draws that keep that chance within alpha/2 to 1.5 · alpha for 80% of draw sets: 101 draws and
  k = 6 at alpha 0.06.
- **The trunk read** comes from the draws, by Tarone's criterion. A pattern too rare to be flagged
  even if the corpus showed a single filler adds no verdict, and its noise raises the bar for
  every other pattern. The least trunk read is the one that leaves the most patterns that could
  be flagged.

**Head rows:**

- **under-shown:** diversity below the draws' mean by more than the bar;
- **absent:** the trunk below the draws' mean trunk by more than the `absent` bar. A random sample
  of this size would show the pattern, and the corpus shows too little of it;
- **within-noise** otherwise, and **unread** below the trunk read.

**Class rows:** D over the node's pooled fillers, against the draws. A node is **under-shown** when
its t exceeds the class bar. Of each branch, the node with the highest t is reported, so the
report names the smallest group of heads that carries the shift. Each class row also lists its
**short heads**, the members whose own rows are under-shown or absent.

### 3.5 Verdicts without a reference

A pattern the corpus shows weakly is weak for a model trained on it, whatever the domain does. A
corpus alone can show this by comparing a head with its siblings:

- Each child of a node of the head tree (a head, or a class of heads) is compared with its
  parent, per filler class and transition kind.
- **The null.** Had the parent's occurrences of that class fallen to its two children at random,
  how many distinct fillers would this child show? 101 random splits are drawn exactly, without
  replacement, at the child's trunk.
- The bar is built from the splits as in §3.4, at the same alpha, and only tests a single filler
  could fail are read.
- Only children of a node inside one filler class are tested. Above the class cut, the two sides
  of a split are different categories (`,`/`the`/`to` against `.`/`:`), and every such test read
  as weak.
- Heads of two tokens are placed by their first token.

Verdicts: **weak**, past the bar; **rare**, shown fewer times than any slot the test can read;
**within-noise**; **unread**, with no sibling to read against.

Three cases are invisible without a reference. A pattern the corpus lacks altogether, and a class
shown weakly as a whole, leave no sibling stronger to compare with. A head restricted enough for
the tree to set it apart from its siblings loses its sibling. The report states these limits and
recommends a reference from the same domain. A reference also separates the corpus's own weakness
from the domain's: most weak patterns of a natural corpus are its grammar (` to` against the other
prepositions, reporting verbs), and a reference reads them within noise.

### 3.6 The tool

`corpus-diagnose` (Python) takes a corpus and, optionally, a reference. It writes JSON and
Markdown reports: summary, warnings, classes, head rows, class rows and settings, including alpha
and the number of draws.

- With no merge budget, it learns the domain's tokenizer to 1.50 tokens per word, resuming the BPE
  in doubling rounds.
- A given `--merges N` is checked before the diagnosis: outside 1.35–1.80 tokens per word the
  command refuses and names the granularity reached, unless `--force` is passed.

A Streamlit page runs the same diagnosis on any catalogued dataset.

---

## 4. Experimental setup

### 4.1 Corpora

- **Recipe grammar.** A probabilistic grammar with a closed lexicon (imperative verbs, countable
  ingredients, adjectives, adverbs, tools) and four sentence rules. One knob changes how many
  ingredients each verb is ever seen with (1, 2, 4, 8, 16, 20, 24 or all). Every ingredient still
  occurs in other sentences. Reserved verb–ingredient combinations are never seen and serve as
  test items.
- **FineWeb** (Penedo et al., 2024).
  - Our main pool F is 4,600 documents (14.4 MB, ≈2.4M words, 3.4M tokens) from the first 5,000
    streamed documents. The other 400 documents are held out for evaluation.
  - A 6.5 MB sample in 8 slices is used for plantings.
  - BigF is the first 20,000 documents minus the evaluation set (57.8 MB, ≈10M words).
- **Interventions.**
  - Plantings: followers of chosen heads are replaced by one of their k most frequent fillers, or
    the sentences where they take another filler are removed.
  - Document cuts at 10–50% of F: random, perplexity filters, length filters.
  - Sentence-level cuts of a 50% cut of F (r50) that remove every sentence in which a listed head
    is immediately followed by a spaCy-tagged class. They are made at class level, at head level
    (two members only) and as random controls, all at matched bytes.
  - **Labeled cuts** for the tool: the class and head cuts of pronouns and modals, possessives,
    numbers and base verbs (8 cuts), 12 random controls, and two cuts that drop four words each.

### 4.2 Models

A GPT-2-layout decoder with a tied head, trained on CPU.

- **small:** 2 layers, 128 wide, 4 heads, context 128, 1.9M parameters, 3,000 steps of 32 × 128
  tokens, learning rate 1e-3.
- **medium:** 4 layers, 256 wide, 6.2M parameters, learning rate 5e-4.
- **long:** small, 9,000 steps.
- **scale:** medium, 30,000 steps on BigF, about 5 hours per run.
- Each corpus learns its own BPE (11,563 merges on FineWeb). The best checkpoint is chosen on a
  held-out fraction. Seeds 0 and 1 unless noted.

### 4.3 Model-side readings

- **Drop:** for a head, the mean over its evaluation occurrences of log p_model(true next word)
  − log p_reference model, in nats. This is a filler-level reading: it mixes syntax with
  semantics and topic.
- **Generalization ratio (recipe grammar):** after "⟨bos⟩ cut the", the probability given to
  ingredients never seen with that verb, relative to their share (1 means the pattern is
  learned, 0 means memorized).
- **G (natural text):** after a real evaluation prefix ending in head h, the share of comparisons
  in which the model prefers a word of h's slot class that never follows h in the reference over
  a word of an ungrammatical class, matched in log frequency. Truth does not enter: after "she",
  any past-tense verb is valid. Classes come from spaCy (`en_core_web_sm`), never from the
  indicator. Grammatical contrasts:
  - pronouns → past-tense verbs (VBD), against nouns and adjectives;
  - modals → base verbs (VB), against nouns and adjectives;
  - possessives → nouns, against adverbs and VBD/VBZ verbs.
- **BLiMP** (67 paradigms × 1,000 pairs), by total log probability.

### 4.4 Protocol

Every experiment's design, predictions and thresholds were written in the design documents before
it ran. Construction parameters were changed only from corpus-side counts, before any model was
scored, and each change is recorded. Failed predictions are kept and listed in Appendix A.

The indicator was first built with hand-set thresholds (a diversity floor of 0.5, a 3-sigma
margin over 8 draws, a minimum trunk of 20, a class floor of 0.5 nats). Those were then replaced
by the settings of §3.4–3.5, and every corpus-side experiment was rerun with the final tool.
Models were not retrained: the corpora are the same, and only their reading changed. Where a
corpus-side number in §5 still comes from the first version, it is marked.

---

## 5. Results

### 5.1 Held-out measures do not see syntactic insufficiency (recipe grammar)

| Measure | Result |
|---|---|
| Per-verb coverage vs. generalization ratio | Spearman 0.891 (one seed); 0.940–0.975 over 3 seeds × 2 sizes |
| Mean ratio, all ingredients vs. one per verb | 0.909 vs. 0.011 |
| Held-out loss and n-gram predictability | Rank the variants in **reverse** on every seed and size |

The model generalizes late:

- The small model's ratio at 16 fillers is only 0.17–0.19, and it crosses 0.5 at coverage
  0.67–0.80.
- The larger model needs more: 0.554 against 0.901 at 24 fillers. "Sufficient" depends on model
  size, a point we return to in §6.
- Minimal pairs of the type "cut the tomato / cut the slowly", meant as an easy control, tracked
  the ratio (0.25 with one filler) instead of staying near 100%.

### 5.2 Head level on natural text: planted gaps and dose

**Detection, with the final tool.** Five heads of a FineWeb sample (*people, she, But, only,
even*) were planted in one slice, and each slice was diagnosed against the sample.

| Planting | Heads found (of 5) | Other rows flagged |
|---|---|---|
| sentences removed, k = 2 | 4 | 0 |
| fillers replaced, k = 2, 4, 8, 16 | 5 at each k | 0 |
| fillers replaced, k = 32 | 4 | 0 |
| 7 unplanted slices | 0 | 0 |

- Every found head is found through its row of determiner fillers, at diversity 0.03–0.17 for
  k = 2 and 0.27–0.35 for k = 32, against about 0.9 in the unplanted slices. Most are also read
  `absent` in a second filler class.
- No row outside the planted heads is flagged in any slice, of 2,900–6,600 rows read.
- The class level flags one class row in each of two of the seven unplanted slices (§7).
- With the first version's 3-sigma margin, the same sweep found 5, 5, 5, 5, 4 heads on FineWeb and
  19, 20, 19, 17, 12 of 20 heads over four sources (FineWeb, Wikipedia, arXiv, C4). Removing
  sentences found 3 of 5: two heads left with 6 and 23 occurrences read just above a fixed
  0.5 floor.

**Model cost.**

- Models trained on the planted corpora give the true next word 2.17–4.60 nats less (k = 2) and
  1.12–3.60 nats less (k = 16) at every planted head and seed.
- Comparison heads stay within 0.3 nats in 19 of 20 readings.
- The whole-text loss moves only 0.03 bits per byte: it registers the damage, but does not say
  where.

**Dose.** Fourteen heads were planted to diversity 0.1–0.7, with two assignments and two seeds.

- The loss is monotone in the dose (Spearman 0.76–0.92).
- Heads at 0.1 lose 2.8–3.3 nats.
- The mild end is not free: 0.38–0.57 nats at 0.5–0.7. Diversity is therefore not the effect
  size.
- D is the effect size (drop ≈ −0.68 · D, R² 0.77, first version's reading), although on plantings
  it orders heads no better than diversity.

### 5.3 Realistic cuts of a corpus

Five 25–50% cuts of F were diagnosed against F, and six models were trained per seed (§4.2).

| Cut | Tokens/word | Under-shown | Absent | Class rows flagged |
|---|---|---|---|---|
| random 50% | 1.49 | 0 | 0 | 0 |
| random 25% | 1.49 | 0 | 0 | 0 |
| random 10% | 1.52 | 0 | 0 | 0 |
| least Wikipedia-like 25% | 1.39 | 2 | 11 | 4 |
| most formulaic 25% (lowest perplexity) | 1.61 | 1 | 4 | 1 |

**Random cuts read as random.** They flag nothing, rows or classes. The first version's 3-sigma
margin flagged 2–8 rows on each, at the rate chance gave it.

**Filters are the only cuts flagged**, but mildly. The Wikipedia-like filter flags "," and ":"
before determiners and labels; the formulaic filter flags ". Our". Their absent rows are a few
thin heads ("50", " a day", " they should"). We had predicted at least 10 under-shown rows for
each filter, and both failed.

**The z of a head predicts what the cut costs the model better than its frequency.** In the 10%
cut, Spearman(z, drop) over 416 heads is 0.157 / 0.181 (two seeds), and log trunk alone 0.037 /
0.018. With the 3-sigma margin it was 0.10 / 0.12, with the trunk at 0.10 / 0.11.

**Filters move slots.** A cut's score is its heads' excess E = D − mean D over 8 random cuts
of F of its size, weighted by F's filler shares, and z divides it by the draws' leave-one-out
spread. The perplexity filter that keeps the 25% most formulaic documents reads z = +18.6. The one
that keeps the documents least like Wikipedia reads z = +24.8. The CCNet-style Wikipedia filter
reads z = +9.7. Random cuts read |z| ≤ 1.5. With the first version's reading of word-like tokens,
these were +15.0, +20.6, +3.9 and |z| < 0.6.

**The model pays where the excess points.**

- Within a filtered cut, heads with larger E lose more than a random cut of the same size costs:
  Spearman(E, drop − random drop) from −0.24 to −0.37, on six filters and both seeds (−0.21 to
  −0.38 in the first version).
- Raw D's apparent success on random cuts was sample size alone. In the 10% cut, D reads −0.42 /
  −0.45 against the drop, and E only −0.05 / −0.11.

**Choosing among cuts.**

- Over ten 25% cuts, the corpus score orders the models' bits per byte only partly: Spearman 0.65
  / 0.55, against a predicted 0.5 on both seeds (0.55 / 0.45 in the first version, which failed).
- Document count, a proxy for topical coverage, matters as much (−0.47 / −0.44). The score orders
  what document count leaves at 0.58 / 0.56.
- Along one filter's dose, with documents held fixed, the score orders the models (Spearman
  0.90). When the dose also adds documents, it does not (0.60).

### 5.4 Syntax is learned at the class level

**First reading, negative.** Among realistic cuts, G followed size (F 0.748, 25% cuts
0.700–0.736, 10% 0.69) and document count, not the score (−0.16 / −0.33). G per head reproduces
across seeds (Pearson 0.87–0.91). The reason, found next: realistic filters thin a head's slot but
never empty the pattern of its class.

**Head versus class removal.** From r50, at matched bytes, three corpora are built:

- `class_cut` removes every sentence where a pronoun (*I you he she we they*) or a modal is
  followed by a verb;
- `head_cut` removes only those of *she, they, can, will*;
- `control` removes random sentences.

| Class | Class cut ΔG | Removed heads: head cut / class cut | Siblings in head cut |
|---|---|---|---|
| pronouns | −0.195 | −0.044 / −0.177 (25%) | −0.028 |
| modals (VB items) | −0.126 | −0.040 / −0.081 (49%) | −0.008 |
| possessives | −0.303 | −0.194 / −0.343 (57%) | +0.046 |

- **Removing the class pattern removes the syntax of every member.** Removing it from two heads
  costs them a quarter to a little over half as much, and leaves their siblings intact.
- **The class level of the indicator agrees with the models.**
  - In `class_cut` the tool reports the modal node (t 319) and the pronoun node *I you we they*
    (t 254), against a class bar of 13.9.
  - In `head_cut` it reports no pronoun node, and a five-modal node holding *can* and *will* at
    t 33, a tenth of the class cut's.
  - Head-level D, by contrast, says *she* and *they* lost the same in both cuts (+1.46 / +1.58).
- **Syntax is learned at both levels, mostly at the class.** How much a head's own evidence adds
  depends on the class: least for pronouns, most for possessives.

### 5.5 Attribution: corpus or model?

The flagged class cuts and their unflagged size-matched controls are trained in three
configurations (small, medium, long), alongside F. The tool flags the class rows of each class cut
(pronouns and modals t 254 and 319, possessives t 165), and no class row of any control.

| Class | Best on F | Best on control (unflagged) | Best on flagged cut |
|---|---|---|---|
| pronouns | 0.981 | 0.961 | 0.846 |
| modals | 0.999 | 0.988 | 0.870 |
| possessives | 0.872 | 0.859 | 0.600 |

- **Flagged.** No configuration recovers the class (0.13–0.27 below F). More capacity makes it
  worse: pronouns 0.85 → 0.75 → 0.69, modals 0.87 → 0.80 → 0.75. A larger model memorizes the
  pattern's absence more firmly.
- **Unflagged.** The best control is within 0.02 of F. On a smaller unflagged cut (10%), changing
  the model moved G by up to 0.06. That is the engineer's lever.

**At ten times the data.** BigF, its class cut and its control were trained with the medium model
for 30,000 steps.

| Model | BLiMP | G pronouns | G modals |
|---|---|---|---|
| BigF | 0.629 | 0.910 | 0.902 |
| control | 0.632 | 0.822 | 0.895 |
| class cut | 0.622 | 0.761 | 0.565 |

- **BLiMP rises from 0.57 to 0.63.**
- **Modals: the attribution result is stronger at scale.** The cut leaves modals near chance.
- **Pronouns: the gap shrinks to 0.06, and our predictions failed.** At this data size almost
  every frequent past-tense verb follows each pronoun in the reference, which leaves 3–7 unseen
  items per pronoun, and the per-head readings become erratic. We cannot separate low test power
  from a real reduction. More data also brings more indirect evidence (noun subjects), as FiCT
  reports.

### 5.6 The tool's detection

The 22 labeled cuts of §4.1 were diagnosed against F at 1.50 tokens per word.

**Class rows.** Every removed class is found, each in its own branch of the tree, and no control
is flagged.

| Cut | Node reported | t (class bar 13.9) |
|---|---|---|
| pronoun and modal class cut | *can will would may could should* (8 heads) | 319 |
| | *I you we they i* | 254 |
| possessive class cut | *your his my their* | 165 |
| number class cut | *two … seven* | 104 |
| base-verb class cut | *get make see take find leave* | 101 |
| pronoun and modal head cut | *can will would could should* | 33 |
| possessive head cut | *his their*, exactly the removed heads | 85 |
| number and base-verb head cuts | nothing | |
| 12 random controls | nothing | |

**Head rows.** The class cut and head cut flag 39 and 13 head rows; the 12 controls flag none.

**Absent rows.** Every one of the 8 dropped words is found. On 8 fresh random controls, two rows
are read absent, one in each of two controls (` luck`, shown 0 times against the draws' 7.0 ± 1.8, and
`Maybe`).

**Random samples of the reference.** 48 random whole-document cuts of F, read against one fixed
set of draws: 2 head flags (4%), no class row and no `absent`. alpha is 0.06 per family.

In the first version, with 64 head classes and a 0.5-nat class floor, the class rows found 4 of 5
removed classes, missing the pronouns at +0.47 nats. The possessives were found only through their
short heads, because they shared an induced class with *the* and *a*. The first `absent` rule fired
79–114 times on random controls.

### 5.7 Comparison with corpus metrics and with BLiMP

**One-number metrics.** On the class cuts, head cuts and eight random controls we computed:
perplexity under F's bigram and under a Wikipedia bigram, unigram KL from F, MATTR, distinct-2,
and the Vendi score of MiniLM embeddings.

- **The planned test (3 sd of the controls) flags at most one class cut per metric.** That test
  was lenient: the controls mixed sizes, and the mean perplexity had an outlier.
- **Size-matched, as an exploratory reading.**
  - Unigram KL and distinct-2 do register the two large class cuts, which removed 18,126 and
    10,315 sentences (KL +0.019 and +0.011 against noise of about 0.001).
  - They stay within noise on the numbers (1,972 sentences) and base verbs (3,642), which the
    indicator flags at t 104 and 101.
  - No metric says which pattern changed.
- **The claim this supports is localization and sensitivity to small, pattern-specific losses.**
  It is not that other metrics see nothing.

**BLiMP.**

- Our small and medium models read 0.56–0.58 (chance 0.5), and 0.63 at scale. Subject–verb
  agreement sits at about 0.52 at small scale.
- Between control and class cut, BLiMP moves at most 0.017, while G moves 0.11–0.34. A single
  score over 67 paradigms dilutes a class-specific loss.
- G and BLiMP measure different syntax. G reads local slot classes, which our models learn well
  (0.96–1.00 for pronouns and modals). BLiMP's paradigms mostly turn on longer dependencies.

### 5.8 No arbitrary settings

Each setting of the first version was either derived from the data or measured against the
verdicts on the 22 labeled cuts of §5.6.

**Granularity.** One tokenizer was learned on F to 60,000 merges and cut at each granularity.

| Tokens/word | Merges | Classes found (of 5) | Controls with a class flag (of 12) | Controls with a head flag (of 12) | Dropped words |
|---|---|---|---|---|---|
| 1.35 | 30,526 | 5 | 1 | 0 | 8/8 |
| 1.40 | 20,608 | 5 | 0 | 0 | 8/8 |
| 1.50 | 11,381 | 5 | 0 | 0 | 8/8 |
| 1.60 | 7,220 | 5 | 1 | 1 | 8/8 |
| 1.70 | 4,944 | 5 | 0 | 3 | 8/8 |
| 1.80 | 3,548 | 5 | 0 | 0 | 8/8 |

- The verdicts hold over the whole range, and the control flags (6 of 144 readings) are near the
  designed rate.
- Between adjacent granularities, under-shown and within-noise swap in 26–37 of 330,000–690,000
  patterns. What moves is whether a rare pattern is read at all.
- 1.30 does not fit in 16 GB: 45,748 tokens to cluster. So the range has no measured edge on
  either side, and the tool's target, 1.50, sits inside it with margin.
- A fixed budget means nothing across corpora: 4,000 merges leave 1.76 tokens per word on F,
  while on 30 recipe documents the merges run out at 269, near whole words.
  The tool therefore learns to a granularity, not a budget, and refuses a budget outside the range.

**The class settings.** The coarse cut was forced from 8 to 64 classes, and the context from 250
to 4,000 tokens (51% to 87% of running tokens). At every value all five classes are found, no
control's class row is flagged, head flags on controls stay at 0–2 of 12, and all dropped words
are found. Class rows do not depend on the coarse cut at all: they are read on the whole tree,
and the class bar is 13.92 throughout. A head row is a head followed by one filler class, so a
different cut asks a different question of the same head; pattern-level agreement with the
12-class cut falls from 97% (8 classes) to 63% (64).

**The bar and the draws.** With 16 draws and the maximum as bar, 6 of 48 random cuts were flagged
against one draw set, and none with fresh draws for each cut: the bar was calibrated on average
over draw sets, but the maximum of 16 draws is itself noisy (5.6 to 9.6 across sets). A pool of
200 draws matched the Beta law of §3.4 at 16–128 draws. With 101 draws and k = 6, 2 of 48 are
flagged.

**What these replaced.** The diversity floor of 0.5 and the 3-sigma margin would have needed a
recalibration for every new corpus: when every token became a filler, the class floor of 0.5 nats
missed every removed class but one, and no floor separated the class shifts that cost the models
their syntax from those that did not. The minimum trunk of 20, swept from 2 to 40, changed no
verdict between 5 and 40 on F but moved both bars, and it would not carry to a corpus of another
size.

### 5.9 Without a reference

**Saturation alone does not flag a thinned slot.** The pronoun → verb pattern was thinned to
5–50% of its sentences at equal bytes, and each corpus was read on its own. The slot's missing
mass rises from 0.028 to about 0.05, but the median node of every corpus reads 0.40: any
tolerance that flagged the thinned slot would flag most of the corpus. Removed outright, the slot
reads 0.028 again, as saturated as the control.

**Siblings do.** Read with the sibling test of §3.5:

- In `head_cut`, *they* shows 3 verbs where a random split of its parent gives 26.5 (t 9.6, bar
  6.3). The tool flags *she* and *they*, and no pronoun of the control.
- **Replication.** Control's two random halves, read on control's own tree and classes so that
  every test is the same test, share 264 weak tests, of 332 and 397.
- On the recipe grammar, 11 of 14 restricted verbs are weak. The other three share their only
  ingredient with their sibling. No verb is weak in the diverse recipes.
- On FineWeb the tool flags 95 head rows in `control` and 120 in `head_cut`, in 2.5 minutes.

**What the flags are.** Most are heads that grammar narrows within their filler class: *to* takes
fewer fillers than the other prepositions (524 against 871), reporting verbs (*said, says, means*)
fewer than other verbs. They are weak in the corpus, which is what a model trained on it sees. A
reference reads them within noise, because the domain has them too.

**What it cannot see.** Thinning every pronoun at once leaves no sibling stronger: the pronoun node
reads 0.94–1.06 of its expected fillers when thinned to 25% or 10%, and `class_cut` shows its slot
19 times. In a synthetic subject–verb corpus, *she* and *they* restricted to two verbs form their
own class and lose their sibling.

**Rejected.** Comparing each head with its whole coarse class, adding a test of how often a head
takes each filler class, and grouping heads by context flagged 500–2,000 heads and 3,000–15,000
class rows in the control, every pronoun among them. Their halves shared 12–17% of the flags.

---

## 6. Discussion

**What the indicator certifies.**

- A flagged class pattern is a corpus fault. In our experiments, no configuration recovered it,
  and capacity made it worse. The remedy is data: text in which the class's heads take other
  fillers of their slot.
- An unflagged corpus lets the reference's syntax be reached at its size. A failure there points
  at the model.
- This division of labor is the practical point of the work. The data side certifies that the
  patterns are served. Finding the model that learns them is left to model engineering.

**Why class level.**

- Models share evidence across heads of a class. A head with a thin slot borrows from its
  siblings, which agrees with the indirect-evidence results of FiCT and the AANN study.
- Realistic filters thin heads but rarely empty classes. This explains why, among realistic cuts,
  class-level syntax followed size and topic rather than the score (§5.4).
- A diagnostic aimed at syntax must therefore read classes, while a diagnostic of which word a
  model will predict reads heads. Reading every node of the head tree, rather than one cut, lets
  the report name the group that carries a shift: the whole modal class in one cut, two
  possessives in another.

**Settings from the data.** A threshold set on the corpora where a method was developed is a
claim about those corpora. The indicator's bar instead says, for each corpus, how far a random
sample of the reference strays at its worst; the reader is left with one number to judge,
alpha, which is a tolerance and not a property of the data. The cost is time: 101 draws make a
diagnosis 4–9 minutes at our sizes, against about 4 minutes before.

**With and without a reference.**

- Without a reference, the corpus is judged by its own evidence: a head weaker than its siblings
  is weak for a model trained on it. This is the reading a practitioner has before they find a
  reference, and it flags real gaps (*she* and *they* in `head_cut`).
- A reference does two further things. It sees what the corpus lacks altogether, or lacks for a
  whole class, and it separates the corpus's weakness from the domain's.
- Both readings are relative. "Unflagged" means as good as a random sample of the reference of
  this size, or as good as a random share of the head's parent; it does not mean as good as the
  reference at any size.
- In our data, a random cut of a large, well-built corpus already preserves the patterns as well
  as any filter. The indicator's practical role is mainly to reject cuts and filters that bias
  syntax.

**Effect sizes.**

- Whole-model effects of slot damage are small in bits per byte (0.017 over a whole filter
  ladder).
- Per-pattern effects are large: 1–4.6 nats for planted heads, 0.13–0.33 in G for class removals.
- A per-pattern diagnostic is needed precisely because whole-corpus and whole-model numbers dilute
  local losses.

---

## 7. Limitations

- **Scale.** The models go up to 6.2M parameters, and the corpora up to about 10M words, on CPU.
  On BLiMP the models read 0.56–0.63. The claims concern local syntax learned by small models.
  Whether the class-level attribution holds for models that learn long-range syntax is untested.
- **Local syntax only.** Heads of one or two tokens and the next filler. Long-range dependencies
  (agreement across attractors, filler–gap, binding) are out of scope.
- **One language and one natural source**: English, FineWeb, plus a synthetic grammar. The
  settings were measured on FineWeb; that they hold on another domain is expected from how they
  are derived, not shown.
- **Classes tested at depth.** Three classes (pronouns, modals, possessives) trained under
  removal; two more (numbers, base verbs) checked on the corpus side. At scale, the pronoun
  result was not confirmed, because its test lost power.
- **Seeds.** Two seeds for most experiments; one seed for the new model configurations and the
  scale runs.
- **The test's level.** alpha = 0.06 is per family. A random corpus can be flagged in any of three
  families, so its chance of some flag is at most 18%; measured, it was 4% (2 of 48). Splitting
  alpha three ways would need about 300 draws. In the planted slices, one class row was flagged
  in each of two of seven unplanted slices.
- **The granularity range** is bounded by what we could measure: no edge appeared from 1.35 to
  1.80, and 1.30 does not fit in 16 GB.
- **Without a reference**, a pattern the corpus lacks, a class shown weakly as a whole, and a head
  the tree sets apart from its siblings are not seen.
- **D on plantings.** The dose fit of §5.2 (drop ≈ −0.68 · D) and the head-level D of §5.4 come
  from the first version's reading of word-like tokens.
- **Model size.** On the recipe grammar the diversity a model needs grew with model size. The
  indicator says where a corpus departs from its reference; how much departure a given model
  tolerates is not part of it.
- **Syntax, not semantics.** A corpus can pass and still fail a model on topic or meaning: a
  topic-narrow corpus failed models document-wide without any slot flagged.

---

## 8. Conclusion

We asked whether a corpus can be checked, before training, for the local syntactic patterns its
domain needs, so that a generalization failure can be attributed to the corpus or to the model.
The answer, within the limits above, is yes.

An indicator that compares each slot's fillers with random samples of a reference, at every node
of an induced tree of heads, flags corpus faults that no model configuration we tried recovers.
Where it flags nothing, a model reaches the reference's syntax. None of its settings is set by
hand: the granularity, the trunk read and the number of samples come from the data, and the test's
level is stated in every report. Without a reference, the same tree lets a corpus be judged by its
own evidence. Along the way, we found that models learn local syntax mostly at the class level,
which explains both why realistic data filters rarely break syntax and why a diagnostic must read
classes. The tool, code and every preregistered prediction, including the failed ones, are
released with the paper.

**Future work:**

- larger models and corpora, with GPU;
- long-range dependencies;
- more languages and domains, to test that the derived settings carry;
- the tolerance of a given model size to a flagged departure;
- a semantic counterpart for knowledge domains.

---

## Appendix A. Predictions that failed

Kept as written in the preregistration documents.

**Indicator and models (first version).**

- Recipe grammar: minimal pairs "cut the slowly" did not stay near 100% (0.25 with one filler).
  Corpus-size predictions failed at 500 documents.
- Plantings: one comparison head moved −0.51 nats. Whole-text loss rose 0.03 bits per byte
  (predicted < 0.02). Detection at k = 32 was still frequent (12/20).
- Topic-narrow corpus: flagged heads did not lose more. The loss was topical and document-wide.
- Subset selection: mild natural flags (diversity 0.26–0.50) did not cost the model. In r10,
  Spearman(z, drop) was 0.10 / 0.12 (bar 0.15).
- Dose: the mild end cost 0.38–0.57 nats (predicted ≤ 0.3). The threshold was crossed at 0.64.
- Divergence: on plantings, D ordered heads no better than diversity. Its first cut, κ, failed
  because of a sample-size floor.
- Cut choice: over ten cuts, Spearman(score, bits per byte) was 0.55 / 0.45 (predicted ≥ 0.5 on
  both seeds). With the current reading it holds (0.65 / 0.55).
- Ladders: the ladder that adds documents had Spearman 0.60. The per-head reading failed at dose
  25%.
- Syntactic evaluation of realistic cuts: all six predictions failed. This led to the class-level
  hypothesis.
- Class patterns: P2 −0.086 (bar −0.10), with an invalid first modal test. P4: noise between
  models exceeded the bar.
- Replication: the possessive head cut cost 57% (bar 50%).
- Tool: the pronoun class was missed (+0.47 against a 0.5 floor). The first `absent` rule gave
  79–114 false flags. One control gave 5 at m/2.
- BLiMP: F's models read below 0.60.
- Scale: the pronoun gap was 0.06 (bar 0.10), and the control was 0.09 below the reference.

**Settings from the data.**

- Granularity: adjacent granularities were predicted to agree on 90% of patterns; they agreed on
  85–94%, almost all of the difference being whether a pattern is read. With a trunk of 20, the
  numbers and base verbs were missed at 1.35 (predicted found over the whole sweep).
- Class floor: each class whose removal cost the models their syntax was predicted to read a
  larger excess than every one that did not. It failed: at 64 head classes over every token,
  pronouns and possessives shared one class with *the* and *a*, and moved it by +0.045 and
  +0.007. This led to class rows at every node.
- The 16-draw bar: 6 of 48 random cuts were flagged against one draw set (design: about 1 in 17).
  This led to alpha and 101 draws.
- Without a reference: saturation (missing mass) did not flag a thinned slot. Three groupings of
  heads gave flags that the corpus's halves did not share.
- Subset selection, rerun: both filters were predicted to flag at least 10 rows; they flagged 2
  and 1.

## Appendix B. Reproducibility

Every experiment has a folder under `experiments/` with its script, its results (JSON) and a
README. `docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md` holds the first version's designs,
predictions and outcomes, in order, and `docs/superpowers/specs/2026-10-11-settings-from-data-design.md` the
derivation of every setting. Results of the first version are kept beside the reruns
(`results/three_sigma/` in `experiments/subset_selection`, the first-margin tables in
`experiments/planted_gap`). `corpus-diagnose` and the Streamlit page are in
`src/corpus_generalization/`. Runs are CPU-only; a diagnosis takes 4–9 minutes at our sizes after
reading the reference once. Run times are in each README.

---

## References

**Verified at source for this draft:**

- Patil, A., Jumelet, J., Chiu, Y. Y., Lapastora, A., Shen, P., Wang, L., Willrich, C., &
  Steinert-Threlkeld, S. (2024). Filtered Corpus Training (FiCT) Shows that Language Models Can
  Generalize from Indirect Evidence. *TACL* 12, 1597–1615.
- Luo, J., & Steinert-Threlkeld, S. (2026). Exposure is Optional: Learning Unlike Coordination in
  Language Models. arXiv:2607.20251.
- Misra, K., & Mahowald, K. (2024). Language Models Learn Rare Phenomena from Less Rare Phenomena:
  The Case of the Missing AANNs. *EMNLP 2024*.
- Wei, J., Garrette, D., Linzen, T., & Pavlick, E. (2021). Frequency Effects on Syntactic Rule
  Learning in Transformers. *EMNLP 2021*, 932–948.
- Miranda, B., Lee, A., Sundar, S., Casasola, A., Schaeffer, R., Obbad, E., & Koyejo, S. (2023).
  Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural
  Language Data. arXiv:2306.13840.
- DataProphet: Demystifying Supervision Data Generalization in Multimodal LLMs (2026). *ICLR 2026*.
  arXiv:2603.19688.
- Wettig, A., et al. (2024). QuRating: Selecting High-Quality Data for Training Language Models.
  arXiv:2402.09739.
- Xie, S. M., Santurkar, S., Ma, T., & Liang, P. (2023). Data Selection for Language Models via
  Importance Resampling. arXiv:2302.03169.
- Albalak, A., et al. (2024). A Survey on Data Selection for Language Models. arXiv:2402.16827.
- Guo, Y., et al. (2024). Benchmarking Linguistic Diversity of Large Language Models.
  arXiv:2412.10271.
- Baayen, R. H. (2009). Corpus linguistics in morphology: morphological productivity. In
  *Corpus Linguistics: An International Handbook*.
- Marvin, R., & Linzen, T. (2018). Targeted Syntactic Evaluation of Language Models.
  arXiv:1808.09031.
- Hu, J., Gauthier, J., Qian, P., Wilcox, E., & Levy, R. (2020). A Systematic Assessment of
  Syntactic Generalization in Neural Language Models. arXiv:2005.03692.
- Warstadt, A., et al. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English.
  *TACL* 8.

**Recommended, to verify before submission:**

- Tarone, R. E. (1990). A modified Bonferroni method for discrete data. *Biometrics* 46.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing*. Wiley.
- Linzen, T., Dupoux, E., & Goldberg, Y. (2016). Assessing the ability of LSTMs to learn
  syntax-sensitive dependencies. *TACL*.
- Wenzek, G., et al. (2020). CCNet: Extracting High Quality Monolingual Datasets from Web Crawl
  Data. *LREC*.
- Penedo, G., et al. (2024). The FineWeb Datasets.
- Warstadt, A., et al. (2023). Findings of the BabyLM Challenge.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with
  Subword Units. *ACL*.
- Good, I. J. (1953). The population frequencies of species and the estimation of population
  parameters. *Biometrika*.
- Witten, I. H., & Bell, T. C. (1991). The zero-frequency problem. *IEEE Trans. Inf. Theory*.
- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *JASA*.
- Honnibal, M., et al. (2020). spaCy: Industrial-strength NLP.
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian knot: the moving-average
  type–token ratio (MATTR). *J. Quantitative Linguistics*.
- Li, J., et al. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models
  (distinct-n). *NAACL*.
- Friedman, D., & Dieng, A. B. (2023). The Vendi Score: A Diversity Evaluation Metric for Machine
  Learning. *TMLR*.
- Lake, B., & Baroni, M. (2018). Generalization without systematicity (SCAN). *ICML*.
- Kim, N., & Linzen, T. (2020). COGS: A Compositional Generalization Challenge. *EMNLP*.
- Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
