# Blame the Data, or the Model? Diagnosing a Domain Corpus's Syntax Before Training

**Marco Fura Mendoza**\
Pontificia Universidad Católica de Chile\
mjfura27@gmail.com

*Draft, October 2026. Not for distribution.*

---

## Abstract

When a language model fails to generalize a pattern of its domain, the cause can lie in the model
or in the corpus, and existing data-quality measures cannot tell these apart. A knowledge domain,
such as recipes, is a bounded subset of a language's patterns; we address its syntactic layer. We propose a
corpus-only indicator of local syntactic sufficiency. It reads every local syntactic pattern of a
corpus at two levels, a head followed by a grammatical class (*she* → verb) and a class of heads
induced from the corpus (pronoun → verb), and compares the variety of its fillers with
size-matched random samples of a reference corpus; without a reference, each head is compared
with its siblings in the corpus. No threshold is tuned by hand: the least evidence read and the
number of random samples are derived from the data, the tokenizer's granularity is set inside a
range measured in five domains, and the remaining settings are shown not to change the verdicts,
leaving only the test's level. We validate the indicator with preregistered corpus interventions and trained
models. On a synthetic grammar it predicts which patterns a model generalizes (Spearman
0.94–0.98), while held-out loss ranks the corpora in reverse. On FineWeb it finds planted gaps
without false alarms at the head level, at heads where models lose 1–4.6 nats on the true next word. Removing a
pattern from a whole class removes it from every member, while removing it from two heads costs
them only 25–57% as much: local syntax is learned mostly at the class level. When the indicator
flags a class pattern, no model configuration we tried recovers it; when it flags nothing, models of
up to 6.2M parameters reach the reference's syntax within 0.02. Unlike one-number corpus metrics, the indicator
localizes the damaged pattern. Its scope is adjacency syntax, validated with models of up to 6.2M
parameters.

**Keywords:** knowledge domains; domain-specific corpora; syntactic sufficiency; training data
diagnostics; syntactic generalization; filler diversity; language models

---

## 1. Introduction

A practitioner who trains a language model on a domain corpus and finds that it fails to
generalize faces a diagnostic question before any engineering one. Is the corpus missing
something, or is the model wrong for it? The two call for opposite actions. Collecting or
rebalancing data cannot fix a model that is too small or badly trained, and tuning the
architecture cannot fix a corpus that never shows a pattern with enough variety. Today this
question is answered, if at all, by training and inspecting.

We frame the question through the notion of a **knowledge domain**: a bounded subset of the
syntactic and semantic patterns of a language, such as recipes, which are mostly imperative
actions on ingredients. This paper addresses only the syntactic layer of a domain. By syntax we
mean the well-formed frame of a sentence regardless of its truth. "Cut the milk" is syntactically
correct; that it makes no sense is a matter of semantics and lies outside our scope. Our premise
follows a long-standing observation in corpus linguistics (the productivity measures of Baayen,
2009) and in language-model learning (Wei et al., 2021; Misra & Mahowald, 2024): a pattern is
learned as a pattern, and not as a memorized word, when the corpus shows its slot with enough
different fillers. We call a corpus **syntactically sufficient** for its domain when it shows
every local pattern of the domain in this way.

Existing tools do not measure this. Data-quality and selection methods, such as perplexity
filtering in the CCNet style (Wenzek et al., 2020), DSIR, QuRating and the diversity coefficient,
give one number per document or dataset, aimed at perplexity or downstream accuracy, and a single
score cannot say which pattern is missing. Held-out loss can mislead: on a controlled grammar we
show that it ranks corpora in reverse of syntactic generalization, because a restricted corpus is
restricted in its held-out split too and therefore looks more predictable. Finally,
filtered-corpus training (FiCT; Patil et al., 2024) and related studies remove a construction and
test what the trained model learns. These studies are post-hoc by design, and their main message,
that models generalize from indirect evidence, makes the question of which removals matter a real
one.

A diagnostic of this kind is only as trustworthy as its settings. A threshold tuned on the
corpora where the method was developed can produce, on a new corpus, flags that are artifacts of
the threshold. We therefore require every setting of the indicator either to be derived from the
data or to be shown not to change its verdicts, and we report the measurements (§5.8).

Our contributions are the following:

1. **An indicator computed on the corpus alone.** It reads local syntactic patterns at the head
   level and at every node of an induced tree of heads, comparing them with size-matched random
   samples of a reference or, without one, with sibling heads in the corpus (§3); it is packaged
   as a command-line tool and a user interface.
2. **Settings from the data.** A measured range of tokenizer granularity, a test level that sets
   the number of random samples, and a trunk read chosen by Tarone's criterion replace every
   hand-set threshold (§3.1, §3.4, §5.8).
3. **Causal validation by corpus interventions** in a synthetic grammar and in natural text,
   covering planted gaps, dose–response, realistic filters, and removal of a pattern from a head
   versus from its whole class (§5).
4. **A finding about the level at which syntax is learned.** A head that loses its slot keeps its
   syntax while its class keeps the pattern, whereas a class that loses the pattern loses the
   syntax of every member (§5.4).
5. **An attribution test.** A flagged class pattern is a corpus fault that no tried model
   configuration fixes, and an unflagged corpus lets small models reach the reference's syntax (§5.5);
   this is the guarantee a practitioner needs.
6. **Comparisons with one-number corpus metrics and BLiMP**, with an honest account of where they
   do and do not see the same thing (§5.7), together with a list of every failed prediction
   (Appendix A).

---

## 2. Related work

*Data quality and selection for pretraining.* Perplexity filtering against a reference language
model, as in the CCNet pipeline, is a standard quality filter. DSIR (Xie et al., 2023) selects
data by importance resampling on hashed n-gram features, and QuRating (Wettig et al., 2024) uses
language-model judgments of writing quality, expertise and educational value. The diversity
coefficient (Miranda et al., 2023) measures a dataset's variability through Task2Vec embeddings.
Training-free predictors of data influence, such as DataProphet (Qi et al., 2026), rank
supervision datasets before training. Albalak et al. (2024) survey the field. All of these
methods score documents or datasets as wholes. None reports which syntactic patterns a corpus
under-serves, and none separates a corpus fault from a model fault. We find that
perplexity-based filters are the only cuts of a corpus the indicator flags (§5.3).

*Learning syntax from manipulated corpora.* Wei et al. (2021) show that BERT applies
subject–verb agreement to pairs never seen together, but struggles with infrequent items and
strong priors. FiCT (Patil et al., 2024) removes a construction from the training corpus and finds
that both LSTMs and transformers generalize to it from indirect evidence; Luo &
Steinert-Threlkeld (2026) extend this result to unlike coordination. Misra & Mahowald (2024) show
that a rare construction (AANN, "a beautiful five days") is learned from related, more frequent
ones, and is learned better with more variability in the input. Our results agree with these
findings at the head level: a pronoun that loses its verb slot keeps its syntax through its
siblings. We add the class level, where removal does prevent learning, and an indicator that says
before training which removals matter.

*Productivity and diversity measures.* Baayen's (2009) productivity measures use hapax legomena
to estimate how open a morphological category is. The Good–Turing missing mass (Good, 1953),
singletons over trunk, applies the same idea to a syntactic slot; we show that on its own it does
not flag a thinned slot (§5.9). Measures of the lexical, syntactic and semantic diversity of model
outputs (Guo et al., 2025) assess generated text rather than training corpora, and they are not
tied to what a model learns.

*Multiple testing with discrete statistics.* The indicator tests tens of thousands of patterns at
once. Its bar is the maximum statistic over random samples, a resampling family-wise bar (Westfall
& Young, 1993), and patterns too rare ever to be flagged are left out of the family, following
Tarone (1990).

*Targeted syntactic evaluation.* Minimal pairs (Linzen et al., 2016; Marvin & Linzen, 2018), the
systematic assessment of Hu et al. (2020) and BLiMP (Warstadt et al., 2020) evaluate trained
models. Our model-side reading is a minimal-pair score built to exclude semantics: unseen
class-valid fillers compete against frequency-matched ungrammatical ones, with classes taken from
an external tagger (§4.3). We also report BLiMP.

---

## 3. The indicator

Figure 1 summarizes the indicator; the subsections below define each step.

![**Figure 1.** The indicator. The corpus is tokenized at 1.50 tokens per word and its tokens are clustered into one tree: every node is a class of heads, and one cut gives the filler classes. Each head pattern and class pattern is read (a) against 101 size-matched random samples of a reference, with a bar taken from the samples' own leave-one-out maxima at level α, or (b) without a reference, against random splits of the head's parent in the tree. The inset is schematic.](figures/fig1_method.png){width=100%}


### 3.1 Units: tokens, transitions, classes

The corpus is tokenized with a byte-level BPE (Sennrich et al., 2016) learned on the domain's
text, which is the reference's text or, without a reference, the corpus's own. The tokenizer is
trained to the fewest merges that leave 1.50 tokens per word; on the FineWeb pool this is 11,381
merges. We measured that the verdicts hold from 1.35 to 1.70 tokens per word in five domains, on
domains of at least 14.4 MB, except where a dropped word's uses sit in a few documents (§5.8).
Every token takes part in the analysis: any token found in at least two documents can be a head or
a filler, whether it is a word, a word piece, punctuation or a number.

A transition is the head's last token followed by the filler. We call it *syntactic* when the
filler starts a new pre-token of the tokenizer (a new word, punctuation run, number or
contraction), and *lexical* when the filler continues the word of the token before it
(` model` → `s`). Because no merge crosses a pre-token boundary, this rule is exact. The two kinds
are counted as separate patterns, and the syntactic rows form the diagnosis.

Classes come from a single tree. Each token is described by its left and right neighbor
distributions over the most frequent 1,000 tokens. The rows are square-rooted, so that Euclidean
distance is Hellinger distance, and are clustered with Ward's (1963) method into one tree. Filler
classes are obtained by cutting the tree at the largest jump in merge height between 8 and 64
classes, which gives 12 classes on FineWeb. On the recipe grammar this cut recovers the grammar's
categories exactly. On natural text the classes are positions in the sentence rather than parts of
speech (§5.8): nouns and verbs come out, while determiners, possessives and adjectives share the
class of what comes before a noun. A finer cut starts grouping fillers by meaning (what can be poured against
what can be chopped), which would bring semantics into a syntactic measure. Head classes, by
contrast, require no cut: every internal node of the same tree is a class of heads, and class
patterns are read at every node (§3.2).

### 3.2 Patterns

A head pattern is a head of one or two tokens followed by a filler of class Y, together with its
transition kind. Occurrences are counted in distinct contexts, defined by the token before the
head and the token after the filler, so that exact copies add nothing. We call a pattern's number
of distinct contexts its trunk. A class pattern is a node of the head tree together with the
fillers that follow any of its one-token heads, pooled.

### 3.3 Readings

The first reading is diversity, defined as fillers / expected fillers, where expected fillers =
Σ_y [1 − (1 − p_y)^n], with n the trunk and p_y the class shares of the corpus. It compares heads
of different traffic against free draws from the class.

The second reading is the divergence D = KL(R ‖ S), in nats. Here R is the reference's filler
distribution for the slot, and S is the corpus's, with Witten–Bell backoff (Witten & Bell, 1991)
to the corpus's filler unigram, adding one count per reference filler. D has the units of the
loss: a model that learned S and is scored on text from R loses on average KL(R ‖ S) nats per slot
occurrence. On planted gaps the drop in the model's log probability is about −0.72 · D
(R² = 0.78, §5.2).

Every reading depends on the corpus's size, so a smaller corpus always reads worse. We therefore
compare each corpus with size-matched draws: random whole-document samples of the reference of
the same size in bytes. The reference is tokenized and classified once and reused for every
corpus diagnosed against it.

### 3.4 Verdicts with a reference

Each pattern's t is how far the corpus falls from the draws' mean, in units of the draws' spread.
The reading is diversity for head rows, trunk for *absent*, and D for class rows. The spread
includes the reading's own rounding, since diversity moves in steps of one filler and a trunk in
steps of one: sd' = √(sd² + step²/12).

The bar comes from the draws. Each draw in turn is left out and read against the others, and its
largest t over all patterns is recorded. A corpus is flagged only where it strays further than
the k-th largest of these maxima, so that a random sample of the reference passes the bar with
probability about k / (n + 1).

The one setting the user states is alpha, the chance that a random sample of the reference is
flagged in a family of verdicts. Its default is 0.06 per family, the families being head rows,
*absent* and class rows. Alpha is the test's level, a tolerance rather than a property of the
data. The number of draws follows from alpha. The k-th largest of n draw maxima is itself noisy:
over draw sets, the chance of passing it follows Beta(k, n + 1 − k). The tool uses the fewest
draws that keep that chance within alpha/2 to 1.5 · alpha for 80% of draw sets, which gives 101
draws and k = 6 at alpha 0.06. The trunk read also comes from the draws, by Tarone's criterion. A
pattern too rare to be flagged even if the corpus showed a single filler adds no verdict, and its
noise raises the bar for every other pattern. The least trunk read is therefore the one that
leaves the most patterns that could be flagged.

A head row is *under-shown* when its diversity lies below the draws' mean by more than the bar. It
is *absent* when its trunk lies below the draws' mean trunk by more than the *absent* bar: a
random sample of this size would show the pattern, and the corpus shows too little of it.
Otherwise it is *within-noise*, and below the trunk read it is *unread*.

A class row reads D over the node's pooled fillers against the draws, and a node is
*under-shown* when its t exceeds the class bar. Of each branch, the node with the highest t is
reported, so that the report names the smallest group of heads that carries the shift. Each class
row also lists its *short heads*, the members whose own rows are under-shown or absent.

### 3.5 Verdicts without a reference

A pattern the corpus shows weakly is weak for a model trained on it, whatever the domain does. A
corpus alone can reveal this by comparing a head with its siblings. Each child of a node of the
head tree, whether a head or a class of heads, is compared with its parent, per filler class and
transition kind. The null asks how many distinct fillers this child would show had the parent's
occurrences of that class fallen to its two children at random. To answer it, 101 random splits
are drawn exactly, without replacement, at the child's trunk. The bar is built from the splits as
in §3.4, at the same alpha, and only tests that a single filler could fail are read.

Two further restrictions apply. Only children of a node inside one filler class are tested: above
the class cut, the two sides of a split are different categories (`,`/`the`/`to` against
`.`/`:`), and every such test read as weak. Heads of two tokens are placed by their first token.

A pattern is *weak* when it lies past the bar, and *rare* when it is shown fewer times than any
slot the test can read. Otherwise it is *within-noise*, or *unread* when it has no sibling to read
against.

Three cases are invisible without a reference. A pattern the corpus lacks altogether, and a class
shown weakly as a whole, leave no stronger sibling to compare with. A head restricted enough for
the tree to set it apart from its siblings loses its sibling. The report states these limits and
recommends a reference from the same domain. A reference also separates the corpus's own weakness
from the domain's: most weak patterns of a natural corpus reflect its grammar (` to` against the
other prepositions, reporting verbs), and a reference reads them within noise.

### 3.6 The tool

`corpus-diagnose` (Python) takes a corpus and, optionally, a reference. It writes JSON and
Markdown reports covering a summary, warnings, classes, head rows, class rows and settings,
including alpha and the number of draws. When no merge budget is given, it learns the domain's
tokenizer to 1.50 tokens per word, resuming the BPE in doubling rounds. A budget given with
`--merges N` is checked before the diagnosis: outside 1.35–1.70 tokens per word the command
refuses and names the granularity reached, unless `--force` is passed. It warns when the domain,
the reference or else the corpus, holds less than 14.4 MB (§5.8). A Streamlit page runs the same
diagnosis on any catalogued dataset.

---


## 4. Experimental setup

### 4.1 Corpora

We use a synthetic recipe grammar, natural text from FineWeb, and a set of interventions on the
latter.

The recipe grammar is a probabilistic grammar with a closed lexicon (imperative verbs, countable
ingredients, adjectives, adverbs, tools) and four sentence rules. A single knob changes how many
ingredients each verb is ever seen with (1, 2, 4, 8, 16, 20, 24 or all), while every ingredient
still occurs in other sentences. The reserved verb–ingredient combinations are never seen and
serve as test items.

Our natural text is drawn from FineWeb (Penedo et al., 2024). The main pool F consists of 4,600
documents (14.4 MB, ≈2.4M words) from the first 5,000 streamed documents; the other
400 documents are held out for evaluation. A 6.5 MB sample, divided into 8 slices, is used for
plantings. BigF is the first 20,000 documents minus the evaluation set (57.8 MB, ≈10M words).

The interventions are of four kinds. In plantings, the followers of chosen heads are replaced by
one of their k most frequent fillers, or the sentences in which they take another filler are
removed. Document cuts at 10–50% of F are random or made by perplexity filters or length filters.
Sentence-level cuts of a 50% cut of F (r50) remove every sentence in which a listed head is
immediately followed by a spaCy-tagged class (Honnibal et al., 2020); they are made at the class
level, at the head level (two members only) and as random controls, all at matched bytes. Finally,
the *labeled cuts* for the tool comprise the class and head cuts of pronouns and modals,
possessives, numbers and base verbs (8 cuts), 12 random controls, and two cuts that drop four
words each.

### 4.2 Models

The model is a decoder with the GPT-2 layout (Radford et al., 2019) and a tied head, trained on
CPU, in four configurations:

- **small:** 2 layers, 128 wide, 4 heads, context 128, 1.9M parameters, 3,000 steps of 32 × 128
  tokens, learning rate 1e-3;
- **medium:** 4 layers, 256 wide, 6.2M parameters, learning rate 5e-4;
- **long:** small, 9,000 steps;
- **scale:** medium, 30,000 steps on BigF, about 5 hours per run.

Each corpus learns its own BPE for the models (11,563 merges on FineWeb, a fixed budget; the
indicator learns its own, 11,381 merges at 1.50 tokens per word, §3.1), and the best checkpoint is chosen on a
held-out fraction. We use seeds 0 and 1 unless noted. On the recipe grammar, whose tokenizer stops at 1,000 merges, the small and medium models have 0.5M and 3.3M parameters.

### 4.3 Model-side readings

We read trained models in four ways. The *drop* of a head is the mean, over its evaluation
occurrences, of log p_model(true next word) − log p_reference model, in nats. It is a
filler-level reading, and therefore mixes syntax with semantics and topic.

On the recipe grammar, the *generalization ratio* is the probability the model gives, after
"⟨bos⟩ cut the", to ingredients never seen with that verb, relative to their share; 1 means that
the pattern is learned and 0 that it is memorized.

On natural text, *G* is read after a real evaluation prefix ending in head h. It is the share of
comparisons in which the model prefers a word of h's slot class that never follows h in the
reference over a word of an ungrammatical class, matched in log frequency. Truth does not enter:
after "she", any past-tense verb is valid. Classes come from spaCy (`en_core_web_sm`), never from
the indicator. The grammatical contrasts are:

- pronouns → past-tense verbs (VBD), against nouns and adjectives;
- modals → base verbs (VB), against nouns and adjectives;
- possessives → nouns, against adverbs and VBD/VBZ verbs.

Finally, we score *BLiMP* (67 paradigms × 1,000 pairs) by total log probability.

### 4.4 Protocol

Every experiment's design, predictions and thresholds were written in the design documents before
it ran. Construction parameters were changed only from corpus-side counts, before any model was
scored, and each change is recorded. Failed predictions are kept and listed in Appendix A.

The indicator was first built with hand-set thresholds (a diversity floor of 0.5, a 3-sigma
margin over 8 draws, a minimum trunk of 20, a class floor of 0.5 nats). Those were replaced by the
settings of §3.4–3.5 (§5.8), and every corpus-side experiment was rerun with the final tool. Models
were not retrained, because the corpora are the same and only their reading changed. The earlier
version's results are collected in Appendix B.

---

## 5. Results

### 5.1 Held-out measures do not see syntactic insufficiency (recipe grammar)

| Measure | Result |
|---|---|
| Per-verb coverage vs. generalization ratio | Spearman 0.891 (one seed); 0.940–0.975 over 3 seeds × 2 sizes |
| Mean ratio, all ingredients vs. one per verb | 0.909 vs. 0.011 |
| Held-out loss and n-gram predictability | Rank the variants in **reverse** on every seed and size (Spearman 0.98–1.00 between loss and ratio) |

The model generalizes late. The small model's ratio at 16 fillers is only 0.17–0.19, and it
crosses 0.5 at coverage 0.67–0.80. The larger model needs more, reaching 0.554 against 0.901 at
24 fillers, so that "sufficient" depends on model size, a point we return to in §6. Minimal pairs
of the type "cut the tomato / cut the slowly", meant as an easy control, tracked the ratio (0.25
with one filler) instead of staying near 100%. Figure 2 plots both readings.

![**Figure 2.** Recipe grammar. (a) Per-verb coverage of the ingredient slot against the generalization ratio, over three seeds and two model sizes (points: verbs; lines: variant means). Coverage predicts which verbs generalize (Spearman 0.94–0.98), and the larger model needs more coverage. (b) Across corpus variants (labels: ingredients per verb), lower held-out loss goes with lower generalization: the most restricted corpora are the most predictable.](figures/fig2_recipe.png){width=100%}


### 5.2 Head level on natural text: planted gaps and dose

*Detection, with the final tool.* We planted five heads of a FineWeb sample (*people, she, But,
only, even*) in one slice and diagnosed each slice against the sample.

| Planting | Heads found (of 5) | Other rows flagged |
|---|---|---|
| sentences removed, k = 2 | 4 | 0 |
| fillers replaced, k = 2, 4, 8, 16 | 5 at each k | 0 |
| fillers replaced, k = 32 | 4 | 0 |
| 7 unplanted slices | 0 | 0 |

Every found head is found through its row of pre-noun fillers (the class of *the*, *a*,
possessives and adjectives, §5.8), at diversity 0.03–0.17 for
k = 2 and 0.27–0.35 for k = 32, against about 0.9 in the unplanted slices. Most are also read
*absent* in a second filler class. No row outside the planted heads is flagged in any slice, of
2,900–6,600 rows read. The class level flags one class row in each of two of the seven unplanted
slices (§7).

*Model cost.* Models trained on the planted corpora give the true next word 2.17–4.60 nats less
(k = 2) and 1.12–3.60 nats less (k = 16) at every planted head and seed, while comparison heads
stay within 0.3 nats in 19 of 20 readings. The whole-text loss moves only 0.03 bits per byte: it
registers the damage, but does not say where.

*Dose.* We planted fourteen heads to diversity 0.1–0.7, with two assignments and two seeds. The
loss is monotone in the dose (Spearman 0.76–0.92), and heads at 0.1 lose 2.8–3.3 nats. The mild
end is not free: heads at diversity 0.5–0.7 lose 0.38–0.57 nats; diversity is therefore not the effect size. D is
the effect size (drop ≈ −0.72 · D, R² 0.78), although on plantings
it orders heads no better than diversity. Figure 3 shows every planted head.

![**Figure 3.** Dose and response on FineWeb. Fourteen heads planted to diversity 0.1–0.7 (two assignments, two seeds each), against the drop in the log probability the model gives the true next word at the head, relative to a model trained on the unplanted corpus. The loss is monotone in the dose (Spearman 0.76–0.92) and does not vanish at the mild end. Unplanted comparison heads mostly stay near zero.](figures/fig3_dose.png){width=60%}


### 5.3 Realistic cuts of a corpus

Five cuts of 10–50% of F were diagnosed against F, and six models were trained per seed (§4.2).

| Cut | Tokens/word | Under-shown | Absent | Class rows flagged |
|---|---|---|---|---|
| random 50% | 1.49 | 0 | 0 | 0 |
| random 25% | 1.49 | 0 | 0 | 0 |
| random 10% | 1.52 | 0 | 0 | 0 |
| least Wikipedia-like 25% | 1.39 | 2 | 11 | 4 |
| most formulaic 25% (lowest perplexity) | 1.61 | 1 | 4 | 1 |

*Random cuts read as random.* They flag nothing, neither rows nor classes.

*Filters are the only cuts flagged, but mildly.* The Wikipedia-like filter flags "," and ":"
before the pre-noun class and labels, and the formulaic filter flags ". Our". Their absent rows are a
few thin heads ("50", " a day", " they should"). We had predicted at least 10 under-shown rows for
each filter, and both predictions failed.

*The z of a head predicts what the cut costs the model better than its frequency.* In the 10%
cut, Spearman(z, drop) over 416 heads is 0.157 / 0.181 (two seeds), and that of log trunk alone
is 0.037 / 0.018.

*Filters move slots.* A cut's score is its heads' excess E = D − mean D over 8 random cuts of F
of its size, weighted by F's filler shares, and z divides it by the draws' leave-one-out spread.
The perplexity filter that keeps the 25% most formulaic documents reads z = +18.6, the one that
keeps the documents least like Wikipedia reads z = +24.8, and the CCNet-style Wikipedia filter
reads z = +9.7. Random cuts read |z| ≤ 1.5.

*The model pays where the excess points.* Within a filtered cut, heads with larger E lose more
than a random cut of the same size costs: Spearman(E, drop − random drop) ranges from −0.24 to
−0.37, on six filters and both seeds. Raw D's apparent success on random cuts was sample size
alone. In the 10% cut, D reads −0.42 / −0.45 against the drop, and E only −0.05 / −0.11.

*Choosing among cuts.* Over ten 25% cuts, the corpus score orders the models' bits per byte only
partly: Spearman 0.65 / 0.55, against a prediction of at least 0.5 on both seeds. Document count, a proxy for
topical coverage, matters as much (−0.47 / −0.44), and the score orders what document count
leaves at 0.58 / 0.56. Along one filter's dose, with documents held fixed, the score orders the
models (Spearman 0.90); when the dose also adds documents, it does not (0.60).

### 5.4 Syntax is learned at the class level

*First reading, negative.* Among realistic cuts, G, read then against any open class and
before the contrasts of §4.3, followed size (F 0.748, 25% cuts
0.700–0.736, 10% 0.69) and document count, not the score (−0.16 / −0.33). G per head reproduces
across seeds (Pearson 0.87–0.91). The reason, which we found next, is that realistic filters thin
a head's slot but never empty the pattern of its class.

*Head versus class removal.* From r50, at matched bytes, we build three corpora. The class cut
removes every sentence where a pronoun (*I you he she we they*) or a modal is followed by a verb;
the head cut removes only those of *she, they, can, will*; and the control removes random
sentences.

| Class | Class cut ΔG | Removed heads: head cut / class cut | Siblings in head cut |
|---|---|---|---|
| pronouns | −0.195 | −0.044 / −0.177 (25%) | −0.028 |
| modals (VB items) | −0.126 | −0.040 / −0.081 (49%) | −0.008 |
| possessives | −0.303 | −0.194 / −0.343 (57%) | +0.046 |

Removing the class pattern removes the syntax of every member. Removing it from two heads costs
them a quarter to a little over half as much, and leaves their siblings intact.

The class level of the indicator agrees with the models. In the class cut the tool reports the
modal node (t 319) and the pronoun node *I you we they* (t 254), against a class bar of 13.9. In
the head cut it reports no pronoun node, and a five-modal node holding *can* and *will* at t 33,
a tenth of the class cut's. Head-level D, by contrast, says *she* and *they* lost nearly the same in
both cuts (*she* +1.60 / +1.73, *they* +1.28 / +1.39, head cut / class cut, against the control).

Syntax is therefore learned at both levels, mostly at the class. How much a head's own evidence
adds depends on the class: least for pronouns, most for possessives.

### 5.5 Attribution: corpus or model?

The flagged class cuts and their unflagged size-matched controls are trained in three
configurations (small, medium, long), alongside F. The tool flags the class rows of each class cut
(pronouns and modals t 254 and 319, possessives t 165), and no class row of any control.

| Class | Best on F | Best on control (unflagged) | Best on flagged cut |
|---|---|---|---|
| pronouns | 0.981 | 0.961 | 0.846 |
| modals | 0.999 | 0.988 | 0.870 |
| possessives | 0.872 | 0.859 | 0.600 |

On the flagged cuts, no configuration recovers the class (0.13–0.27 below F). More capacity makes
it worse: pronouns go 0.85 → 0.75 → 0.68 and modals 0.87 → 0.80 → 0.75, as a larger model
memorizes the pattern's absence more firmly. On the unflagged controls, the best control is within
0.02 of F. On a smaller unflagged cut (10%), changing the model moved G by up to 0.06; that is the
engineer's lever.

*At four times the data.* BigF (57.8 MB against F's 14.4 MB), its class cut and its control were trained with the medium model
for 30,000 steps.

| Model | BLiMP | G pronouns | G modals |
|---|---|---|---|
| BigF | 0.629 | 0.910 | 0.902 |
| control | 0.632 | 0.822 | 0.895 |
| class cut | 0.622 | 0.761 | 0.565 |

BLiMP rises from 0.57 to 0.63. The unflagged control no longer reaches F's syntax within
0.02: its pronouns read 0.09 below BigF's. For modals, the attribution result is stronger at scale: the cut
leaves modals near chance. For pronouns, the gap shrinks to 0.06, and our predictions failed. At
this data size almost every frequent past-tense verb follows each pronoun in the reference, which
leaves 3–7 unseen items per pronoun, and the per-head readings become erratic. We cannot separate
low test power from a real reduction. More data also brings more indirect evidence (noun
subjects), as FiCT reports. Figure 4 collects the class-level results of §5.4 and §5.5.

![**Figure 4.** Corpus or model. (a) G of models trained on the full pool F, on unflagged size-matched controls, on the flagged class cuts and on an unflagged 10% cut, across model configurations; "scale" is the medium model trained on four times the data (BigF and its cuts). No configuration recovers a flagged class, while unflagged controls stay within 0.02 of F at small scale (at scale, the control's pronouns read 0.09 below BigF's). (b) Change in G against the control: removing a pattern from the whole class costs every member, while removing it from two heads costs them 25–57% as much and leaves their siblings intact.](figures/fig4_attribution.png){width=100%}


### 5.6 The tool's detection

We diagnosed the 22 labeled cuts of §4.1 against F at 1.50 tokens per word. The class rows find
every removed class, each in its own branch of the tree, and flag no control.

| Cut | Node reported | t (class bar 13.9) |
|---|---|---|
| pronoun and modal class cut | 8 heads, among them *can will would may could should* | 319 |
| | *I you we they i* | 254 |
| possessive class cut | *your his my their* | 165 |
| number class cut | *two … seven* | 104 |
| base-verb class cut | *get make see take find leave* | 101 |
| pronoun and modal head cut | *can will would could should* | 33 |
| possessive head cut | *his their*, exactly the removed heads | 85 |
| number and base-verb head cuts | nothing | |
| 12 random controls | nothing | |

The head rows separate cuts from controls in the same way: the class cut and the head cut flag 39
and 13 head rows, and the 12 controls flag none. The *absent* rows find every one of the 8 dropped
words. On 8 fresh random controls, two rows are read absent, one in each of two controls: ` luck`,
shown 0 times against the draws' 7.0 ± 1.8, and `Maybe`.

Finally, we read 48 random whole-document cuts of F, which are random samples of the reference,
against one fixed set of draws. They yield 2 head flags (4%), no class row and no *absent*. alpha
is 0.06 per family.

### 5.7 Comparison with corpus metrics and with BLiMP

*One-number metrics.* On the class cuts, the head cuts and eight random controls, we computed
perplexity under F's bigram and under a Wikipedia bigram, unigram KL from F, MATTR (Covington &
McFall, 2010), distinct-2 (Li et al., 2016), and the Vendi score (Friedman & Dieng, 2023) of
MiniLM sentence embeddings (Wang et al., 2020; Reimers & Gurevych, 2019). The planned test (3 sd
of the controls) flags at most one class cut per metric. That test was lenient: the controls mixed
sizes, and the mean perplexity had an outlier. In a size-matched reading, which we treat as
exploratory, unigram KL and distinct-2 do register the two large class cuts, which removed 18,126
and 10,315 sentences (KL +0.019 and +0.011 against noise of about 0.001). They stay within noise,
however, on the numbers (1,972 sentences) and the base verbs (3,642), which the indicator flags at
t 104 and 101, and no metric says which pattern changed. The claim these results support is
therefore localization and sensitivity to small, pattern-specific losses, not that other metrics
see nothing.

*BLiMP.* Our small and medium models read 0.56–0.58 on BLiMP (chance 0.5), and 0.63 at scale;
subject–verb agreement sits at about 0.52 at small scale. Between control and class cut, BLiMP
moves at most 0.017, while G moves 0.11–0.34: a single score over 67 paradigms dilutes a
class-specific loss. G and BLiMP also measure different syntax. G reads local slot classes, which
our models learn well (0.96–1.00 for pronouns and modals), whereas BLiMP's paradigms mostly turn on
longer dependencies.

### 5.8 No arbitrary settings

Each hand-set threshold of the earlier version (§4.4) was either derived from the data or measured
against the verdicts on the 22 labeled cuts of §5.6.

*Granularity.* One tokenizer was learned on F to 60,000 merges and cut at each granularity.

| Tokens/word | Merges | Classes found (of 5) | Controls with a class flag (of 12) | Controls with a head flag (of 12) | Dropped words |
|---|---|---|---|---|---|
| 1.35 | 30,526 | 5 | 1 | 0 | 8/8 |
| 1.40 | 20,608 | 5 | 0 | 0 | 8/8 |
| 1.50 | 11,381 | 5 | 0 | 0 | 8/8 |
| 1.60 | 7,220 | 5 | 1 | 1 | 8/8 |
| 1.70 | 4,944 | 5 | 0 | 3 | 8/8 |
| 1.80 | 3,548 | 5 | 0 | 0 | 8/8 |

The verdicts hold over the whole range, and the control flags (6 of 144 readings) are near the
designed rate. Between adjacent granularities, under-shown and within-noise swap in 26–37 of
330,000–690,000 patterns; what moves is whether a rare pattern is read at all. A granularity of
1.30 does not fit in 16 GB, as it leaves 45,748 tokens to cluster. On F alone, the range has no
measured edge on either side; its edges come from the other domains, below. A fixed
budget, by contrast, means nothing across corpora: 4,000 merges leave 1.76 tokens per word on F,
while on 30 recipe documents the merges run out at 269, near whole words. The tool therefore
learns to a granularity, not a budget, and refuses a budget outside the range.

*Domains, size and pool.* We repeated the reading at 11 granularities from 1.30 to 2.50 on five
corpora: TinyStories, FineWeb, Wikipedia, and two knowledge domains, PubMed abstracts and the facts
of European Court of Human Rights cases. Each corpus was read at 0.9, 3.6, 14.4 and 57.6 MB, on
three disjoint pools at each size up to 14.4 MB and one at 57.6 MB, each pool with its own 22
labeled cuts: 550 readings in all. A granularity holds when every removed class is found and every
dropped word that is one token is read absent.

| Size | Pools where 1.50 holds | Range common to all corpora |
|---|---|---|
| 0.9 MB | 1 of 15 | none |
| 3.6 MB | 6 of 15 | none |
| 14.4 MB | 12 of 15 (14 of 15¹) | 1.35–1.70 |
| 57.6 MB | 5 of 5 | 1.35–2.00 |

¹ Leaving out a dropped word that no granularity of the pool reads absent (see below).

From 14.4 MB, the verdicts hold from 1.35 to 1.70 in all five domains, on every pool but one once
unresolvable dropped words are left out¹; FineWeb's third pool fails on one dropped word (below).
1.30 fails in FineWeb's second pool and PubMed's merges run out before it, and 1.80 fails in two of
legal's three pools. The upper edge moves with the pool and
grows with size: at 57.6 MB, Wikipedia holds to 2.50 and legal to 2.20. False flags on the
controls stay under twice the test's level in every one of the 50 corpora. The tool's range is
therefore 1.35–1.70, and 1.50 stays inside it.

Below 14.4 MB the readings are erratic, and the reference is the small pool itself. Removed
classes are missed in 86 of the 134 failing readings at 0.9 MB, most often the base verbs and the
numbers. The class tree's settings are not the cause: the noise floor of its cut never binds, and
a context matched by count to the 14.4 MB measurement recovered TinyStories' possessives but none
of Wikipedia's or legal's classes. The tool therefore warns when its domain holds less than 14.4
MB.

The failures at 14.4 MB are single dropped words whose uses sit in few documents: *assessment* has
59% of its uses in one FineWeb document, *retrieved* 68% in three, *oil* 80% in one Wikipedia
article. Against random whole-document draws, the absence of such a word looks like a draw that
missed its document, at every granularity in the range (*retrieved* is read absent only at
1.30). The eight words chosen by hand for F are spread over 5%
to 15% of the documents, and none failed.

*The classes against a tagger.* We compared the cut and the tree with spaCy's universal tags
(`en_core_web_sm`, never the indicator) on one 14.4 MB pool of each domain at 1.50. A whole-word
token takes the tag that holds 90% of its 20 or more occurrences; these tokens are 39–46% of
running tokens, the rest being word pieces, punctuation, numbers and ambiguous words. Against tags
shuffled over the same tokens, the cut's purity is 0.67–0.82 (shuffles at most 0.51) and its
adjusted Rand index 0.39–0.69 (at most 0.03). Nouns and verbs each have a node of the tree at F1
0.76–0.92 in all five domains, adjectives at 0.64–0.81. Closed classes are not classes of their
own: *the*, *a*, possessives and adjectives share the class of what comes before a noun in
FineWeb, PubMed and legal, and *of*, *and*, *to* share the verbs' class in FineWeb, Wikipedia and
PubMed. Every tag read beats every shuffle in every domain. The chosen cut ranks 2nd to 11th of
the 57 cuts from 8 to 64 classes by adjusted Rand index. The classes are therefore distributional
positions, and a removal confined to one closed class (only the possessives) is read at a node
inside its class, which is why class rows are read at every node.

*The class settings.* We forced the coarse cut from 8 to 64 classes, and the context from 250 to
4,000 tokens (51% to 87% of running tokens). At every value all five classes are found, no
control's class row is flagged, head flags on controls stay at 0–2 of 12, and all dropped words
are found. Class rows do not depend on the coarse cut at all: they are read on the whole tree, and
the class bar is 13.92 throughout. A head row, in contrast, is a head followed by one filler class,
so a different cut asks a different question of the same head; pattern-level agreement with the
12-class cut falls from 97% (8 classes) to 63% (64).

*The bar and the draws.* With 16 draws and the maximum as bar, 6 of 48 random cuts were flagged
against one draw set, and none with fresh draws for each cut. The bar had been calibrated on
average over draw sets, but the maximum of 16 draws is itself noisy (5.6 to 9.6 across sets). A
pool of 200 draws matched the Beta law of §3.4 at 16–128 draws. With 101 draws and k = 6, 2 of 48
are flagged.

*What these replaced.* The diversity floor of 0.5 and the 3-sigma margin would have needed a
recalibration for every new corpus. When every token became a filler, the class floor of 0.5 nats
missed every removed class but one, and no floor separated the class shifts that cost the models
their syntax from those that did not. The minimum trunk of 20, swept from 2 to 40, changed no
verdict between 5 and 40 on F but moved both bars, and it would not carry to a corpus of another
size.

### 5.9 Without a reference

Saturation alone does not flag a thinned slot. We thinned the pronoun → verb pattern to 5–50% of
its sentences at equal bytes and read each corpus on its own. The slot's missing mass rises from
0.028 to about 0.05, but the median node of every corpus reads 0.40, so any tolerance that flagged
the thinned slot would flag most of the corpus. Removed outright, the slot reads 0.028 again, as
saturated as the control.

Comparing a head with its siblings does flag a thinned slot. Read with the sibling test of §3.5,
*they* in the head cut of §5.4 shows 3
verbs where a random split of its parent gives 26.5 (t 9.6, bar 6.3). The tool flags *she* and
*they*, and no pronoun of the control. The flags replicate: the control's two random halves, read
on the control's own tree and classes so that every test is the same test, share 264 weak tests,
of 332 and 397. On the recipe grammar, 11 of 14 restricted verbs are weak; the other three share
their only ingredient with their sibling, and no verb is weak in the diverse recipes. On FineWeb
the tool flags 95 head rows in the control and 120 in the head cut, in 2.5 minutes.

Most of these flags are heads that grammar narrows within their filler class: *to* takes fewer
fillers than the other prepositions (524 against 871), and reporting verbs (*said, says, means*)
fewer than other verbs. They are weak in the corpus, which is what a model trained on it sees. A
reference reads them within noise, because the domain has them too.

The test cannot see some cases. Thinning every pronoun at once leaves no sibling stronger: the
pronoun node reads 0.94–1.06 of its expected fillers when thinned to 25% or 10%, and the class cut
shows its slot 19 times. In a synthetic subject–verb corpus, *she* and *they* restricted to two
verbs form their own class and lose their sibling.

We rejected three alternatives. Comparing each head with its whole coarse class, adding a test of
how often a head takes each filler class, and grouping heads by context flagged 500–2,000 heads
and 3,000–15,000 class rows in the control, every pronoun among them. Their halves shared 12–17%
of the flags.

---

## 6. Discussion

*What the indicator certifies.* A flagged class pattern is a corpus fault. In our experiments no
configuration recovered it, and capacity made it worse. The remedy is data: text in which the
class's heads take other fillers of their slot. Conversely, an unflagged corpus lets the
reference's syntax be reached at its size, and a failure there points at the model. This division
of labor is the practical point of the work. The data side certifies that the patterns are
served; finding the model that learns them is left to model engineering.

*Why class level.* Models share evidence across heads of a class: a head with a thin slot borrows
from its siblings, which agrees with the indirect-evidence results of FiCT and the AANN study.
Realistic filters thin heads but rarely empty classes, which explains why, among realistic cuts,
class-level syntax followed size and topic rather than the score (§5.4). A diagnostic aimed at
syntax must therefore read classes, while a diagnostic of which word a model will predict reads
heads. Reading every node of the head tree, rather than one cut, lets the report name the group
that carries a shift: the whole modal class in one cut, two possessives in another.

*Settings from the data.* A threshold set on the corpora where a method was developed is a claim
about those corpora. The indicator's bar instead says, for each corpus, how far a random sample of
the reference strays at its worst. The reader is left with one number to judge, alpha, which is a
tolerance and not a property of the data. The cost is time: 101 draws make a diagnosis 4–9 minutes
at our sizes, against about 4 minutes before.

*With and without a reference.* Without a reference, the corpus is judged by its own evidence: a
head weaker than its siblings is weak for a model trained on it. This is the reading a
practitioner has before they find a reference, and it flags real gaps (*she* and *they* in the
head cut). A reference does two further things. It sees what the corpus lacks altogether, or lacks
for a whole class, and it separates the corpus's weakness from the domain's. Both readings are
relative. "Unflagged" means as good as a random sample of the reference of this size, or as good
as a random share of the head's parent; it does not mean as good as the reference at any size. In
our data, a random cut of a large, well-built corpus already preserves the patterns as well as any
filter. The indicator's practical role is mainly to reject cuts and filters that bias syntax.

*Effect sizes.* Whole-model effects of slot damage are small in bits per byte (0.017 over a whole
filter ladder), whereas per-pattern effects are large: 1–4.6 nats for planted heads and 0.13–0.30
in G for class removals (§5.4). A per-pattern diagnostic is needed precisely because whole-corpus and
whole-model numbers dilute local losses.

---

## 7. Limitations

The first limitation is scale. The models go up to 6.2M parameters, and the corpora up to about
10M words, on CPU, and on BLiMP the models read 0.56–0.63. Our claims therefore concern local
syntax learned by small models; whether the class-level attribution holds for models that learn
long-range syntax is untested. The indicator itself covers local syntax only, that is, heads of
one or two tokens and the next filler. Long-range dependencies (agreement across attractors,
filler–gap, binding) are out of scope.

The model-side evidence comes from one language and one natural source, English and FineWeb, plus
a synthetic grammar. On the corpus side, the granularity range was measured in five English
domains, from 0.9 to 57.6 MB, with three pools per size up to 14.4 MB and one at 57.6 MB; other
languages and larger corpora are untested. Below 14.4 MB the verdicts do not hold, so the tool
needs a domain of at least that size; the plantings of §5.2 were read against a 6.5 MB sample,
before this minimum was measured, and held there. A dropped word whose uses sit in a few documents cannot be
read absent inside the granularity range: the indicator resolves the absence of words spread over many
documents, not of words concentrated in a few. Only some classes were tested at depth: three
(pronouns, modals, possessives) were trained under removal, and two more (numbers, base verbs) were
checked on the corpus side. At scale, the pronoun result was not confirmed, because its test lost
power. Most experiments used two seeds; the new model configurations and the scale runs used one.

The test's level, alpha = 0.06, is per family. A random corpus can be flagged in any of three
families, so its chance of some flag is at most 18%; measured, it was 4% (2 of 48). Splitting
alpha three ways would need about 300 draws. In the planted slices, one class row was flagged in
each of two of seven unplanted slices. The granularity range's upper edge moves with the pool and
grows with size (to 2.00 at 57.6 MB), so 1.70 is the edge common to every domain at 14.4 MB, not a
property of each corpus.

Without a reference, a pattern the corpus lacks, a class shown weakly as a whole, and a head the
tree sets apart from its siblings are not seen.

Two further limits concern what the indicator measures. On the recipe grammar, the diversity a
model needs grew with model size. The indicator says where a corpus departs from its reference;
how much departure a given model tolerates is not part of it. Finally, the indicator addresses
syntax, not semantics. A corpus can pass and still fail a model on topic or meaning: a
topic-narrow corpus failed models document-wide without any slot flagged. The classes, too, are
distributional rather than grammatical: against a tagger, nouns and verbs come out as classes and
adjectives as a node of the tree, but determiners, possessives and adjectives share the class of
what comes before a noun, and *of*, *and*, *to* share the verbs' class (§5.8); word pieces,
punctuation, numbers and ambiguous words, 54–61% of running tokens, have no tag to check against.

---

## 8. Conclusion

We asked whether a corpus can be checked, before training, for the local syntactic patterns its
domain needs, so that a generalization failure can be attributed to the corpus or to the model.
The answer, within the limits above, is yes.

An indicator that compares each slot's fillers with random samples of a reference, at every node
of an induced tree of heads, flags corpus faults that no model configuration we tried recovers.
Where it flags nothing, small models reach the reference's syntax. None of its settings is
tuned by hand: the trunk read and the number of samples come from the data, the granularity sits
inside a measured range, and the test's level is stated in every report. Without a reference, the same tree lets a corpus be judged by its
own evidence. Along the way, we found that models learn local syntax mostly at the class level,
which explains both why realistic data filters rarely break syntax and why a diagnostic must read
classes. The tool, code and every preregistered prediction, including the failed ones, will be
released with publication.

Future work includes larger models and corpora, with GPU; long-range dependencies; more languages,
to test that the range measured in five English domains carries; the tolerance of a given model size to a
flagged departure; and a semantic counterpart for knowledge domains.

---

## Appendix A. Predictions that failed

The predictions are kept as written in the preregistration documents.

**Indicator and models (earlier version of the indicator).**

- **Recipe grammar.** The minimal pairs "cut the slowly" did not stay near 100% (0.25 with one
  filler). The corpus-size predictions failed at 500 documents.
- **Plantings.** One comparison head moved −0.51 nats. Whole-text loss rose 0.03 bits per byte
  (predicted < 0.02). Detection at k = 32 was still frequent (12/20).
- **Topic-narrow corpus.** The flagged heads did not lose more; the loss was topical and
  document-wide.
- **Subset selection.** Mild natural flags (diversity 0.26–0.50) did not cost the model. In r10,
  Spearman(z, drop) was 0.10 / 0.12 (bar 0.15).
- **Dose.** The mild end cost 0.38–0.57 nats (predicted ≤ 0.3), and the threshold was crossed at
  0.64.
- **Divergence.** On plantings, D ordered heads no better than diversity. Its first cut, κ, failed
  because of a sample-size floor.
- **Cut choice.** Over ten cuts, Spearman(score, bits per byte) was 0.55 / 0.45 (predicted ≥ 0.5
  on both seeds). With the current reading it holds (0.65 / 0.55).
- **Ladders.** The ladder that adds documents had Spearman 0.60. The per-head reading failed at
  dose 25%.
- **Syntactic evaluation of realistic cuts.** All six predictions failed, which led to the
  class-level hypothesis.
- **Class patterns.** P2 came out at −0.086 (bar −0.10), with an invalid first modal test. In P4,
  the noise between models exceeded the bar.
- **Replication.** The possessive head cut cost 57% (bar 50%).
- **Tool.** The pronoun class was missed (+0.47 against a 0.5 floor). The earlier *absent* rule
  gave 79–114 false flags, and one control gave 5 at m/2.
- **BLiMP.** F's models read below 0.60.
- **Scale.** The pronoun gap was 0.06 (bar 0.10), and the control was 0.09 below the reference.

**Settings from the data.**

- **Granularity.** Adjacent granularities were predicted to agree on 90% of patterns; they agreed
  on 85–94%, and almost all of the difference was whether a pattern is read. With a trunk of 20,
  the numbers and base verbs were missed at 1.35, although they were predicted to be found over
  the whole sweep.
- **Class floor.** Each class whose removal cost the models their syntax was predicted to read a
  larger excess than every one that did not. The prediction failed: at 64 head classes over every
  token, pronouns and possessives shared one class with *the* and *a*, and moved it by +0.045 and
  +0.007. This led to class rows at every node.
- **The 16-draw bar.** 6 of 48 random cuts were flagged against one draw set (design: about 1 in
  17). This led to alpha and 101 draws.
- **Without a reference.** Saturation (missing mass) did not flag a thinned slot, and three
  groupings of heads gave flags that the corpus's halves did not share.
- **Subset selection, rerun.** Both filters were predicted to flag at least 10 rows; they flagged
  2 and 1.
- **Domains, size and pool.** 1.50 was predicted to hold at every size and on every pool; it held
  on 1 of 15 pools at 0.9 MB and 6 of 15 at 3.6 MB, and failed on three of fifteen 14.4 MB pools,
  each on one dropped word concentrated in a few documents. The 57.6 MB readings were predicted to
  match 14.4 MB; Wikipedia's and legal's ranges widened instead. PubMed was predicted to need more
  merges than Wikipedia to reach 1.50 (15,066 against 16,022), and its range and legal's were
  predicted to contain 1.30–2.00 (legal's ends at 1.70). This led to the 14.4 MB minimum and the
  range 1.35–1.70.
- **Classes against a tagger.** Nouns, verbs, adjectives, adverbs, prepositions, determiners and
  pronouns were each predicted to have a node of the tree beating every shuffle of the tags in at
  least four of five domains. Determiners had fewer than 20 tagged tokens in TinyStories and legal
  and were read in three; every tag that was read beat the shuffles.

## Appendix B. Earlier version of the indicator

The indicator was first built with hand-set thresholds (§4.4). Its results, replaced in the body by
those of the final tool, were the following.

- **Planted gaps (§5.2).** With a 3-sigma margin, the replacement sweep found 5, 5, 5, 5 and 4 of
  5 heads on FineWeb at k = 2, 4, 8, 16 and 32, and 19, 20, 19, 17 and 12 of 20 heads over four
  sources (FineWeb, Wikipedia, arXiv, C4). Removing sentences found 3 of 5: two heads left with 6
  and 23 occurrences read just above a fixed diversity floor of 0.5.
- **Realistic cuts (§5.3).** The 3-sigma margin flagged 2–8 rows on each random cut, at the rate
  chance gave it. In the 10% cut, Spearman(z, drop) was 0.10 / 0.12, with log trunk alone at
  0.10 / 0.11. Reading word-like tokens only, the cut scores were z = +15.0 (formulaic filter),
  +20.6 (least Wikipedia-like) and +3.9 (CCNet-style), and |z| < 0.6 for random cuts.
  Spearman(E, drop − random drop) was −0.21 to −0.38, and Spearman(score, bits per byte) over ten
  cuts was 0.55 / 0.45.
- **Tool detection (§5.6).** With 64 head classes and a 0.5-nat class floor, the class rows found
  4 of 5 removed classes, missing the pronouns at +0.47 nats. The possessives were found only
  through their short heads, because they shared an induced class with *the* and *a*. The first
  *absent* rule fired 79–114 times on random controls.
- **Dose fit and head-level D (§3.3, §5.2, §5.4).** Read on word-like tokens, the dose fit was
  drop ≈ −0.68 · D (R² 0.77), and head-level D gave *she* +1.46 / +1.58 in the head and class
  cuts.

## Appendix C. Reproducibility

Every experiment has its own script, its results (JSON) and a record of its design, predictions
and outcome, written before it ran. The diagnosis is packaged as a command-line tool
(`corpus-diagnose`) with a web interface. Runs are CPU-only; a diagnosis takes 4–9 minutes at our
sizes after reading the reference once. Code, data and preregistration records are available from
the author on request and will be released with publication.

---

## Declaration of generative AI use

During this work the author used Claude (Anthropic) to assist in writing code, running and
analyzing experiments, and drafting the manuscript. The author directed the study, reviewed and
edited all content, and takes full responsibility for it.

---

## References

- Albalak, A., Elazar, Y., Xie, S. M., Longpre, S., Lambert, N., Wang, X., et al. (2024). A Survey on Data Selection for Language Models. *Transactions on Machine Learning Research*. arXiv:2402.16827.
- Baayen, R. H. (2009). Corpus linguistics in morphology: Morphological productivity. In A. Lüdeling & M. Kytö (Eds.), *Corpus Linguistics: An International Handbook* (Vol. 2, pp. 899–919). Berlin: Mouton de Gruyter. https://doi.org/10.1515/9783110213881.2.899
- Covington, M. A., & McFall, J. D. (2010). Cutting the Gordian Knot: The Moving-Average Type–Token Ratio (MATTR). *Journal of Quantitative Linguistics*, 17(2), 94–100. https://doi.org/10.1080/09296171003643098
- Friedman, D., & Dieng, A. B. (2023). The Vendi Score: A Diversity Evaluation Metric for Machine Learning. *Transactions on Machine Learning Research*. arXiv:2210.02410.
- Good, I. J. (1953). The Population Frequencies of Species and the Estimation of Population Parameters. *Biometrika*, 40(3–4), 237–264. https://doi.org/10.1093/biomet/40.3-4.237
- Guo, Y., Shang, G., & Clavel, C. (2025). Benchmarking Linguistic Diversity of Large Language Models. *Transactions of the Association for Computational Linguistics*, 13, 1507–1526. https://doi.org/10.1162/tacl.a.47
- Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength Natural Language Processing in Python. Zenodo. https://doi.org/10.5281/zenodo.1212303
- Hu, J., Gauthier, J., Qian, P., Wilcox, E., & Levy, R. (2020). A Systematic Assessment of Syntactic Generalization in Neural Language Models. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 1725–1744). https://doi.org/10.18653/v1/2020.acl-main.158
- Li, J., Galley, M., Brockett, C., Gao, J., & Dolan, B. (2016). A Diversity-Promoting Objective Function for Neural Conversation Models. In *Proceedings of NAACL-HLT 2016* (pp. 110–119). https://doi.org/10.18653/v1/N16-1014
- Linzen, T., Dupoux, E., & Goldberg, Y. (2016). Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies. *Transactions of the Association for Computational Linguistics*, 4, 521–535. https://doi.org/10.1162/tacl_a_00115
- Luo, J., & Steinert-Threlkeld, S. (2026). Exposure is Optional: Learning Unlike Coordination in Language Models. arXiv:2607.20251.
- Marvin, R., & Linzen, T. (2018). Targeted Syntactic Evaluation of Language Models. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 1192–1202). https://doi.org/10.18653/v1/D18-1151
- Miranda, B., Lee, A., Sundar, S., Casasola, A., Schaeffer, R., Obbad, E., & Koyejo, S. (2023). Beyond Scale: The Diversity Coefficient as a Data Quality Metric for Variability in Natural Language Data. arXiv:2306.13840.
- Misra, K., & Mahowald, K. (2024). Language Models Learn Rare Phenomena from Less Rare Phenomena: The Case of the Missing AANNs. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 913–929). https://doi.org/10.18653/v1/2024.emnlp-main.53
- Patil, A., Jumelet, J., Chiu, Y. Y., Lapastora, A., Shen, P., Wang, L., Willrich, C., & Steinert-Threlkeld, S. (2024). Filtered Corpus Training (FiCT) Shows that Language Models Can Generalize from Indirect Evidence. *Transactions of the Association for Computational Linguistics*, 12, 1597–1615. https://doi.org/10.1162/tacl_a_00720
- Penedo, G., Kydlíček, H., Ben Allal, L., Lozhkov, A., Mitchell, M., Raffel, C., et al. (2024). The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. In *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track* (pp. 30811–30849). arXiv:2406.17557.
- Qi, X., He, L., Roth, D., & Fu, X. (2026). Demystifying Supervision Data Generalization in Multimodal LMs. In *International Conference on Learning Representations (ICLR 2026)*. arXiv:2603.19688.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI technical report.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982–3992). https://doi.org/10.18653/v1/D19-1410
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 1715–1725). https://doi.org/10.18653/v1/P16-1162
- Tarone, R. E. (1990). A Modified Bonferroni Method for Discrete Data. *Biometrics*, 46(2), 515–522. https://doi.org/10.2307/2531456
- Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., & Zhou, M. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. In *Advances in Neural Information Processing Systems 33* (pp. 5776–5788). arXiv:2002.10957.
- Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective Function. *Journal of the American Statistical Association*, 58(301), 236–244. https://doi.org/10.1080/01621459.1963.10500845
- Warstadt, A., Parrish, A., Liu, H., Mohananey, A., Peng, W., Wang, S.-F., & Bowman, S. R. (2020). BLiMP: The Benchmark of Linguistic Minimal Pairs for English. *Transactions of the Association for Computational Linguistics*, 8, 377–392. https://doi.org/10.1162/tacl_a_00321
- Wei, J., Garrette, D., Linzen, T., & Pavlick, E. (2021). Frequency Effects on Syntactic Rule Learning in Transformers. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing* (pp. 932–948). https://doi.org/10.18653/v1/2021.emnlp-main.72
- Wenzek, G., Lachaux, M.-A., Conneau, A., Chaudhary, V., Guzmán, F., Joulin, A., & Grave, E. (2020). CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data. In *Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020)* (pp. 4003–4012).
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing: Examples and Methods for p-Value Adjustment*. New York: Wiley.
- Wettig, A., Gupta, A., Malik, S., & Chen, D. (2024). QuRating: Selecting High-Quality Data for Training Language Models. In *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235, 52915–52971.
- Witten, I. H., & Bell, T. C. (1991). The Zero-Frequency Problem: Estimating the Probabilities of Novel Events in Adaptive Text Compression. *IEEE Transactions on Information Theory*, 37(4), 1085–1094. https://doi.org/10.1109/18.87000
- Xie, S. M., Santurkar, S., Ma, T., & Liang, P. (2023). Data Selection for Language Models via Importance Resampling. In *Advances in Neural Information Processing Systems 36* (pp. 34201–34227). arXiv:2302.03169.
