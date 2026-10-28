# Settings from the data: every token in the patterns, and the granularity range

**Status:** done (2026-10-15). Steps 1–4 closed, and the paper's corpus-side readings re-run (commit `fac3b67`). Step 1's range is being extended by `2026-10-25-granularity-across-domains-design.md` and `2026-10-28-granularity-robustness-design.md`.

Agreed with the user on 2026-10-11.

**Goal:** diagnose before training, from the corpus alone (patterns in its token-walk graph),
whether it shows its domain's syntactic patterns with enough variety for a model to generalize
them, so a failure can be attributed to the corpus or the model. A reference corpus is optional:
it only separates an under-shown pattern from one that is rare in the domain itself.

**Criterion:** the diagnosis must not depend on arbitrary choices. Every parameter is derived from
the data, or shown not to change the verdicts.

**Path:**

1. Measure the optimal tokens-per-word range.
2. Replace the fixed 4000/1000 with a better strategy.
3. CLI: accept `--merges` and validate it against the range before the analysis (`--force`
   overrides); search the best merges itself when none is given.
4. Class patterns without a reference, by saturation.

The paper's experiments are re-run after the changes. If the Goal is not met at the end of the
path, the path is revised.

## What the 4,000 cap hides: measured at 11,563 merges on the FineWeb pool

The pool is `natural_gap.split()`: 4,600 documents, 14.3 MB, 1.46 tokens per word. It is the same
set-up as `experiments/class_patterns`.

**The cap is not the limit.** Only 4,313 tokens are word-like (`word_like_tokens`), and the cap
keeps 4,000 of them. Lifting the cap classifies 4,285, and the share of running tokens with a
class goes from 0.475 to 0.478.

**The word-like filter is the limit.** It leaves 52% of running tokens without a class:

| Tokens | Share of running tokens | Examples |
|---|---|---|
| Word continuations | 17.4% | `s`, `ing`, `ed` |
| Space-prefixed word starts that are usually continued | 14.4% | ` out`, ` S`, ` C` |
| Punctuation | 12.0% | `.`, `,`, `(` |
| Capitalized, no leading space | 3.5% | `The`, `I` |
| Digits | 2.3% | |
| Whitespace | 2.0% | |
| Contractions | 0.6% | `'s`, `'t` |

The filter was added on 2026-09-28 ("Word-like tokens, diversity against chance, and a reference
corpus") to clear word pieces from the list of restricted patterns. It has two costs:

- It removes half the walk from the patterns. Punctuation, sentence-initial words and every word
  BPE splits have no slot.
- It adds two arbitrary settings: the `^ [A-Za-z]+$` text rule and `WORD_FINAL_SHARE = 0.9`.

## Change: every token in the patterns, each transition lexical or syntactic

No token is filtered. Every token found in at least two documents is clustered, and every token can
be a head or a filler.

What separates spelling from syntax is the transition, not the token. A transition is the
head's last token followed by the filler. Its kind comes from the tokenizer's pre-tokenization,
which no merge crosses:

- **Syntactic:** the filler starts a pre-token. It begins a new word, a punctuation run, a
  number, a contraction, or it is `<eos>`.
- **Lexical:** the filler continues the pre-token of the token before it, so both are inside one
  word.

"The salesman sold models." tokenizes as `The | sal | esman | sold | model | s | .`

| Transition | Filler starts a pre-token | Kind |
|---|---|---|
| `The` → ` sal` | yes | syntactic |
| ` sal` → `esman` | no | lexical |
| `esman` → ` sold` | yes | syntactic |
| ` sold` → ` model` | yes | syntactic |
| ` model` → `s` | no | lexical |
| `s` → `.` | yes | syntactic |

The encoder records each token's flag while it encodes, so the rule is exact. It does not
re-split the decoded text, which can differ: byte-level tokens may cut a UTF-8 character.

A pattern's key gains the kind: (head, filler class, lexical). Occurrences of one head and class
of different kinds are counted as two patterns, so neither kind inflates the other. Class
patterns are split the same way. The verdict rules do not change. The report gives syntactic and
lexical rows in separate sections, and syntactic rows are the diagnosis.

### The cap: classify every eligible token

With every token eligible, frequency rank covers this share of running tokens:

| Top k tokens | 1,000 | 2,000 | 4,000 | 6,000 | 8,000 | all 11,636 |
|---|---|---|---|---|---|---|
| Share of running tokens | 0.681 | 0.772 | 0.866 | 0.921 | 0.958 | 0.999 |

Ward on all 11,636 tokens took 173 s and 1.25 GB, against 26 s and 0.50 GB at 4,000.
`MAX_CLASSIFIED_TOKENS` is removed, and the only condition left is `min_documents = 2`: a token
seen in one document has no context that is not that document's.

The cost is cut by building the Ward tree once. `induce_classes` fitted it twice (the full tree,
then the cut), and the head classes fitted it twice more. `class_tree` builds it once with SciPy's
`linkage`, whose tree and heights equal sklearn's, and every cut replays its merges.

### Measured cost (2026-10-11)

One diagnosis of the 2,220-document `control` cut against F at 11,563 merges, one-token heads:

| Stage | Word-like filter | Every token, first version | Every token, optimized |
|---|---|---|---|
| Tokenizer, framing, classes (reference, once) | 190 s | 140 s | 145 s |
| Each of the 8 noise draws | 3.7 s | 20 s | 2.6 s |
| Report and class patterns | 43 s | 54 s | 13 s |
| Peak memory | 1.0 GB | 1.3 GB | 1.1 GB |

The optimized counting is vectorized and gives exactly what the loops gave: the same patterns,
counts and floats on randomized inputs. The Ward tree on 11,636 tokens is now the largest cost,
about 100 s, and it is paid once per reference. The verdict counts of the run are identical
before and after the optimization.

### Settings still to justify

These remain fixed and are checked by the sweeps in later steps. Each one must be derived from the
data or shown not to change the verdicts.

- `CONTEXT_TOKENS = 1000`: covers 68% of running tokens; the rest share one "other" column.
  Shown not to change the verdicts from 250 to 4,000 (step 2, below).
- `min_classes = 8`, `max_classes = 64`, `NOISE_FLOOR = 0.05`: the coarse cut.
  Shown not to change the verdicts: the cut forced from 8 to 64 classes (step 2, below).
- `head_classes = 64`. Removed: class rows are read at every node.
- `min_trunk = 20`, `low = 0.5`, `sigmas = 3`, `draws = 8`, `class_floor = 0.5`, and the `m/4`
  absent rule. Removed or derived in step 1: the trunk from the draws, `draws` from alpha, the
  rest replaced by the draws' bar. `low = 0.5` and trunk 20 remain only without a reference.

## Step 1: the tokens-per-word range

1.4–1.5 tokens per word was chosen at the plateau of FineWeb's whole-word coverage
(`experiments/granularity`). It was never checked against the verdicts.

**Sweep:**

- One tokenizer is learned on the pool at the largest budget and cut for each granularity. The
  first k merges of a longer run are a run at k (`tests/test_diagnose.py`).
- At each granularity, `prepare_reference` reads the pool once.
- The labeled cuts of `experiments/class_patterns` are diagnosed against it.
- The granularities are 1.30, 1.35, 1.40, 1.50, 1.60, 1.70 and 1.80 tokens per word.

**Readings per granularity:**

- the share of transitions that are lexical;
- detection: the class rows that flag the removed classes in the `*_class_cut` corpora;
- false flags: `under-shown` and `absent` rows on the controls (`control`, `ctrl_a`–`ctrl_d`,
  `r25b`–`r25e`);
- `absent` on `drop4` and `drop4b` against the controls;
- stability: the agreement of syntactic verdicts between adjacent granularities, over the
  patterns read at both.

**Predictions (written before running):**

1. The lexical share falls as merges rise, near `(t − 1) / t` for `t` tokens per word. This is
   nearly an identity, so it is a check of the flags, not a finding.
2. Detection of the numbers and base-verb class cuts holds over the whole sweep. At 11,563 merges
   their excess was +2.62 and +0.53.
3. The controls stay at zero false flags at 1.40–1.60.
4. The syntactic verdicts of adjacent granularities agree on at least 90% of the patterns read
   at both, over a range that contains 1.46. The range where this holds is the measured range.
   Its edges are the result, and are not set in advance.
5. Below the range, rare words lose occurrences to merges that make them single tokens, so more
   patterns fall below `min_trunk` and go `unread`. Above it, word pieces split the slots, so
   detection weakens.

If prediction 4 finds no range, the granularity changes the verdicts by itself, and the step is
revised with the user before the CLI work.

## First sweep, 1.40–1.80 (2026-10-11)

`experiments/granularity_range`. On the 22 labeled cuts, at 1.40, 1.50, 1.60, 1.70 and 1.80
tokens per word:

- No class row is flagged on the controls or the head cuts, at any granularity.
- `absent` finds all 8 removed words in `drop4` and `drop4b`, with 0–1 rows on the 8 fresh
  controls.
- The removed classes are found only once: the modals, at 1.60. At 1.50 the numbers and base-verb
  class rows read D 12 and 15 sd above their draws, but excess 0.04–0.12, under `class_floor`.
- Adjacent granularities agree on 85–94% of syntactic patterns (prediction 4 fails at 1.50–1.60).
- 1.30 does not fit in memory: 45,748 tokens to cluster, 8.4 GB for the distance matrix alone.

Two causes:

1. **`class_floor` = 0.5 nats belongs to the old scale.** With every token a filler, punctuation
   and pieces carry most of a class's filler mass, so a removed slot moves D far less.
2. **The cuts no longer remove a whole class.** They removed the members of the old word-like
   class. The new class of `two` also holds the digits, and the class of `make` holds `be`.

## Recalibrating `class_floor` on the trained models

The floor separates class shifts that cost a model its syntax from those that do not. The models
of `experiments/class_patterns` were trained on these cuts, and their losses are measured:

| Cut | Class-level ΔG | Model |
|---|---|---|
| `class_cut`, pronouns | −0.195 | lost the slot |
| `class_cut`, modals | −0.126 | lost the slot |
| `poss_class_cut`, possessives | −0.303 | lost the slot |
| `head_cut` (pronouns, modals), `poss_head_cut` | −0.03 to −0.04 for the class | kept it |
| the three controls | 0 by construction | kept it |

The class rows of these six cuts are re-read with the new method at 11,563 merges, from the same
60,000-merge tokenizer. No model is trained.

**Predictions (written before running):**

1. Each class where the model lost the slot reads a larger excess than every row where it kept it.
2. If prediction 1 holds, the new floor is the geometric midpoint between the smallest "lost"
   excess and the largest "kept" one (the old 0.5 was placed by eye between +0.04 and +1.82). If it fails, no floor separates them, and the class rows need a different reading.

**Result (2026-10-11): prediction 1 fails.**

| Cut | Modal class row (16 heads) | Class row holding pronouns and possessives (90 heads) |
|---|---|---|
| `class_cut` (model lost both) | +0.969 | +0.045 |
| `head_cut` (kept) | +0.100 | −0.009 |
| `control` | −0.031 | −0.012 |
| `poss_class_cut` (model lost) | +0.001 | +0.007 |
| `poss_head_cut` (kept) | −0.006 | −0.005 |
| `poss_control` | −0.007 | −0.010 |

The modals separate cleanly: a floor near 0.31 (√(0.969 · 0.100)) splits them. The pronouns and
the possessives do not have a class of their own. At 64 head classes over every token they
share one 90-token class with `the`, `a`, `that` and `this`, and their removal moves that row by
+0.045 and +0.007, less than the modals' kept `head_cut`. No floor separates "lost" from "kept",
so the floor is not the first thing to fix: the fixed 64 head classes are.

## Class rows at every node of the head tree (exploration, 2026-10-11)

Instead of cutting the tree at 64 head classes, every internal node of the Ward tree (11,635 at
11,563 merges; 10,279 with a reference trunk of at least 20) gets a class row: its heads' pooled
fillers, D against F, and the 8 draws' mean and sd. Scratch code; nothing in `src/` yet.

A node is flagged when its t = (D − mean) / sd exceeds the one-sided critical value of Student's t
with `draws − 1` degrees of freedom at a family-wise error of 0.05 over the nodes tested
(Bonferroni): 11.3 here. This replaces both `sigmas` and `class_floor`; it is set by the number of
draws and of nodes, not by hand. Of each branch, the node with the highest t is reported.

| Cut | Reported (t, excess, heads) |
|---|---|
| `class_cut` | modals (276, 1.97, 7); I/you/we/they (237, 1.36, 5); He/She (24); You/We/They (28); plus side shifts |
| `head_cut` | can/cannot (69, 1.30); will/would (34, 0.33): exactly the removed heads |
| `poss_class_cut` | your/my/his/their (161, 1.82) |
| `poss_head_cut` | his/their (89, 1.56): exactly the removed heads |
| `num_class_cut` | four/two/three (104, 1.97) |
| `verb_class_cut` | see/get/find/leave/make/take (126, 0.69) |
| `control`, `ctrl_a` | one node of acronyms (VD, SR, CN…; t 17 and 12) |
| `poss_control`, `ctrl_b`, `r25b` | nothing |

Every removed class and every removed head pair is found, in its own branch. The controls show one
false flag, the same acronym node, in 2 of 5.

**The acronym flag is a shortfall of 8 draws.** With the same corpus and node, t falls from 17.3
(8 draws) to 5.1 (16) and 3.3 (32), under critical values of 11.3, 6.5 and 5.3; the removed
`can`/`cannot` pair stays at t 69–80. Eight draws had underestimated the node's spread. The tool
now draws 16 by default; whether 16 and 32 give the same verdicts is checked on the sweep.

### In the tool

`diagnose` reads every internal node of the head tree whose reference trunk reaches `min_trunk`
(since replaced by a trunk the draws set; see "The trunk read, from the draws" below),
per kind (syntactic, lexical); of each branch only the node with the highest t is reported.
`head_classes` and `class_floor` are removed, and `draws` defaults to 16. Vectorized D
(`node_divergences`) equals `filler_divergence` on randomized inputs.

**The bar comes from the draws, not from a normal law.** With 16 draws the Bonferroni value fell
to 6.5, and 9 of the 12 controls at 1.50 then flagged a node of rare pieces (`De`/`Ste`,
`sour`/`pow`) at t 7–9. Leaving each draw out in turn and reading its t against the other 15 shows
why: the largest t a random draw reaches over all nodes is 4–10 most of the time and 35–44 once in
16, when one document dominates a small node. The spread has long tails, and a normal-theory bar
underestimates them. The bar is now that leave-one-out maximum: a node is under-shown only when
the corpus strays further than any random sample of the reference strayed at any node. With n
exchangeable draws a random corpus passes it with probability about 1/(n + 1); no rate is chosen.

At 1.50 (11,381 merges) on the 22 cuts, with bars of 39–94:

| Cut | Reported |
|---|---|
| `class_cut` | modals (t 348); I/you/we/they (207) |
| `poss_class_cut` | your/his/my/their (154) |
| `num_class_cut` | two…seven (99) |
| `verb_class_cut` | get/make/see/take/find/leave (95) |
| `poss_head_cut` | his/their (112) |
| other head cuts, 12 controls | nothing |

On the synthetic subject–verb grammar of the tests, restricting `she` and `they` flags no node
holding a subject, but it does flag the node of the sentence openers (`then`, `now`, `so`…). Fewer
verbs after `she` leave fewer distinct contexts for `she` after an opener: counting fillers in
distinct contexts carries a restriction one slot back. It is a property of the counting, kept in
mind when reading real cuts.

## Second sweep, 1.35–1.80, every node and the draws' bar (2026-10-11)

| Tokens/word | Merges | Removed classes found (of 5) | Head cuts flagged | Control flags | `absent` on 8 fresh controls | Dropped words found |
|---|---|---|---|---|---|---|
| 1.35 | 30,526 | 3: modals, pronouns, numbers | will/would | 0 | 1 | 8/8 |
| 1.40 | 20,608 | 5 | his/their | 0 | 0 | 8/8 |
| 1.50 | 11,381 | 5 | his/their | 0 | 2 | 8/8 |
| 1.60 | 7,220 | 5 | his/their | 0 | 0 | 8/8 |
| 1.70 | 4,944 | 5 | will/would, his/their | 0 | 1 | 8/8 |
| 1.80 | 3,548 | 5 | can/will, his/their | 0 | 1 | 8/8 |

The five removed classes are pronouns, modals, possessives, numbers and base verbs.

- **Class rows hold from 1.40 to 1.80 and fail below.** At 1.35 the possessives and base verbs are
  missed. Their t rises as tokens per word rise (modals 319 → 690), but nothing breaks at 1.80, so
  the sweep finds a lower edge (between 1.35 and 1.40) and no upper one.
- **No class row is flagged on any control**, at any granularity.
- **Head rows are less stable.** Adjacent granularities agree on 85–94% of syntactic head
  patterns (89% at 1.35–1.40, 85% at 1.50–1.60), so prediction 4 fails. Head rows still use the
  fixed `low = 0.5` and `sigmas = 3`, the next settings to derive.
- 1.30 does not fit in 16 GB (45,748 tokens to cluster).

Prediction 1 (lexical share near (t − 1)/t) held loosely: 0.085 at 1.35 against 0.26, since
punctuation and numbers are pre-tokens of their own; it falls with merges as expected. Prediction
2 (numbers and base verbs found over the whole sweep) failed at 1.35. Prediction 3 held:
no control flag. Prediction 5 held for the lower edge.

## Head rows and `absent` against the draws' bar (2026-10-11)

`sigmas = 3`, `low = 0.5` (with a reference) and the `m/4` absent rule are removed. A head row is
under-shown when its diversity falls further below the draws' mean, in their spread, than any draw
falls below the other draws at any pattern; `absent` reads trunks the same way. Diversity moves in
steps of one filler (1 / expected fillers) and a trunk in steps of one, so a spread below its
step is rounding: the spread used is √(sd² + step²/12). Without it, patterns whose class is
covered in full every draw had sd ≈ 0.0001 and set a bar of 37 on the recipe tests, hiding
restricted verbs at t 24. With a reference the verdicts are `under-shown`, `absent`,
`within-noise` and `unread`; `low` remains only without a reference, until step 4. A corpus
at least as large as its reference makes every draw the whole reference, and every verdict
`unread`.

Third sweep, same 22 cuts:

| Tokens/word | Classes found (of 5) | Head flags on 12 controls | Head flags, `class_cut` / `head_cut` | `absent` on 8 fresh controls | Dropped words |
|---|---|---|---|---|---|
| 1.35 | 3 | 2 | 23 / 6 | 0 | 8/8 |
| 1.40 | 5 | 0 | 32 / 9 | 0 | 8/8 |
| 1.50 | 5 | 0 | 38 / 11 | 0 | 8/8 |
| 1.60 | 5 | 2 | 25 / 7 | 0 | 8/8 |
| 1.70 | 5 | 1 | 25 / 8 | 1 | 8/8 |
| 1.80 | 5 | 2 | 20 / 7 | 1 | 8/8 |

- `absent` without `m/4` still finds every dropped word and flags 0–1 fresh controls.
- Head flags on the controls (0–2 of 12 per granularity) match the bar's design: a random corpus
  passes it somewhere about once in 17.
- **Stability.** Adjacent granularities agree on 86–91% of patterns, but 99.8% of the
  disagreements are `unread` against `within-noise`: whether a pattern has enough occurrences to
  be read. Among patterns read at both, under-shown and within-noise swap 22 times in 93,321
  (1.40–1.50) and 27 in 116,601 (1.50–1.60). 62–76% of flagged patterns keep their flag; the
  rest sit at the bar.

**Step 1 result.** The verdicts hold from 1.40 to 1.80 tokens per word: every removed class is
found and no control's class row is flagged, and the read verdicts barely move. Below 1.40 (at
1.35) two classes are lost. No upper edge appears by 1.80. What still moves with granularity is
which patterns are read at all, set by `min_trunk = 20`, the last fixed setting of this step.

## The trunk read, from the draws (2026-10-12)

`min_trunk = 20` decided which patterns were read. Swept at 1.50 tokens per word (fixed 2, 5, 10,
20, 40), it changed no verdict on the labeled cuts between 5 and 40, but it was not free: below 20
rare patterns raised the class bar from 52 to 93 (the number class then sat at t 99, near the
edge) and the head bar from 6.6 to 8.4; above it, half as many patterns were read. A plateau
measured on one domain at one size does not carry to another corpus, whose trunks scale with its
size.

**Design.** A pattern too rare for its corpus to be flagged even when it shows a single filler
adds no verdict, but its noise raises the bar for every other pattern. Tarone's criterion for
discrete tests (1990) drops such untestable tests. `diagnose` now picks the least trunk read as the
one that leaves the most flaggable patterns: for each candidate trunk it reads the draws from that
trunk, computes their bar, and counts the patterns whose drop to a single filler (1 / expected
fillers) would exceed it. Class patterns do the same over the draws' mean node trunk, with a
single filler meaning that the node shows only the reference's most frequent filler. Two is the floor: one
occurrence has diversity 1 by definition. A given `min_trunk` still fixes both. Without a reference
there are no draws, so the trunk stays 20 until step 4.

**A fix found on the way.** The rounding floor used the draws' step. A control that showed
` angels` 4 times, where the draws showed it more often, moves in steps of 0.26 against their
0.04, and was flagged. The floor now uses the tested value's own step: the corpus's for a row, and
each held-out draw's for the bar.

Fourth sweep, same 22 cuts; the derived trunk is 2–9 for patterns and 2–78 for class nodes,
varying by cut:

| Tokens/word | Classes found (of 5) | Class flags on controls | Head flags on 12 controls (trunk 20 before) | Head patterns read per cut (trunk 20 before) | Dropped words |
|---|---|---|---|---|---|
| 1.35 | 5 | 0 | 1 (2) | 25,326 (4,370) | 8/8 |
| 1.40 | 5 | 0 | 0 (0) | 26,990 (4,629) | 8/8 |
| 1.50 | 5 | 0 | 5 (0) | 24,873 (5,082) | 8/8 |
| 1.60 | 5 | 0 | 1 (2) | 18,131 (5,486) | 8/8 |
| 1.70 | 5 | 0 | 2 (1) | 18,645 (5,790) | 8/8 |
| 1.80 | 5 | 0 | 0 (2) | 10,960 (4,848) | 8/8 |

- **Class patterns.** The class bar falls from 39–94 to 10–29, and 1.35 now finds all five
  removed classes (it found three). No control's class row is flagged.
- **Head patterns.** Four to six times as many patterns are read, at about the same false-flag rate: 9
  of 72 control readings against 7 of 72 with trunk 20. Both are above the bar's design rate of
  about 1 in 17 (4 of 72). The flagged patterns have trunks of 6–9 (` >`, `It`, ` So`,
  ` reporters`). Eight of the 12 controls are random sentences, not whole documents like the
  draws, so they are not strictly exchangeable with them. This remains open.
- **Stability.** Under-shown and within-noise swap 19–29 times in 290,000–590,000 patterns read
  at adjacent granularities, about 0.5 in 10,000, against 2.4 with trunk 20. The rest of the
  disagreements (99.9%) are still `unread` against `within-noise`.
- **Cost.** A cut takes 100–140 s instead of 60–100 s: the search over trunks, mostly for class
  nodes.

**Step 1 closed** (draw count revisited below). No fixed setting of step 1 is left. The verdicts hold from 1.35 to 1.80 tokens
per word, so the lower edge moved below 1.35, and 1.30 does not fit in 16 GB. Still open: head
false flags on the controls run at about twice the design rate, with or without the derived
trunk.

## The draw count and the test's level (2026-10-12)

**Investigation.** The excess head flags on controls were not the controls' doing. 48 random
whole-document cuts of F (4.66 MB, exchangeable with the draws) met the same 16 draws (seed 0),
as every diagnosis does: 6 were flagged with the derived trunk and 8 with trunk 20. With fresh
draws for each cut, none was flagged. The bar is calibrated over draw sets, about 1 in 17 on
average, but the maximum of 16 draws is itself noisy: across draw sets it ran from 5.6 to 9.6.
Seed 0 set a low bar (6.7, 5.2 with trunk 20), and every corpus read against it inherited a
false-flag rate near 12%. The pattern `Beach` was how this showed. Its draws had 2–7 occurrences,
all with distinct fillers, so their diversity sat at 1.00 with a spread of 0.001, while one cut
held a document that repeated it 10 times.

For the k-th largest of n exchangeable draw maxima, a random corpus's chance of passing it
follows Beta(k, n + 1 − k) over draw sets. A pool of 200 draws confirmed this: the measured
spread across sets matched the Beta at 16, 32, 64 and 128 draws.

**Design.**
- `alpha` (default 0.06) is explicit: a random corpus's chance of a flag in each family (head
  rows, `absent`, class rows). It is the test's level, a tolerance rather than a property of the
  data. 1/17 had been implicit in 16 draws.
- The bar is the k-th largest draw maximum, with k = round(alpha · (draws + 1)).
- `draws` unset is the fewest draws that keep that chance within alpha/2 to 1.5·alpha for 80% of
  draw sets (`_draws_for`). At 0.06 that is 101 draws and k = 6. A pattern is still read when all
  draws but two read it: at 128 draws this reads as many patterns as 14 of 16 did (16,524 against
  16,336).
- The leave-one-out sums are vectorized; the loop over draws took minutes at 100 draws.

**Check.** The same 48 random cuts against one fixed set of 101 draws: 2 head flags (4%), with
no class row and no `absent` flagged.

Fifth sweep, same 22 cuts:

| Tokens/word | Classes found (of 5) | Class flags on 12 controls | Head flags on 12 controls | `absent` on 8 fresh controls | Head flags, `class_cut` / `head_cut` | Dropped words | s per cut |
|---|---|---|---|---|---|---|---|
| 1.35 | 5 | 1 | 0 | 3 | 28 / 9 | 8/8 | 537 |
| 1.40 | 5 | 0 | 0 | 2 | 36 / 11 | 8/8 | 528 |
| 1.50 | 5 | 0 | 0 | 2 | 39 / 13 | 8/8 | 505 |
| 1.60 | 5 | 1 | 1 | 0 | 29 / 8 | 8/8 | 348 |
| 1.70 | 5 | 0 | 3 | 1 | 32 / 10 | 8/8 | 254 |
| 1.80 | 5 | 0 | 0 | 0 | 26 / 9 | 8/8 | 227 |

- **Head flags on controls:** 4 of 72 readings, the designed 6% (it was 9 of 72). Detection rises
  on the cuts with a planted gap.
- **`absent` flags:** all come from two fixed corpora, `r25c` (` luck`, shown 0 times against the
  draws' 7.0 ± 1.8) and `r25e` (`Maybe`), flagged again at several granularities because they are
  the same corpora. Across the 60 random corpora read at 1.50, 2 had an `absent` flag (3%).
- **Class flags on controls:** 2 of 72, a county-name node and `San`/`Haw`.
- **Stability:** under-shown and within-noise swap 26–37 times in 330,000–690,000 patterns.
- **Cost:** 101 draws make a cut 3–5 times slower, 4–9 minutes at these sizes.
- **Alpha per family, not split.** A random corpus can be flagged in any of three families, so
  in the worst case its chance of some flag is three times alpha (18%). Measured, it is not: 2 of
  48 random cuts drew any flag (4%), both head flags, because the families are not independent and
  `absent` and class rows run below alpha. Each family answers its own question ("these heads
  lack fillers", "this word is missing", "this class is missing"), and its alpha says how likely
  each flag is to be noise. Splitting alpha three ways would need about 300 draws, 12–27 minutes
  a cut. The report states the level and this choice.

**Step 1 result (final).** Verdicts hold from 1.35 to at least 1.80 tokens per word. The trunk
read and the draw count come from the data and alpha. The one setting left is alpha = 0.06, the
test's level, stated in every report. Without a reference, `low = 0.5` and trunk 20 remain until
step 4.

## Step 2: `CONTEXT_TOKENS` (2026-10-12)

The reading at 1.50 tokens per word was repeated with 250, 500, 2,000 and 4,000 context tokens
(51% to 87% of running tokens) instead of 1,000. The 22 labeled cuts were diagnosed against
each reading (`experiments/class_induction`).

- **Cut-level verdicts are unchanged.**
  - All five removed classes are found at every value.
  - There are no class flags on controls.
  - Head flags on controls stay at the design rate (0 to 2 of 12).
  - The same two absent rows appear on controls at every value.
  - All 8 dropped words are found.
  - The bars move by less than their own noise.
- **The partition changes.** The coarse cut gives 11 to 18 classes, with no trend. Pattern keys
  shift with it.
  - Patterns read at both values agree on 75–86% of verdicts, almost all of the difference
    being read against unread.
  - 33 to about 120 patterns swap between a flag and within-noise.

`CONTEXT_TOKENS` stays at 1,000, shown not to matter. The partition's instability points at the
cut, `min_classes = 8`, `max_classes = 64` and `NOISE_FLOOR = 0.05`, which are measured next.

## Step 2: the coarse cut (2026-10-13)

`min_classes`, `max_classes` and `NOISE_FLOOR` only choose where the Ward tree is cut into
filler classes; the default lands at 12 at 1.50 tokens per word. The cut was forced to 8, 16,
24, 32 and 64 classes and the 22 cuts diagnosed again (`experiments/class_induction`).

- **Class rows are unchanged at every k.** They are read on the whole tree and its filler tokens,
  so the coarse cut never enters them. The same nodes are flagged with the same t, and the class
  bar is 13.92 throughout.
- **Cut-level head and absent verdicts hold.**
  - The class and head cuts keep 33–43 and 10–14 head flags.
  - Controls get at most one head row (k = 64) and one absent row each.
  - All 8 dropped words are found.
  - The bars stay within 4.77–4.98 (head) and 3.73–3.88 (absent).
- **Pattern-level agreement with 12 classes falls from 97% (8) to 63% (64).** A head row is the
  head followed by one filler class, so a different cut asks a different question. It is not a
  verdict on the same pattern changing.

The three settings stay as they are, shown not to change the verdicts. **Step 2 is closed**:
`CONTEXT_TOKENS` and the coarse cut do not move the cut-level verdicts within the ranges
measured (250 to 4,000 context tokens; 8 to 64 classes).

## Step 3: the merge budget from the granularity (2026-10-13)

A merge budget means nothing across corpora: on the FineWeb pool, 4,000 merges (the old default)
leave 1.76 tokens per word, while on 30 recipe documents the merges run out at 269, near whole
words. What the verdicts were shown to depend on is tokens per word (step 1), so the budget is now
read from it.

- **Budget unset** (library and command): `learn_to_granularity` learns the domain's tokenizer
  (the reference's, or the corpus's without one) to the fewest merges that reach
  `GRANULARITY_TARGET = 1.50` tokens per word. Learning resumes in doubling rounds from 1,024
  merges, so at most twice the merges needed are learned. On the FineWeb pool it stops at
  11,381 merges, exactly the budget step 1 read from a 60,000-merge run and a prefix of that
  run, in 110 s.
- **The target.** Any granularity from 1.35 to 1.80 gives the same verdicts (step 1), so the
  target is shown not to matter within the range. 1.50 has margin on both sides, and it is
  where the test's level and the class settings were checked (steps 1 and 2).
- **`--merges N`, given:** the command learns the tokenizer and reads the domain's tokens per
  word *before* the diagnosis. Outside `GRANULARITY_RANGE = (1.35, 1.80)` it refuses (exit 2)
  and names the granularity reached, unless `--force` is passed. A domain whose merges run out
  before 1.50 is refused the same way. The library does not refuse, because experiments read
  granularities outside the range on purpose; it adds a warning to the report instead, which
  replaces the old fixed "more than 1.6 tokens per word" warning.
- `tokens_per_word_by_budget` moved from `experiments/granularity_range` to
  `tokenizer/granularity.py`. Its curve equals the encoding at every budget
  (`tests/test_granularity.py`).

## Step 4: patterns without a reference (2026-10-13)

The goal makes a reference optional. Without one, `low = 0.5` and a trunk of 20 decided
`sufficient` or `restricted`, the last fixed settings. The agreed reading: a pattern the corpus
shows weakly is weak, objectively, because that is what a model trained on it sees. A reference
added later refines the verdict: a pattern weak in both is the domain's own (`fixed`).

**Saturation alone does not find a weak pattern** (`experiments/saturation`). The slope of a
slot's filler-accumulation curve is its Good-Turing missing mass, so no regrowing is needed.
The pronoun → VERB pattern was thinned to 5–50% at equal bytes:
- Its missing mass rose from 0.028 to about 0.05.
- But the median node of every corpus reads 0.40, so no tolerance flags the thinned slot
  without flagging most of the corpus.
- Removed outright, the slot reads 0.028 again, as saturated as the control.

**Against its sibling, the corpus can judge.**
- Each child of a head-tree node (a head, or a class of heads) is compared with its parent, per
  filler class.
- The null: had the parent's occurrences of that class fallen to its two children at random,
  how many distinct fillers would this child show?
- The draws are exact random splits, without replacement, 101 per test.
- The bar is the draws' maxima at alpha. Only tests a single filler could fail are read
  (Tarone).
- Two-token heads are placed by their first token.
- Only nodes inside a coarse class are tested.

Verdicts: `weak`, past the bar; `rare`, shown fewer times than any slot the test can read;
`within-noise`; `unread`, with no sibling to read against. `low` and the reference-free trunk
are gone (`analysis/siblings.py`).

Results:
- In head_cut, ` they` shows 3 verbs where a random split gives 26.5 (t 9.6, bar 6.3). No
  pronoun of the control is flagged.
- On control's two halves, read on control's tree and classes, 264 weak tests are shared, of
  332 and 397.
- In the restricted recipes, 11 of 14 verbs are weak. The other three share their only
  ingredient with their sibling. No verb is weak in the diverse recipes.
- On FineWeb the integrated tool flags 95 head rows in control and 120 in head_cut, ` she` and
  ` they` among them, in 2.5 minutes.

**Rejected on FineWeb.** These variants were rejected because their flags did not replicate:
- comparing each head with its whole coarse class;
- adding a test of how often a head takes each filler class (with `absent`);
- grouping heads by their left context, or by the context past the pattern.

They flagged 2,000 heads and 14,000 classes in the control, every pronoun among them. Their
halves shared 12–17% of the flags (`experiments/saturation/README.md`).

**Limits.** Without a reference, three cases go unseen:
- a pattern the corpus lacks altogether;
- a class shown weakly as a whole;
- a head restricted enough for the tree to set it apart from its siblings.

The report says so and points to a reference from the same domain. Most of the corpus's weak
patterns are its grammar (` to` against the other prepositions, reporting verbs). Read without a
reference, that is correct: they are weak in the corpus.

**Step 4 closed (2026-10-13)**, with these limits. Next: re-run the paper's experiments.

