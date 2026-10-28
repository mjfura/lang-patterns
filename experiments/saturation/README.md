# Saturation: class patterns without a reference

Step 4 of `docs/superpowers/specs/2026-10-11-settings-from-data-design.md`. Without a reference, can a corpus show
by its own evidence that a class pattern is under-shown? A slot that is still discovering
fillers has a filler-accumulation curve that has not flattened. The expected slope of that curve
at the full corpus is the Good-Turing missing mass: fillers seen once, divided by the slot's
trunk. So no subsampling is needed.

**Cuts.** The pronoun → VERB pattern of `experiments/class_patterns` is thinned rather than
removed. `thin_<p>` keeps a random share p of the pronoun target sentences. Every corpus is cut
to the bytes of `class_cut`, which keeps none. Modal target sentences are removed in all of them,
so the modal slot is the removed-pattern case throughout. Each corpus is read on its own, with
no reference: its own tokenizer at 1.50 tokens per word and its own classes.

    poetry run python experiments/saturation/saturation.py segment
    poetry run python experiments/saturation/saturation.py read <control|class_cut|thin_0.05|...>
    poetry run python experiments/saturation/saturation.py summarize

## Result (2026-10-13): saturation does not detect a thinned class pattern

**1. The pronoun slot as a whole.** This is the six heads ` I you he she we they` pooled,
followed by any filler.

| Corpus | Trunk | Fillers | Missing mass | Nodes of similar trunk more saturated |
|---|---|---|---|---|
| control | 22,639 | 1,791 | 0.028 | 13% |
| thin_0.5 | 13,819 | 1,439 | 0.046 | 3.5% |
| thin_0.25 | 10,900 | 1,173 | 0.054 | 1.7% |
| thin_0.1 | 9,027 | 845 | 0.051 | 0.9% |
| thin_0.05 | 8,350 | 704 | 0.048 | 1.3% |
| class_cut | 7,720 | 433 | 0.028 | 0.4% |

The predicted shape holds: thinning nearly doubles the missing mass, and removal brings it back
to the control's level. The size of the change does not make it detectable, though:

- Even thinned to 5%, the slot is among the most saturated nodes of its size.
- The median node of every corpus has a missing mass of about 0.40.
- Any tolerance that flagged 0.05 would flag nearly the whole corpus.
- The modal slot, removed everywhere, reads 0.026–0.027. The control's own modal slot reads
  0.039, so removal makes a slot look *more* saturated.

**2. Only the verb fillers.** The thinned part is the verbs, so the missing mass was also read
within the induced filler class that holds ` know` ` think` ` get` ` see`. This does not give a
stable reading either, because each corpus induces its own classes and the thinning changes
them:

- In control, the verbs share one class of 821 fillers with the auxiliaries (` have` ` was`
  ` are`), with missing mass 0.008.
- In thin_0.1 they form their own class: 135 fillers, missing mass 0.134.
- In thin_0.05 they fall in with determiners (` a` ` the` ` first`), with missing mass 0.289.
- In class_cut, 3 verb tokens remain.

Even the clean case, 0.134 at trunk 462, is below the typical node of that size.

**Reading.** At this corpus size (4.7 MB) the missing mass rises when a pattern is thinned, but
only relative to the same corpus unthinned, which is to say relative to a reference. Read alone,
a thinned class pattern looks as saturated as, or more than, the corpus's ordinary slots. A
pattern removed outright looks more saturated still.

## Second approach (2026-10-13): each child against its siblings

`siblings.py` asks the question within the corpus. For every child of a head-tree node (a single
head, or a class of heads) and every filler class, it computes how many distinct fillers the
child would show had its parent's occurrences of that class fallen to the children at random.
The expectation is exact: rarefaction without replacement. Ratio = shown / expected.

    poetry run python experiments/saturation/siblings.py read <name>
    poetry run python experiments/saturation/siblings.py summarize

**Head level: works.** In `head_cut` (the verb after ` she` and ` they` removed), ` they` shows
3 verbs where a random split of its parent would give 26.6, a ratio of 0.11. In control the
pronouns read 0.92–1.00. ` she` keeps only 2 verb occurrences, too few to read.

**Class level, thinned for every pronoun: invisible to siblings, as predicted.** The siblings
are thinned too. The pronoun node reads 0.94 (thin_0.25) and 1.06 (thin_0.1). The thinning shows
only in how often the slot occurs: the pronoun node's verb-class trunk is 16,422 in control,
7,565 at 25%, 484 at 10% and 24 in class_cut.

**A head-level gap does not stand out from the corpus's own heterogeneity.** Every corpus,
control included, has tests this extreme:

| Corpus | Tests | Ratio < 0.5 | Ratio < 0.25 |
|---|---|---|---|
| control | 60,822 | 764 | 70 |
| head_cut | 69,710 | 706 | 48 |
| class_cut | 88,582 | 1,095 | 67 |

So ` they` in head_cut becomes one more of the corpus's ~70 strongly weak child patterns, not an
outlier among them. Read without a reference, that is the right reading: the corpus does show
those patterns weakly. Whether the domain shares them is what a reference settles.

### Calibrated, within classes (2026-10-13)

**Method.**
- Each test draws 101 random splits of the parent's occurrences of the filler class, exactly and
  without replacement, at the child's trunk.
- The bar is the 6th largest of the draws' maxima (alpha 0.06), as with a reference.
- A test is read only if a child showing a single filler could cross the bar (Tarone).
- Only children of a node inside one coarse class are tested. Above the class cut, the two sides
  of a split are different categories (`,`/`the`/`to` against `.`/`:`), and those tests
  dominated the flags.
- An earlier run used independent filler presences. That overstated the spread: ` they` in
  head_cut could not have been flagged even showing a single filler.

**Results.**

| Corpus | Read | Bar | Weak (of which single heads) | ` they` → verbs | Pronoun node → verbs |
|---|---|---|---|---|---|
| control | 41,511 | 6.46 | 504 (71) | 364 vs 367, t 0.4 | 732 vs 825, t 14.6, **weak** |
| head_cut | 48,157 | 6.32 | 478 (80) | 3 vs 26.5, t 9.6, **weak** | t 1.3 |
| thin_0.25 | 51,110 | 6.42 | 515 (68) | t −1.9 | t 6.0 |
| thin_0.1 | 57,738 | 5.91 | 657 (102) | t 0.2 | t −2.6 |
| class_cut | 55,615 | 6.77 | 415 (64) | 1 occurrence | 19 vs 19, t 0.0 |

**Replication.** Control's two random halves were read on control's own tokenizer, tree and
classes, so every test is the same test in both halves. 264 weak tests are shared, out of 332
(half a) and 397 (half b). The pronoun node is weak in both halves (t 9.1 and 10.9). So the flags
are reproducible properties of the corpus, not noise of the split.

**What the flags are.** They are mostly heads that grammar narrows within their coarse class:
- ` to` takes fewer `, the to and` fillers than the prepositions (524 vs 871).
- Reporting verbs (` said` ` says` ` means`) take fewer than other verbs.
- Auxiliaries take almost no word-piece fillers.

Read without a reference, these are weak patterns of the corpus. A reference would mark them as
the domain's own (`fixed`). The planted gap (` they` in head_cut) is flagged, and it looks like
them. Class-level thinning (every pronoun at once) leaves no sibling stronger to compare with,
and is not flagged.

## Groupings tried for the tool, and the one kept (2026-10-13)

`siblings.py tool <name>` runs `corpus-diagnose` itself, without a reference, at heads of one
token, and writes to `results/tool/`.

**Variants rejected on FineWeb.** Three variants also tested how often a head takes each filler
class (a trunk test, with `absent` where it never does), and compared each head with its whole
coarse class:

| Grouping of heads | Control: weak / absent / class rows | Flags shared by the halves |
|---|---|---|
| left context only | 522 / 63 / 3,163 | 51 of ~400 |
| context past the pattern | 2,017 / 254 / 13,852 | 171 of ~1,000 |
| both sides, beside the head | 1,987 / 266 / 14,822 | 118 of ~1,000 |

In every one, every pronoun of the control read weak in most filler classes. The coarse classes
of FineWeb hold thousands of heads across categories (` the` ` a` ` I` ` it` ` you` in one), and
the trunk test read those category differences as weakness.

**Kept: each child against its sibling, diversity only, on the corpus's own head tree.**

| Corpus | Weak head rows | Pronoun head rows weak | Class rows | Head bar / class bar |
|---|---|---|---|---|
| control | 95 | none | 1,490 | 5.61 / 6.19 |
| head_cut | 120 | ` she` (` the to and`), ` they` (` be do get`, ` the to and`) | 1,635 | 5.38 / 5.80 |

The halves here are read with their own trees and classes, so their flags cannot be matched test
by test. The matched replication is the one above: 264 shared of 332 and 397.

**Known limits, without a reference.** A pattern the corpus lacks altogether, and a class shown
weakly as a whole (every pronoun thinned, `class_cut`), leave no sibling stronger to compare with.
A head restricted enough for the tree to set it apart also loses its sibling: in the synthetic
subject-verb corpus, ` she` and ` they` with two verbs form their own class. A reference sees all
three.
