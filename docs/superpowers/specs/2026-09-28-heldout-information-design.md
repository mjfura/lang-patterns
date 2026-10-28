# Held-out information gain: does the corpus hold structure that carries to unseen text?

**Status:** done. `analysis/information.py`; result in `experiments/information/README.md`. Its "Next" items were not taken up: the syntactic-sufficiency indicator replaced them.

**Goal:** the core learnability indicator, agreed on 2026-09-28. The graph curves measure the
topology of the transitions; a model learns the **conditional probabilities**, and "learnable"
means that what is estimated on one part of a corpus predicts another part. This measures that
directly, on the corpus alone, before any training. The curves stay as diagnosis (where the
structure is), and every report puts this measure next to them.

## Definition

The corpus's documents are split in two halves by a seeded shuffle, **A** (fit) and **B**
(evaluation). A byte-level BPE tokenizer with the run's merge budget is learned on A, as
`corpus-train` learns it on its train split. Each document is framed `<bos>` + tokens + `<eos>`.

For each order `k` in 0–3, an interpolated n-gram model is fitted on A's counts: order 0 is the
unigram, order `k` conditions on the previous `k` tokens. Smoothing is **Witten-Bell
interpolation** down to a uniform distribution over the tokenizer's vocabulary, so every token of B
has a nonzero probability and every conditional distribution sums to 1. Witten-Bell rather than
Kneser-Ney keeps the estimator short and parameter-free; this measure compares corpora, it does
not compete on perplexity.

Cross-entropy is reported in **bits per byte**: the total `−log2 p` of every predicted token of a
half (every token after `<bos>`, `<eos>` included) divided by the UTF-8 bytes of that half's
documents. Per byte rather than per token, the number does not depend on how many bytes a token
spans, which removes most of the merge-budget confound `experiments/granularity` found.

- `heldout_bpb[k]`: cross-entropy of the order-`k` model on B;
- `fit_bpb[k]`: the same on A, the half it was fitted on;
- **information gain** `gain[k] = heldout_bpb[0] − heldout_bpb[k]`: the bits per byte the previous
  `k` tokens save on unseen text;
- **generalization gap** `heldout_bpb[k] − fit_bpb[k]`: how much of what order `k` learned was
  specific to A.

**Revised after `experiments/information`:** bits per byte make the **best held-out bits per
byte** (the minimum over orders) independent of the merge budget, within 0.04 across 1000–16000
merges; the gain over order 0 is not, because larger tokens absorb within-word structure into the
unigram. The readings used are the best held-out bits per byte (predictability) and the
word-shuffled null's best minus the corpus's (order). The halves keep every copy of a text on
one side (`split_halves`); split independently, duplicates leaked into the held-out half.

Readings as first planned: a gain above zero that grows with `k` is structure that carries to unseen text; a gain
near zero is text with no usable order (the shuffled contamination); a large gap at high `k` is
structure the corpus does not repeat often enough to generalize.

## Implementation — `src/corpus_generalization/analysis/information.py`

`heldout_information(documents, merge_budget, orders=(0, 1, 2, 3), seed=0)` returns a frozen
dataclass with the four per-order readings, the halves' document, token and byte counts, and the
vocabulary size. Counts are `Counter`s keyed by context tuple; Witten-Bell needs, per context, its
total count and number of distinct continuations, both kept alongside.

## Tests (before the code)

1. **A proper distribution:** for fitted contexts and an unseen one, the order-`k` probabilities
   over the vocabulary sum to 1.
2. **Random bytes:** on documents of uniformly random bytes with no merges, order 0 reads close
   to 8 bits per byte and the gain at every order is close to zero.
3. **Structure:** on a deterministic cycle ("a b c a b c ..." as words), `gain[1]` is large and
   the order-1 held-out bits per byte are near zero.
4. **Order destroyed:** the same grammar-generated documents, and their words shuffled within
   each document, give a clearly lower `gain[2]` for the shuffled version.
5. **Determinism:** the same seed gives the same readings; another seed gives another split.

## Experiment — `experiments/information/`

On corpora already built: the contamination variants (seed 0, every kind and dose) and the clean
sources' first samples, at 1000, 4000 and 16000 merges. The questions:

- Does the gain separate shuffled text, duplicates and Python from the clean control, where red
  and blue did not?
- Is it comparable across domains, and stable across merge budgets in bits per byte?

## Next, not in this step

Mutual-information decay with distance (long-range structure) and missing mass counted with
context support (sample sufficiency), the other two parts of the indicator.
