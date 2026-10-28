# The class tree without the n² table

**Status:** experiment done (2026-10-28), result in `experiments/ward_memory/README.md`. Not yet applied: `class_tree` in `src/` still uses SciPy alone. Open for the user: the size at which it switches to `linkage_vector`, and whether ties get a fixed order.

Agreed with the user on 2026-10-27. Design and predictions, written before running.

## Why

`class_tree` builds the Ward tree with `scipy.cluster.hierarchy.linkage(features, "ward")`.
That call first computes every pairwise distance between the clustered tokens and keeps them in
RAM: n(n − 1)/2 doubles.

| Reading | Tokens clustered | Pairwise table alone | Result |
|---|---|---|---|
| F at 1.35 | 29,274 | ~3.4 GB | ran |
| F at 1.30 | 45,748 | ~8.4 GB | did not fit in 16 GB (step 1) |
| Wikipedia at 1.35 | 38,580 | ~6.0 GB | skipped by the memory rule |
| Wikipedia at 1.30 | 57,942 | ~13.4 GB | skipped by the memory rule |

The rest of the diagnosis grows with the text, not with n². So this table sets the finest
granularity a corpus can be read at, and it is what a larger corpus would hit first, since a
larger corpus needs more merges, and so more tokens, to reach the same granularity.

`fastcluster.linkage_vector(features, method="ward")` computes the same Ward clustering from the
feature vectors, without the table. Its memory grows with n · d, the size of the features
themselves, which are already in RAM. The metric must be Euclidean, and ours is: features are
square-rooted distributions, so Euclidean distance is Hellinger distance.

## What "the same" means

Both compute exact Ward, but by different arithmetic:

- SciPy updates the table with the Lance–Williams formula;
- `linkage_vector` recomputes distances from centroids.

They can differ in the last bits, and where two merges are nearly tied, in their order. So the
check is in three levels, from strictest to what the paper reports:

1. **Tree:**
   - the same sequence of merged pairs, read as the clusters they form, so labels can differ;
   - heights equal to a relative 1e-9.

   Where the sequences differ, the differing steps are listed with the height gap between the
   two candidate merges.
2. **Classes:** the coarse cut (`cut()`, the largest height jump between 8 and 64 classes)
   gives the same partition of tokens, and every internal node, read as its set of tokens,
   exists in both trees. Class patterns are read at every node, so a node present in only one
   tree is a pattern only one reading has.
3. **Verdicts:** the 22 labeled cuts diagnosed against each reading give the same verdict for
   every pattern, head rows, class rows and `absent`.

## Readings

- **Synthetic:** the recipe grammar's tree, in a test.
- **Natural:** the three domains' pools at 1.50 tokens per word (TinyStories, F, Wikipedia).
  Levels 1 and 2 are read on all three. Level 3 is read on TinyStories, whose 22 cuts take about
  35 minutes, and on F.
- **Memory and time:** peak resident memory and wall time of the clustering step for each
  reading, and for F at 1.30 and Wikipedia at 1.35 and 1.30, which do not fit today.

## Predictions

1. **Level 1 holds** on the recipe grammar and on the three pools, or fails only at steps whose
   two candidate merges are within 1e-9 of each other relative to their height.
2. **Level 2 holds** on all three pools: the same coarse partition and the same set of nodes.
3. **Level 3 holds:** no verdict changes on either domain's 22 cuts.
4. **Memory.** `linkage_vector` clusters F at 1.30 and Wikipedia at 1.30 within 16 GB, with a
   clustering-step peak under 4 GB.
5. **Time.** `linkage_vector` is slower: no more than 5 times SciPy's clustering time on the
   three pools at 1.50.

## What follows

- **Predictions 1–3 hold:** `class_tree` switches to `linkage_vector`, with a test that pins the
  equality against SciPy on the recipe grammar. `fastcluster` becomes a dependency. The memory
  rule in `experiments/granularity_domains` is dropped, and the readings it skipped are run
  (F 1.30, Wikipedia 1.30 and 1.35).
- **Level 3 fails anywhere:** the switch is not made. The differing patterns are reported for the
  user to judge, since a near-tie that changes a verdict would mean the verdict was fragile
  before.
- **Prediction 5 fails badly:** the switch is made only below a size, decided with the user.
