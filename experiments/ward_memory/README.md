# The class tree without the n² table

["The class tree without the n² table"](../../docs/superpowers/specs/2026-10-27-memory-light-ward-design.md) sets
out the design and five predictions, written before running.

`class_tree` builds its Ward tree with `scipy.cluster.hierarchy.linkage`, which keeps every
pairwise distance in RAM, n(n − 1)/2 doubles. That table is what kept F at 1.30 tokens per word,
and Wikipedia at 1.30 and 1.35, out of 16 GB. This experiment compares it with
`fastcluster.linkage_vector`, which computes the same Ward clustering from the feature vectors
without the table.

    poetry run python experiments/ward_memory/ward_memory.py features <domain> <t>
    poetry run python experiments/ward_memory/ward_memory.py link <domain> <t> scipy|vector
    poetry run python experiments/ward_memory/ward_memory.py compare <domain> <t>
    WORKERS=2 poetry run python experiments/ward_memory/ward_memory.py verdicts <domain>
    poetry run python experiments/ward_memory/ward_memory.py summarize
    WORKERS=2 poetry run python experiments/ward_memory/ward_memory.py sweep <domain> <t>

The features are exactly what `class_tree` clusters for a domain's reading. Each `link` runs in
its own process so that its peak memory is its own. Features and trees go to `runs/ward_memory/`
(gitignored). The level-3 readings in `results/vector/` are gitignored; the comparisons, link
records and `summary.json` are kept.

## Result (2026-10-28)

| Reading | Tokens | SciPy | `linkage_vector` |
|---|---|---|---|
| TinyStories 1.50 | 1,300 | 0.3 s, 0.40 GB | 5 s, 0.41 GB |
| F 1.50 | 11,459 | 44 s, 1.38 GB | 442 s, 0.74 GB |
| Wikipedia 1.50 | 15,893 | 91 s, 1.76 GB | 856 s, 0.88 GB |
| Wikipedia 1.35 | 38,582 | not run (~6 GB table) | 1 h 29 min, 1.61 GB |
| F 1.30 | 45,748 | did not fit in 16 GB (step 1) | 2 h 8 min, 1.84 GB |
| Wikipedia 1.30 | 57,944 | not run (~13 GB table) | 10 h 16 min, 2.01 GB |

Peak memory is the whole process's, the features included.

**The trees.** Where no two merges tie, the two trees are the same:

- **Same merges:** the same clusters in the same order.
- **Heights:** equal to within 7·10⁻¹⁵ relative.
- **Coarse classes:** the same partition in all three domains (11, 12 and 9 classes).

They differ only at exact ties. A tie is a set of tokens at exactly the same distance from each
other, and each library breaks it in its own order:

| Domain | Steps that differ | Where | Nodes in only one tree |
|---|---|---|---|
| TinyStories | 0 of 1,299 | — | none |
| F | 2 of 11,458 | height 0: four word pieces with identical neighbors (`GISTR`, `itage`, `zheimer`, `secutive`) | 1 each way |
| Wikipedia | 12 of 15,892 | heights 0, 1.0, 1.0855 and √2, each exactly repeated | 3 each way |

**The verdicts.** On the 22 labeled cuts of TinyStories and of F at 1.50, every verdict is the
same. That covers every head row, every class row and every `absent` row.

### Against the predictions

1. **Failed as written.** The tree differs at 2 steps in F and 12 in Wikipedia. Every difference
   is an exact tie: the prediction allowed near-ties within 1e-9, and these are at a gap of 0.
2. **Failed as written.** The coarse partition is the same in all three domains. A few nodes
   formed inside the ties exist in only one tree: 1 in F, 3 in Wikipedia.
3. **Held.** No verdict changes on either domain's 22 cuts.
4. **Held.** `linkage_vector` clusters F at 1.30 at a 1.84 GB peak and Wikipedia at 1.30 at
   2.01 GB, both under 4 GB.
5. **Failed.** `linkage_vector` is 9.4 to 10.8 times slower, against a bound of 5. The time grows
   faster than n² at the largest size: Wikipedia at 1.30 took 4.8 times as long as F at 1.30 for
   1.27 times the tokens, where n² predicts 1.6.

## What this means

`linkage_vector` computes the same Ward tree. The only differences are how exact ties are broken,
and none of them changes a verdict, so the memory problem has an exact solution.

The cost moves to time. At the tool's target, 1.50 tokens per word, SciPy fits easily (1.8 GB at
most here) and is 10 times faster. `linkage_vector` is needed only where the table does not fit,
and there it takes hours.

`sweep` diagnoses a domain's 22 cuts at one granularity on the saved `linkage_vector` tree. It read
the three readings the memory rule had skipped in `experiments/granularity_domains`: F at 1.30 and
Wikipedia at 1.35 and 1.30, about 1.5 hours each on top of the tree. All three hold, so every
declared granularity from 1.30 to 2.00 holds in all three domains.

So the design's last rule applies: the switch is made only above a size, decided with the user.
Two things are still open: where that size is, and whether a tie-breaking rule should make the
two trees identical by construction rather than only in their verdicts.
