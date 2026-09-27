# Merge sweep: how do the curves and islands move with the merge budget?

`merge_sweep.py <actions|states-x-space|fineweb>` learns a byte-level tokenizer on the whole corpus
at each merge budget in 0 / 16 / 64 / 250 / 1000 / 4000 / 16000, and scores that tokenizer's
order-1 `<bos>`/`<eos>`-framed graph: the four curves' linear and log AUC, and the island
decomposition at count >= 2. Token count and bytes per token are recorded next to them because
both change with the budget, and log AUC grows with token count (`experiments/subsampling`).
Results: `<corpus>_results.json`, one row per budget. Run with
`poetry run python experiments/merge_sweep/merge_sweep.py fineweb`; the first run downloads the
5000 documents `analysis-15-fineweb` reads and caches them in `fineweb_documents.json` (gitignored).

The question: the graph is built on the model's own tokenizer, so its nodes range from bytes (few
merges) to near-words (many). Does the reading change with that, and does the profile across
budgets tell corpora apart?

## Result (2026-09-27; red log AUC, shares at count >= 2)

`states-x-space` (`analysis-09`, two hand-built domains; saturates at 330 merges, 305 tokens):

| Merges | Bytes/token | Tokens | red | blue | orange | green | Mainland | Islands | Isolated |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 4,447 | 0.499 | 0.640 | 0.277 | 0.481 | 92% | 0% | 8% |
| 16 | 1.3 | 3,485 | 0.510 | 0.751 | 0.310 | 0.620 | 94% | 0% | 6% |
| 64 | 1.8 | 2,526 | 0.366 | 0.617 | 0.196 | 0.585 | 88% | 0% | 12% |
| 250 | 2.8 | 1,610 | 0.114 | 0.363 | 0.014 | 0.479 | 49% | 9% (10) | 42% |
| 330 (saturated) | 3.1 | 1,450 | 0.065 | 0.299 | 0.007 | 0.376 | 21% | 3% (5) | 76% |

`actions` (`analysis-04`, templates over 18 words; saturates at 64 merges):

| Merges | Bytes/token | Tokens | red | blue | Mainland | Isolated |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 2,143 | 0.695 | 0.816 | 100% | 0% |
| 16 | 1.7 | 1,262 | 0.698 | 0.877 | 100% | 0% |
| 64 (saturated) | 4.4 | 492 | 0.563 | 0.801 | 100% | 0% |

FineWeb (5000 documents; never saturates):

| Merges | Bytes/token | Tokens | red | blue | orange | green | Mainland | Isolated |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.0 | 15.6M | 0.497 | 0.569 | 0.308 | 0.538 | 98.9% | 1.1% |
| 16 | 1.2 | 13.0M | 0.570 | 0.626 | 0.372 | 0.599 | 99.0% | 1.0% |
| 64 | 1.5 | 10.3M | 0.637 | 0.720 | 0.453 | 0.698 | 99.2% | 0.8% |
| 250 | 2.0 | 7.9M | 0.656 | 0.816 | 0.555 | 0.804 | 99.5% | 0.5% |
| 1000 | 2.6 | 6.0M | 0.588 | 0.772 | 0.479 | 0.841 | 99.8% | 0.2% |
| 4000 | 3.4 | 4.6M | 0.447 | 0.630 | 0.280 | 0.746 | 99.9% | 0.1% |
| 16000 | 4.1 | 3.8M | 0.277 | 0.458 | 0.133 | 0.519 | 99.4% | 0.6% |

- **The reading depends on the budget.** Every corpus reads cohesive at the byte level (red
  0.50–0.70): any text in one language repeats its letter patterns. What separates corpora is
  what survives as nodes approach words: `actions` keeps its whole vocabulary on the mainland at
  word level, `states-x-space` falls to 0.065 with 76% of its tokens isolated.
- **The profile is not monotonic.** Every corpus rises over the first merges (FineWeb peaks at
  250), since the first merges pack the most frequent pairs into single nodes before the
  vocabulary grows enough to spread the counts thin.
- **The drop is not mainly fewer tokens.** From its peak FineWeb loses ~2 doublings of tokens
  (7.9M → 3.8M), which the subsampling slope (~+0.03 per doubling) puts at ~0.06 of red log AUC;
  the measured drop is 0.38. `states-x-space` loses ~1.6 doublings for a 0.43 drop, though that
  slope was measured on FineWeb, not a corpus this small.
- **Islands at count >= 2 say nothing on a large corpus.** On FineWeb nearly every token has some
  transition seen twice at every budget, so the mainland stays at ~99% and there are no islands.
  A fixed threshold of 2 is only a meaningful bar for a small corpus; a large one needs a
  threshold that scales with its size.

Consequence: a corpus-quality score has to be read per merge budget, and the informative part is
where the profile falls, not the byte-level end. The islands threshold needs to scale with corpus
size before it can be used beyond small corpora.
