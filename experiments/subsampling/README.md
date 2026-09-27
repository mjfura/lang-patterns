# Subsampling: does an AUC measure structure or corpus size?

`subsample_auc.py <wikipedia|fineweb>` fixes the tokenizer (byte-level, 3000 merges, learned once
on the first 5000 documents, as `analysis-15-fineweb` / `analysis-17-wikipedia-en` do), loads
20,000 documents, and scores random subsets of 1250 / 2500 / 5000 / 10,000 / 20,000 documents
(3 seeds below the full set) on the four curves of the order-1 `<bos>`/`<eos>`-framed graph, each
with its linear and its log AUC. Results: `<corpus>_results.json`, one row per sample. Run with
`poetry run python experiments/subsampling/subsample_auc.py wikipedia`; the first run downloads
the documents and caches their token ids in `<corpus>_ids.json` (gitignored).

## Result (2026-09-27, means over seeds; linear / log AUC)

| Corpus | Docs | Tokens | max count | red | blue | orange | green |
|---|---|---|---|---|---|---|---|
| FineWeb | 1,250 | 1.1M | 8,654 | 0.008 / 0.399 | 0.042 / 0.604 | 0.027 / 0.213 | 0.266 / 0.738 |
| FineWeb | 2,500 | 2.3M | 17,867 | 0.008 / 0.442 | 0.042 / 0.634 | 0.028 / 0.271 | 0.268 / 0.766 |
| FineWeb | 5,000 | 4.6M | 35,245 | 0.007 / 0.476 | 0.042 / 0.657 | 0.026 / 0.319 | 0.271 / 0.787 |
| FineWeb | 10,000 | 9.4M | 71,258 | 0.007 / 0.508 | 0.042 / 0.679 | 0.025 / 0.366 | 0.268 / 0.803 |
| FineWeb | 20,000 | 18.5M | 142,083 | 0.007 / 0.535 | 0.042 / 0.697 | 0.026 / 0.411 | 0.269 / 0.818 |
| Wikipedia | 1,250 | 2.6M | 12,697 | 0.011 / 0.472 | 0.066 / 0.667 | 0.016 / 0.242 | 0.231 / 0.751 |
| Wikipedia | 2,500 | 5.3M | 25,351 | 0.011 / 0.507 | 0.067 / 0.691 | 0.016 / 0.293 | 0.231 / 0.772 |
| Wikipedia | 5,000 | 10.8M | 51,174 | 0.011 / 0.538 | 0.067 / 0.711 | 0.016 / 0.340 | 0.232 / 0.791 |
| Wikipedia | 10,000 | 21.8M | 103,194 | 0.011 / 0.566 | 0.067 / 0.729 | 0.016 / 0.383 | 0.232 / 0.807 |
| Wikipedia | 20,000 | 43.9M | 208,381 | 0.011 / 0.590 | 0.067 / 0.744 | 0.016 / 0.422 | 0.233 / 0.820 |

Seed-to-seed spread at a fixed size is ≤ 0.01 on every AUC.

- **Linear AUC is size-invariant but barely discriminates.** Scaling every count by `c` scales the
  maximum by `c` too, so `threshold / maximum` — and the area — is unchanged. But the axis is
  owned by the single highest count: on these corpora all the structure lives in its first ~1%.
- **Log AUC discriminates but grows with size**, ~+0.03 per doubling of the corpus, regularly,
  because `log(c·x) / log(c·max)` moves every threshold right.
- **At matched token counts Wikipedia still reads above FineWeb** (red log 0.472 at 2.6M tokens vs
  0.442 at 2.3M; 0.538 at 10.8M vs 0.508 at 9.4M): part of the gap is structure, not size.

Consequence: log AUC is only comparable between corpora at an equal token budget; how it grows
with size (its slope against `log(tokens)`) is a candidate indicator in its own right.
