# Clean domains: do the single-corpus readings have domain-free values?

`clean_domains.py` scores clean text from eight sources on the two single-corpus readings
`experiments/contamination` found: the **red gap** (red log AUC over `count` minus over
`context_support`, at 16000 merges; redundancy) and **green over `context_support`** at 4000
merges (order). Each source gives two consecutive samples of 6,495,298 bytes, the size of the
contamination base, as whole documents. `download <name>` caches a source's documents
(`<name>_documents.json`, gitignored), `score <name>` writes `<name>_results.json`, and
`dedup <name>` scores the first sample again after removing exact duplicate documents
(`<name>_dedup_results.json`). FineWeb reuses `experiments/merge_sweep`'s cache, so its first
sample is the contamination base.

Sources: FineWeb, FineWeb-Edu, C4 (web); Wikipedia; PubMed and arXiv abstracts
(`ccdv/*-summarization`); US case law (`HFforLegal/case-law`); MultiNLI premises (short
sentences from several genres). The `pubmed`, `arxiv_dataset` and `pile-of-law` datasets of
`analysis-20`/`22`/`23` no longer load (dataset scripts are no longer supported by `datasets`),
nor does `Samsung/samsum`; the replacements above cover the same domains where one exists.

## Result (2026-09-27; sample 0 / sample 1)

| Source | Documents | Bytes/token (16000) | Red gap (16000) | Green context (4000) | Green count (4000) |
|---|---|---|---|---|---|
| arXiv | 3,906 | 4.83 | 0.005 / 0.005 | 0.642 / 0.650 | 0.583 / 0.596 |
| PubMed | 5,254 | 5.07 | 0.007 / 0.008 | 0.655 / 0.654 | 0.661 / 0.663 |
| FineWeb | 2,000 | 4.14 | 0.008 / 0.008 | 0.711 / 0.706 | 0.718 / 0.696 |
| Wikipedia | 216 | 4.37 | 0.008 / 0.010 | 0.739 / 0.731 | 0.734 / 0.732 |
| FineWeb-Edu | 1,380 | 4.39 | 0.009 / 0.009 | 0.731 / 0.734 | 0.739 / 0.731 |
| C4 | 2,910 | 4.33 | 0.010 / 0.010 | 0.707 / 0.707 | 0.710 / 0.711 |
| Case law | 689 | 3.89 | 0.012 / 0.013 | 0.651 / 0.652 | 0.647 / 0.635 |
| MultiNLI | 57,477 | 4.47 | −0.007 / −0.006 | 0.663 / 0.661 | 0.667 / 0.668 |

For reference, FineWeb with 10% added (`experiments/contamination`, three seeds): red gap 0.016–0.017
with duplicates, 0.015–0.016 with Python; green context 0.678–0.685 with shuffled text.

MultiNLI repeats some premises (14.5% of the first sample's documents are exact duplicates).
Removed, its red gap falls from −0.007 to −0.023 at 16000 merges and from −0.035 to −0.044 at
4000; green context at 4000 merges moves from 0.663 to 0.668.

- **The red gap is stable within a source and narrow across long-document sources**: 0.005–0.013
  over seven sources of web, encyclopedic, scientific and legal text, with the two samples of a
  source within 0.002. Contamination at 30% or more (0.022–0.042) sits clearly above that range;
  at 10% (0.015–0.017) only just above its top (case law, 0.013).
- **It follows redundancy within a corpus, but its level depends on document length.** MultiNLI,
  whose documents are single sentences, reads negative although it holds duplicates; removing
  them lowers the gap further, the direction the contamination experiment predicts. With very
  short documents most transitions touch `<bos>`/`<eos>`, and each reading's log axis is
  normalized by its own maximum, which differs between `count` and `context_support` (in
  deduplicated MultiNLI at 16000 merges: 37,956 vs 7,590). A hypothesis, not yet tested.
- **Green over `context_support` has no domain-free value.** Clean sources span 0.642–0.739; the
  shuffled contamination (0.678–0.685) lands inside that range. It separates shuffled text only
  when compared with clean text of the same source.

Consequence: neither reading can take one threshold for every corpus. The red gap comes close for
long-document text but not at a 10% dose and not for short documents; the green level is
domain-dependent. A corpus needs a reference of its own. A candidate that needs no second corpus is
a null model built from the corpus itself: the same documents with their words shuffled (for
order) and with duplicates removed (for redundancy), scored with the same tokenizer budget, so that
each reading is the corpus's distance from its own null.
