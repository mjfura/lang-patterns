# Do existing corpus metrics see what the indicator sees?

["Do existing corpus metrics see what the indicator sees?"](../../docs/2026-09-28-syntactic-sufficiency-design.md#do-existing-corpus-metrics-see-what-the-indicator-sees)
sets out the design and predictions, written before computing.

Six one-number corpus metrics are computed on 4 class cuts, their 4 head cuts and 8 random
sentence-removal controls, all from `r50` and stored by `experiments/class_patterns`:

- perplexity under F's bigram and under a Wikipedia bigram;
- unigram KL from F;
- MATTR;
- distinct-2;
- Vendi score of MiniLM embeddings.

    poetry run python experiments/baselines/baselines.py compute
    poetry run python experiments/baselines/baselines.py summarize

## Result (2026-10-05)

Both predictions held as written: no baseline flags more than one class cut, or separates
more than one class cut from its head cut, at 3 control sd. That test was lenient: the
controls vary in size and the mean perplexity has an outlier.

A size-matched reading, exploratory, changes the picture.

- Unigram KL and distinct-2 do register the two large class cuts (18,126 and 10,315 sentences
  removed).
- They stay within noise on the numbers (1,972) and base verbs (3,642), which the indicator
  flags at the class level.
- No baseline says which pattern changed.

The indicator's edge is localization and sensitivity to small, pattern-specific losses.
