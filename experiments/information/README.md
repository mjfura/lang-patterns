# Held-out information: predictability and order on unseen text

`information.py` scores the measure of `docs/superpowers/specs/2026-09-28-heldout-information-design.md` on the
corpora already built: the contamination variants (seed 0) and the clean sources' first samples,
each at 1000, 4000 and 16000 merges, and each with its word-shuffled null (every copy of a text
shuffled the same way, so the null keeps the corpus's duplicates). Two readings:

- **predictability**: the best held-out bits per byte over n-gram orders 0–3 (lower is more
  predictable);
- **order**: the null's best held-out bits per byte minus the corpus's — how much of the
  predictability comes from word order.

`information.py contamination <kind> <dose> [null]`, `information.py clean <source> [null]`,
`<target>[-null]_results.json`. One seed, one sample.

The split into fit and held-out halves keeps every copy of a text on one side. With copies split
independently, FineWeb with 50% duplicates read 1.72 bits per byte against the control's 2.23: the
held-out half was predicted from text the fit half had memorized.

## Result (2026-09-28; 1000 / 4000 / 16000 merges)

FineWeb base and additions (50% dose unless stated):

| Corpus | Predictability | Order |
|---|---|---|
| base | 2.301 / 2.301 / 2.285 | 0.290 / 0.253 / 0.233 |
| control 10% / 30% / 50% (at 4000) | 2.272 / 2.244 / 2.232 | 0.267 / 0.281 / 0.289 |
| duplicate 10% / 30% / 50% (at 4000) | 2.308 / 2.308 / 2.295 | 0.252 / 0.253 / 0.255 |
| shuffled 10% / 30% / 50% (at 4000) | 2.319 / 2.368 / 2.386 | 0.219 / 0.157 / 0.136 |
| python 10% / 30% / 50% (at 4000) | 2.369 / 2.426 / 2.472 | 0.268 / 0.309 / 0.389 |

Clean sources (6.5 MB each):

| Source | Predictability | Order |
|---|---|---|
| arXiv | 1.669 / 1.645 / 1.659 | 0.406 / 0.376 / 0.332 |
| PubMed | 1.688 / 1.659 / 1.669 | 0.422 / 0.425 / 0.357 |
| MultiNLI | 2.004 / 1.977 / 1.973 | 0.408 / 0.403 / 0.351 |
| Case law | 2.083 / 2.067 / 2.108 | 0.371 / 0.360 / 0.302 |
| FineWeb-Edu | 2.172 / 2.162 / 2.153 | 0.275 / 0.266 / 0.223 |
| Wikipedia | 2.206 / 2.188 / 2.196 | 0.278 / 0.300 / 0.222 |
| C4 | 2.216 / 2.213 / 2.196 | 0.296 / 0.261 / 0.242 |
| FineWeb | 2.301 / 2.301 / 2.285 | 0.290 / 0.253 / 0.233 |

- **Predictability does not depend on the merge budget**: every corpus reads within 0.04 bits per
  byte across 1000–16000 merges, where the graph curves moved with it
  (`experiments/granularity`). The gain over order 0 does depend on it, because larger tokens
  absorb within-word structure into the unigram, so gain is not used as a reading.
- **It separates the contamination kinds by their effect on unseen text.** New FineWeb documents
  make held-out text more predictable (2.301 → 2.232); duplicates leave it where the base is
  (2.295–2.308); shuffled text makes it less predictable (2.386 at 50%).
- **Order separates shuffled text at every dose** (0.219 / 0.157 / 0.136 against the control's
  0.267–0.289) and is left unchanged by duplicates.
- **Python is not flagged as unlearnable, and should not be by this measure**: code is highly
  ordered (order rises to 0.389), and the mixed corpus is less predictable only because it holds
  two distributions. Detecting text foreign to a domain is a separate question from learnability.
- **Across domains the readings differ, and now by the text rather than the tokenization**:
  scientific abstracts are the most predictable (1.65–1.67), web text the least (2.15–2.30).

Consequence: predictability and order are merge-invariant readings that behave as a learnability
indicator should on every contamination except the off-domain one, which they are not built to
detect. They give a prediction that can be written down before training: a model's held-out loss,
in bits per byte, should order these corpora as their predictability does. That prediction is
close to what an n-gram baseline would make by construction, which is why the graph metrics must
be tested for what they add beyond it.
