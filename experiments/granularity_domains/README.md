# The tokens-per-word range across domains, and its upper edge

["The tokens-per-word range across domains"](../../docs/superpowers/specs/2026-10-25-granularity-across-domains-design.md)
sets out the design and six predictions, written before running, and the amendments made along
the way (each marked with when it was made and why). Corpus side only: no model is trained.

Step 1 (`experiments/granularity_range`) measured that the verdicts hold from 1.35 to at least
1.80 tokens per word on FineWeb alone. This experiment repeats the reading in two more domains,
TinyStories (simple) and Wikipedia (rich), each pooled to F's 14.4 MB with its own 22 labeled
cuts. It also extends all three domains up to 2.50.

    uv run python experiments/granularity_domains/granularity_domains.py download
    uv run python experiments/granularity_domains/granularity_domains.py prepare <domain>
    WORKERS=2 uv run python experiments/granularity_domains/granularity_domains.py diagnose <domain> <t>
    uv run python experiments/granularity_domains/granularity_domains.py summarize

`<domain>` is `tinystories` or `wikipedia` for `prepare`. FineWeb keeps step 1's pool, merges and
cuts, and its readings at 1.35–1.80 are step 1's.

`diagnose` skips a reading whose class tree would need more memory than SciPy's pairwise table
can have in 16 GB. The three readings it skipped (F at 1.30, Wikipedia at 1.30 and 1.35) were read
on the tree `fastcluster.linkage_vector` builds without the table, which
[`experiments/ward_memory`](../ward_memory/README.md) showed gives the same verdicts:

    uv run python experiments/ward_memory/ward_memory.py sweep <domain> <t> Pools, merges and cuts go to
`runs/granularity_domains/` (gitignored). The per-target readings `results/<domain>/target_*.json`
(tens of MB) are gitignored and rebuilt by `diagnose`. `prepared.json` and `summary.json` are
kept.

## Result (2026-10-27, finest readings added 2026-10-28)

| Domain | Merges at 1.50 | Measured range | First failure above | Read on `linkage_vector`'s tree |
|---|---|---|---|---|
| TinyStories | 1,223 | 1.30–2.00 | 2.20: the modals' class; 2.50: possessives too | — |
| FineWeb | 11,381 | 1.30–2.50 | none by 2.50 | 1.30 |
| Wikipedia | 16,022 | 1.30–2.20 | 2.50: the base verbs' class | 1.30, 1.35 |

Every declared granularity was read in every domain, and every one at or below 2.00 holds.

A granularity holds when (a) every removed class is flagged by a class row and (b) every dropped
word that is one token at that granularity is read `absent`.

**False flags, pooled per domain.** A control flagged at several granularities counts once.
Class rows are read on the four whole-document controls; head rows and `absent` on all eight
random controls.

| Domain | Class rows | Head rows | `absent` |
|---|---|---|---|
| TinyStories | 0 of 44 | 4 of 88 (4.5%) | 2 of 88 (2.3%) |
| FineWeb | 1 of 44 (2.3%) | 4 of 88 (4.5%) | 2 of 88 (2.3%) |
| Wikipedia | 1 of 44 (2.3%) | 4 of 88 (4.5%) | 1 of 88 (1.1%) |

The finest readings added two distinct flagged controls, both in Wikipedia: a class row on
`r25c` at 1.35 and `absent` on `r25e` at 1.30.

**Dropped words that are one token**, of the eight per domain:

| Domain | 1.30 | 1.35 | 1.40 | 1.50 | 1.60 | 1.70 | 1.80 | 1.90 | 2.00 | 2.20 | 2.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyStories | 8 | 8 | 7 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| FineWeb | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 6 | 4 | 1 | 0 |
| Wikipedia | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 1 | 0 |

Every dropped word that was one token was found, at every granularity, in every domain.

**Stability.** Between adjacent granularities inside each range:

| Domain | Swaps between under-shown and within noise | Patterns read at both |
|---|---|---|
| TinyStories | 17–57 | 16,000–102,000 |
| FineWeb | 18–64 | 54,000–341,000 |
| Wikipedia | 10–44 | 76,000–305,000 |

The largest share inside a range is 0.085%, in TinyStories.

### Against the predictions

1. **Held.** 1.50 is inside the measured range of all three domains.
2. **Held.** TinyStories reaches 1.50 with 1,223 merges, and reads 1.30 within memory.
3. **Held.** Wikipedia needs 16,022 merges for 1.50. At 1.30 and 1.35 it clusters 57,942 and
   38,580 tokens, above the 29,274 that fit SciPy's table (F at 1.35). Both were later read on
   `linkage_vector`'s tree, and both hold.
4. **Failed.** An upper edge appears in two domains, TinyStories (after 2.00) and Wikipedia
   (after 2.20), but not in FineWeb by 2.50. It is class detection (a) that fails first, not the
   dropped words (b). The dropped words stop being testable before that: from 1.70 in
   TinyStories, none of the eight is still a token.
5. **Held.** Pooled false flags stay within 2 × alpha in every family of every domain.
6. **Failed at the coarse end.** Swaps stay under 0.1% inside every range (largest 0.085%), but
   TinyStories' 2.20–2.50 step, outside its range, reaches 0.19%.

## What this means

The tool's target, 1.50 tokens per word, holds in all three domains: a simple one, a web one and
an encyclopedic one. The ranges differ in where they end. Their intersection is **1.30–2.00**:

- below, no domain fails: 1.30 is the finest granularity declared, so the lower end is where the
  test stopped, not an edge;
- above, TinyStories loses the modals' class at 2.20.

The tool's current `GRANULARITY_RANGE = (1.35, 1.80)` is not contradicted, but it is narrower
than the evidence on both sides. Below 1.40 the reading needs `linkage_vector` for F and
Wikipedia, at 1.5–10 hours for the tree alone.

Granularity has a second effect, separate from the verdicts: it sets which words the tool can
see. A word that is not one token cannot be reported missing. At 1.50, TinyStories has only
1,223 merges, so words seen 250–420 times in its 14 MB (`river`, `seek`) are still split, while
in Wikipedia every dropped word is a token up to 2.00. Tokens per word fixes running-text
granularity, not vocabulary coverage, and a domain with a small vocabulary reaches 1.50 while
covering fewer word types.

Two construction effects showed up in TinyStories, a domain full of dialogue. Both are reported
in the design's amendments:

- **Sentence controls orphan quotes.** Random sentence cuts remove the sentence that opens a
  quotation and leave its closing quote, and the class rows flag the closing-quote class (`."`,
  `!"`). That is a real change the cut made.
- **Matched controls concentrate the pattern.** Removing only sentences without the pattern
  concentrates it where it is common (46% of TinyStories' sentences), and the class rows flag
  the shift.

Still open: English only, one pool size per domain, and no lower edge: nothing below 1.30 was
read.
