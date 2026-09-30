# `corpus-diagnose`: the syntactic-sufficiency diagnosis as a tool

**Goal:** one command (and one UI page) that takes any corpus, and optionally a reference corpus of
the same domain, and reports which syntactic patterns the corpus shows restricted — the indicator
of `docs/2026-09-28-syntactic-sufficiency-design.md`, packaged from the pieces its experiments
validated. It diagnoses syntactic sufficiency only: a topic-narrow corpus can pass and still fail a
model outside its topics (`experiments/natural_gap`).

## Pipeline — `src/corpus_generalization/diagnose.py`

`diagnose(documents, reference=None, merge_budget=4000, draws=8, min_trunk=20, low=0.5,
sigmas=3.0, seed=0, progress=None, head_lengths=(1, 2)) -> DiagnosisReport`

1. **Tokenizer** — byte-level BPE with `merge_budget` merges, learned on the reference when given,
   else on the corpus; every text framed `<bos>`/`<eos>`.
2. **Word-like tokens** (`word_like_tokens`) and **classes** (`induce_classes` with
   `candidates=` the word-like tokens), on the reference when given, else on the corpus.
3. **Corpus patterns** — `syntactic_patterns(..., heads=word-like)`; a pattern is
   read when its trunk is at least `min_trunk`.
   Heads of one and two tokens are read because a determiner separates a verb from its object.
4. **With a reference**: the reference's patterns, and `draws` random samples of reference
   documents of the corpus's size in bytes (seeded) give each pattern's noise — mean and standard
   deviation of its diversity. Verdict per read pattern:
   - `sufficient` — diversity >= `low`;
   - `under-shown` — below `low`, below mean − `sigmas`·sd of its draws, reference >= `low`: the
     domain uses more fillers here than the corpus shows (missing data);
   - `fixed` — below `low` and reference below `low`: the domain itself restricts the slot;
   - `within-noise` — below `low`, reference >= `low`, but not below its margin;
   - `unread` — no reference pattern, or fewer than `draws − 2` draws read it.
5. **Without a reference**: `sufficient` (>= `low`) or `restricted` (< `low`: a candidate — either
   missing data or a construction; a reference or a reader decides).
6. **Warnings** — fewer than 1,000 documents or 50,000 words (classes and correlations degraded on
   500 recipe documents); a corpus larger than its reference; a vocabulary the tokenizer leaves at
   more than 1.6 tokens per word (word pieces fill the classes).

`DiagnosisReport` holds: sizes (documents, words, tokens per word, merge budget), the number of
classes and each class's 12 most frequent tokens, counts per verdict, one row per read pattern
(head, class, the class's top tokens, trunk, fillers, diversity, reference diversity, margin mean
and sd, verdict, the corpus's 5 most frequent fillers of that slot), the warnings, and the
settings. Rows are sorted by verdict severity (under-shown, restricted, fixed, within-noise,
sufficient, unread), then by trunk.

## Reading it

- The report is **per pattern**: one number for a whole corpus dilutes local gaps (the recipe
  area moved 0.958 → 0.902 while the restricted verbs read 0.93 → 0.09).
- `under-shown` rows are where a model trained on the corpus is expected to tie the head to the
  fillers shown (`experiments/planted_model`: the true next word 9–100 times less likely there).
  The fix is more text in which the head takes other fillers of its class.
- `fixed` rows are the language's or the domain's constructions ("due → to"); not defects.
- `low` = 0.5 is permissive: on the recipe grammar the small model generalized from diversity
  ~0.8, a larger one later. A threshold is calibrated per model family and size.

## Command

    corpus-diagnose --files PATH [PATH ...] [--reference PATH [PATH ...]] [--merges N]
                    [--head-lengths N [N ...]] [--limit N] [--output DIR]

Files are text files of documents separated by `<|endoftext|>` lines (`sources.read_documents_from`).
`--limit` keeps the first N documents of each. Writes `report.json` and `report.md` (summary,
warnings, classes, the flagged rows) to `--output` (default `diagnoses/<date>/`, gitignored), and
prints the summary.

## Tests (before the code)

1. Recipe grammar, restricted variant (1 filler per verb) against the diverse one: every verb's
   noun slot is `under-shown`; no pattern of the diverse variant read against itself is
   `under-shown`.
2. Without a reference, the restricted variant's verb noun slots are `restricted`.
3. A corpus of 300 documents gets the size warning.
4. The command writes `report.json` and `report.md` and exits 0; a missing file is reported and
   exits non-zero.
