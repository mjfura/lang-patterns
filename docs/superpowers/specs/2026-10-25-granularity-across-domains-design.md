# The tokens-per-word range across domains, and its upper edge

**Status:** experiment done (2026-10-28), result in `experiments/granularity_domains/README.md`. Not yet applied: the change to `GRANULARITY_RANGE` and the paper's §5.8 and §7, which wait for `2026-10-28-granularity-robustness-design.md`.

Agreed with the user on 2026-10-25. Design and predictions, written before running.

## Why

Step 1 of `docs/superpowers/specs/2026-10-11-settings-from-data-design.md` measured that the verdicts hold from
1.35 to at least 1.80 tokens per word. The tool rests on that result: it learns every domain's
tokenizer to `GRANULARITY_TARGET = 1.50` and refuses a corpus outside
`GRANULARITY_RANGE = (1.35, 1.80)`. Two gaps weaken it:

1. **One domain.** The range was measured on the FineWeb pool alone. A domain with a smaller or
   larger vocabulary could place its range elsewhere, and 1.50 could fall outside it.
2. **No upper edge.** Nothing failed at 1.80, and nothing above it was read. The lower end was
   set by memory (1.30 did not fit in 16 GB), not by a failure. So the range has no measured edge
   on either side. A corpus whose merges run out above 1.80 (a small corpus diagnosed without a
   reference) is refused for a reason that was never measured.

The question is not whether the range is the same in every domain. It is: **in each domain, is
1.50 inside the range where the verdicts hold, and where does the range end above 1.80?**

## Domains

Two domains, chosen to lie on either side of FineWeb:

- **TinyStories** (`roneneldan/TinyStories`, `train`, field `text`): short stories in a small
  vocabulary, with few rare words. It is the simple end, and a corpus commonly used to train
  small models like ours.
- **Wikipedia** (`wikimedia/wikipedia`, `20231101.en`): encyclopedic register, with many rare
  words, names and numbers. It is the rich end. Its 1,500 articles (32.6 MB) are already cached
  by `experiments/clean_domains`.

FineWeb is also re-read above 1.80 with the existing 60,000 merges. Another language is out of
scope: it would change the pre-tokenizer's word boundaries and the morphology together, a
different question.

## Set-up

Each domain is built exactly as FineWeb was in step 1, so that only the domain changes.

- **Pool.** The domain's documents in cache order until the pool matches F's 14.3 MB. TinyStories
  is downloaded to `experiments/clean_domains` the same way the other domains were.
- **Wikipedia articles stay whole.** The pool then has about 650 documents against F's 4,600,
  so the size-matched draws are coarser. Splitting articles would need an arbitrary length. The
  control false-flag rate measures whether coarser draws matter.
- **Half pool (the analogue of `r50`).** A random 50% of the pool's documents, seed 0.
- **Tokenizer.** One tokenizer per domain, learned on its pool to the budget of its lowest
  reachable target. Each granularity is a prefix of it, as in step 1.
- **Workers.** At most two cuts at a time, the memory limit measured on this machine.

### The 22 labeled cuts per domain

They come from the half pool, built by the code of `experiments/class_patterns` with the
domain's own pool and tree.

- **Four class cuts, four head cuts, four matched controls**, as in FineWeb:
  - pronouns and modals → VERB;
  - possessives → NOUN;
  - the anchor `two`'s class → NOUN;
  - the anchor `make`'s class → DET.

  Pronouns, modals and possessives are fixed lists. The number and verb classes are the
  domain's own: the members of the anchor's class in the 64-class cut of the domain's tree, as
  `tool_check.class_members` reads them on F. The head cuts remove the same two heads as on
  FineWeb (`she`/`they`/`can`/`will`, `his`/`their`, `five`/`six`, `make`/`get`). When one is
  not a member, the most frequent other member replaces it.
- **Eight fresh controls**: four random sentence cuts of the half pool (`ctrl_a`–`ctrl_d`) and
  four random 25% document cuts of the pool (`r25b`–`r25e`), as in `experiments/class_patterns`.
- **Two drop cuts**, four words each. FineWeb's words were picked by hand. Here they are set by a
  rule:
  - lowercase alphabetic words;
  - not a head of any cut;
  - inside the frequency-rank band that FineWeb's eight words occupy in F;
  - sampled with seed 0.

**Feasibility gate (before any sweep).** A removed class must give its cut at least as much
to remove as the smallest FineWeb cut that was found at every granularity: numbers, 1,972 target
sentences of r50's 62,025 (3.2%). Below that, no measured cut shows the tool can test it. The
report does not expose whether one class node was read, so the gate counts what the cut removes.
Numbers in TinyStories are the likely case. A class that fails the gate is replaced, in this
order: prepositions (`in on at with from for`) → DET, then determiners (`the a an this that`) →
NOUN. The replacement is recorded, not hidden.

## Sweep

- **Granularities:** 1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.20 and 2.50 tokens
  per word, in each new domain.
- **FineWeb:** 1.90, 2.00, 2.20 and 2.50.
- **Unreachable or too large.** A granularity a domain cannot reach (its merges run out), or one
  that does not fit in 16 GB, is recorded as such and is not a failure of the range. Memory is
  checked before reading: a reading that would cluster more tokens than F at 1.35 (the largest
  that fit) is recorded as too large and not run, since a memory-pressure kill takes the machine
  down with it.
- **Cost estimate** (from FineWeb's 4–9 minutes per cut):
  - each new domain: 11 × 22 cuts ≈ 10 hours with two workers;
  - FineWeb's extension: ≈ 3 hours;
  - about a day of compute in all.

## When the verdicts hold at a granularity

The same readings as step 1, with the pass rule written down before running.

**Per granularity (required):**

- **(a)** every removed class is flagged by a class row;
- **(b)** every dropped word is read `absent`.

**Pooled over a domain's sweep (judged across granularities):** false flags on the 12 controls,
for each family (class rows, head rows, `absent` on the fresh controls).

- **What passes:** at most 2 × alpha = 12% of control readings in each family.
- **Why pooled:** twelve readings per granularity are too few to judge a 6% rate. Three flags of
  12 occur by chance about 3% of the time, and FineWeb's 1.70 had exactly three head flags.

**The measured range** of a domain is the longest run of adjacent granularities where (a) and
(b) hold, given that the pooled false-flag rate passes.

**Also recorded, not part of the pass rule:**

- head flags on the class and head cuts;
- the swap count between adjacent granularities;
- the merges per granularity;
- the lexical share of transitions.

## Predictions

1. **1.50 is inside the measured range of both new domains.** This is the result the tool needs.
2. **TinyStories reaches 1.50 with far fewer merges than FineWeb** (under 3,000 against
   11,381), because its vocabulary is small. It reaches 1.30, and 1.30 fits in memory, since few
   distinct tokens remain to cluster.
3. **Wikipedia needs more merges than FineWeb for 1.50** (over 11,381), because of its rare
   words and names. 1.30 does not fit in 16 GB.
4. **An upper edge appears below 2.50 in at least one of the three domains, and (b) fails first.**
   - The mechanism: at a coarse granularity a mid-frequency word is no longer one token. Its
     drop then removes pieces that other words share, and the pieces are not missing.
   - Class detection (a) holds to 2.50 in FineWeb: class rows grew stronger as tokens per word
     rose (the modals' t went from 319 to 690 between 1.35 and 1.80).
5. **The pooled false-flag rate passes in every domain**, Wikipedia's coarser draws included.
6. **Adjacent granularities swap under 0.1% of the patterns read at both**, in every domain, as
   on FineWeb (26–37 of 330,000–690,000).

## What follows from the outcome

- **If prediction 1 holds,** 1.50 stays the target. `GRANULARITY_RANGE` becomes the intersection
  of the three measured ranges, changed with a test that pins it. The paper's §5.8 reports the
  three domains, and §7 drops the one-domain limitation.
- **If 1.50 is outside a domain's range,** a single target does not carry across domains. The
  path is revised with the user before any change to the tool. The likely revision is to derive
  the target from each domain.
- **If no upper edge appears by 2.50,** the range is reported as open above. The refusal above
  1.80 is relaxed to the highest granularity measured in all three domains.

## Code

- `experiments/granularity_domains/`: one script with `prepare <domain>`, `gate <domain>`,
  `diagnose <domain> <target>` and `summarize`, plus its README with the result.
- It reuses `granularity_range`'s reading of each report and `class_patterns`'s cut builders,
  parameterized by pool instead of reading F.
- No change to `src/` unless the outcome calls for one, and then with a test first.

## Amendments before any new domain was diagnosed (2026-10-25)

Made after preparing TinyStories and re-reading step 1's FineWeb readings with this experiment's
code, before any TinyStories or Wikipedia diagnosis was run.

1. **A removed class is found** when a flagged class row of its cut holds any of the class's
   heads. Step 1 asked for an anchor (`two`, `make`, …) among the row's 12 listed heads, but
   TinyStories' tree puts `two` in a large class of adjectives (`big`, `little`, `good`, …),
   where it need not be listed. On FineWeb both rules find the same five classes at every
   granularity.
2. **A false flag that recurs on the same control counts once.** The same controls are read at
   every granularity, so a flag that recurs on one control belongs to that corpus, not to the
   granularity. Counted per reading, FineWeb's `absent` family would fail (7 of 48, 15%), all from
   `r25c` (` luck`) and `r25e` (`Maybe`), the two corpora step 1 had already traced. Counted
   once, it is 2 of 48. The pooled rate is the number of distinct controls flagged in a family
   over the readings; the per-granularity counts are kept in the summary.
3. **Stability counts swaps between under-shown and within noise** among patterns read at both
   granularities, as step 1 did (26–37 on FineWeb, reproduced). A pattern moving between read and
   `unread` is not a swap.

## Amendments after TinyStories at 1.50 (2026-10-25)

The first TinyStories reading at 1.50 exposed two bugs and two design questions. The bugs were
fixed and both domains prepared again; the two questions were decided by the user. These
amendments come after one result was seen, and are reported as such.

- **Bug: the random sentence controls.** `ctrl_a`–`ctrl_d` were drawn from sentences without a
  pronoun or modal followed by a verb, where step 1 drew them from every sentence. In TinyStories
  those sentences are 46% of the text, so the controls concentrated the pattern and were flagged.
- **Bug: case.** Dropped words were compared case-sensitively with the reported heads (`Jill`).
- **Dropped words that are not one token (user's choice).** At 1.50 TinyStories has 1,223 merges,
  and words seen 250–420 times are still split (`river` = ` r` + `iver`). The tool cannot report
  as missing what is not a unit. Rule (b) is judged on the dropped words that are one token at
  that granularity; the others are counted in the summary as a coverage reading, not a failure.
- **Matched controls are not false-flag controls (user's choice).** A matched control removes
  only sentences without the pattern, so where the pattern is common it concentrates it (46% of
  TinyStories' sentences for pronouns and modals, 29% in FineWeb). False flags are read on the
  eight random controls (`ctrl_a`–`ctrl_d`, `r25b`–`r25e`); the matched controls keep their
  original role, the comparison with the head and class cuts, and their flags are recorded.
- **Class false flags on whole-document controls (user's choice, after TinyStories' sweep).** A
  random sentence cut removes the sentence that opens a quotation and leaves its closing quote
  (`…sew my shirt?" After they finished…`). The class rows flag the closing-quote class (`."`,
  `!"`) on all four sentence controls at most granularities of TinyStories: a real change the
  cut made, absent from FineWeb's prose. Class false flags are read on the four whole-document
  controls (`r25b`–`r25e`); head and `absent` flags stay on all eight random controls, and the
  sentence controls' class flags are recorded.

## Amendment after the memory-light Ward (2026-10-28)

`experiments/ward_memory` measured that `fastcluster.linkage_vector` builds the same Ward tree as
SciPy without the pairwise table, and gives the same verdicts on TinyStories' and F's 22 cuts at
1.50. The user then asked for every declared granularity to be read.

- **The memory rule is lifted for the three readings it skipped:** F at 1.30, Wikipedia at 1.30
  and 1.35. Each is read on the tree `linkage_vector` built for it, saved by `ward_memory.py`
  (`sweep <domain> <t>`); every other step of the diagnosis is unchanged. Every other reading
  keeps SciPy's tree.
- **Prediction 3 is judged on the tokens to cluster**, against the most that fit SciPy's table
  (29,274, F at 1.35), not on whether the reading was skipped. It is a prediction about that
  table, and the table still does not fit at Wikipedia's 1.30 (57,942 tokens).
