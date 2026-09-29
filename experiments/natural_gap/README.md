# A natural gap: a topic-narrow corpus against a random one of the same size

"A natural gap" in `docs/2026-09-28-syntactic-sufficiency-design.md`, with its predictions written
before this ran (the narrow corpus's cluster rule was revised before any reading, as recorded
there). FineWeb's 5,000 cached documents: 400 random ones evaluate, the other 4,600 are the pool
and the reference. The **narrow** corpus is the topical clusters with most text — health, sports,
education (923 documents, 3.5 MB) — leaving out two register clusters (news; conversational
blogs); the **random** corpus is 3.5 MB of random pool documents; 8 further random draws give each
pattern's margin. The small model is trained on each (seeds 0 and 1). Per head read in the narrow
corpus: its **drop**, the narrow model's mean log probability of the true next word on the
evaluation text minus the random model's (up to 50 occurrences).

    poetry run python experiments/natural_gap/natural_gap.py prepare
    THREADS=3 poetry run python experiments/natural_gap/natural_gap.py train <narrow|random> <seed>
    poetry run python experiments/natural_gap/natural_gap.py summarize

About 40 minutes to prepare and 35 per run, two at a time. Results in `results/`.

## Result (2026-09-29)

The indicator flags 6 patterns in the narrow corpus — "found", "mean", "open", "president",
"right", "without" — and none in the random one, of 813 and 896 heads read.

| Seed | Evaluation bits per byte, narrow / random | Mean drop, flagged heads (6) | Unflagged heads (806) | Spearman(z, drop) |
|---|---|---|---|---|
| 0 | 2.179 / 2.133 | +0.156 | −0.049 | 0.044 |
| 1 | 2.182 / 2.130 | −0.002 | −0.053 | 0.048 |

Largest drops: "sales" −3.55 / −3.97, "desire" −1.58 / −3.97, "prepared" −1.48 / −2.58, "South",
"beyond", "human", "local" — none of them flagged.

Against the predictions:

1. **Held.** 6 flagged in the narrow corpus, 0 in the random one.
2. **Failed.** The flagged heads do not drop more than the others; on seed 0 they drop less.
3. **Failed.** The standardized diversity and the drop are uncorrelated (0.04, 0.05).

## Why

The heads where the narrow model fails most have **normal diversity** in the narrow corpus: "sales"
0.61 against a margin mean of 0.70, "desire" 0.29 against 0.41, "human" 0.83 against 0.65. The
narrow corpus shows them with as many different fillers as a random corpus does — but different
fillers: those of health, sports and education. The evaluation text uses others. What a
topic-narrow corpus lacks is not variety in the slot but the domain's own words in it.

That is the line the frame draws between syntax and semantics. Diversity measures whether a slot
is shown open (syntax); a topic-narrow corpus keeps its slots open and changes what fills them
(semantics, and the domain's vocabulary). On this natural gap the indicator does what it claims —
it flags no syntactic restriction that is not there — and the model's failure lies outside what it
claims to measure.

What it leaves for the indicator: diversity diagnoses a syntactic restriction (planted, or a
construction), not a mismatch between the corpus's fillers and the domain's. The second is what
limits a model trained on a narrow corpus of real text, and it needs a measure that compares
*which* fillers each slot shows against a reference of the target domain — the recipe grammar's
own notion of coverage, read against the domain rather than the corpus's class.
