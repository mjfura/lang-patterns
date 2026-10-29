# A planted gap in natural text

"A planted gap in natural text" in `docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md`, with its
predictions written before each run. FineWeb's first 6.5 MB sample (`experiments/clean_domains`) is
the reference, cut into 8 disjoint slices of 250 documents, the targets. Five heads are chosen on
the reference automatically (word-like, not among the 100 most frequent tokens, trunk 200–2,000,
diversity at least 0.8); in slice 0 each is restricted to its two most frequent fillers. A pattern
is called under-shown when its target diversity is below 0.5 and below its own margin (the mean
minus three standard deviations of its diversity over the 8 unplanted slices), and its reference
diversity is at least 0.5.

    uv run python experiments/planted_gap/planted.py run fineweb <remove|replace>
    uv run python experiments/planted_gap/planted.py show fineweb <remove|replace>

About 18 minutes per run, most of it learning the 11,563-merge tokenizer. Results:
`fineweb_<mode>_results.json`.

## Result (2026-09-29)

Heads chosen, their two kept fillers, and their diversity over the 8 unplanted slices:

| Head | Kept fillers | Reference | Slices, mean (sd) |
|---|---|---|---|
| people | to, who | 0.815 | 0.786 (0.076) |
| she | said, was | 0.978 | 0.902 (0.081) |
| But | it, the | 0.895 | 0.970 (0.104) |
| these | days, things | 0.816 | 0.892 (0.067) |
| only | a, to | 0.907 | 0.908 (0.060) |

In the planted slice (diversity, occurrences in parentheses):

| Head | `remove` (560 sentences removed) | `replace` (words replaced) |
|---|---|---|
| people | 0.164 (15) — under-shown | 0.036 (129) — under-shown |
| she | 0.294 (29) — under-shown | 0.136 (209) — under-shown |
| But | 0.525 (23) — not | 0.197 (93) — under-shown |
| these | 0.510 (6) — not | 0.056 (125) — under-shown |
| only | 0.212 (11) — under-shown | 0.036 (130) — under-shown |

Other patterns read (at least 20 occurrences) below their margin: 0 of 320 in the `remove` slice,
0 of 330 in the `replace` slice, 0 of 330 in the same slice unplanted.

Against the predictions:

1. **Nearly held.** Every chosen head reads sufficient in every unplanted slice (lowest 0.702);
   standard deviations 0.060–0.081, but 0.104 for "But", above the 0.1 predicted.
2. **Failed with `remove`, held with `replace`.** Removing each sentence where a head took another
   filler also removed most of the head's occurrences (6 left for "these"); a head seen 6 times
   with 2 fillers is no evidence of a restriction, and diversity, normalized for traffic, rightly
   does not call it one. Replacing the other filler with a kept one leaves the head its traffic and
   takes its variety, as the recipe restriction did: then all five are under-shown, far below
   their margins (0.036–0.197 against margins of 0.56–0.73).
3. **Held.** No other pattern falls below its margin, planted or not: the margin gives no false
   alarm on this sample.

What this establishes: on real text, a restriction that keeps a head's traffic but ties it to two
fillers is found for every planted head and nowhere else, once the verdict asks for a drop beyond
the pattern's own sampling noise. What it does not: the margin comes from 8 slices of one source;
a gap milder than two fillers (a head restricted to, say, half its usual fillers) is untested; and
replacement makes some sentences ungrammatical ("But it days"), which a real under-shown corpus
would not contain. The replaced word is also sometimes another chosen head ("these" after "But"),
which slightly lowers that head's traffic.

## Sensitivity: how strong a restriction must be to be found (2026-09-29)

Predictions written in the design document before this ran. The replacement planting, with each
chosen head kept to its `k` most frequent reference fillers for `k` in {2, 4, 8, 16, 32}, on
FineWeb, Wikipedia, arXiv and C4 (`planted.py run <source> sweep`; about 25 minutes per source, the
four in parallel). Results: `<source>_sweep_results.json`.

Heads found under-shown, of 5 per source:

| Source (slice size) | k = 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| FineWeb (250 documents) | 5 | 5 | 5 | 5 | 4 |
| Wikipedia (27) | 5 | 5 | 5 | 5 | 3 |
| arXiv (488) | 5 | 5 | 5 | 3 | 1 |
| C4 (363) | 4 | 5 | 4 | 4 | 4 |
| **All (of 20)** | **19** | **20** | **19** | **17** | **12** |

Planted diversity rises smoothly with `k` (FineWeb "people": 0.04, 0.07, 0.14, 0.27, 0.37; arXiv
"system": 0.08, 0.17, 0.34, 0.59, 0.92). No other pattern fell below its margin at any `k` in any
source (0 of 240–359 per reading), nor in the unplanted slices.

Against the predictions:

1. **Failed by one head.** Every head was found at `k` = 2 and 4 except C4's "who" at `k` = 2
   (0.51). The restriction was planted — after "who" the slice reads "are" 184 times — but the
   planting only replaces words made of letters, so contractions survive ("who don’t", "who
   hasn’t", tokenized as " don" + "’t"), leaving about ten fillers; and the class of "who"'s
   fillers is the small class of auxiliaries, where ten fillers are a large share. A flaw of the
   planting, not of the measure, but recorded as the prediction's failure.
2. **Failed.** Twelve of twenty heads are still found at `k` = 32: a head with a hundred or more
   occurrences in a slice uses far more than 32 fillers, so keeping 32 is still a restriction. The
   indicator is more sensitive than predicted.
3. **Held in three sources of four;** C4 is not monotone because of "who" hovering at 0.5.
4. **Held.** No false alarm at any `k`.

What this adds: the detection is not tuned to the extreme two-filler case. Across four sources it
finds a head restricted to 16 of its fillers 17 times in 20, with no false alarm among some 1,200
other patterns read per `k`; what limits it is the 0.5 cut on diversity, not the noise margin. On
the recipe grammar the model began to generalize only near diversity 0.8, so every restriction
found here is one the model would also be expected to learn as a restriction.

## Through the diagnosis tool, with the settings from the data (2026-10-15)

The plantings above used this experiment's own margin (8 slices, 3 sigma, diversity below 0.5).
`planted.py tool fineweb` reruns them through `corpus-diagnose` itself, whose settings all come
from the data (`docs/superpowers/specs/2026-10-11-settings-from-data-design.md`): the reference is learned to 1.50
tokens per word, read once with one-token heads, and each slice is diagnosed against it. Heads
are chosen as before; at the new granularity "even" replaces "these" (kept fillers "though",
"if"). A head is found when one of its rows is under-shown or absent. Controls are the unplanted
slices 1–7. About 25 minutes. Results: `fineweb_tool_results.json`.

| Planting | Heads found (of 5) | Other rows flagged | Class rows flagged |
|---|---|---|---|
| `remove`, k = 2 | 4 (not "But") | 0 | 2 |
| `replace`, k = 2 | 5 | 0 | 3 |
| `replace`, k = 4 | 5 | 0 | 3 |
| `replace`, k = 8 | 5 | 0 | 2 |
| `replace`, k = 16 | 5 | 0 | 2 |
| `replace`, k = 32 | 4 (not "But") | 0 | 2 |
| Controls, 7 slices | 0 | 0 | 2 (slices 5 and 7, one each) |

Every head is found through its row of determiner fillers (" the" class) under-shown, and most
also through an absent class (" be", "’"): diversity 0.03–0.17 at k = 2, rising to 0.27–0.35 at
k = 32, against slices that read about 0.9. FineWeb's sweep row is the same as with the old
margin (5, 5, 5, 5, 4), and `remove` finds one head more (4 against 3), now that a thin head is
judged against the reference's draws rather than a fixed 0.5 cut. No row outside the planted
heads is flagged in any planted or control slice, of 2,900–6,600 rows read each. The class level
does flag one class row in two of the seven controls: the class rows of a planted slice are not
all the planting's doing.
