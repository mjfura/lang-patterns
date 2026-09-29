# A planted gap in natural text

"A planted gap in natural text" in `docs/2026-09-28-syntactic-sufficiency-design.md`, with its
predictions written before each run. FineWeb's first 6.5 MB sample (`experiments/clean_domains`) is
the reference, cut into 8 disjoint slices of 250 documents, the targets. Five heads are chosen on
the reference automatically (word-like, not among the 100 most frequent tokens, trunk 200–2,000,
diversity at least 0.8); in slice 0 each is restricted to its two most frequent fillers. A pattern
is called under-shown when its target diversity is below 0.5 and below its own margin (the mean
minus three standard deviations of its diversity over the 8 unplanted slices), and its reference
diversity is at least 0.5.

    poetry run python experiments/planted_gap/planted.py run fineweb <remove|replace>
    poetry run python experiments/planted_gap/planted.py show fineweb <remove|replace>

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
