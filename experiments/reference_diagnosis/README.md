# Fixed construction or missing data: each pattern read against a reference corpus

Predictions 1 to 4 of "Word-like tokens, diversity against chance, and a reference corpus" in
`docs/2026-09-28-syntactic-sufficiency-design.md`, written before this ran. Patterns over
word-like tokens only; each pattern's `diversity` is its fillers over the fillers free draws from
its class would give at its traffic; a target's low-diversity patterns are called **under-shown**
when the same slot is diverse in a reference corpus of the same domain, **fixed** when it is not.
One tokenizer, word-like set and class cut, learned on the reference, read both corpora; "low" is
diversity below 0.5.

    poetry run python experiments/reference_diagnosis/diagnosis.py recipes
    poetry run python experiments/reference_diagnosis/diagnosis.py natural <source>
    poetry run python experiments/reference_diagnosis/diagnosis.py show <recipes|source>

`recipes`: the restricted variants (seed 0, 2000 documents) as targets, the diverse variant as
reference, verbs' `<verb> the` → noun slot. `natural`: the first 1/8 of a clean source's first
6.5 MB sample as target, the whole sample as reference; target patterns with at least 20
occurrences. A few minutes per source. Results: `<name>_results.json`.

## Result (2026-09-29)

Recipe grammar — the verbs' noun slot:

| Fillers per verb | Target diversity | Reference diversity | Verdicts (14 verbs) |
|---|---|---|---|
| 1 | 0.135 | 0.952 | 14 under-shown |
| 2 | 0.146 | 0.952 | 14 under-shown |
| 4 | 0.204 | 0.952 | 14 under-shown |
| 8 | 0.314 | 0.952 | 14 under-shown |
| 16 | 0.562 | 0.952 | 14 sufficient |
| 24 | 0.823 | 0.952 | 14 sufficient |

Natural text — target patterns read (at least 20 occurrences) and their verdicts:

| Source | Target / reference documents | Classes | Sufficient | Fixed | Under-shown |
|---|---|---|---|---|---|
| arXiv | 488 / 3,906 | 10 | 220 | 67 | 17 |
| PubMed | 656 / 5,254 | 11 | 223 | 78 | 20 |
| MultiNLI | 7,184 / 57,477 | 13 | 383 | 53 | 22 |
| C4 | 363 / 2,910 | 8 | 378 | 57 | 11 |
| FineWeb-Edu | 172 / 1,380 | 8 | 370 | 77 | 16 |
| FineWeb | 250 / 2,000 | 9 | 375 | 57 | 9 |
| Wikipedia | 27 / 216 | 10 | 315 | 64 | 14 |
| Case law | 86 / 689 | 8 | 261 | 151 | 14 |

Against the predictions:

1. **Held.** Diverse 0.952 (≥ 0.8); one and two fillers 0.135 and 0.146 (≤ 0.2).
2. **Held.** With 1–8 fillers every verb is under-shown. With 16 fillers the verbs read
   *sufficient* (0.562), yet the small model trained on that corpus did not generalize (ratio
   0.18, `experiments/syntactic_sufficiency`); it did from 24 fillers (0.823 here, ratio 0.90).
   The 0.5 cut is too permissive against the model: on this grammar the model's rise sits near
   diversity 0.8.
3. **Held where readable.** "due", "able", "pursuant" and "such" (in its "such as" slot) are fixed
   in every source where they reach 20 occurrences (diversity 0.03–0.27 in target and reference).
   The contractions are not read at all: the token after "didn" is "'t", which is not word-like,
   so the slot has no class. That is by construction, not a verdict.
4. **Held.** Fixed patterns outnumber under-shown ones in every source (from 57 to 9 in FineWeb
   to 151 to 14 in case law).

What the lists show:

- **Fixed, clearly:** diversity below about 0.15 in both corpora — "due" (0.046 in arXiv), "black"
  (0.047, black hole), "associated" (0.059, associated with), "based" (0.088, based on), "able"
  (0.073), "such" (0.076), "Academy" (0.052). These are the language's and the domains' own
  constructions, and the reference confirms them.
- **Under-shown, weakly:** mostly prepositions and auxiliaries a little below the cut in the target
  and a little above it in the reference — "in" 0.394 against 0.722 in arXiv, "of" 0.459 against
  0.561 in case law, "levels" 0.498 against 0.606 in PubMed. Many other function words read
  0.35–0.5 in both and are called fixed. Between about 0.3 and 0.6 the verdict is a cut through a
  continuum, not a finding; a smaller sample of documents is also lumpier by topic, which lowers
  the target's diversity on its own.

So on natural text the reference separates the strong cases — fixed constructions — as it should,
and the recipe grammar shows it finds real gaps when they are large. The middle band needs a margin
(target and reference apart by more than their sampling noise, measured over several target
samples) before an "under-shown" verdict means missing data, and the cut itself belongs near 0.8,
where the model started to generalize.
