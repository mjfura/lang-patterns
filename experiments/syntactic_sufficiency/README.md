# Syntactic sufficiency: does coverage predict whether the model ties a verb to its nouns?

The first trained test of `docs/2026-09-28-syntactic-sufficiency-design.md`, whose predictions
were written before this ran. The recipe grammar (`synthetic/recipes.py`) is generated with each
verb seen with 1, 2, 4, 8, 16 or all 28 of its unreserved ingredients (2000 documents, seed 0, the
same reserved combinations in every variant); the same model (2 layers, 4 heads, 128 wide, 0.5M
parameters, 3000 steps, learning rate 1e-3, BPE at 1000 merges so every word is one token) is
trained on each. Per verb:

- **coverage** (corpus): the `<verb> the` → singular-ingredient pattern's fillers over its class,
  read by `analysis.syntactic` on the model's own tokens;
- **ratio** (model): after `<bos><verb> the`, the share of the ingredient probability that goes to
  ingredients never seen with that verb, over their share of the ingredients. 1 = the unseen are
  treated like the seen (pattern learned), 0 = only the seen get probability (memorized).

`THREADS=2 poetry run python experiments/syntactic_sufficiency/sufficiency.py run <variant>`, then
`... summarize`. About 4 minutes per variant on CPU. One seed, one model size.

## Result (2026-09-28)

| Fillers per verb | Coverage | Ratio | Verb pairs | Agreement pairs | Held-out loss | n-gram bpb |
|---|---|---|---|---|---|---|
| 1 | 0.091 | 0.011 | 0.250 | 1.000 | 1.389 | 0.490 |
| 2 | 0.087 | 0.021 | 0.500 | 1.000 | 1.483 | 0.550 |
| 4 | 0.167 | 0.025 | 0.679 | 1.000 | 1.579 | 0.585 |
| 8 | 0.312 | 0.047 | 0.929 | 1.000 | 1.675 | 0.630 |
| 16 | 0.533 | 0.171 | 1.000 | 1.000 | 1.778 | 0.658 |
| all (28) | 0.931 | 0.909 | 1.000 | 1.000 | 1.832 | 0.675 |

Per-verb Spearman(coverage, ratio) over the 84 verbs: **0.891**.

Against the predictions:

1. **Held.** Coverage and ratio rise together verb by verb (0.891 > 0.8).
2. **Held.** Diverse ratio 0.909 (≥ 0.8); one filler 0.011 (≤ 0.3).
3. **Held, and more strongly than stated.** The held-out loss and the n-gram predictability rank
   the variants in exactly the reverse order of generalization: the one-filler corpus is the most
   predictable (1.389 nats per token, 0.490 bits per byte) and the diverse one the least. Held-out
   text of a restricted corpus is restricted the same way, so predicting it well says nothing
   about whether the domain's pattern was learned.
4. **Failed.** Verb minimal pairs were predicted near 100% everywhere, on the reasoning that
   "the slowly" alone gives the bad sentence away. They read 0.25 with one filler: after "cut the"
   the restricted model gives an ingredient it never saw with "cut" less probability than an adverb
   that never follows "the" at all. The pair is a harsher test than expected, and it tracks the
   ratio. Agreement pairs are 1.0 everywhere (their test only reserves the adjective, as noted in
   the design).

**The ratio is far from linear in coverage.** Half the ingredients per verb (16 of 28, coverage
0.53) still gives a ratio of 0.17: the model gives the other half a sixth of their share. Only near
full coverage does it treat unseen ingredients like seen ones. The nouns sharing every state
sentence did not make the model generalize from the class alone.

Limits, before this is read as a threshold: one seed; one model size; about 200 passes over a
small corpus, with the checkpoint chosen by a held-out loss that — as prediction 3 shows — rewards
learning the restriction. A larger model, fewer passes or more varied text could move where the
ratio rises; that is the next thing to vary, not a settled number.
