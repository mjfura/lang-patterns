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

## Replication: three seeds, two model sizes (2026-09-28)

Predictions written in the design document before this ran. `fillers_per_verb` in {1, 2, 4, 8,
16, 20, 24, all} × seeds {0, 1, 2} (each seed draws its own reserved combinations, fillers,
documents and initialization) × `small` (2 layers, 128 wide, 0.5M parameters) and `medium`
(4 layers, 256 wide, 3.3M). `sufficiency.py jobs | xargs -P 4 ...` (see the script's docstring);
48 runs, about two and a half hours on CPU four at a time. Results in `results/<size>/`.

Mean over seeds [min, max]:

| Fillers | Coverage | Ratio, small | Ratio, medium | Verb pairs, small | Verb pairs, medium |
|---|---|---|---|---|---|
| 1 | 0.098 [0.091, 0.111] | 0.009 [0.008, 0.011] | 0.006 [0.005, 0.006] | 0.143 | 0.179 |
| 2 | 0.099 [0.087, 0.111] | 0.016 [0.012, 0.021] | 0.014 [0.013, 0.016] | 0.476 | 0.381 |
| 4 | 0.185 [0.167, 0.203] | 0.030 [0.025, 0.033] | 0.028 [0.019, 0.041] | 0.536 | 0.512 |
| 8 | 0.282 [0.267, 0.312] | 0.045 [0.038, 0.050] | 0.043 [0.036, 0.050] | 0.833 | 0.810 |
| 16 | 0.533 | 0.179 [0.171, 0.194] | 0.125 [0.090, 0.150] | 1.000 | 1.000 |
| 20 | 0.667 | 0.380 [0.340, 0.404] | 0.180 [0.130, 0.218] | 1.000 | 0.976 |
| 24 | 0.800 | 0.901 [0.868, 0.938] | 0.554 [0.408, 0.792] | 1.000 | 1.000 |
| all (28) | 0.932 [0.931, 0.933] | 0.944 [0.909, 0.991] | 0.981 [0.949, 1.016] | 1.000 | 1.000 |

Per-verb Spearman(coverage, ratio), 112 verbs per seed: small 0.942 / 0.972 / 0.952, medium
0.952 / 0.975 / 0.940. Spearman(held-out loss, ratio) across the 8 variants: 0.976 / 1.000 /
1.000 for both sizes — the lower the held-out loss, the less the model generalizes.

Against the predictions:

1. **Held.** Spearman above 0.8 for every seed and size (0.940–0.975).
2. **Held.** Small's ratio at 16 fillers is 0.171–0.194 on every seed.
3. **Held.** The medium model generalizes later: at 20 fillers 0.180 against 0.380, at 24 fillers
   0.554 against 0.901, with no overlap between the sizes' seed ranges at 20 and 24.
4. **Held.** The held-out loss ranks the variants in reverse of the ratio for every seed and size.

**The rise is sharp, and where it happens depends on the model.** The small model crosses a ratio
of 0.5 between coverage 0.67 and 0.80, the medium one at about 0.80. Below coverage 0.3 neither
gives unseen ingredients more than a twentieth of their share. A larger model, able to memorize
more, needs more coverage before it treats a pattern as filler-agnostic — consistent with a claim
made for a fixed model family and budget, not for "a model" in general.

**Coverage is relative to the induced class**, which is why 1 and 2 fillers read almost the same
(0.098, 0.099): in the restricted variants the ingredients split into those some verb uses and
those seen only in state sentences, and the verbs' class is the smaller first one. A first version
of the script read every verb against the class of "tomato" and gave 0 when "tomato" fell in the
other one (seed 1, 1–4 fillers); `verb_coverage` now takes the ingredient class each verb is seen
with most, and `sufficiency.py recoverage` recomputed every result from its run's tokenizer. The
automatic cut also fell to 2 classes on seed 2 with 8 fillers (see the next section); with the
floor of 8 classes that variant's mean coverage moved from 0.291 to 0.282, and no ratio or
correlation changed.

Still open: one grammar, one corpus size (2000 documents), one training length (3000 steps with
the checkpoint chosen by held-out loss).

## Corpus size and training dynamics (2026-09-28)

Predictions written in the design document before this ran. The small model; seeds {0, 1, 2};
`fillers_per_verb` in {1, 4, 8, 16, 20, 24, all}; 500, 2000 and 8000 documents; 3000 steps, with
the ratio and the minimal pairs read at steps 250, 500, 1000, 2000, 3000 and at the best held-out
checkpoint (`train(checkpoint_steps=...)`, `load_run(checkpoint=...)`). `sufficiency.py dynamics
<variant> <documents> <seed>`; 63 runs, about 90 minutes four at a time. Results in
`results/dynamics/docs<N>/`.

**A measurement fault found on the way.** On 500 documents the automatic class cut chose 2 classes
— one of 95 tokens mixing " the", " .", " is", " are" and the ingredients — on every run, and on
one 2000-document run: the jump to 2 classes nearly tied the real one (1.35 against 1.43) and won.
`induce_classes` now searches only cuts of at least 8 classes (`min_classes`), which recovers the
categories on 500 documents; `recoverage-dynamics` and `recoverage` recomputed every coverage.

Mean ratio over seeds (coverage is the mean over verbs and seeds):

| Documents | Fillers | Coverage | Step 250 | 500 | 1000 | 2000 | 3000 | Best | Best step |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 1 | 0.098 | 0.905 | 0.104 | 0.017 | 0.039 | 0.042 | 0.062 | 567 |
| 500 | 16 | 0.503 | 0.960 | 0.940 | 0.666 | 0.501 | 0.511 | 0.938 | 467 |
| 500 | 24 | 0.702 | 0.933 | 0.938 | 0.811 | 0.663 | 0.651 | 0.944 | 467 |
| 500 | all | 0.773 | 0.942 | 0.950 | 0.721 | 0.692 | 0.663 | 0.954 | 467 |
| 2000 | 1 | 0.098 | 0.912 | 0.230 | 0.013 | 0.006 | 0.004 | 0.009 | 1367 |
| 2000 | 16 | 0.533 | 0.957 | 0.955 | 0.895 | 0.257 | 0.132 | 0.179 | 2333 |
| 2000 | 24 | 0.800 | 0.930 | 0.922 | 0.905 | 0.813 | 0.716 | 0.901 | 1200 |
| 2000 | all | 0.932 | 0.960 | 0.969 | 0.968 | 0.908 | 0.885 | 0.936 | 1500 |
| 8000 | 1 | 0.098 | 0.903 | 0.262 | 0.012 | 0.005 | 0.004 | 0.004 | 2933 |
| 8000 | 16 | 0.533 | 0.962 | 0.942 | 0.909 | 0.552 | 0.278 | 0.281 | 2967 |
| 8000 | 24 | 0.800 | 0.935 | 0.930 | 0.923 | 0.928 | 0.892 | 0.896 | 2933 |
| 8000 | all | 0.933 | 0.950 | 0.925 | 0.937 | 0.927 | 0.946 | 0.938 | 2833 |

(4, 8 and 20 fillers: `sufficiency.py summarize-dynamics`.) Per-verb Spearman(coverage, best
ratio): 0.787 / 0.773 / 0.815 on 500 documents, 0.968 / 0.967 / 0.958 on 2000, 0.974 / 0.955 /
0.968 on 8000.

Against the predictions:

1. **Failed.** Coverage is not the same at every corpus size: on 500 documents a verb is not seen
   with every filler it is allowed (0.503 against 0.533 at 16 fillers; 0.773 against 0.932 with
   all). The measure read that correctly; the prediction assumed every allowed combination would
   occur.
2. **Failed.** The best-checkpoint ratio does not fall with corpus size: at 16 fillers it reads
   0.938, 0.179, 0.281 on 500, 2000, 8000 documents. What moves it is where training stands, not
   the size (below).
3. **Held for 16–24 fillers**: the ratio at step 250 is above the one at step 3000 on every seed.
   With all fillers it stays high on 2000 and 8000 documents, but falls on 500 (0.942 to 0.663).
4. **Failed on 500 documents** (0.773–0.815), held on 2000 and 8000 (0.955–0.974).

**Every model generalizes first and learns the restriction later.** At step 250 every variant,
one filler included, gives unseen ingredients about their full share (0.90–0.96); from 4 fillers
up the verb minimal pairs are already at 1.0 there, so the model has learned that an ingredient
follows "<verb> the" and treats them all alike. Training on, it learns which ingredients each verb
was shown, and the ratio falls — to 0.004–0.03 with 1–4 fillers, much less far with 24 or all.
The best held-out checkpoint catches this process at different points: on 500 documents at step
~470, before the restriction is learned (hence the high ratios); on 8000 documents near step 3000,
with the restriction still being learned (at 16 fillers the ratio is still falling, 0.552 to
0.278 between steps 2000 and 3000).

Consequences for the indicator:

- Coverage predicts the **end point**: what a model has taken from the corpus once it has learned
  it well (per-verb Spearman 0.955–0.974 on 2000 and 8000 documents). Early in training the ratio
  is high whatever the corpus, and says the model has not yet learned anything verb-specific.
- **Sufficiency is relative to the model and the training budget**, not only to the corpus: the
  same corpus reads 0.28 or 0.70 depending on how far training went. The paper's claim stays the
  one made for a fixed model family and budget, and "trained to convergence" is part of the budget.
- On 500 documents both the class induction and the correlation degrade: the indicator needs a
  corpus large enough to show its classes.
