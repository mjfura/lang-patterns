# Corpus or model? Attributing a syntactic failure

["Corpus or model? Attributing a syntactic failure"](../../docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md#corpus-or-model-attributing-a-syntactic-failure)
sets out the design and predictions, written before training.

The question is whether the tool's flags tell an engineer that a failure is the corpus's (flagged)
or the model's (unflagged). Corpora:

- F, the reference;
- `r10`;
- the unflagged controls and flagged class cuts of `experiments/class_patterns`, in pairs of one
  size.

Each is trained in three configurations, seed 0:

- `small`: 2 layers, 128 wide, 3000 steps; these runs already existed;
- `medium`: 4 layers, 256 wide, learning rate 5e-4;
- `long`: small, 9000 steps.

G is read per head on pronouns (→ VBD), modals (→ VB) and possessives (→ NOUN), with
ungrammatical contrasts.

    THREADS=3 uv run python experiments/model_sweep/model_sweep.py train <corpus> <medium|long>
    uv run python experiments/model_sweep/model_sweep.py items
    uv run python experiments/model_sweep/model_sweep.py summarize

A medium run takes about 28 minutes and a long one about 35. Keep the machine on power with
the lid open: sleeping runs stall past the two-hour background limit.

## Result (2026-10-05)

All three predictions held.

| Class | Best on F | Best on control | Best on flagged cut | Best on `r10` |
|---|---|---|---|---|
| pronouns | 0.981 | 0.961 | 0.846 | 0.907 |
| modals | 0.999 | 0.988 | 0.870 | 0.974 |
| possessives | 0.872 | 0.859 | 0.600 | 0.850 |

- A flagged class pattern is not recovered by any configuration (0.13–0.27 below F).
- Larger and longer-trained models do worse on it.
- Unflagged controls of the same size reach F within 0.02.
- On the smaller `r10`, the model choice moves G by up to 0.06.
