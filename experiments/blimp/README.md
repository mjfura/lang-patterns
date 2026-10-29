# The models on BLiMP

["The models on BLiMP"](../../docs/superpowers/specs/2026-09-28-syntactic-sufficiency-design.md#the-models-on-blimp)
sets out the predictions, written before scoring.

The 18 models of `model_sweep` and the 12 of `class_patterns` are scored on BLiMP's 67
paradigms (`nyu-mll/blimp` on Hugging Face). A pair counts as correct when the grammatical
sentence has the higher total log probability after `<bos>`.

    uv run python experiments/blimp/blimp.py score
    uv run python experiments/blimp/blimp.py summarize

## Result (2026-10-05)

| Prediction | Result |
|---|---|
| 1. F's models ≥ 0.60 overall | FAILED: 0.555 / 0.571 / 0.576 (small / medium / long); subject–verb agreement about 0.52 |
| 2. BLiMP gap smaller than G gap, control vs class cut | HELD: BLiMP −0.016 to +0.017 against G +0.11 to +0.34 |

By the field's measure these models know little syntax beyond the local. G reads local slot
classes, which they learn well. BLiMP's paradigms mostly turn on longer dependencies. The
paper's claims are about local syntactic patterns, and scale is the main open limit.
