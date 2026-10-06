# Scale: more data and longer training

["Scale: more data and longer training"](../../docs/2026-09-28-syntactic-sufficiency-design.md#scale-more-data-and-longer-training)
sets out the design and predictions, written before training.

BigF holds FineWeb's first 20,000 documents minus the 400 evaluation documents: 57.8 MB, about
10M words. From it come `big_class_cut` (no pronoun/modal + VERB sentence) and `big_control`
(random sentences removed to the same 38.8 MB). Each trains the medium model (6.2M parameters)
for 30,000 steps, about 4.9 hours on this CPU.

    poetry run python experiments/scale/scale.py prepare
    nohup poetry run python experiments/scale/scale.py train <bigf|big_class_cut|big_control> &
    poetry run python experiments/scale/scale.py items
    poetry run python experiments/scale/scale.py summarize

Downloads, corpora and runs go to `runs/scale/` (gitignored). Train detached and keep the
machine on power: a run outlives the two-hour background limit.

## Result (2026-10-05)

| Model | BLiMP | G pronouns | G modals |
|---|---|---|---|
| BigF | 0.629 | 0.910 | 0.902 |
| control | 0.632 | 0.822 | 0.895 |
| class cut | 0.622 | 0.761 | 0.565 |

- BLiMP rises from 0.57 to 0.63 (prediction 1 held).
- The modal result replicates more strongly: the cut leaves modals near chance.
- The pronoun predictions failed. The gap is 0.06 and the control is 0.09 below BigF.
- At this scale few unseen past-tense verbs remain for pronouns (3–7 pairs for most of them),
  and the per-head readings are erratic.
