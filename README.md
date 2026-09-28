# corpus-generalization

Graph-topology indicators of whether a training corpus holds the patterns a language model needs
to generalize a knowledge domain — measured on the corpus alone, before any model is trained.

A corpus is tokenized with byte-level BPE and turned into a directed transition graph: every token
is a node, every observed transition `a → b` an edge weighted by how often it occurs. Four coverage
curves sweep an intensity threshold over that graph and report which fraction of the vocabulary
still holds together:

| Curve | Question | Mechanism |
|---|---|---|
| red — `red-cluster` | Do the strong transitions form one cohesive piece? | Largest weakly connected component of the edges with `count >= t` |
| blue — `weighted-core` | Is each token embedded in a dense neighborhood? | Weighted (strength) k-core peeling |
| orange — `reciprocal-cluster` | Which transitions run both ways? | Pairs scored by the harmonic mean of `A→B` and `B→A` |
| green — `reciprocal-core` | Which tokens keep both receiving and emitting flow? | k-core peeling on the harmonic mean of in/out strength: directed cycles |

Each curve collapses to an area under it (AUC), on a linear and on a log-normalized threshold
axis. `experiments/subsampling/` measures how each depends on corpus size.

## Usage

Requires Python `>=3.13,<3.14` and Poetry.

```sh
make install
make analyze CORPUS=analysis-17-wikipedia-en   # writes analysis/<corpus>/<date>/
make check                                     # lint, format, typecheck, tests
```

Installed in another project, `corpus-analyze --config path/to/tokenizer_analysis.json
[--output DIR]` analyzes that project's own corpus: the config's `files` and the default
`analysis/` resolve against the working directory.

Each corpus is a folder `configs/corpora/<name>/` holding `tokenizer_analysis.json`: local
`files` (the hand-built corpora are under `data/kid_learning/`) or a `huggingface_dataset`, plus
the tokenizer, merge budget and which analyses to run (the full list of keys is in
`corpus_generalization/analyze.py`'s docstring). `export_neo4j: true` needs `NEO4J_PASSWORD`
in the environment or in `.env`.

## Training a model on a corpus

To test whether a corpus's readings predict generalization, the same small GPT-2-style model is
trained on each corpus (`docs/2026-09-28-training-design.md`), on the CPU:

```sh
make train CONFIG=path/to/training.json   # writes runs/<config folder>/<date>/
corpus-generate --run runs/<name>/<date> --prompt "The" --temperature 0.8
make ui                                   # a Streamlit page to try every trained run
```

A `training.json` names the documents (`files` or a `huggingface_dataset`) and optionally the
merge budget, the held-out fraction, the model's size and the optimization; the full list of keys
and their defaults is in `corpus_generalization/model/train.py`'s docstring. A run holds its
settings, tokenizer merges, best checkpoint (lowest held-out loss) and every evaluation's train
and held-out loss.
