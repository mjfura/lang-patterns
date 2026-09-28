# Training: a fixed small language model to test whether a corpus generalizes

**Goal:** calibrate the corpus indicators against real generalization. The same model is trained
on each corpus, and the corpus's readings are compared with the model's held-out loss (and later
with the syntactic validity of what it generates). The user can load every trained model in a
Streamlit page and try it directly, instead of relying only on reported numbers.

The model is built here in PyTorch, not in eva-llm (the user's decision), and stays fixed across
corpora so the corpus is the only variable.

## Model — `src/corpus_generalization/model/gpt.py`

A decoder-only transformer in the GPT-2 layout, the usual small baseline for this kind of study
(TinyStories-scale models): learned token and position embeddings, pre-LayerNorm blocks of causal
self-attention (`torch.nn.functional.scaled_dot_product_attention`) and a 4× GELU MLP, a final
LayerNorm, and an output head tied to the token embedding. Dropout on attention, residual and
embedding outputs. Sizes come from the config (`n_layer`, `n_head`, `d_model`, `context`,
`dropout`); the defaults (4 layers, 4 heads, 256 wide, 256 context) are a starting point to be
justified by a learning-rate/size check, not copied values.

## Data — `src/corpus_generalization/model/data.py`

- Documents come from the same sources as the analysis (`files` and/or `huggingface_*`, read
  with `sources`).
- They are split **by document** into train and held-out (`heldout_fraction`, default 0.1,
  seeded), so no held-out document is ever seen during training.
- The tokenizer (byte-level BPE, `merge_budget`) is learned on the **train split only**; the
  held-out loss is then a fair measure of generalization, not of tokenizer leakage. It is the
  same tokenizer family and framing the graph analysis reads, so the model learns the routes the
  indicators measured.
- Each split is framed (`<bos>` + tokens + `<eos>`, `packing.framed_documents`) and concatenated
  into one stream; batches are random windows of `context + 1` tokens (inputs and targets shifted
  by one).

## Training — `src/corpus_generalization/model/train.py`, command `corpus-train`

AdamW (β = 0.9, 0.95; weight decay on matrices only), linear warmup then cosine decay to 10% of the
peak rate, gradient clipping at 1.0, seeded. Every `eval_every` steps: mean loss over
`eval_batches` batches of each split. The checkpoint kept is the one with the lowest held-out
loss. Device: CPU (`device` can name another, but MPS is never picked on its own: the user
measured it as far slower than CPU for this kind of model in eva-llm).

    corpus-train --config PATH [--output DIR]

`PATH` is a `training.json`; the run goes to `runs/<config folder name>/<timestamp>/` (`runs/` is
gitignored) with:

- `config.json`: the resolved settings, defaults filled in;
- `tokenizer.json`: the learned merges, reloaded with `ByteLevelBPETokenizer.from_merges`;
- `model.pt`: the best checkpoint's weights;
- `metrics.json`: every evaluation (step, tokens seen, learning rate, train and held-out loss),
  the best step, and the final train/held-out gap; plus document and token counts per split.

## Generation — `src/corpus_generalization/model/generate.py`, command `corpus-generate`

Autoregressive sampling from a prompt: temperature, top-k, a maximum number of new tokens, and a
seed; it stops at `<eos>`. Temperature 0 is greedy decoding.

    corpus-generate --run DIR --prompt TEXT [--max-tokens N] [--temperature T] [--top-k K] [--seed S]

## UI — `src/corpus_generalization/ui/app.py`, `make ui`

A Streamlit page: pick a run from `runs/`, see its corpus, model size and best held-out loss, its
loss curves (train and held-out against step), and generate from a prompt with the sampling
controls above, several samples at once. It uses the same loading and generation functions as
`corpus-generate`, so what the page shows is what the command produces.

## Tests (before the code)

- **Model:** logits have shape `(batch, time, vocab)`; causality — changing a later token never
  changes an earlier position's logits; the head is tied to the embedding.
- **Data:** the split is disjoint, deterministic for a seed, and covers every document; the
  tokenizer never sees held-out text; batch targets are the inputs shifted by one.
- **Training:** a tiny model overfits a tiny corpus (training loss falls well below its start);
  the run folder holds the four files; a reloaded run gives the same logits as before saving.
- **Generation:** greedy decoding is deterministic; sampling is deterministic for a seed; it stops
  at `<eos>`; top-k 1 equals greedy.
- **Command and UI:** `corpus-train` on a tiny config writes a run; a bad config is reported; the
  Streamlit page runs against a temp `runs/` folder (`streamlit.testing.v1.AppTest`) and
  generates text.

## Out of scope for now

Syntactic-validity evaluation (it needs the synthetic grammar's validator), several model sizes
(decided when the calibration experiment is designed), and mixed precision.
