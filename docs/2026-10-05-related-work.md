# Related work and positioning (notes for the paper)

Written 2026-10-05 from a targeted search; every work below was checked at its source (links
at the end). These are notes, not a survey: the paper's related-work section starts here.

## 1. Data quality and data selection for pretraining

- Perplexity filtering, CCNet-style; `experiments/subset_selection` uses a bigram stand-in.
- DSIR, importance resampling on hashed n-grams (Xie et al. 2023).
- QuRating, LLM-judged quality ratings (Wettig et al. 2024).
- The diversity coefficient, Task2Vec variability of a dataset (Miranda, Lee et al. 2023/2025).
- Training-free influence prediction such as DataProphet (ICLR 2026, multimodal supervision).
- A survey of the whole area: Albalak et al. 2024.

**What they share.**

- One score per document or per dataset.
- Aimed at perplexity or downstream accuracy.

**What none of them do.**

- Say which syntactic patterns a corpus under-serves.
- Compare against a reference, with size-matched noise.
- Separate a corpus fault from a model fault.

**Our evidence against them.**

- On the recipe grammar, held-out loss and n-gram predictability rank corpora in reverse of
  syntactic generalization (`syntactic_sufficiency`).
- On FineWeb, perplexity filters of the CCNet kind move the slots far beyond random:
  `self25` z +15 and `wikihigh25` z +20.6 (`divergence`, `cut_choice`).

## 2. Learning syntax from manipulated training corpora

- **Filtered Corpus Training, FiCT** (Patil, Jumelet et al., TACL 2024). It removes a
  construction from the training corpus, trains LSTMs and transformers, and finds that both
  generalize to the unseen construction from indirect evidence.
- **Unlike coordination by FiCT** (Luo & Steinert-Threlkeld, arXiv 2607.20251, 2026). Models
  learn unlike coordination from alike coordination alone.
- **The missing AANNs** (Misra & Mahowald, EMNLP 2024, outstanding paper). A rare construction
  ("a beautiful five days") is learned from related, more frequent ones, and the learning is
  stronger with more variability in the input.
- **Frequency effects on syntactic rule learning** (Wei, Garrette, Linzen & Pavlick, EMNLP
  2021). BERT applies subject–verb agreement to pairs never seen together, but struggles
  against strong training priors and on infrequent items.

**How we relate.** These works manipulate a corpus and read the trained model. Their message is
that models generalize from indirect evidence, and that variability helps.

- **Agreement at the head level.** In `class_patterns`, `she` and `they` keep their syntax
  when only they lose the verb slot (−0.04 G): indirect evidence from their siblings.
- **Our addition at the class level.** When the whole class loses the pattern, every member
  loses its syntax (−0.13 to −0.30 G), and no model configuration recovers it
  (`model_sweep`).
- **Our addition on the corpus side.** The indicator computes, before training, which removals
  matter, and it attributes a failure to the corpus or to the model.

**What none of them do.**

- Offer a pre-training diagnostic of which patterns a corpus fails to serve.
- Separate head-level from class-level evidence.

## 3. Diversity and productivity measures

- **Baayen's productivity measures** (corpus morphology): the hapax ratio P. Our missing mass
  (singletons / trunk) is this measure applied to a syntactic slot, and the paper must credit
  it as the antecedent.
- **Linguistic diversity of model outputs**: lexical, syntactic (dependency-tree graphs) and
  semantic (Guo et al. 2024, the "curious decline" line). These measure generated text, not
  training corpora, and are not tied to what a model learns.

## 4. Targeted syntactic evaluation

- Minimal pairs: Marvin & Linzen 2018, the SyntaxGym assessment of Hu et al. 2020, BLiMP
  (Warstadt et al. 2020).
- Our G is a minimal-pair reading with unseen class-valid fillers against frequency-matched
  ungrammatical ones, and classes from an external tagger.
- The paper should also report the trained models on BLiMP, so readers can compare.

## Where this leaves the contribution

The combination is new:

1. a corpus-only, per-pattern indicator, read against a reference with size-matched noise, at
   two levels (head and induced class);
2. causal validation by corpus interventions (plantings, cuts, class versus head removal);
3. an attribution test: a flag is a corpus fault no model fixes, and no flag means the
   reference's syntax is reachable.

**Gaps a reviewer will raise, in order of weight.**

- **Scale.** Our models go up to 4 layers by 256 wide, on corpora of 1.4–14 MB. FiCT and AANN
  train on human-scale corpora, BabyLM-sized.
- **Baselines.** The existing data-quality scores (perplexity, type–token ratio or n-gram
  diversity, the diversity coefficient) have not been shown, on the same cuts, to miss what
  the indicator finds.
- **A standard benchmark.** BLiMP on the existing models would tie G to the field's measure.
- **One language, one natural source** (English FineWeb).

## Sources

- [FiCT, TACL 2024](https://aclanthology.org/2024.tacl-1.87/)
- [Exposure is Optional: unlike coordination, arXiv 2607.20251](https://arxiv.org/abs/2607.20251)
- [The missing AANNs, arXiv 2403.19827 / EMNLP 2024](https://arxiv.org/abs/2403.19827)
- [Frequency effects on syntactic rule learning, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.72/)
- [Beyond Scale: the diversity coefficient, arXiv 2306.13840](https://arxiv.org/abs/2306.13840)
- [DataProphet, arXiv 2603.19688](https://arxiv.org/pdf/2603.19688)
- [QuRating, arXiv 2402.09739](https://arxiv.org/html/2402.09739v3)
- [DSIR, arXiv 2302.03169](https://arxiv.org/pdf/2302.03169)
- [A survey on data selection for language models, arXiv 2402.16827](https://arxiv.org/html/2402.16827v1)
- [Benchmarking linguistic diversity of LLMs, arXiv 2412.10271](https://arxiv.org/pdf/2412.10271)
- [Baayen, morphological productivity](https://quantling.org/~hbaayen/publications/BaayenHSK2009.pdf)
- [Targeted syntactic evaluation, arXiv 1808.09031](https://arxiv.org/pdf/1808.09031)
- [A systematic assessment of syntactic generalization, arXiv 2005.03692](https://arxiv.org/pdf/2005.03692)
- [BLiMP, TACL 2020](https://aclanthology.org/2020.tacl-1.25.pdf)
