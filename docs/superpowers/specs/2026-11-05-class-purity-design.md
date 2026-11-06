# Do the Ward classes match parts of speech?

**Status:** experiment done (2026-11-05), result in `experiments/class_purity/README.md`:
predictions 1, 2 and 4 held, 3 failed as written (DET has too few tokens in two domains). Design
and predictions written before running.

## Why

Step 2 of the indicator clusters tokens on their neighbor distributions (Ward) and cuts the tree
at the largest jump in merge height between 8 and 64 classes. The only direct check that the
classes are grammatical is the recipe grammar, where the categories are known. On natural text,
the evidence is indirect: removed classes are found in five domains
(`experiments/granularity_robustness`). That shows the classes are good enough for the
diagnosis, not that they are parts of speech.

Drawing the recipe tree at 1.50 tokens per word showed that the cut there rests on a jump of only
×1.09 (k = 9): the grammar's verbs, ingredients and adjectives come out exactly, and the other six
classes mix function words with word pieces. This experiment measures, on natural text, how far
the classes and the tree agree with an outside tagger.

## Design

- **Pools.** Pool #1 at 1× (14.4 MB) of the five domains of `experiments/granularity_robustness`
  (TinyStories, FineWeb, Wikipedia, PubMed, ECtHR), the size `MIN_DOMAIN_BYTES` sets. Pool #1
  because every domain has its merges in `runs/granularity_robustness/` (pool #0 of three domains
  was read in an earlier experiment). The tokenizer is those merges cut at the pool's 1.50 budget,
  the tree and cut are `class_tree` and `ClassTree.cut` as `diagnose` runs them.
- **Truth.** spaCy `en_core_web_sm` (tagger only), never the indicator. A word's tag is its
  majority universal tag over its occurrences after whitespace, case kept, when that tag holds
  ≥ 90% of ≥ 20 occurrences (`syntactic_eval`'s `WORD_SHARE` and `WORD_OCCURRENCES`). Every
  universal tag counts, closed classes included.
- **Evaluated tokens.** Tokens of the tree that are a space and a whole word (`" cut"`) with a tag.
  Word pieces and ambiguous words (` run`: NOUN and VERB) are left out; their share of running
  tokens is reported.
- **Measures, each against a null of permuted tags** (tags shuffled over the evaluated tokens,
  class sizes kept; 1,000 permutations for the cut, 200 for the tree):
  1. *Purity of the cut*: the share of evaluated tokens whose class's majority tag is their own.
  2. *Adjusted Rand index* (ARI) of the cut against the tags. Zero is chance.
  3. *Best node per tag*: for each tag with ≥ 20 evaluated tokens, the internal node of the tree
     whose evaluated leaves best match the tag's tokens (F1). This is what step 7 reads.
  4. *The cut's choice*: ARI of every cut from 8 to 64 classes, and where the chosen k ranks.

## Predictions

1. In all five domains the cut's purity is above every one of 1,000 permutations.
2. In all five domains the cut's ARI is above zero by more than every permutation's.
3. NOUN, VERB, ADJ, ADV, ADP, DET and PRON each have a node whose F1 beats every one of 200
   permutations, in at least four of five domains.
4. The chosen k is in the top quarter of ARI over k = 8 to 64 in at least three of five domains.

## Not in scope

Changing the class induction. If the classes fail, the finding goes to the paper as a limitation
and to a new spec; this experiment only measures.
