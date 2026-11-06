# Do the Ward classes match parts of speech?

Design and predictions: `docs/superpowers/specs/2026-11-05-class-purity-design.md`, written
before running. Corpus side only: no model is trained.

Each domain's pool #1 at 1× (14.4 MB) from `experiments/granularity_robustness` is tokenized at
its 1.50 budget, clustered and cut as `diagnose` does. spaCy `en_core_web_sm` tags the same pool.
A whole-word token (`" court"`, mostly ending its word) gets the universal tag that holds ≥ 90% of
its ≥ 20 occurrences after whitespace; the cut and the tree are compared with those tags, each
measure against tags shuffled over the same tokens.

    uv run python experiments/class_purity/class_purity.py read <corpus>    # 1-3 minutes each
    uv run python experiments/class_purity/class_purity.py summarize

`results/<corpus>.json` holds every measure, the composition of each class and the ARI of every
cut from 8 to 64 classes; `results/summary.json` the predictions.

## Result (2026-11-05)

**The classes are grammatical well above chance, but they are positions in the sentence more than
parts of speech.** Nouns and verbs each have a node of the tree that matches them closely in every
domain; determiners, pronouns and prepositions do not have classes of their own.

| Domain | Classes | Word tokens read | Their share of running tokens | Purity | Null purity (max of 1,000) | ARI | Null ARI (max) | Chosen k's rank of 57 | Best k (ARI) |
|---|---|---|---|---|---|---|---|---|---|
| TinyStories | 10 | 368 | 44% | 0.67 | 0.34 | 0.51 | 0.02 | 11 | 19 (0.57) |
| FineWeb | 11 | 3,781 | 39% | 0.82 | 0.44 | 0.69 | 0.02 | 3 | 8 (0.69) |
| Wikipedia | 13 | 5,917 | 41% | 0.73 | 0.39 | 0.39 | 0.01 | 6 | 9 (0.41) |
| PubMed | 8 | 4,840 | 46% | 0.68 | 0.51 | 0.45 | 0.01 | 2 | 9 (0.45) |
| ECtHR | 11 | 861 | 43% | 0.71 | 0.42 | 0.65 | 0.03 | 2 | 13 (0.65) |

Best node per tag (F1 of the tree's closest subtree to the tag's tokens; tags with ≥ 20 tokens):

| Tag | TinyStories | FineWeb | Wikipedia | PubMed | ECtHR |
|---|---|---|---|---|---|
| NOUN | 0.90 | 0.92 | 0.78 | 0.82 | 0.91 |
| VERB | 0.87 | 0.84 | 0.81 | 0.76 | 0.86 |
| ADJ | 0.81 | 0.79 | 0.68 | 0.64 | 0.66 |
| ADV | 0.50 | 0.64 | 0.68 | 0.60 | 0.57 |
| ADP | · | 0.70 | 0.67 | 0.65 | 0.76 |
| PRON | 0.54 | 0.57 | 0.28 | · | 0.40 |
| DET | · | 0.57 | 0.51 | 0.48 | · |
| PROPN | · | 0.69 | 0.56 | 0.30 | 0.31 |

`·`: fewer than 20 tokens with that tag. Every tag read, in every domain, beats the best of 200
shuffles (null maxima 0.09–0.68; NOUN's is highest because nouns are most of the tokens).

- **Open classes come out.** NOUN and VERB have a node at F1 0.76–0.92 in all five domains, ADJ
  at 0.64–0.81. In ECtHR one class is 318 of 356 nouns (` applicant`, ` Court`, ` proceedings`)
  and another 172 of 183 verbs (` found`, ` stated`, ` lodged`); in TinyStories the noun class
  is all nouns (70 of 70).
- **Closed classes are grouped by position, not by tag.** The class that holds ` the` and ` a` also
  holds possessives and adjectives in FineWeb, PubMed and ECtHR (FineWeb: 346 of its 485 tagged
  tokens are ADJ): everything that comes before a noun. Prepositions and conjunctions (` of`, ` and`, ` to`) share a
  class with verbs in FineWeb, Wikipedia and PubMed: what comes after a noun. Pronouns sit with
  sentence openers. This is what neighbor distributions see, and it is a syntactic grouping, but
  not the tagger's.
- **Proper nouns merge with nouns and with capitalized pieces** (PROPN 0.30–0.69).
- **More than half of running tokens are not read** (39–46% are): word pieces, punctuation,
  numbers and ambiguous words have no tag to compare with. Their classes (` p`, ` s`; `ing`, `ed`;
  digits) group pieces by where they fall in a word, as on the recipe grammar.
- **The cut's choice is near the best** in four domains (rank 2–6 of 57; FineWeb's 11 classes
  reach 0.69 against a best of 0.69). TinyStories is the exception: 10 classes give 0.51, 19
  give 0.57.

## Predictions

1. Purity above every permutation in all five domains — **HELD** (0.67–0.82 against at most
   0.51).
2. ARI above every permutation in all five domains — **HELD** (0.39–0.69 against at most 0.03).
3. NOUN, VERB, ADJ, ADV, ADP, DET and PRON each beat every permutation in at least four of five
   domains — **FAILED** as written: DET is read in three domains and has fewer than 20 tokens in
   TinyStories and ECtHR, so it counts there as not beating. Every tag that was read beat the
   permutations in every domain.
4. The chosen k in the top quarter of ARI over 8–64 in at least three domains — **HELD** (all
   five: ranks 11, 3, 6, 2, 2 of 57).

## What it means for the indicator

The class tree carries real grammatical structure on natural text, which is why removed classes
are found (`experiments/granularity_robustness`). But "class" in the paper should be read as
*distributional position*: the pre-noun class gathers determiners, possessives and adjectives,
and a removal confined to one of them (only the possessives) is a node inside that class, which is
why step 7 reads every node of the tree and not only the cut.
