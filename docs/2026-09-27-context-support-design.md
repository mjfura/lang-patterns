# Context support: credit a transition by its distinct contexts, not its occurrences

**Goal:** stop duplicated text from inflating the curves. `experiments/contamination` showed that
adding exact copies of base documents raises every curve at least as much as adding new documents
from the same domain, `document_cluster_coverage_curve` included: each copy is a separate document,
so `document_support` and `effective_document_support` count it as independent evidence. A copy
adds occurrences, not patterns.

## Definition

An occurrence of the transition `s → t` at position `i` of a sequence (at order `k`, the window
`sequence[i : i + k]` followed by `sequence[i + k]`) has a context: the token just before the
window and the token just after the transition,

    context = (sequence[i - 1], sequence[i + k + 1])

with `None` where the sequence ends first (not a negative id: test tokenizers use those for
`<bos>`/`<eos>`).
`context_support(s → t)` is the number of distinct contexts over every occurrence in the corpus. It
becomes a fourth edge attribute of `build_transition_graph`, next to `count`, `document_support` and
`effective_document_support`, so every curve reads it through its existing `weight_key` parameter
with no new curve code.

Properties:

- `1 <= context_support <= count`.
- **Exact duplicates add nothing:** a copied document repeats the contexts its original already
  contributed. The same holds within a document that repeats itself.
- **Near-duplicates add little:** they change few neighbors, so few new contexts.
- **New text adds support** when it uses the transition in new surroundings.

## What it does not fix

Shuffled text puts transitions in many random contexts, so it raises `context_support` further. The
two failures are meant to be caught by different means: in `experiments/contamination`, the green
curve (flow persistence) already separates shuffled text and code from the control at high merge
budgets. The question for this measure is whether green read over `context_support` catches
duplicates as well, keeping the separation it already has.

A context of one token on each side is the smallest that separates a copy from new use. Wider
contexts separate more finely but converge on "every occurrence is distinct" (`context_support ==
count`) in a large corpus; the width stays at 1 until an experiment asks for more.

## Implementation

In `build_transition_graph`'s existing loop, collect a set of `(left, right)` pairs per `(state,
token)` and store its size. Memory is one pair per distinct context, bounded by the corpus's token
count; for the 6.5 MB FineWeb base this is a few million small tuples.

## Tests (before the code)

1. **Duplicate document:** appending a copy of a document doubles `count` and leaves
   `context_support` unchanged on every edge.
2. **New surroundings:** the same transition in a sentence with different neighbors adds one.
3. **Bounds:** `1 <= context_support <= count` on every edge of a varied corpus.
4. **Boundaries:** at the first and last position, the sentinel is the context's missing side, and
   `include_boundaries=True` makes `<bos>`/`<eos>` the neighbors instead.
5. **Order 2:** the context is the token before the two-token window and the token after the
   transition.
6. **Curves:** `red_cluster_coverage_curve(graph, weight_key="context_support")` is unchanged when
   every document is duplicated.

## Experiment

Rerun `experiments/contamination` (every seed) with green, red and blue read over
`context_support`. Success: `duplicate` no longer rises with the dose, and green still falls with
`python` and `shuffled` at 4000+ merges.
