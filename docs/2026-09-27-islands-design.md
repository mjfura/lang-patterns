# Islands: which part of the vocabulary sits outside the main region

**Goal:** diagnose a corpus by saying, for each level of repetition demanded, which part of its
vocabulary already forms one connected region, which part forms separate regions (islands), and
which part is left isolated. It is the first step of a corpus-quality indicator: it points at the
tokens that do not repeat enough to be learned, and therefore at where to add data.

It replaces the idea of reading island size off the gap between the red and blue curves. That gap
is ill-defined because the two x axes are in different units: red's is one transition's count,
blue's is a token's total weight.

## Definition

The same graph and threshold as the red curve (`red_cluster_coverage_curve`): at threshold `t`,
keep the edges with `count >= t` and take their weakly connected components. Every node of the
**full** graph falls into exactly one of three classes:

| Class | What it is | Reading |
|---|---|---|
| **Mainland** | the largest component | equal to red coverage at `t` |
| **Islands** | every other component with 2 or more nodes | patterns that repeat, but are disconnected from the mainland at that level |
| **Isolated** | nodes with no edge reaching `t` | tokens whose transitions do not repeat enough to be learned |

The three fractions, divided by the full graph's node count, sum to exactly 1. That is the central
property: the red curve becomes one part of a complete decomposition instead of a single number
that hides where the rest of the vocabulary is.

**Ties for the mainland:** if two components share the largest size, one is taken as the mainland
and the other as an island. The fractions do not depend on which.

**Nodes whose only edge reaching `t` is a self-loop (`a→a`):** they form a one-node component.
They count as isolated, unless that component is the largest in the graph, in which case it is the
mainland. This keeps `mainland` exactly equal to the red curve, which counts it too.

## Output

Everything goes in `markov_chain.json`, under an `islands` field per order.

- **Per threshold** (the same observed levels the red curve sweeps): `mainland`, `islands` and
  `isolated` (fractions), `island_count` and `largest_island` (in nodes).
- **At `WELL_SUPPORTED_THRESHOLD` (2):** the members of each island, largest first, as token ids
  and decoded text. This is the actionable part: it says which tokens form each island. The data
  keeps every island of 2 or more nodes; the report shows only islands of 3 or more, so it is not
  flooded with lone `a→b` pairs.
- **Plot:** `islands_order<N>.png`, the three fractions stacked against the threshold on a log
  axis. The bottom band is the red curve.

## Implementation

One descending union-find sweep, the same as `_component_coverage_curve`, plus a `Counter` of
component sizes updated on each union, so every threshold costs O(1) beyond the unions themselves.
Island members are read once, when the sweep passes threshold 2. The `weight_key` parameter
(default `"count"`) is kept, like every other function in the module, but only the `count` version
is reported in this stage.

## Tests (before the code)

1. **Two disjoint triangles with different counts, plus a node with count-1 edges:** exact
   mainland, islands and isolated at every threshold.
2. **Consistency:** on varied graphs, `mainland` equals `red_cluster_coverage_curve` point by point.
3. **Sum:** `mainland + islands + isolated == 1` at every threshold.
4. **Tie:** two components of the same size give one mainland and one island.
5. **Self-loop:** a node whose only surviving edge is `a→a` counts as isolated.
6. **Members at `t = 2`:** exact sets, ordered by size.

## Out of scope for this stage

- The `effective_document_support` version.
- An AUC for islands: it is not yet clear what it would claim.
- Counting domains. Here an island is "something that does not connect at level `t`", not a domain.
