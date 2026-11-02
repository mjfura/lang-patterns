# Granularity Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read the 517 new cells of the tokens-per-word grid and judge whether the range 1.30–2.00
holds across corpus size, pool draw and two knowledge domains. The grid is 5 corpora × (1/16, ¼
and 1× in three pools, plus 4× in one) × 11 granularities.

**Architecture:**

- A new experiment, `experiments/granularity_robustness/granularity_robustness.py`, holds:
  - the pure rules, tested in `tests/test_granularity_robustness.py`: variants, the pool layout,
    the SciPy/`linkage_vector` switch, workers and judging;
  - the commands: `download`, `verify`, `prepare`, `costs`, `diagnose`, `calibrate`, `run` and
    `summarize`.
- It reuses `experiments/granularity_domains` through four functions extracted from that
  script's `prepare`, `diagnose_target` and `summarize`, with no change in behavior.
- `src/` changes in one place: `huggingface_documents` gains `data_files`, for PubMed's chunks.

**Tech Stack:** Python 3.13, uv, `datasets` (Hugging Face), SciPy and `fastcluster` Ward, spaCy
`en_core_web_sm`, pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-10-28-granularity-robustness-design.md`. It builds on
`docs/superpowers/specs/2026-10-25-granularity-across-domains-design.md` and its amendments.

## Global Constraints

- **Language:** chat with the user in Spanish; code, comments, README and commits in English.
- **No commits or branches unless the user asks.** Where a task ends, run `make check`, and the
  user commits.
- **Machine and memory:**
  - 64 GB RAM, 18 cores;
  - memory ceiling `MEMORY_CEILING = 48 * 10**9` bytes;
  - at most `MAX_WORKERS = 9` diagnosis workers. `OMP_NUM_THREADS` stays `"2"`, as in
    `granularity_domains`, so 9 × 2 ≤ 18 cores.
- **CPU only;** no MPS.
- **Granularities:** `TARGETS = (1.30, 1.35, 1.40, 1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.20,
  2.50)`, reused from `granularity_domains.TARGETS`.
- **Unit of size:** `UNIT = base.size(F)` bytes (F = `natural_gap.split()[1]`, 4,600 documents,
  14.4 MB).
  - Sizes are `SIZES = (0.0625, 0.25, 1.0, 4.0)`.
  - 4× exists for pool 0 only.
  - Each corpus needs `6 * UNIT` bytes.
- **Corpora:** `CORPORA = ("tinystories", "fineweb", "wikipedia", "pubmed", "legal")`.
- **Already read, never re-read:** `DONE = {"tinystories@1#0", "fineweb@1#0", "wikipedia@1#0"}`.
  Their verdicts come from `experiments/granularity_domains/results/summary.json`.
- **Unchanged from the 10-25 design:**
  - the pass rule, alpha 0.06, pooled false flags at most 12%;
  - the 22 cuts and their seeds;
  - the gate and its replacements.
- **Long runs:**
  - run in the background with a log;
  - a wait on a log is scoped with `sed -n '/<marker>/,$p'`;
  - the shell is zsh, so `$var` is not word-split.
- **Stop for the user** when a step's predictions fail, when pool #0 does not reproduce, or before
  the long steps (Task 9).

## Review Focus

1. **A pool that runs short.** Legal may not reach `6 * UNIT`. The layout must drop legal's pool #2
   and record it, not crash and not return a short pool. Tested in Task 3
   (`test_layout_drops_pool_2_when_short`).
2. **A granularity a small pool cannot reach** (1/16 at 1.30). It must be written as
   `unreachable`, and judging must leave it out instead of counting it as a failure. Tested in
   Task 3 (`test_agrees_skips_unreachable`).
3. **A variant name typed wrong, or 4× with pool 1 or 2.** It must be rejected with a message, not
   create an empty run. Tested in Task 3 (`test_parse_variant_rejects`).
4. **FineWeb's pool #0 rebuilt from a new stream.** F must be the same 4,600 documents as before, or
   every FineWeb variant silently changes. `verify` asserts F's text equals `natural_gap.split()`
   (Task 5).
5. **A process that runs many readings and reports the wrong peak.** `ru_maxrss` is a lifetime
   peak, so `run` starts each reading in its own subprocess (Task 7).

---

### Task 1: `huggingface_documents` takes `data_files`

**Files:**
- Modify: `src/corpus_generalization/sources.py:61-83`
- Test: `tests/test_sources.py` (create)

**Interfaces:**
- Produces: `huggingface_documents(name: str, split: str, text_field: str, limit: int | None,
  config_name: str | None = None, data_files: list[str] | None = None) -> list[str]`.
  `data_files` is passed to `load_dataset(..., data_files=data_files)` only when it is not `None`,
  so every existing call is unchanged.

- [x] **Step 1: Write the failing tests**

```python
def test_huggingface_documents_passes_data_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load(name: str, **kwargs: object) -> list[dict[str, str]]:
        calls.append({"name": name, **kwargs})
        return [{"content": "a"}, {"content": ""}, {"content": "b"}]

    monkeypatch.setattr("datasets.load_dataset", fake_load)
    files = ["chunk/pubmed23n1000.jsonl"]
    assert huggingface_documents("MedRAG/pubmed", "train", "content", None, None, files) == ["a", "b"]
    assert calls[0]["data_files"] == files and calls[0]["streaming"] is False


def test_huggingface_documents_omits_data_files_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same fake; assert "data_files" not in calls[0].
```

- [x] **Step 2: Run them to see them fail**

  Run `uv run pytest tests/test_sources.py -v`. Expected: FAIL, because the function does not take
  a 6th argument.

- [x] **Step 3: Add the `data_files` parameter.** Build the keyword arguments as a dict, add
  `data_files` only when it is given, and extend the docstring by one sentence.

- [x] **Step 4: Run the tests and `make check`.** Expected: PASS, 445 tests.

---

### Task 2: Extract `granularity_domains`' reusable steps

Five extractions, so that a pool can be prepared, read and judged without a domain name. The
behavior must stay identical; Task 5 checks it by reproducing TinyStories' `prepared.json` byte for
byte.

**Files:**
- Modify: `experiments/granularity_domains/granularity_domains.py`:
  - `prepare` (lines 259-407);
  - `diagnose_target` (lines 476-536);
  - `summarize` (lines 644-708).

**Interfaces (Produces):**
- **`prepare_pool(pool: list[str], runs: Path, results: Path) -> None`.** The body of `prepare`
  after `pool = pool_of(domain)`. It writes `runs / "merges.json"`, `runs / "corpora" / *.json`
  and `results / "prepared.json"`, and raises `SystemExit` when 1.50 is unreachable.
  `prepare(domain)` becomes the FineWeb guard plus
  `prepare_pool(pool_of(domain), RUNS / domain, RESULTS / domain)`.
- **`diagnose_pool(pool: list[str], merges: Sequence[Merge], budget: int | None, target: str,
  corpora: Path, out: Path, workers: int = WORKERS) -> dict[str, Any]`.** It:
  - writes `{"target", "status": "unreachable"}` when `budget` is `None`;
  - otherwise writes the `status: "read"` record, with the same keys as today;
  - returns what it wrote;
  - has no memory rule.

  `diagnose_target` keeps its memory rule, then calls it.
- **`judge_all(read: dict[str, dict[str, Any]], record: dict[str, Any], merges: Sequence[Merge])
  -> dict[str, dict[str, Any]]`.** The `per_target` comprehension of `summarize`.
- **`pooled_false_flags(per_target: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]`.** The
  `pooled` dict of `summarize`, with its comment.
- **`longest_run(per_target: dict[str, dict[str, Any]]) -> list[str]`.** The `best` loop: the
  longest run of consecutive holding targets.

- [x] **Step 1: Extract the five functions.** Move code as it is, and keep its comments with it.
  `summarize` calls the new functions.

- [x] **Step 2: Run `make check`.** Expected: PASS, ruff clean on `experiments/`. `summarize`
  cannot be re-run here: its per-target readings are gitignored and were not copied. Task 5
  checks the refactor instead.

---

### Task 3: The experiment's rules (pure, tested)

**Files:**
- Create: `experiments/granularity_robustness/granularity_robustness.py`. The module docstring
  lists the commands; there is a `sys.path` set-up like `ward_memory.py`'s (imports
  `granularity_domains as domains`); no command runs on import.
- Create: `tests/test_granularity_robustness.py`, loading the module with `importlib`, as
  `tests/test_dose_response.py` does.
- Modify: `.gitignore`. Add `experiments/granularity_robustness/results/**/target_*.json`.

**Interfaces (Produces):**
- **Constants** (Global Constraints values): `CORPORA`, `SIZES`, `POOLS = (0, 1, 2)`, `DONE`,
  `MEMORY_CEILING`, `MAX_WORKERS`, `SCIPY_OVERHEAD = 2 * 10**9`, `CUTS = 22`.
  - `OUT`, `RESULTS = OUT / "results"`, `RUNS = Path("runs") / "granularity_robustness"`.
  - `STEPS: dict[str, list[str]]` maps `"small"`, `"pools"`, `"domains"` and `"4x"` to their
    variant names, in the spec's order 4–7.
- **`@dataclass(frozen=True) class Variant`**, with fields `corpus: str`, `size: float` and
  `pool: int`.
  - Its property `name` is `f"{corpus}@{size:g}#{pool}"`, for example `"wikipedia@0.25#1"` and
    `"tinystories@4#0"`.
- **`parse_variant(name: str) -> Variant`.** It raises `ValueError` naming the problem, for:
  - an unknown corpus or size;
  - a pool outside 0–2;
  - 4× with a pool other than 0.
- **`variants() -> list[Variant]`.** All 50. **`new_variants() -> list[Variant]`.** Those not in
  `DONE`: 47.
- **`layout(sizes: Sequence[int], unit: int, first: Sequence[int], rest_from: int) -> dict[tuple[float, int], list[int]]`.**
  Indices into the cache, keyed `(size, pool)`:
  - `(1.0, 0)` is `first`;
  - `(4.0, 0)` is `first` followed by indices from `rest_from` on, until its bytes reach
    `4 * unit`;
  - `(1.0, 1)` and `(1.0, 2)` are the next slices after the 4× pool, each reaching `unit`;
  - `(s, k)` for s < 1 is the leading indices of `(1.0, k)` reaching `s * unit`.

  Every slice uses `subset_selection.fill`'s rule: take documents until the bytes reach the
  target, overshooting by at most one document. If pool #2 cannot be filled, its three keys are
  left out. A shortfall before pool #2 raises `ValueError`.
- **`scipy_peak_bytes(tokens: int) -> int`.** Returns `8 * tokens * (tokens - 1) + SCIPY_OVERHEAD`.
  **`uses_scipy(tokens: int) -> bool`.** True when `scipy_peak_bytes(tokens) <= MEMORY_CEILING`.
- **`clustering(tokens: int) -> Iterator[str]`** (a `contextlib.contextmanager`).
  - When `uses_scipy(tokens)` is false, it swaps `syntactic.linkage` for a `fastcluster.linkage_vector`
    wrapper, as in `ward_memory._vector_linkage`.
  - It yields `"scipy"` or `"vector"`, and always restores `syntactic.linkage`.
- **`workers_for(parent_peak: int, child_peak: int) -> int`.** Returns
  `max(1, min(MAX_WORKERS, CUTS, (MEMORY_CEILING - parent_peak) // child_peak))`.
- **`legal_document(paragraphs: Sequence[str]) -> str`.** The stripped, non-empty paragraphs,
  joined by `"\n"`.
- **`pubmed_chunk(number: int) -> str`.** Returns `f"chunk/pubmed23n{number:04d}.jsonl"`.
- **`agrees(variant: dict[str, bool | None], base: dict[str, bool]) -> bool`.** True when every
  target with a non-`None` verdict in `variant` equals `base`'s. `None` means unreachable.
- **`first_failure_above(holds: dict[str, bool | None], floor: str = "2.00") -> str | None`.** The
  lowest target above `floor` whose verdict is `False`.

- [x] **Step 1: Write the failing tests.** One test per rule, with these exact assertions:

```python
def test_variant_names(rb: ModuleType) -> None:
    assert rb.Variant("wikipedia", 0.25, 1).name == "wikipedia@0.25#1"
    assert rb.Variant("tinystories", 4.0, 0).name == "tinystories@4#0"
    assert rb.parse_variant("pubmed@0.0625#2") == rb.Variant("pubmed", 0.0625, 2)
    assert len(rb.variants()) == 50 and len(rb.new_variants()) == 47
    assert sum(len(v) for v in rb.STEPS.values()) == 47
    assert len(rb.STEPS["small"]) == 30 and len(rb.STEPS["4x"]) == 5


@pytest.mark.parametrize("name", ["legal@4#1", "fineweb@2#0", "news@1#0", "tinystories@1#3", "x"])
def test_parse_variant_rejects(rb: ModuleType, name: str) -> None:
    with pytest.raises(ValueError):
        rb.parse_variant(name)


def test_layout_slices_are_nested_and_disjoint(rb: ModuleType) -> None:
    sizes = [10] * 100  # unit 40 -> 4 docs per 1x pool
    cut = rb.layout(sizes, 40, first=[0, 1, 2, 3], rest_from=4)
    assert cut[(1.0, 0)] == [0, 1, 2, 3]
    assert cut[(4.0, 0)] == list(range(16))
    assert cut[(1.0, 1)] == [16, 17, 18, 19] and cut[(1.0, 2)] == [20, 21, 22, 23]
    assert cut[(0.25, 1)] == [16] and cut[(0.0625, 2)] == [20]


def test_layout_fineweb_first_pool_is_not_a_prefix(rb: ModuleType) -> None:
    cut = rb.layout([10] * 100, 40, first=[1, 3, 5, 7], rest_from=10)
    assert cut[(4.0, 0)][:4] == [1, 3, 5, 7] and cut[(4.0, 0)][4] == 10


def test_layout_drops_pool_2_when_short(rb: ModuleType) -> None:
    cut = rb.layout([10] * 22, 40, first=[0, 1, 2, 3], rest_from=4)
    assert (1.0, 1) in cut and not any(k[1] == 2 for k in cut)
    with pytest.raises(ValueError):
        rb.layout([10] * 18, 40, first=[0, 1, 2, 3], rest_from=4)


def test_scipy_rule_bounds_the_measured_peaks(rb: ModuleType) -> None:
    assert rb.scipy_peak_bytes(11_459) >= 1.38e9 and rb.scipy_peak_bytes(15_893) >= 1.76e9
    assert rb.uses_scipy(57_944) and rb.uses_scipy(75_000) and not rb.uses_scipy(76_000)


def test_workers_for(rb: ModuleType) -> None:
    assert rb.workers_for(4 * 10**9, 2 * 10**9) == 9
    assert rb.workers_for(10 * 10**9, 5 * 10**9) == 7
    assert rb.workers_for(47 * 10**9, 5 * 10**9) == 1


def test_legal_document_and_pubmed_chunk(rb: ModuleType) -> None:
    assert (
        rb.legal_document(["1. The applicant ", "", "2. He was"]) == "1. The applicant\n2. He was"
    )
    assert rb.pubmed_chunk(998) == "chunk/pubmed23n0998.jsonl"


def test_agrees_skips_unreachable(rb: ModuleType) -> None:
    base = {"1.30": True, "1.50": True, "2.20": False}
    assert rb.agrees({"1.30": None, "1.50": True, "2.20": False}, base)
    assert not rb.agrees({"1.30": None, "1.50": True, "2.20": True}, base)


def test_first_failure_above(rb: ModuleType) -> None:
    assert (
        rb.first_failure_above({"1.50": False, "2.00": True, "2.20": True, "2.50": False}) == "2.50"
    )
    assert rb.first_failure_above({"2.20": True, "2.50": None}) is None


def test_clustering_restores_linkage(rb: ModuleType) -> None:
    from corpus_generalization.analysis import syntactic

    original = syntactic.linkage
    with rb.clustering(80_000) as tree:
        assert tree == "vector" and syntactic.linkage is not original
    assert syntactic.linkage is original
    with rb.clustering(1_000) as tree:
        assert tree == "scipy" and syntactic.linkage is original
```

- [x] **Step 2: Run them to see them fail.** Run
  `uv run pytest tests/test_granularity_robustness.py -v`. Expected: FAIL, because the module does
  not exist.
- [x] **Step 3: Implement the constants and the eleven functions above.**
- [x] **Step 4: Run the tests, then `make check`.** Expected: PASS.

---

### Task 4: `download`, then run it

**Files:**
- Modify: `experiments/granularity_robustness/granularity_robustness.py`

**Interfaces:**
- Consumes: `huggingface_documents(..., data_files=...)` (Task 1), and `legal_document` and
  `pubmed_chunk` (Task 3).
- Produces: `cache_of(corpus: str) -> Path`, which is `OUT / f"{corpus}_documents.json"`
  (gitignored by `*_documents.json`), and `download(corpus: str) -> None`.

The sources:

| Corpus | Call | Also writes the old cache (its prefix) |
|---|---|---|
| tinystories | `huggingface_documents("roneneldan/TinyStories", "train", "text", 100_000)` | first 25,000 to `domains.TINYSTORIES` |
| fineweb | `huggingface_documents("HuggingFaceFW/fineweb", "train", "text", 30_000)` | first 5,000 to `clean_domains.cache_of("fineweb")` |
| wikipedia | `huggingface_documents("wikimedia/wikipedia", "train", "text", 4_000, "20231101.en")` | first 1,500 to `domains.WIKIPEDIA` |
| pubmed | chunks 1000, 999, … one call each (`limit=None`, `data_files=[pubmed_chunk(n)]`, field `content`), until the abstracts reach `6 * UNIT`; stored in ascending chunk order | none |
| legal | `load_dataset("coastalcph/lex_glue", "ecthr_a")` splits train, validation, test, each row through `legal_document(row["text"])` | none |

- The old caches are written only when absent, so that `granularity_domains`, `natural_gap` and
  `ward_memory` find their data.
- `download` prints the documents, the bytes and `bytes / UNIT`, then calls `os._exit(0)`, as
  `granularity_domains.download` does.
- PubMed's and legal's `UNIT` needs F, so fineweb is downloaded first.

- [x] **Step 1: Implement `cache_of` and `download`, and register the `download <corpus>`
  subcommand.**
- [x] **Step 2: Run the downloads in order:** fineweb, tinystories, wikipedia, pubmed, legal. Each
  is `uv run python experiments/granularity_robustness/granularity_robustness.py download <corpus>`,
  run in the background with a log. Expected: the four that are not legal print `bytes / UNIT >= 6`.
- [x] **Step 3: Check legal's size.** If legal prints below 6, note it: `layout` will drop pool #2
  (Review Focus 1), and the summary reports it.
- [x] **Step 4: Run `make check`.**

---

### Task 5: `verify`: pool #0 and the refactor reproduce

**Interfaces:**
- Consumes: `layout` (Task 3), `cache_of` (Task 4), and `prepare_pool` (Task 2).
- Produces:
  - `pools(corpus: str) -> dict[tuple[float, int], list[str]]`: the documents of every slice.
    - FineWeb's `first` is the cache positions of F: the 5,000 leading documents shuffled with
      `random.Random(0)`, the first 400 dropped, the rest sorted. That is `natural_gap.split()`'s
      rule, and `rest_from=5000`.
    - For the other corpora, `first` is `fill(sizes, range(n), UNIT)` and `rest_from=len(first)`.
  - `pool_of(variant: Variant) -> list[str]`.
  - `verify() -> None`.

What `verify` checks:

1. **FineWeb's pool #0 is F.** `pools("fineweb")[(1.0, 0)] == natural_gap.split()[1]`.
2. **The three general corpora reach the recorded merges.** For each, it learns pool #0's
   tokenizer to 1.30 (`learn_to_granularity`). Every target's budget must equal `merge_budget` in
   `experiments/granularity_domains/results/summary.json`: TinyStories 3,322 at 1.30 and 1,223 at
   1.50; FineWeb 51,214 and 11,381; Wikipedia 70,215 and 16,022.
3. It writes `results/verify.json` with every comparison, and exits non-zero on any mismatch.

- [x] **Step 1: Implement `pools`, `pool_of` and `verify`, and register `verify`.**
- [x] **Step 2: Run `verify` in the background.** Wikipedia's 70,215 merges are the slow part.
  Expected: every budget matches.
- [x] **Step 3: Check the refactor.** Run
  `uv run python experiments/granularity_domains/granularity_domains.py prepare tinystories`, then
  `git diff --stat experiments/granularity_domains/results/tinystories/prepared.json`. Expected: no
  diff.
- [x] **Step 4: On any mismatch, stop and tell the user.** Every result depends on pool #0.

---

### Task 6: `prepare` and `costs`, then prepare all 47 variants

**Interfaces:**
- Consumes: `pool_of`, `clustering`, `uses_scipy` and `scipy_peak_bytes`, and
  `domains.prepare_pool`.
- Produces: `prepare(name: str) -> None` and `costs() -> None`.

**`prepare(name)`:**

1. It refuses names in `DONE`.
2. It calls `domains.prepare_pool(pool, RUNS / name, RESULTS / name, before_reading=...)`.
   - `prepare_pool` gains a keyword-only parameter,
     `before_reading: Callable[[ByteLevelBPETokenizer], AbstractContextManager[object]] | None = None`,
     whose default keeps Task 2's behavior.
   - It is called with the 1.50 tokenizer, and its context wraps the `prepare_reference` and
     `anchor_class` calls.
   - Here it is `lambda tokenizer: clustering(domains.clustered_tokens(pool, tokenizer))`, so the
     trees at 1.50 use `linkage_vector` when SciPy would not fit.
3. It then adds to `results/<name>/prepared.json`:
   - `"clustered_tokens"`: `{target: count}`, for every reachable target;
   - `"tree"`: `{target: "scipy" | "vector"}`;
   - `"projected_scipy_bytes"`: `{target: int}`;
   - `"pool"`: `{"corpus", "size", "pool"}`.

`prepare all` loops over `new_variants()` in `STEPS` order, each in its own subprocess, and skips
the variants already prepared.

**`costs()`** prints and writes `results/costs.json`, one row per new reading, with:

- the variant and the target;
- the merges;
- the clustered tokens;
- the tree;
- `projected_scipy_bytes`;
- reachable or not.

It ends with totals per step and per tree.

- [x] **Step 1: Write the failing test.** `test_prepare_refuses_done` asserts that
  `rb.prepare("fineweb@1#0")` raises `SystemExit`.
- [x] **Step 2: Implement `prepare`, `prepare all`, `costs` and `before_reading`, and run the
  test and `make check`.**
- [x] **Step 3: Run `prepare all` in the background.** Expected: 47 `prepared.json` files.
  - Gate replacements appear in the log as `under the gate ...; replaced`.
  - A variant whose 1.50 is unreachable stops with `does not reach 1.50`: note it as a prediction 1
    failure.
- [x] **Step 4: Run `costs`.**
- [x] **Step 5: Judge predictions 8, 9 and 12** from `prepared.json` and `costs.json`:
  - 8: PubMed's 1.50 budget is above 16,022; legal's is between 1,223 and 16,022;
  - 9: no `replaced` slot in `pubmed@1#0` or `legal@1#0`;
  - 12: only 4× readings at ≤ 1.50 have `tree: "vector"`.
- [x] **Step 6: Report to the user, in Spanish:** the costs table, which readings need
  `linkage_vector`, and predictions 8, 9 and 12. Wait for their go-ahead before Task 9.

---

### Task 7: `diagnose`, `calibrate` and `run`

**Interfaces:**
- Consumes: `domains.diagnose_pool` (Task 2), `clustering` and `workers_for` (Task 3), and
  `pool_of` (Task 5).
- Produces:
  - **`diagnose(name: str, target: str, workers: int) -> None`.** It runs `diagnose_pool` inside
    `clustering(tokens)` for the target's clustered tokens, then adds to the target JSON:
    - `"tree"`;
    - `"workers"`;
    - `"peak_parent_bytes"`, from `resource.getrusage(RUSAGE_SELF).ru_maxrss`, which is in bytes
      on macOS;
    - `"peak_child_bytes"`, from `RUSAGE_CHILDREN`;
    - `"seconds"`.

    Its output is `results/<name>/target_<t>.json`.
  - **`calibrate() -> None`.** It runs `diagnose("wikipedia@1#1", "1.50", 2)` in a subprocess and
    writes `results/calibration.json`, with:
    - `seconds`;
    - both peaks;
    - `workers = workers_for(parent, child)`.
  - **`run(step: str) -> None`.** For each variant of `STEPS[step]`, it runs each target from 2.50
    down to 1.30, skipping targets already written.
    - Each target runs in its own subprocess (`subprocess.run([sys.executable, __file__,
      "diagnose", name, target, "--workers", str(w)], check=True)`).
    - The workers are `w = workers_for(max parent peak, max child peak)` over the readings already
      written in this step. Before the first reading, the step uses the calibration's peaks for
      `"small"`, `"pools"` and `"domains"`, and 2 workers for `"4x"`.
    - It prints one line per reading: variant, target, seconds, workers, tree.

- [x] **Step 1: Write the failing test.** `test_run_order` asserts that `rb.reading_order("small")`
  starts with `(rb.STEPS["small"][0], "2.50")`, ends with `(rb.STEPS["small"][-1], "1.30")`, and
  has `30 * 11` items. `reading_order(step) -> list[tuple[str, str]]` is the pure part of `run`.
- [x] **Step 2: Implement `reading_order`, `diagnose`, `calibrate` and `run`, and register them;
  then run the test and `make check`.**
- [x] **Step 3: Run `calibrate` in the background.** Expected: `calibration.json`, with workers
  between 1 and 9.
- [x] **Step 4: Report to the user** the calibration's time and workers, and the projected total
  time: readings × the calibration's seconds, scaled by pool size. Wait for the go-ahead.

---

### Task 8: `summarize` and the twelve predictions

**Interfaces:**
- Consumes: `domains.judge_all`, `pooled_false_flags` and `longest_run` (Task 2), and `agrees` and
  `first_failure_above` (Task 3).
- Produces:
  - **`summarize() -> None`**, which writes `results/summary.json`:
    `{"variants": {name: {...}}, "base": {corpus: {target: bool}}, "predictions": {"1": ..., "12": ...}}`.
  - **`predictions(summary: dict[str, Any]) -> dict[str, str]`**, with values `HELD`, `FAILED` or
    `unreadable`, like `granularity_domains.predictions`.

What each variant records:

- `holds` per target: `None` when unreachable;
- `measured_range`;
- `pooled_false_flags`;
- `range_valid`;
- `agrees`;
- `first_failure_above_2.00`;
- `tree` per target;
- the dropped words that are not tokens;
- swaps between adjacent targets (`domains.swaps`).

Where the base verdicts come from:

- for tinystories, fineweb and wikipedia, from `granularity_domains/results/summary.json`;
- for pubmed and legal, from `pubmed@1#0` and `legal@1#0`.

Each prediction reads only the variants that are done, and says `unreadable` until its variants
exist.

- [x] **Step 1: Write the failing test.** `test_predictions_on_a_fixture` builds a two-variant
  summary dict by hand and asserts two outcomes:
  - prediction 1 is `HELD` when both hold at 1.50, and `FAILED` when one does not;
  - prediction 4 is `unreadable` without 4× variants.
- [x] **Step 2: Implement `summarize` and `predictions`, and register `summarize`; then run the
  test and `make check`.**

---

### Task 9: Run the grid, step by step

Each step ends with `summarize`, a short report to the user in Spanish (what held, what failed,
time taken), and a stop if its predictions failed.

- [x] **Step 1: Spec step 4, `run small`** (330 readings: 1/16 and ¼, five corpora, three pools).
  Then run `summarize`. It judges predictions 1, 2, 3 and 5 on small sizes, and 6 on small pools.
- [x] **Step 2: Report; stop on any failed prediction.**
- [x] **Step 3: Spec step 5, `run pools`** (110 readings: 1× pools #1 and #2). Then run
  `summarize`, which judges predictions 6 and 7.
- [x] **Step 4: Report; stop on any failure.**
- [x] **Step 5: Spec step 6, `run domains`** (22 readings: `pubmed@1#0`, `legal@1#0`). Then run
  `summarize`, which judges predictions 10 and 11, and re-judges every PubMed and legal variant
  against its now-known base.
- [x] **Step 6: Report; stop on any failure.**
- [x] **Step 7: Spec step 7, `run 4x`** (55 readings, from 2.50 down). Then run `summarize`, which
  judges predictions 4 and 12 (the trees used).
- [x] **Step 8: Report the whole outcome**, against the spec's "What follows from the outcome".

---

### Task 10: README and statuses

**Files:**
- Create: `experiments/granularity_robustness/README.md`. Like
  `experiments/granularity_domains/README.md`, it gives:
  - the question;
  - the grid;
  - a results table per corpus: size × pool rows, holds per target;
  - the twelve predictions with HELD or FAILED;
  - costs (trees, times);
  - deviations from the spec, dated.
- Modify: the `**Status:**` line of the spec. It reads "experiment done (date), result in
  `experiments/granularity_robustness/README.md`. Not yet applied." Also check off this plan's
  steps.

- [x] **Step 1: Write the README from `summary.json` and `costs.json`.**
- [x] **Step 2: Update the status, then run `make check`.**
- [x] **Step 3: Ask the user how to apply the outcome.** The spec's "What follows" lists:
  - `GRANULARITY_RANGE`, changed with a test that pins it;
  - §5.8 and §7 in EN and ES;
  - the PDFs;
  - the status of the 10-25 spec.

  Those are a separate decision, made with the user.
