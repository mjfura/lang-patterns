"""The models on BLiMP, the field's benchmark of minimal pairs.

"The models on BLiMP" in `docs/2026-09-28-syntactic-sufficiency-design.md` records the
predictions before scoring. A pair is correct when the grammatical sentence gets the higher total
log probability after `<bos>`. No model is trained.

    blimp.py score       every model of model_sweep and class_patterns on the 67 paradigms
    blimp.py summarize   the predictions

BLiMP comes from Hugging Face (`nyu-mll/blimp`). No command runs on import.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

OUT = Path(__file__).parent
RESULTS = OUT / "results"
EXPERIMENTS = OUT.parent
sys.path[:0] = [str(EXPERIMENTS / "model_sweep"), str(EXPERIMENTS / "class_patterns")]

import class_patterns  # noqa: E402
import model_sweep  # noqa: E402
from class_patterns import divergence, syntactic_eval  # noqa: E402

BATCH = 256
PHENOMENA = ("subject_verb_agreement", "determiner_noun_agreement", "irregular_forms")


def models() -> dict[str, Path]:
    found = {
        f"{corpus}_{config}": model_sweep.run_path(corpus, config)
        for corpus in model_sweep.CORPORA
        for config in model_sweep.CONFIGS
    }
    for name in (*class_patterns.CORPUS_NAMES, "poss_class_cut", "poss_head_cut", "poss_control"):
        for seed in (0, 1):
            found[f"{name}_small_seed{seed}"] = class_patterns.RUNS / f"{name}_seed{seed}"
    return found


def pairs() -> list[dict[str, str]]:
    from datasets import get_dataset_config_names, load_dataset

    rows = []
    for paradigm in get_dataset_config_names("nyu-mll/blimp"):
        for row in load_dataset("nyu-mll/blimp", paradigm, split="train"):
            rows.append(
                {
                    "good": row["sentence_good"],
                    "bad": row["sentence_bad"],
                    "paradigm": row["UID"],
                    "term": row["linguistics_term"],
                }
            )
    return rows


def log_probabilities(run: syntactic_eval.LoadedRun, sentences: list[str]) -> list[float]:
    """Total log probability of each sentence's tokens after `<bos>`, batched with right padding:
    a causal model's earlier positions never see the padding."""
    result = []
    for start in range(0, len(sentences), BATCH):
        encoded = [
            [run.tokenizer.bos_id, *run.tokenizer.encode(s, recognize_markers=True)][
                : run.model.context
            ]
            for s in sentences[start : start + BATCH]
        ]
        width = max(len(ids) for ids in encoded)
        batch = torch.tensor([ids + [0] * (width - len(ids)) for ids in encoded])
        with torch.no_grad():
            log_probs = torch.log_softmax(run.model(batch), dim=-1)
        for row, ids in enumerate(encoded):
            targets = torch.tensor(ids[1:])
            result.append(float(log_probs[row, : len(ids) - 1].gather(1, targets[:, None]).sum()))
    return result


def score() -> None:
    rows = pairs()
    good = [r["good"] for r in rows]
    bad = [r["bad"] for r in rows]
    results: dict[str, dict[str, float]] = {}
    for name, path in models().items():
        divergence.progress(f"Scoring {name}")
        run = syntactic_eval.load_run(path)
        correct = [
            g > b
            for g, b in zip(log_probabilities(run, good), log_probabilities(run, bad), strict=True)
        ]
        by_paradigm: dict[str, list[bool]] = defaultdict(list)
        by_term: dict[str, list[bool]] = defaultdict(list)
        for row, ok in zip(rows, correct, strict=True):
            by_paradigm[row["paradigm"]].append(ok)
            by_term[row["term"]].append(ok)
        paradigm_accuracy = {p: sum(v) / len(v) for p, v in by_paradigm.items()}
        results[name] = {
            "overall": statistics.mean(paradigm_accuracy.values()),
            **{f"term:{t}": sum(v) / len(v) for t, v in by_term.items()},
            **{f"paradigm:{p}": a for p, a in paradigm_accuracy.items()},
        }
        print(f"{name}: overall {results[name]['overall']:.4f}")
    divergence.write_json(RESULTS / "scores.json", results)


def summarize() -> None:
    scores = json.loads((RESULTS / "scores.json").read_text())
    sweep = json.loads((EXPERIMENTS / "model_sweep/results/summary.json").read_text())["class_g"]
    held = {"1": True, "2": True}
    for config in model_sweep.CONFIGS:
        overall = scores[f"full_{config}"]["overall"]
        held["1"] &= overall >= 0.60
        print(f"F {config}: BLiMP overall {overall:.4f}")
    for kind, (control, cut) in model_sweep.PAIRS.items():
        for config in model_sweep.CONFIGS:
            blimp_gap = (
                scores[f"{control}_{config}"]["overall"] - scores[f"{cut}_{config}"]["overall"]
            )
            g_gap = sweep[kind][control][config] - sweep[kind][cut][config]
            held["2"] &= abs(blimp_gap) < g_gap
            print(f"{kind:10} {config:6} BLiMP gap {blimp_gap:+.4f}  G gap {g_gap:+.4f}")
    print("phenomena (exploratory), small models:")
    for name in sorted(scores):
        if "_small" in name or name.endswith("_small"):
            print(
                f"  {name:28} overall {scores[name]['overall']:.3f} "
                + " ".join(
                    f"{t.split('_')[0]} {scores[name].get(f'term:{t}', float('nan')):.3f}"
                    for t in PHENOMENA
                )
            )
    predictions = {k: "HELD" if v else "FAILED" for k, v in held.items()}
    predictions["3"] = "exploratory"
    print(f"Predictions: {json.dumps(predictions)}")
    divergence.write_json(RESULTS / "summary.json", {"predictions": predictions})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("score", "summarize"))
    {"score": score, "summarize": summarize}[parser.parse_args().command]()


if __name__ == "__main__":
    main()
