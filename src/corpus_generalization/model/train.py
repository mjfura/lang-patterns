"""Train the fixed GPT on one corpus and save the run — see `docs/2026-09-28-training-design.md`.

    corpus-train --config PATH [--output DIR]

`PATH` is a `training.json`. Its keys, all optional except a source:

    files                 text files of documents separated by `<|endoftext|>` lines
    huggingface_dataset   a Hugging Face dataset instead of, or besides, `files`; with
                          huggingface_config, huggingface_split ("train"),
                          huggingface_text_field ("text") and huggingface_limit (2000)
    merge_budget          byte-level BPE merges, learned on the train split only (4000)
    heldout_fraction      documents held out for the held-out loss (0.1)
    seed                  the split, the initialization and the batches (0)
    n_layer, n_head, d_model, context, dropout
                          the model (4, 4, 256, 256, 0.1)
    batch_size, steps, learning_rate, warmup_steps, weight_decay
                          the optimization (32, 2000, 3e-4, 100, 0.1)
    eval_every, eval_batches
                          how often, and over how many batches per split, losses are measured
                          (100, 20)
    device                "cpu" (the default: MPS was measured far slower for this in eva-llm)

The run goes to `runs/<config folder name>/<timestamp>/` (or `--output`): `config.json` (every
setting, defaults filled in), `tokenizer.json` (the merges), `model.pt` (the weights of the
evaluation with the lowest held-out loss) and `metrics.json` (every evaluation, the best step, the
held-out/train gap there, document, token and parameter counts, and saved checkpoint steps).
When requested through `train(checkpoint_steps=...)`, `checkpoints/step_<n>.pt` stores the
weights after exactly `n` optimizer steps, in the same format as `model.pt`.
"""

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from corpus_generalization.config_file import ConfigFileError, ConfigSchema, ListOf, load_config
from corpus_generalization.model.data import random_batch, split_documents, token_stream
from corpus_generalization.model.gpt import GPT, GPTConfig
from corpus_generalization.sources import (
    EmptyDataFileError,
    MissingDataFileError,
    huggingface_documents,
    read_documents_from,
)
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

RUNS_DIR = Path("runs")
MINIMUM_LEARNING_RATE_FRACTION = 0.1
GRADIENT_CLIP = 1.0
ADAM_BETAS = (0.9, 0.95)


@dataclass(frozen=True)
class TrainingSettings:
    merge_budget: int = 4000
    heldout_fraction: float = 0.1
    seed: int = 0
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    context: int = 256
    dropout: float = 0.1
    batch_size: int = 32
    steps: int = 2000
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    eval_every: int = 100
    eval_batches: int = 20
    device: str = "cpu"


SOURCE_SCHEMA: ConfigSchema = {
    "files": ListOf(str),
    "huggingface_dataset": str,
    "huggingface_config": str,
    "huggingface_split": str,
    "huggingface_text_field": str,
    "huggingface_limit": int,
}
CONFIG_SCHEMA: ConfigSchema = {
    **SOURCE_SCHEMA,
    **{
        field.name: field.type for field in fields(TrainingSettings) if isinstance(field.type, type)
    },
}
Progress = Callable[[dict[str, float]], None]


class TrainingConfigError(ValueError):
    def __init__(self, path: Path, problem: str) -> None:
        super().__init__(f"{path}: {problem}.")


def learning_rate_at(step: int, settings: TrainingSettings) -> float:
    """Linear warmup to `learning_rate`, then cosine decay to a tenth of it at the last step."""
    if step < settings.warmup_steps:
        return settings.learning_rate * (step + 1) / settings.warmup_steps
    decay_steps = max(1, settings.steps - settings.warmup_steps)
    progress = min(1.0, (step - settings.warmup_steps) / decay_steps)
    floor = settings.learning_rate * MINIMUM_LEARNING_RATE_FRACTION
    return floor + 0.5 * (settings.learning_rate - floor) * (1 + math.cos(math.pi * progress))


def _optimizer(model: GPT, settings: TrainingSettings) -> torch.optim.AdamW:
    # Weight decay on matrices only: decaying biases and LayerNorm gains pulls them toward zero
    # for no regularizing benefit.
    decayed = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
    kept = [parameter for parameter in model.parameters() if parameter.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": settings.weight_decay},
            {"params": kept, "weight_decay": 0.0},
        ],
        lr=settings.learning_rate,
        betas=ADAM_BETAS,
    )


def _mean_loss(
    model: GPT, stream: torch.Tensor, settings: TrainingSettings, device: torch.device
) -> float:
    # The same batches at every evaluation, so successive losses differ by the model, not by
    # which windows happened to be drawn.
    generator = torch.Generator().manual_seed(settings.seed + 1)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(settings.eval_batches):
            inputs, targets = random_batch(stream, settings.batch_size, settings.context, generator)
            logits = model(inputs.to(device))
            losses.append(
                functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten()).item()
            )
    model.train()
    return sum(losses) / len(losses)


def train(
    documents: Sequence[str],
    settings: TrainingSettings,
    output: Path,
    sources: dict[str, object] | None = None,
    progress: Progress | None = None,
    *,
    checkpoint_steps: Sequence[int] = (),
) -> Path:
    """Train on `documents` and write the run to `output`, which it returns. `sources` (where
    the documents came from) is recorded in `config.json` beside the settings."""
    requested_steps = set(checkpoint_steps)
    for step in requested_steps:
        if not 0 <= step <= settings.steps:
            raise ValueError(f"Checkpoint step {step} is outside 0..{settings.steps}")
    start = time.time()
    torch.manual_seed(settings.seed)
    device = torch.device(settings.device)
    train_documents, heldout_documents = split_documents(
        documents, settings.heldout_fraction, settings.seed
    )
    tokenizer = ByteLevelBPETokenizer.from_document_sequence(
        train_documents, max_merges=settings.merge_budget, recognize_markers=True
    )
    streams = {
        "train": token_stream(train_documents, tokenizer),
        "heldout": token_stream(heldout_documents, tokenizer),
    }
    model = GPT(
        GPTConfig(
            vocab_size=tokenizer.vocab_size,
            context=settings.context,
            n_layer=settings.n_layer,
            n_head=settings.n_head,
            d_model=settings.d_model,
            dropout=settings.dropout,
        )
    ).to(device)
    optimizer = _optimizer(model, settings)
    generator = torch.Generator().manual_seed(settings.seed)

    evaluations: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] = {}
    best: dict[str, float] = {}
    for step in range(settings.steps + 1):
        if step in requested_steps:
            checkpoint_dir = output / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_config": asdict(model.config),
                    "state_dict": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    },
                },
                checkpoint_dir / f"step_{step}.pt",
            )
        if step % settings.eval_every == 0 or step == settings.steps:
            evaluation = {
                "step": step,
                "tokens_seen": step * settings.batch_size * settings.context,
                "learning_rate": learning_rate_at(step, settings),
                "train_loss": _mean_loss(model, streams["train"], settings, device),
                "heldout_loss": _mean_loss(model, streams["heldout"], settings, device),
            }
            evaluations.append(evaluation)
            if progress is not None:
                progress(evaluation)
            if not best or evaluation["heldout_loss"] < best["heldout_loss"]:
                best = evaluation
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        if step == settings.steps:
            break
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(step, settings)
        inputs, targets = random_batch(
            streams["train"], settings.batch_size, settings.context, generator
        )
        logits = model(inputs.to(device))
        loss = functional.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]  # torch leaves Tensor.backward unannotated
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps({**(sources or {}), **asdict(settings)}, indent=2) + "\n"
    )
    (output / "tokenizer.json").write_text(
        json.dumps({"merges": [[list(pair), new_id] for pair, new_id in tokenizer.merges]}) + "\n"
    )
    torch.save(
        {"model_config": asdict(model.config), "state_dict": best_state}, output / "model.pt"
    )
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "evaluations": evaluations,
                "checkpoints": sorted(requested_steps),
                "best_step": best["step"],
                "best_train_loss": best["train_loss"],
                "best_heldout_loss": best["heldout_loss"],
                "gap_at_best": best["heldout_loss"] - best["train_loss"],
                "documents": {"train": len(train_documents), "heldout": len(heldout_documents)},
                "tokens": {name: len(stream) for name, stream in streams.items()},
                "vocab_size": tokenizer.vocab_size,
                "parameters": model.parameter_count(),
                "seconds": round(time.time() - start, 1),
            },
            indent=2,
        )
        + "\n"
    )
    return output


def _load(path: Path) -> tuple[TrainingSettings, dict[str, object], list[str]]:
    if not path.is_file():
        raise TrainingConfigError(path, "does not exist")
    config = load_config(path, CONFIG_SCHEMA)
    # `load_config` has checked every value against its schema type, so these reads are typed.
    sources: dict[str, Any] = {key: config.pop(key) for key in list(config) if key in SOURCE_SCHEMA}
    files: list[str] = sources.get("files", [])
    dataset: str | None = sources.get("huggingface_dataset")
    if not files and not dataset:
        raise TrainingConfigError(
            path, '"files" has to list at least one text file, or "huggingface_dataset" a dataset'
        )
    documents = list(read_documents_from([Path(file) for file in files]))
    if dataset:
        documents += huggingface_documents(
            dataset,
            sources.get("huggingface_split", "train"),
            sources.get("huggingface_text_field", "text"),
            sources.get("huggingface_limit", 2000),
            sources.get("huggingface_config"),
        )
    return TrainingSettings(**config), sources, documents


def _print_progress(evaluation: dict[str, float]) -> None:
    print(
        f"  step {int(evaluation['step']):>6}  lr {evaluation['learning_rate']:.2e}  "
        f"train {evaluation['train_loss']:.4f}  held-out {evaluation['heldout_loss']:.4f}",
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--config", type=Path, required=True, help="a training.json")
    parser.add_argument(
        "--output", type=Path, help="where the dated run folders go (default runs/NAME/)"
    )
    args = parser.parse_args(argv)
    config_path: Path = args.config
    try:
        settings, sources, documents = _load(config_path)
    except (
        ConfigFileError,
        TrainingConfigError,
        EmptyDataFileError,
        MissingDataFileError,
    ) as error:
        print(error, file=sys.stderr)
        return 1
    name = config_path.resolve().parent.name
    output: Path = args.output if args.output is not None else RUNS_DIR
    run = output / name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"  {len(documents):,} documents; training into {run}", file=sys.stderr)
    train(documents, settings, run, sources, _print_progress)
    metrics = json.loads((run / "metrics.json").read_text())
    print(
        f"best held-out loss {metrics['best_heldout_loss']:.4f} at step {metrics['best_step']} "
        f"(train {metrics['best_train_loss']:.4f}, gap {metrics['gap_at_best']:.4f}); {run}"
    )
    return 0
