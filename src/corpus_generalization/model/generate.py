"""Load a trained run and generate text from it — the same functions the Streamlit page uses, so
what the page shows is what this command produces.

    corpus-generate --run DIR --prompt TEXT [--max-tokens N] [--temperature T] [--top-k K]
                    [--seed S]

Temperature 0 is greedy decoding; generation stops at `<eos>` or after `--max-tokens` tokens.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from corpus_generalization.model.gpt import GPT, GPTConfig
from corpus_generalization.model.train import TrainingSettings
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer


class LanguageModel(Protocol):
    context: int

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class LoadedRun:
    path: Path
    model: GPT
    tokenizer: ByteLevelBPETokenizer
    settings: TrainingSettings
    sources: dict[str, Any]
    metrics: dict[str, Any]


def load_run(path: Path, device: str = "cpu", *, checkpoint: int | None = None) -> LoadedRun:
    checkpoint_path = (
        path / "model.pt" if checkpoint is None else path / "checkpoints" / f"step_{checkpoint}.pt"
    )
    if checkpoint is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint step {checkpoint} does not exist in run {path}")
    config = json.loads((path / "config.json").read_text())
    settings_names = TrainingSettings.__dataclass_fields__
    settings = TrainingSettings(**{k: v for k, v in config.items() if k in settings_names})
    sources = {k: v for k, v in config.items() if k not in settings_names}
    merges = [
        ((pair[0], pair[1]), new_id)
        for pair, new_id in json.loads((path / "tokenizer.json").read_text())["merges"]
    ]
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**saved["model_config"])).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return LoadedRun(
        path=path,
        model=model,
        tokenizer=ByteLevelBPETokenizer.from_merges(merges),
        settings=settings,
        sources=sources,
        metrics=json.loads((path / "metrics.json").read_text()),
    )


def generate(
    model: LanguageModel,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> str:
    """The text `model` continues `prompt` with (the prompt itself not included), on the CPU."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    tokens = [tokenizer.bos_id, *tokenizer.encode(prompt, recognize_markers=True)]
    new_tokens: list[int] = []
    with torch.no_grad():
        for _ in range(max_tokens):
            window = torch.tensor([tokens[-model.context :]], dtype=torch.long)
            logits = model(window)[0, -1].float().cpu()
            if temperature == 0:
                next_token = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k is not None:
                    cutoff = torch.topk(logits, min(top_k, len(logits))).values[-1]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probabilities, 1, generator=generator))
            if next_token == tokenizer.eos_id:
                break
            tokens.append(next_token)
            new_tokens.append(next_token)
    return tokenizer.decode(new_tokens)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--run", type=Path, required=True, help="a run folder from corpus-train")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    loaded = load_run(args.run)
    text = generate(
        loaded.model,
        loaded.tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k,
        args.seed,
    )
    print(args.prompt + text)
    return 0
