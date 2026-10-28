"""`model.train`, `model.generate` and `corpus-train`: a run trains, is saved, reloads and
generates."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from corpus_generalization.model import train as train_command
from corpus_generalization.model.generate import generate, load_run
from corpus_generalization.model.gpt import GPT, GPTConfig
from corpus_generalization.model.train import TrainingSettings, train
from corpus_generalization.tokenizer.bpe import ByteLevelBPETokenizer

DOCUMENTS = tuple(
    f"the {animal} sat on the {thing}"
    for animal in ("cat", "dog", "bird", "fox")
    for thing in ("mat", "rug", "bed", "box", "log")
)
TINY = TrainingSettings(
    merge_budget=20,
    heldout_fraction=0.2,
    seed=0,
    n_layer=1,
    n_head=2,
    d_model=32,
    context=16,
    dropout=0.0,
    batch_size=8,
    steps=120,
    learning_rate=1e-2,
    warmup_steps=10,
    weight_decay=0.0,
    eval_every=40,
    eval_batches=2,
    device="cpu",
)


def test_training_lowers_the_loss_and_writes_the_run(tmp_path: Path) -> None:
    run = train(DOCUMENTS, TINY, tmp_path / "run")

    metrics = json.loads((run / "metrics.json").read_text())
    first, last = metrics["evaluations"][0], metrics["evaluations"][-1]
    assert last["train_loss"] < 0.5 * first["train_loss"]
    assert {"config.json", "tokenizer.json", "model.pt", "metrics.json"} <= {
        path.name for path in run.iterdir()
    }
    assert metrics["documents"] == {"train": 16, "heldout": 4}
    assert metrics["checkpoints"] == []
    assert not (run / "checkpoints").exists()
    assert metrics["best_step"] in [evaluation["step"] for evaluation in metrics["evaluations"]]


@pytest.mark.parametrize("steps", [(0, 40, 120), (120, 40, 0, 40)])
def test_training_saves_requested_checkpoints(tmp_path: Path, steps: tuple[int, ...]) -> None:
    run = train(DOCUMENTS, TINY, tmp_path / "run", checkpoint_steps=steps)

    assert {path.name for path in (run / "checkpoints").iterdir()} == {
        "step_0.pt",
        "step_40.pt",
        "step_120.pt",
    }
    assert json.loads((run / "metrics.json").read_text())["checkpoints"] == [0, 40, 120]
    initial = load_run(run, checkpoint=0)
    final = load_run(run, checkpoint=120)
    saved = torch.load(run / "checkpoints" / "step_120.pt", weights_only=True)
    expected = GPT(GPTConfig(**saved["model_config"]))
    expected.load_state_dict(saved["state_dict"])
    expected.eval()
    tokens = torch.tensor([final.tokenizer.encode("the cat")])

    with torch.no_grad():
        assert not torch.equal(initial.model(tokens), final.model(tokens))
        assert torch.equal(final.model(tokens), expected(tokens))


@pytest.mark.parametrize("step", [121, -1])
def test_invalid_checkpoint_step_writes_nothing(tmp_path: Path, step: int) -> None:
    with pytest.raises(ValueError, match=str(step)):
        train(DOCUMENTS, TINY, tmp_path / "run", checkpoint_steps=(0, step))

    assert list(tmp_path.iterdir()) == []


def test_loading_an_unsaved_checkpoint_names_the_step_and_run(tmp_path: Path) -> None:
    run = train(DOCUMENTS, TINY, tmp_path / "run")

    with pytest.raises(FileNotFoundError) as error:
        load_run(run, checkpoint=7)

    assert "7" in str(error.value)
    assert str(run) in str(error.value)


def test_a_reloaded_run_gives_the_same_logits(tmp_path: Path) -> None:
    run = train(DOCUMENTS, TINY, tmp_path / "run")

    first, second = load_run(run, "cpu"), load_run(run, "cpu")
    tokens = torch.tensor([first.tokenizer.encode("the cat")])

    with torch.no_grad():
        assert torch.equal(first.model(tokens), second.model(tokens))
    assert first.settings == TINY


def test_greedy_generation_is_deterministic_and_top_k_one_is_greedy(tmp_path: Path) -> None:
    loaded = load_run(train(DOCUMENTS, TINY, tmp_path / "run"), "cpu")

    greedy = generate(loaded.model, loaded.tokenizer, "the", 10, temperature=0.0)

    assert greedy == generate(loaded.model, loaded.tokenizer, "the", 10, temperature=0.0)
    assert greedy == generate(loaded.model, loaded.tokenizer, "the", 10, temperature=1.0, top_k=1)


def test_sampling_is_deterministic_for_a_seed() -> None:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=261, context=16, n_layer=1, n_head=1, d_model=16))
    tokenizer = ByteLevelBPETokenizer.from_merges(())

    samples = [generate(model, tokenizer, "a", 8, temperature=1.0, seed=seed) for seed in (1, 1, 2)]

    assert samples[0] == samples[1]
    assert samples[0] != samples[2]


class AlwaysEos(nn.Module):
    def __init__(self, vocab_size: int, eos_id: int) -> None:
        super().__init__()
        self.vocab_size, self.eos_id, self.calls = vocab_size, eos_id, 0
        self.context = 16

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        logits = torch.zeros((*tokens.shape, self.vocab_size))
        logits[..., self.eos_id] = 10.0
        return logits


def test_generation_stops_at_eos() -> None:
    tokenizer = ByteLevelBPETokenizer.from_merges(())
    model = AlwaysEos(tokenizer.vocab_size, tokenizer.eos_id)

    text = generate(model, tokenizer, "hello", 50, temperature=0.0)

    assert text == ""
    assert model.calls == 1


def _config(tmp_path: Path, **overrides: object) -> Path:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n<|endoftext|>\n".join(DOCUMENTS) + "\n")
    settings = {
        "files": [str(corpus)],
        **asdict(TINY),
        "steps": 20,
        **overrides,
    }
    path = tmp_path / "tiny-model" / "training.json"
    path.parent.mkdir()
    path.write_text(json.dumps(settings))
    return path


def test_corpus_train_writes_a_run_named_after_its_config_folder(tmp_path: Path) -> None:
    config = _config(tmp_path)

    exit_code = train_command.main(["--config", str(config), "--output", str(tmp_path / "runs")])

    (run,) = (tmp_path / "runs" / "tiny-model").iterdir()
    assert exit_code == 0
    assert (run / "model.pt").exists()
    assert json.loads((run / "config.json").read_text())["files"] == [str(tmp_path / "corpus.txt")]


def test_corpus_train_reports_an_unknown_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = _config(tmp_path, layers=3)

    exit_code = train_command.main(["--config", str(config), "--output", str(tmp_path / "runs")])

    assert exit_code == 1
    assert "layers" in capsys.readouterr().err
