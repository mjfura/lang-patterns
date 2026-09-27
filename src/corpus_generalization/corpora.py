"""Named corpora: each one a folder `configs/corpora/<name>/` holding its analysis settings."""

import re
from pathlib import Path
from typing import Literal

from corpus_generalization.config_file import CONFIGS_DIR

CORPORA_CONFIG_DIR = CONFIGS_DIR / "corpora"
ANALYSIS_CONFIG_FILENAME = "tokenizer_analysis.json"
# A name becomes a folder, so it is held to what a folder name can safely be: a `/` or `..` would
# let `--corpus` read somewhere else entirely.
CORPUS_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

type TokenizerKind = Literal["reserved", "byte_level"]


class InvalidCorpusNameError(ValueError):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"Corpus name {name!r} must be letters, digits, '-' or '_' only: it names a folder "
            f"under configs/corpora/, and anything else could point outside it."
        )
        self.name = name


class CorpusRequiredError(ValueError):
    def __init__(self, command: str) -> None:
        super().__init__(
            f"{command} needs --corpus <name> (CORPUS=<name> through make). "
            f"Available corpora: {', '.join(available_corpora()) or 'none'}."
        )
        self.command = command


def available_corpora() -> list[str]:
    if not CORPORA_CONFIG_DIR.is_dir():
        return []
    return sorted(
        path.parent.name for path in CORPORA_CONFIG_DIR.glob(f"*/{ANALYSIS_CONFIG_FILENAME}")
    )


def corpus_config_dir(name: str) -> Path:
    if not CORPUS_NAME_PATTERN.fullmatch(name):
        raise InvalidCorpusNameError(name)
    return CORPORA_CONFIG_DIR / name
