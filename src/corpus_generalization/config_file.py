"""JSON configuration files that hold a command's usual arguments, so a run needs no inline flags.

Every `eva` subcommand and the standalone scripts under `scripts/` read one of these — named for
the command, under `configs/` — as the source of its argparse defaults. A flag typed on the
command line still overrides its value for that one run, exactly as `--vary` overrides
`sweep_grid.json` for `eva sweep`; what changes is that the *usual* run no longer needs any flags
at all, and changing what "usual" means is an edit to a file rather than a change to the code.

A config file is optional. Every value it can hold already has a code-level default declared
alongside the argument that reads it, so a fresh checkout — or a config file deleted by hand —
runs exactly as it did before this module existed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Anchored on this file rather than on the working directory, for the same reason
# `sources.DATA_DIR` is: pytest, a shell and an editor all run from different places.
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class ListOf:
    """A schema entry for a value that must be a list, every element of type `element`.

    `nargs="+"`/`nargs="*"` arguments (`--roots`, `--sections`, `--temperatures`) are the only
    ones that need this; every other argument's type is a plain Python type and needs nothing
    beyond `isinstance`.
    """

    element: type


@dataclass(frozen=True)
class Nullable:
    """A schema entry for a value that is either JSON `null` or of type `inner`.

    For a field whose code-level meaning of "absent" (`config.get(key, DEFAULT)`, `DEFAULT` used)
    already differs from its meaning of "explicitly off" -- `huggingface_limit`'s 2000-row safety
    default versus a file that deliberately asks for every row, for one -- only this lets a config
    file say the second without the first swallowing it: leaving the key out still reads as
    `DEFAULT`, exactly as before this wrapper existed, but a file that sets it to `null` now reads
    as `None` instead of failing type validation.
    """

    inner: type


type ConfigSchema = dict[str, type | ListOf | Nullable]


class ConfigFileError(ValueError):
    """One base class for every way a config file cannot be used.

    A caller can no more recover from a malformed value than from an unknown key, so both share a
    type and carry the difference in the message — the same reasoning `CorpusLineError` and
    `MalformedGridFileError` already follow for the corpus and the sweep grid.
    """


class MalformedConfigFileError(ConfigFileError):
    def __init__(self, path: Path, problem: str) -> None:
        super().__init__(f"Config file {path} is not usable: {problem}.")
        self.path = path


class UnknownConfigKeyError(ConfigFileError):
    def __init__(self, path: Path, keys: set[str], schema: ConfigSchema) -> None:
        super().__init__(
            f"Config file {path} sets {', '.join(sorted(keys))}, which it does not take. "
            f"Known keys: {', '.join(sorted(schema))}."
        )
        self.path = path
        self.keys = keys


def load_config(path: Path, schema: ConfigSchema) -> dict[str, Any]:
    """A command's persisted defaults, read from `path` and checked against `schema`.

    Returns `{}` when the file does not exist, so an argparse default written as
    `config.get("steps", DEFAULT_STEPS)` behaves identically whether the file is present, absent,
    or simply silent about that one key.

    Every value is checked against the exact type `schema` declares — `True` does not satisfy an
    `int` field and `1` does not satisfy a `bool` one — for the same reason
    `eva_llm.checkpoint._field_value` checks a checkpoint's travelling config that strictly: a
    loose `int(value)` would let a fractional or boolean JSON value through and quietly build a
    run unlike the one the file describes.
    """
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as unreadable:
        raise MalformedConfigFileError(path, unreadable.msg) from unreadable
    if not isinstance(raw, dict):
        raise MalformedConfigFileError(path, "must be a JSON object mapping flags to values")

    unknown = set(raw) - set(schema)
    if unknown:
        raise UnknownConfigKeyError(path, unknown, schema)
    return {key: _typed_value(path, key, value, schema[key]) for key, value in raw.items()}


def _typed_value(path: Path, key: str, value: object, declared: type | ListOf | Nullable) -> object:
    if isinstance(declared, Nullable):
        return None if value is None else _typed_value(path, key, value, declared.inner)
    if isinstance(declared, ListOf):
        if not isinstance(value, list) or not all(type(item) is declared.element for item in value):
            raise MalformedConfigFileError(
                path, f"{key}={value!r} must be a list of {declared.element.__name__}"
            )
        return value
    if declared is Path:
        if type(value) is not str:
            raise MalformedConfigFileError(path, f"{key}={value!r} must be a string path")
        return Path(value)
    if declared is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise MalformedConfigFileError(path, f"{key}={value!r} must be a number")
        return float(value)
    if type(value) is not declared:
        raise MalformedConfigFileError(path, f"{key}={value!r} must be a {declared.__name__}")
    return value
