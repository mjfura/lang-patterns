"""Where a corpus's documents come from: raw-text files on disk or a Hugging Face dataset."""

from collections.abc import Callable
from pathlib import Path

# What separates one raw-text document from the next, on a line of its own. A corpus of
# single-line documents does not need it — a sentence per line says where each one ends — but a
# document that contains newlines has no other way to say it.
DOCUMENT_SEPARATOR = "<|endoftext|>"

HuggingFaceLoader = Callable[..., list[str]]


class MissingDataFileError(FileNotFoundError):
    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Data file {path} does not exist. The dataset is read from disk, not generated."
        )
        self.path = path


class EmptyDataFileError(ValueError):
    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Data file {path} holds no non-blank lines, so it would produce an empty corpus "
            f"instead of an error."
        )
        self.path = path


def read_documents(path: Path) -> tuple[str, ...]:
    """The documents of a raw-text corpus file, in file order.

    Two shapes, told apart by whether `DOCUMENT_SEPARATOR` appears anywhere in the file. Without
    it, a document is a line; with it, a document is everything between two separators, newlines
    and all. The rule is decidable from the file alone rather than from a flag, so a corpus cannot
    be read the wrong way by a caller who forgot to say which kind it is. Blank documents are
    dropped and an empty file is an error.
    """
    if not path.is_file():
        raise MissingDataFileError(path)
    text = path.read_text(encoding="utf-8")
    if DOCUMENT_SEPARATOR in text:
        documents = tuple(
            stripped
            for chunk in text.split(DOCUMENT_SEPARATOR)
            if (stripped := chunk.strip("\r\n")).strip()
        )
    else:
        documents = tuple(stripped for line in text.splitlines() if (stripped := line.strip()))
    if not documents:
        raise EmptyDataFileError(path)
    return documents


def read_documents_from(paths: list[Path]) -> tuple[str, ...]:
    """Every document across `paths`, in file order, concatenated across files."""
    return tuple(document for path in paths for document in read_documents(path))


def huggingface_documents(
    name: str,
    split: str,
    text_field: str,
    limit: int | None,
    config_name: str | None = None,
) -> list[str]:
    """Every non-empty value of `text_field` in `name`'s `split`, capped at `limit` rows (`None`
    for the whole split, fetched with a plain, non-streaming download since there is then no
    limit to stream up to). Imports `datasets` lazily: most runs never touch a Hugging Face
    dataset at all, and the import is neither cheap nor small.
    """
    from datasets import load_dataset

    rows = load_dataset(
        name,
        name=config_name,
        split=split,
        streaming=limit is not None,
    )
    if limit is not None:
        rows = rows.take(limit)
    return [str(text) for row in rows if (text := row[text_field])]
