"""`huggingface_documents` passes what it is given to `datasets.load_dataset`, and nothing more."""

import pytest

from corpus_generalization.sources import huggingface_documents


def _fake_loader(calls: list[dict[str, object]]) -> object:
    def fake_load(path: str, **kwargs: object) -> list[dict[str, str]]:
        calls.append({"path": path, **kwargs})
        return [{"content": "a"}, {"content": ""}, {"content": "b"}]

    return fake_load


def test_huggingface_documents_passes_data_files(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("datasets.load_dataset", _fake_loader(calls))
    files = ["chunk/pubmed23n1000.jsonl"]
    documents = huggingface_documents("MedRAG/pubmed", "train", "content", None, None, files)
    assert documents == ["a", "b"]
    assert calls[0]["data_files"] == files
    assert calls[0]["streaming"] is False


def test_huggingface_documents_omits_data_files_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr("datasets.load_dataset", _fake_loader(calls))
    assert huggingface_documents("MedRAG/pubmed", "train", "content", None) == ["a", "b"]
    assert "data_files" not in calls[0]
