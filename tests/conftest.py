from pathlib import Path

import pytest

from corpus_generalization import corpora


@pytest.fixture(autouse=True)
def corpora_the_developer_has_not_edited(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """Point every corpus lookup at an empty `configs/corpora/` for the duration of each test: a
    corpus someone added must not change what a test sees, and a test must never write into the
    real one."""
    directory = tmp_path_factory.mktemp("corpora")
    monkeypatch.setattr(corpora, "CORPORA_CONFIG_DIR", directory)
    return directory
