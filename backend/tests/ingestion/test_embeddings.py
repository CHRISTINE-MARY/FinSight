import math
from types import SimpleNamespace

import pytest

from app.config import NVIDIA_BASE_URL
from app.database.models.document_chunk import EMBEDDING_DIMENSIONS
from app.ingestion.embeddings import embed_texts, fit_stored_dimensions


def test_fit_stored_dimensions_truncates_and_normalizes() -> None:
    vector = [float(index + 1) for index in range(2048)]
    fitted = fit_stored_dimensions(vector)

    assert len(fitted) == EMBEDDING_DIMENSIONS
    norm = math.sqrt(sum(value * value for value in fitted))
    assert norm == pytest.approx(1.0)
    assert fitted[0] > 0


def test_fit_stored_dimensions_rejects_short_vectors() -> None:
    with pytest.raises(RuntimeError, match="1536"):
        fit_stored_dimensions([0.1, 0.2])


def test_embed_texts_calls_nvidia_with_input_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Item:
        def __init__(self, index: int, embedding: list[float]) -> None:
            self.index = index
            self.embedding = embedding

    class _Response:
        def __init__(self) -> None:
            self.data = [_Item(0, [1.0] * EMBEDDING_DIMENSIONS)]

    class _Embeddings:
        def create(self, **kwargs: object) -> _Response:
            captured.update(kwargs)
            return _Response()

    class _Client:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.embeddings = _Embeddings()

    monkeypatch.setattr("app.ingestion.embeddings.OpenAI", _Client)
    monkeypatch.setattr(
        "app.ingestion.embeddings.settings",
        SimpleNamespace(
            nvidia_api_key="nvapi-test",
            nvidia_embedding_model="nvidia/nemotron-3.5-embed-1b",
        ),
    )

    vectors = embed_texts(["revenue"], input_type="query")

    client = captured["client"]
    assert isinstance(client, dict)
    assert client["base_url"] == NVIDIA_BASE_URL
    assert client["api_key"] == "nvapi-test"
    assert captured["model"] == "nvidia/nemotron-3.5-embed-1b"
    assert captured["input"] == ["revenue"]
    assert captured["dimensions"] == EMBEDDING_DIMENSIONS
    assert captured["extra_body"] == {"input_type": "query", "truncate": "NONE"}
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIMENSIONS
