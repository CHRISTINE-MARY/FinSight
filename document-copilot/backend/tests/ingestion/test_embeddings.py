import math

import pytest

from app.database.models.document_chunk import EMBEDDING_DIMENSIONS
from app.ingestion.embeddings import fit_stored_dimensions


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
