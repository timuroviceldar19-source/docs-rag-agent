import uuid
from pathlib import Path

import pytest

from docs_rag_agent.ingest.pipeline import _build_documents
from docs_rag_agent.retrieve import Document


@pytest.fixture
def tmp_docs(tmp_path: Path) -> Path:
    """Create a small tree of markdown fixtures."""
    (tmp_path / "tutorial").mkdir()
    (tmp_path / "tutorial" / "index.md").write_text(
        "## Getting Started\n\nInstall FastAPI with pip.\n\n## Requirements\n\nPython 3.8+.\n",
        encoding="utf-8"
    )
    (tmp_path / "advanced").mkdir()
    (tmp_path / "advanced" / "middleware.md").write_text(
        "## Middleware\n\nAdd middleware to your FastAPI app.\n",
        encoding="utf-8"
    )
    return tmp_path


def test_build_documents_returns_list(tmp_docs: Path) -> None:
    files = list(tmp_docs.rglob("*.md"))
    docs = _build_documents(files, base_dir=tmp_docs, max_chars=1500, overlap_chars=150)
    assert isinstance(docs, list)
    assert len(docs) > 0


def test_all_items_are_documents(tmp_docs: Path) -> None:
    files = list(tmp_docs.rglob("*.md"))
    docs = _build_documents(files, base_dir=tmp_docs, max_chars=1500, overlap_chars=150)
    for doc in docs:
        assert isinstance(doc, Document)


def test_document_ids_are_valid_uuids(tmp_docs: Path) -> None:
    files = list(tmp_docs.rglob("*.md"))
    docs = _build_documents(files, base_dir=tmp_docs, max_chars=1500, overlap_chars=150)
    for doc in docs:
        uuid.UUID(doc.id)  # raises ValueError if not a valid UUID


def test_document_ids_are_stable(tmp_docs: Path) -> None:
    """Same files must produce the same IDs on repeated calls."""
    files = sorted(tmp_docs.rglob("*.md"))
    docs1 = _build_documents(files, base_dir=tmp_docs, max_chars=1500, overlap_chars=150)
    docs2 = _build_documents(files, base_dir=tmp_docs, max_chars=1500, overlap_chars=150)
    ids1 = [d.id for d in docs1]
    ids2 = [d.id for d in docs2]
    assert ids1 == ids2


def test_metadata_has_source_and_heading(tmp_docs: Path) -> None:
    files = list(tmp_docs.rglob("*.md"))
    docs = _build_documents(files, base_dir=tmp_docs, max_chars=1500, overlap_chars=150)
    for doc in docs:
        assert "source" in doc.metadata
        assert "heading" in doc.metadata
