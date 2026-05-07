import hashlib
import uuid as uuid_module
from pathlib import Path

import typer
from qdrant_client import QdrantClient

from docs_rag_agent.config import Settings
from docs_rag_agent.embeddings import FastEmbedLocalEmbedder, FastEmbedSparseEmbedder
from docs_rag_agent.embeddings.sparse import SparseEmbedder
from docs_rag_agent.ingest.chunk import chunk_markdown
from docs_rag_agent.ingest.fetch import clone_or_pull, collect_markdown_files
from docs_rag_agent.retrieve import Document, VectorStore

app = typer.Typer()


def _build_documents(
    markdown_files: list[Path],
    base_dir: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[Document]:
    """Convert markdown files to Documents. Pure function, no I/O side effects."""
    all_docs: list[Document] = []
    for path in markdown_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
            
        relative_path = str(path.relative_to(base_dir)).replace("\\", "/")
        chunks = chunk_markdown(
            text,
            source=relative_path,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        
        for chunk in chunks:
            uid_str = f"{chunk.source}:{chunk.index}"
            uid = str(
                uuid_module.UUID(
                    hashlib.md5(uid_str.encode()).hexdigest()
                )
            )
            doc = Document(
                id=uid,
                text=chunk.text,
                metadata={"source": chunk.source, "heading": chunk.heading},
            )
            all_docs.append(doc)
            
    return all_docs


@app.command()
def ingest(
    repo_url: str = typer.Option(
        "https://github.com/fastapi/fastapi.git",
        help="Git URL of the FastAPI repository.",
    ),
    cache_dir: Path = typer.Option(
        Path(".cache/fastapi-repo"),
        help="Local directory to clone/pull the repository into.",
    ),
    docs_subdir: str = typer.Option(
        "docs/en/docs",
        help="Subdirectory within the repo containing markdown docs.",
    ),
    max_chars: int = typer.Option(1500, help="Max characters per chunk."),
    overlap_chars: int = typer.Option(150, help="Overlap between hard-split chunks."),
    batch_size: int = typer.Option(64, help="Documents per upsert batch."),
) -> None:
    """Ingest FastAPI documentation into Qdrant."""
    settings = Settings()
    
    clone_or_pull(repo_url, cache_dir)
    docs_root = cache_dir / docs_subdir
    markdown_files = collect_markdown_files(docs_root)
    
    embedder = FastEmbedLocalEmbedder(settings.embedding_model)
    sparse_embedder: SparseEmbedder | None = (
        FastEmbedSparseEmbedder(model_name=settings.sparse_model)
        if settings.hybrid_enabled
        else None
    )
    client = QdrantClient(url=settings.qdrant_url)
    store = VectorStore(
        client=client,
        collection=settings.qdrant_collection,
        embedder=embedder,
        sparse_embedder=sparse_embedder,
        hybrid_fetch_k=settings.hybrid_fetch_k,
    )
    store.ensure_collection()
    
    documents = _build_documents(
        markdown_files,
        base_dir=cache_dir,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    
    total_chunks = len(documents)
    for i in range(0, total_chunks, batch_size):
        batch = documents[i : i + batch_size]
        store.upsert(batch)
        
    typer.echo(f"Ingested {total_chunks} chunks from {len(markdown_files)} files.", err=True)


if __name__ == "__main__":
    app()
