from __future__ import annotations

import argparse
import asyncio
import os
import re
from datetime import date
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import asyncpg

from apps.api.adapters.openai_embeddings import OpenAIEmbeddingService
from apps.api.adapters.pgvector_vector_store import PgvectorVectorStore

CHUNKER_VERSION = "naive-paragraph-v1"
DEFAULT_CHUNK_CHARS = 800
DEFAULT_CHUNK_OVERLAP_CHARS = 100


def _load_dotenv(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Minimal `--- key: value ---` frontmatter parser (scalars and `[a, b]` lists).

    Avoids adding a pyyaml dependency for a handful of sample docs.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text

    header = text[3:end].strip("\n")
    body = text[end + 4 :].lstrip("\n")

    meta: dict[str, Any] = {}
    for line in header.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key] = value.strip('"').strip("'")
    return meta, body


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Greedy paragraph-based chunker: accumulate paragraphs up to chunk_chars,
    carrying the last overlap_chars of the previous chunk into the next one."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if not current or len(candidate) <= chunk_chars:
            current = candidate
            continue
        chunks.append(current)
        overlap = current[-overlap_chars:] if overlap_chars else ""
        current = f"{overlap}\n\n{para}" if overlap else para
    if current:
        chunks.append(current)
    return chunks


def _parse_doc_date(value: Any) -> date | None:
    if not value:
        return None
    return date.fromisoformat(str(value))


def _doc_id_for(path: Path, tenant_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"businessassistant:{tenant_id}:{path.name}"))


def _load_documents(folder: Path) -> list[tuple[Path, dict[str, Any], str]]:
    return [
        (path, *_parse_frontmatter(path.read_text(encoding="utf-8")))
        for path in sorted(folder.glob("*.md"))
    ]


async def main() -> int:
    _load_dotenv()

    ap = argparse.ArgumentParser(
        description="Ingesta mínima: lee documentos .md, los trocea, los embebe (OpenAI) y los sube a pgvector"
    )
    ap.add_argument("--docs-dir", default=os.getenv("INGEST_DOCS_DIR", "scripts/sample_docs"))
    ap.add_argument("--tenant-id", default=os.getenv("INGEST_TENANT_ID", "tenant_test"))
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    ap.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    ap.add_argument("--chunk-chars", type=int, default=int(os.getenv("INGEST_CHUNK_CHARS", str(DEFAULT_CHUNK_CHARS))))
    ap.add_argument(
        "--chunk-overlap-chars",
        type=int,
        default=int(os.getenv("INGEST_CHUNK_OVERLAP_CHARS", str(DEFAULT_CHUNK_OVERLAP_CHARS))),
    )
    args = ap.parse_args()

    if not args.database_url:
        raise SystemExit("Missing --database-url (or env DATABASE_URL)")

    docs_dir = Path(args.docs_dir)
    if not docs_dir.is_dir():
        raise SystemExit(f"No existe el directorio de documentos: {docs_dir}")

    documents = _load_documents(docs_dir)
    if not documents:
        raise SystemExit(f"No se encontraron documentos .md en {docs_dir}")

    embeddings = OpenAIEmbeddingService(model=args.embedding_model)

    all_chunks: list[dict[str, Any]] = []
    for path, meta, body in documents:
        doc_id = _doc_id_for(path, args.tenant_id)
        title = meta.get("title") or path.stem
        pieces = _chunk_text(body, chunk_chars=args.chunk_chars, overlap_chars=args.chunk_overlap_chars)
        vectors = await embeddings.embed_chunks(texts=pieces)

        for idx, (piece, vector) in enumerate(zip(pieces, vectors)):
            all_chunks.append(
                {
                    "chunk_id": f"{path.stem}::{idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "content": piece,
                    "embedding": vector,
                    "embedding_model": args.embedding_model,
                    "chunker_version": CHUNKER_VERSION,
                    "department": meta.get("department"),
                    "doc_type": meta.get("doc_type"),
                    "tags": meta.get("tags"),
                    "doc_date": _parse_doc_date(meta.get("doc_date")),
                }
            )
        print(f"{path.name}: {len(pieces)} chunk(s)")

    pool = await asyncpg.create_pool(dsn=args.database_url, min_size=1, max_size=3)
    try:
        store = PgvectorVectorStore(pool)
        n_upserted = await store.upsert_chunks(tenant_id=args.tenant_id, chunks=all_chunks)
    finally:
        await pool.close()

    print(f"Ingeridos {len(documents)} documento(s), {n_upserted} chunk(s) subido(s) (tenant={args.tenant_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
