from __future__ import annotations

from typing import Any, Sequence
import asyncpg

from apps.api.services.ports import VectorStore
from apps.api.services.retrieval_service import RetrievedChunk
from packages.shared.schemas.filters import MetaFilters


class PgvectorVectorStore(VectorStore):
    """
    Adapter Postgres + pgvector.
    - Multi-tenant enforced via tenant_id column and WHERE clause
    - Metadata filters on JSONB
    - Vector similarity via cosine distance
    """

    EMBEDDING_DIM: int = 1536

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def search_by_embedding(
        self,
        *,
        tenant_id: str,
        query_embedding: Sequence[float],
        top_k: int,
        filters: MetaFilters,
    ) -> list[RetrievedChunk]:
        # Fail-fast: query embedding dimension
        if len(query_embedding) != self.EMBEDDING_DIM:
            raise ValueError(f"query_embedding debe tener dimensión {self.EMBEDDING_DIM}, got={len(query_embedding)}")

        where_sql, params = self._build_where(tenant_id=tenant_id, filters=filters)

        # cosine distance: lower is better, so score = 1 - distance (rough signal)
        sql = f"""
        SELECT
          chunk_id,
          doc_id,
          title,
          content,
          metadata,
          1 - (embedding <=> $1::vector) AS score
        FROM document_chunks
        WHERE {where_sql}
        ORDER BY embedding <=> $1::vector
        LIMIT {int(top_k)}
        """

        async with self._pool.acquire() as conn:
            # RLS convention: app.tenant_id must be set per connection
            # (still keep explicit filter in SQL as defense-in-depth)
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            rows = await conn.fetch(sql, self._to_pgvector(query_embedding), *params)

        out: list[RetrievedChunk] = []
        for r in rows:
            out.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    title=r["title"],
                    content=r["content"],
                    score=float(r["score"]),
                    metadata=dict(r["metadata"]) if r["metadata"] is not None else {},
                )
            )
        return out

    async def upsert_chunks(self, *, tenant_id: str, chunks: Sequence[dict[str, Any]]) -> int:
        if not chunks:
            return 0

        # Fail-fast: require embedding_model and chunker_version
        for c in chunks:
            em = str(c.get("embedding_model", "") or "").strip()
            cv = str(c.get("chunker_version", "") or "").strip()
            if not em or not cv:
                raise ValueError("upsert_chunks requiere 'embedding_model' y 'chunker_version' no vacíos")
            # Fail-fast: dimensión
            emb = c.get("embedding")
            if not isinstance(emb, (list, tuple)) or len(emb) != self.EMBEDDING_DIM:
                raise ValueError(
                    f"embedding debe ser lista/tupla de longitud {self.EMBEDDING_DIM}, got={0 if emb is None else len(emb)}"
                )

        # First, prepare minimal insertion in documents to comply with FK
        docs_sql = """
        INSERT INTO documents (
          tenant_id, doc_id, title, status, source_type, access_level,
          department, doc_type, tags, doc_date, metadata
        ) VALUES (
          $1, $2::uuid, $3, 'ready', 'upload', 'public',
          $4, $5, $6::text[], $7::timestamptz, $8::jsonb
        )
        ON CONFLICT (tenant_id, doc_id) DO NOTHING
        """

        docs_records: list[tuple[Any, ...]] = []
        seen_docs: set[tuple[str, str]] = set()
        for c in chunks:
            md = dict(c.get("metadata") or {})
            department = c.get("department") or md.get("department")
            doc_type = c.get("doc_type") or md.get("doc_type")
            tags = c.get("tags") or md.get("tags") or []
            doc_date = c.get("doc_date") or md.get("doc_date")

            key = (tenant_id, str(c["doc_id"]))
            if key in seen_docs:
                continue
            seen_docs.add(key)

            docs_records.append(
                (
                    tenant_id,
                    str(c["doc_id"]),
                    str(c.get("title", "")) or "Untitled",
                    department,
                    doc_type,
                    list(map(str, tags)) if isinstance(tags, (list, tuple)) else [str(tags)],
                    doc_date,
                    md,
                )
            )

        # Now, prepare the chunk records
        sql = """
        INSERT INTO document_chunks (
          tenant_id, chunk_id, doc_id, title, content, embedding,
          department, doc_type, tags, doc_date,
          metadata, embedding_model, chunker_version
        )
        VALUES (
          $1, $2, $3::uuid, $4, $5, $6::vector,
          $7, $8, $9::text[], $10::timestamptz,
          $11::jsonb, $12, $13
        )
        ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
          doc_id = EXCLUDED.doc_id,
          title = EXCLUDED.title,
          content = EXCLUDED.content,
          embedding = EXCLUDED.embedding,
          department = EXCLUDED.department,
          doc_type = EXCLUDED.doc_type,
          tags = EXCLUDED.tags,
          doc_date = EXCLUDED.doc_date,
          metadata = EXCLUDED.metadata,
          embedding_model = EXCLUDED.embedding_model,
          chunker_version = EXCLUDED.chunker_version
        """

        records: list[tuple[Any, ...]] = []
        chunk_ids: list[str] = []
        for c in chunks:
            md = dict(c.get("metadata") or {})
            department = c.get("department") or md.get("department")
            doc_type = c.get("doc_type") or md.get("doc_type")
            tags = c.get("tags") or md.get("tags") or []
            doc_date = c.get("doc_date") or md.get("doc_date")
            if tags is None:
                tags = []

            chunk_id = str(c["chunk_id"])
            chunk_ids.append(chunk_id)

            records.append(
                (
                    tenant_id,
                    chunk_id,
                    str(c["doc_id"]),
                    str(c.get("title", "")),
                    str(c.get("content", "")),
                    self._to_pgvector(c["embedding"]),
                    department,
                    doc_type,
                    list(map(str, tags)) if isinstance(tags, (list, tuple)) else [str(tags)],
                    doc_date,
                    md,
                    str(c.get("embedding_model", "")),
                    str(c.get("chunker_version", "")),
                )
            )

        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)

            # Fail-fast: if the chunk already exists, do not allow changing embedding_model/chunker_version
            existing = await conn.fetch(
                """
                SELECT chunk_id, embedding_model, chunker_version
                FROM document_chunks
                WHERE tenant_id = $1 AND chunk_id = ANY($2::text[])
                """,
                tenant_id,
                chunk_ids,
            )
            if existing:
                by_id = {r["chunk_id"]: (r["embedding_model"], r["chunker_version"]) for r in existing}
                for c in chunks:
                    cid = str(c["chunk_id"])
                    cur = by_id.get(cid)
                    if not cur:
                        continue
                    em_new = str(c.get("embedding_model", "") or "").strip()
                    cv_new = str(c.get("chunker_version", "") or "").strip()
                    em_old, cv_old = cur
                    if em_old != em_new or cv_old != cv_new:
                        raise RuntimeError(
                            "VectorStore fail-fast: chunk existente con embedding_model/chunker_version distintos. "
                            f"tenant_id={tenant_id} chunk_id={cid} "
                            f"db=({em_old},{cv_old}) incoming=({em_new},{cv_new}). "
                            "Reindexa explícitamente antes de cambiar modelo/version."
                        )

            if docs_records:
                await conn.executemany(docs_sql, docs_records)
            await conn.executemany(sql, records)

        return len(records)

    async def delete_by_doc_id(self, *, tenant_id: str, doc_id: str) -> int:
        sql = "DELETE FROM document_chunks WHERE tenant_id = $1 AND doc_id = $2::uuid"
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            res = await conn.execute(sql, tenant_id, doc_id)
        # res: "DELETE <n>"
        try:
            return int(res.split()[-1])
        except Exception:
            return 0

    async def delete_by_chunk_id(self, *, tenant_id: str, chunk_id: str) -> int:
        sql = "DELETE FROM document_chunks WHERE tenant_id = $1 AND chunk_id = $2"
        async with self._pool.acquire() as conn:
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            res = await conn.execute(sql, tenant_id, chunk_id)
        try:
            return int(res.split()[-1])
        except Exception:
            return 0

    async def health(self) -> bool:
        async with self._pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        return val == 1

    def _build_where(self, *, tenant_id: str, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        """Translate MetaFilters to SQL.

        Strategy:
        - Common filters go to columns (department/doc_type/tags/doc_date) for performance.
        - The rest goes to JSONB metadata.

        Parameters:
        - $1 is reserved for embedding.
        - From $2 onwards go the rest.
        """
        clauses: list[str] = ["tenant_id = $2"]
        params: list[Any] = [tenant_id]  # $2

        arg_index = 3  # $1=embedding, $2=tenant_id

        def add_param(value: Any) -> str:
            nonlocal arg_index
            placeholder = f"${arg_index}"
            params.append(value)
            arg_index += 1
            return placeholder

        for key, value in filters.items():
            if key == "tenant_id":
                continue

            op = None
            val = None
            if isinstance(value, dict) and "op" in value and "value" in value:
                op = value["op"]
                val = value["value"]
            elif hasattr(value, "op") and hasattr(value, "value"):
                op = getattr(value, "op")
                val = getattr(value, "value")
            else:
                val = value

            # Date range convention
            if key.endswith("__gte"):
                field = key.removesuffix("__gte")
                if field == "doc_date":
                    ph = add_param(val)
                    clauses.append(f"doc_date >= {ph}::timestamptz")
                else:
                    ph = add_param(val)
                    clauses.append(f"(metadata->>'{field}')::timestamptz >= {ph}::timestamptz")
                continue

            if key.endswith("__lte"):
                field = key.removesuffix("__lte")
                if field == "doc_date":
                    ph = add_param(val)
                    clauses.append(f"doc_date <= {ph}::timestamptz")
                else:
                    ph = add_param(val)
                    clauses.append(f"(metadata->>'{field}')::timestamptz <= {ph}::timestamptz")
                continue

            # Column fast-path: department/doc_type
            if key in {"department", "doc_type"}:
                if op is None or op == "$eq":
                    ph = add_param(str(val))
                    clauses.append(f"{key} = {ph}")
                    continue

            # Column fast-path: tags contains_any
            if key == "tags" and op == "$contains_any":
                # array overlap: tags && ARRAY[...]
                ph = add_param(list(map(str, val or [])))
                clauses.append(f"tags && {ph}::text[]")
                continue

            # JSONB fallback operators
            if op == "$eq":
                ph = add_param(str(val))
                clauses.append(f"(metadata->>'{key}') = {ph}")
                continue

            if op == "$in":
                ph = add_param(list(map(str, val or [])))
                clauses.append(f"(metadata->>'{key}') = ANY({ph}::text[])")
                continue

            if op == "$contains_any":
                # JSONB array contains any of strings
                # (metadata->'tags') ?| ARRAY['a','b']
                ph = add_param(list(map(str, val or [])))
                clauses.append(f"(metadata->'{key}') ?| {ph}::text[]")
                continue

            if op == "$gte":
                ph = add_param(val)
                clauses.append(f"(metadata->>'{key}')::timestamptz >= {ph}::timestamptz")
                continue

            if op == "$lte":
                ph = add_param(val)
                clauses.append(f"(metadata->>'{key}')::timestamptz <= {ph}::timestamptz")
                continue

            # Fallback: equality
            ph = add_param(str(val))
            clauses.append(f"(metadata->>'{key}') = {ph}")

        where_sql = " AND ".join(clauses)
        return where_sql, params

    @staticmethod
    def _to_pgvector(v: Sequence[float]) -> str:
        # asyncpg does not have a native vector type; pgvector accepts string type '[1,2,3]'
        return "[" + ",".join(f"{x:.8f}" for x in v) + "]"
