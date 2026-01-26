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
        for c in chunks:
            md = dict(c.get("metadata") or {})

            # Prefer explicit keys; fallback to metadata for denormalized columns
            department = c.get("department") or md.get("department")
            doc_type = c.get("doc_type") or md.get("doc_type")
            tags = c.get("tags") or md.get("tags") or []
            doc_date = c.get("doc_date") or md.get("doc_date")

            if tags is None:
                tags = []

            records.append(
                (
                    tenant_id,
                    str(c["chunk_id"]),
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

    async def health(self) -> bool:
        async with self._pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
        return val == 1

    def _build_where(self, *, tenant_id: str, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        """Traduce MetaFilters a SQL.

        Estrategia:
        - Filtros comunes van a columnas (department/doc_type/tags/doc_date) por performance.
        - Resto cae a metadata JSONB.

        Parámetros:
        - $1 se reserva para el embedding.
        - A partir de $2 van el resto.
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
        # asyncpg no tiene tipo vector nativo; pgvector acepta string tipo '[1,2,3]'
        return "[" + ",".join(f"{x:.8f}" for x in v) + "]"
