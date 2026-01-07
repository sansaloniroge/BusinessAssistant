from __future__ import annotations

from typing import Any, Sequence
import asyncpg

from apps.api.services.ports import VectorStore
from apps.api.services.retrieval_service import RetrievedChunk


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
        filters: dict[str, Any],
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

    def _build_where(self, *, tenant_id: str, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        """
        Interpreta filtros planos + FieldFilter(op=...) y los traduce a SQL.
        Convención:
          - keys simples: metadata->>'key' == value
          - tags contains_any: metadata->'tags' ?| array[...]
          - rangos: key__gte / key__lte
        """
        clauses: list[str] = ["tenant_id = $2"]
        params: list[Any] = [tenant_id]  # NOTE: $2 because $1 is embedding

        # empezamos en $3 (porque $1=embedding, $2=tenant_id)
        arg_index = 3

        def add_clause(sql_fragment: str, value: Any | None = None) -> None:
            nonlocal arg_index
            if value is None:
                clauses.append(sql_fragment)
                return
            clauses.append(f"{sql_fragment} ${arg_index}")
            params.append(value)
            arg_index += 1

        # filters contiene tenant_id también; lo ignoramos aquí porque ya va en columna
        for key, value in filters.items():
            if key == "tenant_id":
                continue

            # FieldFilter (pydantic) llega como dict o como objeto según tu codepath.
            # Soportamos ambos sin complicar:
            op = None
            val = None
            if isinstance(value, dict) and "op" in value and "value" in value:
                op = value["op"]
                val = value["value"]
            elif hasattr(value, "op") and hasattr(value, "value"):
                op = getattr(value, "op")
                val = getattr(value, "value")

            # Convención rango: doc_date__gte / doc_date__lte
            if key.endswith("__gte"):
                field = key.removesuffix("__gte")
                add_clause(f"(metadata->>'{field}')::timestamptz >= ", val)
                continue

            if key.endswith("__lte"):
                field = key.removesuffix("__lte")
                add_clause(f"(metadata->>'{field}')::timestamptz <= ", val)
                continue

            # Operadores soportados
            if op == "$eq":
                add_clause(f"(metadata->>'{key}') = ", str(val))
                continue

            if op == "$in":
                # val debe ser lista
                add_clause(f"(metadata->>'{key}') = ANY(", list(map(str, val)))
                clauses[-1] = clauses[-1] + ")"  # cierra ANY($n)
                continue

            if op == "$contains_any":
                # tags como JSON array -> usamos operador ?| sobre jsonb array de strings
                # requiere que metadata.tags sea array de strings
                add_clause(f"(metadata->'{key}') ?| ", list(map(str, val)))
                continue

            if op == "$gte":
                add_clause(f"(metadata->>'{key}')::timestamptz >= ", val)
                continue

            if op == "$lte":
                add_clause(f"(metadata->>'{key}')::timestamptz <= ", val)
                continue

            # Fallback: igualdad simple para strings/escalares
            add_clause(f"(metadata->>'{key}') = ", str(value))

        where_sql = " AND ".join(clauses)
        return where_sql, params

    @staticmethod
    def _to_pgvector(v: Sequence[float]) -> str:
        # asyncpg no tiene tipo vector nativo; pgvector acepta string tipo '[1,2,3]'
        return "[" + ",".join(f"{x:.8f}" for x in v) + "]"
