from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import asyncpg


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


async def main() -> int:
    _load_dotenv()
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL no está configurada (ni en entorno ni en .env)")

    conn = await asyncpg.connect(dsn)
    try:
        # Forzar que RLS se aplique también a roles con posibilidad de bypass
        # (en psql: SET row_security = on)
        await conn.execute("SET row_security = on")

        # 1) extensión pgvector
        has_vector = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        print(f"pgvector extension: {bool(has_vector)}")
        if not has_vector:
            raise SystemExit("La extensión 'vector' no está instalada")

        # 2) listar tablas relevantes
        tables = await conn.fetch(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY tablename
            """
        )
        table_names = [r["tablename"] for r in tables]
        print("tablas:", ", ".join(table_names))

        for required in ("documents", "document_chunks"):
            if required not in table_names:
                raise SystemExit(f"Falta tabla requerida: {required}")

        # 3) RLS: set tenant_id por sesión
        tenant_id = "tenant_test"
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)

        # 4) insert mínimo + select
        doc_id = uuid4()
        chunk_id = "c1"

        await conn.execute(
            """
            INSERT INTO documents (tenant_id, doc_id, title, status, source_type, access_level, tags, metadata)
            VALUES ($1, $2::uuid, 'Doc test', 'ready', 'upload', 'public', ARRAY['a','b']::text[], '{}'::jsonb)
            ON CONFLICT (tenant_id, doc_id) DO NOTHING
            """,
            tenant_id,
            str(doc_id),
        )

        # embedding dummy 1536d (una sola vez)
        # IMPORTANTE: no usar el vector cero (norma 0) porque en cosine distance puede producir NaN.
        # Usamos un patrón determinista y finito.
        emb = "[" + ",".join(["0.01" for _ in range(1536)]) + "]"

        await conn.execute(
            """
            INSERT INTO document_chunks (
              tenant_id, chunk_id, doc_id, title, content, embedding,
              department, doc_type, tags, doc_date,
              metadata, embedding_model, chunker_version
            ) VALUES (
              $1, $2, $3::uuid, 'Doc test', 'Contenido', $4::vector,
              'HR', 'policy', ARRAY['a']::text[], NULL,
              '{}'::jsonb, 'test-emb', 'v1'
            )
            ON CONFLICT (tenant_id, chunk_id) DO UPDATE SET
              content = EXCLUDED.content,
              embedding = EXCLUDED.embedding
            """,
            tenant_id,
            chunk_id,
            str(doc_id),
            emb,
        )

        n_docs = await conn.fetchval("SELECT count(*) FROM documents")
        n_chunks = await conn.fetchval("SELECT count(*) FROM document_chunks")
        print(f"count(documents)={n_docs} count(document_chunks)={n_chunks}")

        # Validar RLS: sin WHERE, debe ver solo el tenant de la sesión
        rows_default = await conn.fetch("SELECT tenant_id, chunk_id FROM document_chunks")
        print(f"select chunks (tenant={tenant_id}, sin WHERE): {len(rows_default)} fila(s)")
        if not rows_default:
            raise SystemExit("RLS/insert falló: no se recuperaron filas para el tenant")
        if any(r["tenant_id"] != tenant_id for r in rows_default):
            raise SystemExit("RLS falló: se devolvieron filas de otro tenant")

        # Validar aislamiento: cambiando tenant, no debe ver las filas anteriores
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", "other_tenant")
        rows_other = await conn.fetch("SELECT tenant_id, chunk_id FROM document_chunks")
        print(f"select chunks (tenant=other_tenant, sin WHERE): {len(rows_other)} fila(s)")
        if rows_other:
            raise SystemExit("RLS parece no estar activo: se devolvieron filas para otro tenant")

        print("OK")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
