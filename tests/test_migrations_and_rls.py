import os
import subprocess
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text


TABLES_WITH_RLS = [
    "documents",
    "document_chunks",
    "runs",
    "eval_cases",
    "eval_runs",
    "eval_results",
    "conversations",
    "messages",
]


def _db_url() -> str:
    # Preferimos reusar el mismo env var que usa la app. DATABASE_URL usa el
    # esquema "postgresql://" sin driver explícito (así lo espera asyncpg, que
    # es lo que usa el resto de la app); SQLAlchemy en cambio necesita el
    # sufijo "+psycopg" para no caer por defecto en psycopg2 (no instalado).
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url
    # fallback local
    return "postgresql+psycopg://app:app@localhost:5432/businessassistant"


def _admin_db_url() -> str:
    # DATABASE_URL apunta al rol de mínimo privilegio (app_runtime), que no
    # tiene permisos sobre alembic_version a propósito. Para comprobar el
    # estado de las migraciones hace falta el rol admin.
    url = os.getenv("ALEMBIC_DATABASE_URL")
    if url:
        return url
    return "postgresql+psycopg://app:app@localhost:5432/businessassistant"


def _rls_test_role() -> str:
    # Rol sin privilegios (sin BYPASSRLS) para poder testear RLS incluso si el user principal es superuser.
    return os.getenv("RLS_TEST_ROLE", "rls_test")


def _assume_rls_role_or_skip(conn) -> str:
    role = _rls_test_role()
    role_exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname=:r"), {"r": role}).scalar()
    if role_exists is None:
        pytest.skip(f"RLS_TEST_ROLE '{role}' no existe. Crea el rol o define RLS_TEST_ROLE.")

    # Cambia a un rol sin BYPASSRLS para que RLS sea efectivo.
    # (SET LOCAL para no contaminar conexiones del pool)
    conn.execute(text(f"SET LOCAL ROLE {role}"))
    conn.execute(text("SET LOCAL row_security = on"))
    return role


@pytest.mark.integration
def test_alembic_upgrade_head_smoke():
    # Smoke: debe aplicar sin excepciones y dejar alembic_version.
    subprocess.check_call(["alembic", "upgrade", "head"], env=dict(os.environ))

    engine = create_engine(_admin_db_url())
    with engine.begin() as conn:
        v = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    assert isinstance(v, str) and v


@pytest.mark.integration
def test_rls_enabled_and_policies_exist():
    engine = create_engine(_db_url())
    with engine.begin() as conn:
        for t in TABLES_WITH_RLS:
            row = conn.execute(
                text(
                    """
                    SELECT c.relrowsecurity, c.relforcerowsecurity
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND c.relname = :t
                    """
                ),
                {"t": t},
            ).one()
            assert row[0] is True, f"RLS not enabled for {t}"
            assert row[1] is True, f"RLS not forced for {t}"

            policies = conn.execute(
                text("SELECT COUNT(*) FROM pg_policies WHERE schemaname='public' AND tablename=:t"),
                {"t": t},
            ).scalar_one()
            assert int(policies) >= 1, f"No RLS policies found for {t}"


@pytest.mark.integration
def test_rls_tenant_isolation_runs():
    engine = create_engine(_db_url())

    tenant_a = str(uuid4())
    tenant_b = str(uuid4())
    run_id = str(uuid4())
    user_id = str(uuid4())
    conv_id = str(uuid4())

    with engine.begin() as conn:
        _assume_rls_role_or_skip(conn)

        # A: insert
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_a})
        conn.execute(
            text(
                """
                INSERT INTO runs (
                  tenant_id, run_id, user_id, conversation_id,
                  question, answer, model
                ) VALUES (
                  :tenant_id, CAST(:run_id AS uuid), CAST(:user_id AS uuid), CAST(:conv_id AS uuid),
                  'q', 'a', 'm'
                )
                """
            ),
            {"tenant_id": tenant_a, "run_id": run_id, "user_id": user_id, "conv_id": conv_id},
        )

        # A: read ok
        got_a = conn.execute(text("SELECT COUNT(*) FROM runs WHERE run_id = CAST(:rid AS uuid)"), {"rid": run_id}).scalar_one()
        assert int(got_a) == 1

        # B: read must be 0
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_b})
        got_b = conn.execute(text("SELECT COUNT(*) FROM runs WHERE run_id = CAST(:rid AS uuid)"), {"rid": run_id}).scalar_one()
        assert int(got_b) == 0


@pytest.mark.integration
def test_rls_tenant_isolation_documents_and_chunks():
    # A diferencia de test_rls_enabled_and_policies_exist (que solo comprueba
    # que la política existe), esto intenta romperlo de verdad: inserta un
    # chunk bajo un tenant y confirma que otro tenant no puede leerlo, sobre
    # la tabla que de hecho contiene el contenido del RAG.
    engine = create_engine(_db_url())

    tenant_a = str(uuid4())
    tenant_b = str(uuid4())
    doc_id = str(uuid4())
    chunk_id = f"isolation-test-{uuid4()}"
    embedding = "[" + ",".join(["0.01"] * 1536) + "]"

    with engine.begin() as conn:
        _assume_rls_role_or_skip(conn)

        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_a})

        conn.execute(
            text(
                """
                INSERT INTO documents (tenant_id, doc_id, title)
                VALUES (:tenant_id, CAST(:doc_id AS uuid), 'isolation test doc')
                """
            ),
            {"tenant_id": tenant_a, "doc_id": doc_id},
        )

        conn.execute(
            text(
                f"""
                INSERT INTO document_chunks (
                  tenant_id, chunk_id, doc_id, title, content, embedding,
                  embedding_model, chunker_version
                ) VALUES (
                  :tenant_id, :chunk_id, CAST(:doc_id AS uuid), 't', 'c', CAST(:embedding AS vector),
                  'test-model', 'v1'
                )
                """
            ),
            {"tenant_id": tenant_a, "chunk_id": chunk_id, "doc_id": doc_id, "embedding": embedding},
        )

        got_a = conn.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE chunk_id = :cid"),
            {"cid": chunk_id},
        ).scalar_one()
        assert int(got_a) == 1

        # Switch to tenant B: must not see the chunk nor the document
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_b})

        got_b = conn.execute(
            text("SELECT COUNT(*) FROM document_chunks WHERE chunk_id = :cid"),
            {"cid": chunk_id},
        ).scalar_one()
        doc_b = conn.execute(
            text("SELECT COUNT(*) FROM documents WHERE doc_id = CAST(:doc_id AS uuid)"),
            {"doc_id": doc_id},
        ).scalar_one()

        assert int(got_b) == 0
        assert int(doc_b) == 0


@pytest.mark.integration
def test_rls_tenant_isolation_conversations_and_messages():
    engine = create_engine(_db_url())

    tenant_a = str(uuid4())
    tenant_b = str(uuid4())
    conversation_id = str(uuid4())
    created_by = str(uuid4())

    with engine.begin() as conn:
        _assume_rls_role_or_skip(conn)

        # Insert under tenant A
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_a})

        conn.execute(
            text(
                """
                INSERT INTO conversations (tenant_id, conversation_id, created_by)
                VALUES (:tenant_id, CAST(:cid AS uuid), CAST(:uid AS uuid))
                """
            ),
            {"tenant_id": tenant_a, "cid": conversation_id, "uid": created_by},
        )

        conn.execute(
            text(
                """
                INSERT INTO messages (tenant_id, message_id, conversation_id, role, content)
                VALUES (:tenant_id, gen_random_uuid(), CAST(:cid AS uuid), 'user', 'hello')
                """
            ),
            {"tenant_id": tenant_a, "cid": conversation_id},
        )

        c_a = conn.execute(
            text("SELECT COUNT(*) FROM conversations WHERE conversation_id = CAST(:cid AS uuid)"),
            {"cid": conversation_id},
        ).scalar_one()
        m_a = conn.execute(
            text("SELECT COUNT(*) FROM messages WHERE conversation_id = CAST(:cid AS uuid)"),
            {"cid": conversation_id},
        ).scalar_one()

        assert int(c_a) == 1
        assert int(m_a) == 1

        # Switch to tenant B: must not see the conversation nor the message
        conn.execute(text("SELECT set_config('app.tenant_id', :t, true)"), {"t": tenant_b})

        c_b = conn.execute(
            text("SELECT COUNT(*) FROM conversations WHERE conversation_id = CAST(:cid AS uuid)"),
            {"cid": conversation_id},
        ).scalar_one()
        m_b = conn.execute(
            text("SELECT COUNT(*) FROM messages WHERE conversation_id = CAST(:cid AS uuid)"),
            {"cid": conversation_id},
        ).scalar_one()

        assert int(c_b) == 0
        assert int(m_b) == 0

