"""least-privilege, non-superuser DB roles: app_runtime + rls_test

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-27

The role used so far for DATABASE_URL (`app`, from POSTGRES_USER in
docker-compose) is a Postgres SUPERUSER: it always bypasses row-level
security, regardless of `ENABLE ROW LEVEL SECURITY` / `FORCE ROW LEVEL
SECURITY` on the tables. RLS policies were never actually being enforced
for real app traffic.

docker-compose.yml already declared APP_DB_USER/APP_DB_PASSWORD for exactly
this purpose (see its comment: "el user 'app' no debe ser superuser en
prod/ci") but nothing ever created that role. This migration does, reusing
those same env vars (default role name/password: app_runtime, to avoid
colliding with the existing superuser literally named "app" on databases
that were already bootstrapped with POSTGRES_USER=app). `app` remains the
role migrations run as.

It also creates `rls_test`: tests/test_migrations_and_rls.py already
expects this role (RLS_TEST_ROLE env var, default "rls_test") to run real
cross-tenant isolation checks via `SET LOCAL ROLE`, but nothing in the repo
ever created it — it only existed as undocumented manual state on one local
DB, so those tests silently self-skip (pytest.skip) for anyone else. This
migration makes that role, and therefore those tests, reproducible from a
clean `alembic upgrade head` instead of tribal knowledge.

No infra/sql/*.sql mirror for this one: those files are mounted into
docker-entrypoint-initdb.d and executed verbatim by Postgres on first boot,
which can't do the env-var/password handling this migration does safely.
"""

from __future__ import annotations

import os

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

TABLES = [
    "documents",
    "document_chunks",
    "runs",
    "eval_cases",
    "eval_runs",
    "eval_results",
    "conversations",
    "messages",
]


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _ensure_login_role(*, role: str, password: str) -> None:
    role_ident = _quote_ident(role)
    password_literal = _quote_literal(password)

    conn = op.get_bind()
    exists = conn.exec_driver_sql(
        f"SELECT 1 FROM pg_roles WHERE rolname = {_quote_literal(role)}"
    ).scalar()

    attrs = "LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION".format(
        password_literal
    )
    if exists:
        op.execute(f"ALTER ROLE {role_ident} WITH {attrs}")
    else:
        op.execute(f"CREATE ROLE {role_ident} {attrs}")

    op.execute(
        f"""
        DO $$
        BEGIN
          EXECUTE format('GRANT CONNECT ON DATABASE %I TO {role_ident}', current_database());
        END $$;
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {role_ident}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(TABLES)} TO {role_ident}")


def _ensure_nologin_role(*, role: str) -> None:
    role_ident = _quote_ident(role)

    conn = op.get_bind()
    exists = conn.exec_driver_sql(
        f"SELECT 1 FROM pg_roles WHERE rolname = {_quote_literal(role)}"
    ).scalar()

    attrs = "NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION"
    if exists:
        op.execute(f"ALTER ROLE {role_ident} WITH {attrs}")
    else:
        op.execute(f"CREATE ROLE {role_ident} {attrs}")

    op.execute(f"GRANT USAGE ON SCHEMA public TO {role_ident}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {', '.join(TABLES)} TO {role_ident}")


def upgrade() -> None:
    app_runtime_role = os.environ.get("APP_DB_USER", "app_runtime")
    app_runtime_password = os.environ.get("APP_DB_PASSWORD", "app_runtime")
    rls_test_role = os.environ.get("RLS_TEST_ROLE", "rls_test")

    _ensure_login_role(role=app_runtime_role, password=app_runtime_password)
    _ensure_nologin_role(role=rls_test_role)

    # SET ROLE a un rol NOLOGIN requiere membresía explícita (un superuser
    # como `app` puede hacerlo igualmente, pero lo dejamos explícito).
    op.execute(f"GRANT {_quote_ident(rls_test_role)} TO {_quote_ident(app_runtime_role)}")


def downgrade() -> None:
    # Revoca lo que este migration otorga; deliberadamente NO hace DROP ROLE.
    # Un rol puede acumular grants fuera de esta migración (p.ej. default
    # privileges, u otras tablas) que un DROP ROLE automático no puede
    # limpiar de forma segura sin conocerlos explícitamente. Si hace falta
    # borrar el rol del todo, es una decisión manual, no automática.
    app_runtime_role = os.environ.get("APP_DB_USER", "app_runtime")
    rls_test_role = os.environ.get("RLS_TEST_ROLE", "rls_test")

    op.execute(f"REVOKE {_quote_ident(rls_test_role)} FROM {_quote_ident(app_runtime_role)}")

    for role in (app_runtime_role, rls_test_role):
        role_ident = _quote_ident(role)
        op.execute(f"REVOKE ALL PRIVILEGES ON {', '.join(TABLES)} FROM {role_ident}")
        op.execute(f"REVOKE USAGE ON SCHEMA public FROM {role_ident}")
