from __future__ import annotations

from fastapi import FastAPI

from apps.api.deps import setup_app
from apps.api.routers import health, chat


def create_app() -> FastAPI:
    app = FastAPI(title="BusinessAssistant API", version="0.1.0")

    setup_app(app)

    # Logueo de errores no controlados (DEV): imprime traceback para diagnosticar 500.
    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request, exc: Exception):  # type: ignore[no-redef]
        import traceback
        from fastapi.responses import JSONResponse

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    app.include_router(health.router)
    app.include_router(chat.router)

    return app
