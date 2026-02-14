from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("apps.api.asgi:app", host=host, port=port, reload=os.getenv("RELOAD", "false").lower() in {"1","true","yes"})


if __name__ == "__main__":
    main()

