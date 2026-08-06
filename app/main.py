import logging

from fastapi import FastAPI

from app.db import (
    DatabaseConfigurationError,
    DatabaseServiceError,
    check_database_connection,
    initialize_chat_schema,
)
from app.routes import router as summarize_router

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="PA Chat Bot", version="1.0.0")
app.include_router(summarize_router)


@app.on_event("startup")
def startup_event() -> None:
    try:
        check_database_connection()
        initialize_chat_schema()
    except DatabaseConfigurationError as exc:
        logger.error("Database startup validation failed: %s", exc)
        raise
    except DatabaseServiceError as exc:
        logger.error("Database schema initialization failed: %s", exc)
        raise


@app.get("/chatbot/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
