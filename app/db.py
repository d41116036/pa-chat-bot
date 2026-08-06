import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from psycopg import Connection, connect
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


def _load_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    app_env = os.getenv("APP_ENV", "dev").lower()
    env_file = project_root / f".env.{app_env}"

    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv(dotenv_path=project_root / ".env")


_load_environment()


class DatabaseConfigurationError(Exception):
    """Raised when database configuration is missing/invalid."""


class DatabaseServiceError(Exception):
    """Raised when a database operation fails."""


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise DatabaseConfigurationError("DATABASE_URL is not set.")
    return database_url


def get_db_connection() -> Connection:
    database_url = get_database_url()
    return connect(database_url, row_factory=dict_row)


def check_database_connection() -> None:
    """Validate DB connectivity during app startup."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
        logger.info("Database connection check passed.")
    except Exception as exc:
        raise DatabaseConfigurationError(
            f"Database connection failed: {exc}"
        ) from exc


def initialize_chat_schema() -> None:
    """Create chat message table if it does not exist."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
        logger.info("Chat schema initialization check passed.")
    except Exception as exc:
        raise DatabaseServiceError(f"Failed to initialize chat schema: {exc}") from exc


def save_chat_message(session_id: str, role: str, content: str) -> None:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES (%s, %s, %s);
                    """,
                    (session_id, role, content),
                )
    except Exception as exc:
        raise DatabaseServiceError(f"Failed to save chat message: {exc}") from exc


def get_chat_history(session_id: str, limit: int = 20) -> list[dict]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at ASC, id ASC
                    LIMIT %s;
                    """,
                    (session_id, limit),
                )
                rows = cur.fetchall()
        return list(rows)
    except Exception as exc:
        raise DatabaseServiceError(f"Failed to fetch chat history: {exc}") from exc
