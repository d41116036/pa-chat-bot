import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _load_environment() -> None:
    project_root = Path(__file__).resolve().parent.parent
    app_env = os.getenv("APP_ENV", "dev").lower()
    env_file = project_root / f".env.{app_env}"
    # override=True so .env.* wins over any stale shell/process value
    load_dotenv(
        dotenv_path=env_file if env_file.exists() else project_root / ".env",
        override=True,
    )


_load_environment()
logger = logging.getLogger(__name__)

DEFAULT_GEMINI_APP_BASE_URL = "http://54.204.110.222/gemini"


class GoogleAIConfigurationError(Exception):
    """Raised when Gemini app client configuration is invalid."""


class GoogleAIServiceError(Exception):
    """Raised when Gemini app API calls fail."""


def _base_url() -> str:
    return os.getenv("GEMINI_APP_BASE_URL", DEFAULT_GEMINI_APP_BASE_URL).rstrip("/")


def _post_json(path: str, payload: dict) -> dict:
    url = f"{_base_url()}{path}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info("Calling gemini-app: url=%s payload_keys=%s", url, list(payload.keys()))
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        if exc.code >= 500 and "not set" in error_body.lower():
            raise GoogleAIConfigurationError(error_body) from exc
        raise GoogleAIServiceError(
            f"gemini-app request failed ({exc.code}) at {url}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise GoogleAIServiceError(
            f"gemini-app unreachable at {url}: {exc}"
        ) from exc


def summarize_text(text: str, max_words: Optional[int] = 150) -> tuple[str, str]:
    payload = {"text": text, "max_words": max_words or 150}
    logger.info(
        "before calling gemini api summarize end point payload=%s",
        payload,
    )
    body = _post_json("/gemini/summarize", payload)
    logger.info(
        "after receiving response from gemini api summarize end point response=%s",
        body,
    )
    summary = (body.get("summary") or "").strip()
    model = body.get("model") or ""
    if not summary:
        raise GoogleAIServiceError("gemini-app returned an empty summary.")
    return summary, model


def generate_chat_reply(prompt: str) -> tuple[str, str]:
    payload = {"prompt": prompt}
    logger.info(
        "before calling gemini api chat end point payload=%s",
        payload,
    )
    body = _post_json("/gemini/chat", payload)
    logger.info(
        "after receiving response from gemini api chat end point response=%s",
        body,
    )
    reply = (body.get("reply") or "").strip()
    model = body.get("model") or ""
    if not reply:
        raise GoogleAIServiceError("gemini-app returned an empty chat reply.")
    return reply, model
