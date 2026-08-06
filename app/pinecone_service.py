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
    load_dotenv(
        dotenv_path=env_file if env_file.exists() else project_root / ".env",
        override=True,
    )


_load_environment()
logger = logging.getLogger(__name__)

DEFAULT_PINECONE_APP_BASE_URL = "http://54.204.110.222/pineconeapp"


class PineconeAppConfigurationError(Exception):
    """Raised when pinecone-app client configuration is invalid."""


class PineconeAppServiceError(Exception):
    """Raised when pinecone-app API calls fail."""


def _base_url() -> str:
    return os.getenv(
        "PINECONE_APP_BASE_URL", DEFAULT_PINECONE_APP_BASE_URL
    ).rstrip("/")


def _request_json(method: str, path: str, payload: Optional[dict] = None) -> dict:
    url = f"{_base_url()}{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    logger.info(
        "Calling pinecone-app: method=%s url=%s payload_keys=%s",
        method,
        url,
        list(payload.keys()) if payload else [],
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise PineconeAppServiceError(
            f"pinecone-app request failed ({exc.code}) at {url}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise PineconeAppServiceError(
            f"pinecone-app unreachable at {url}: {exc}"
        ) from exc


def retrieve_namespaces() -> list[str]:
    body = _request_json("GET", "/pinecone/retrievenamespace")
    namespaces = body.get("namespaces") or []
    return [str(name) for name in namespaces]


def retrieve_documents(
    question: str,
    namespace: str,
    *,
    top_k: int = 3,
) -> list[str]:
    body = _request_json(
        "POST",
        "/pinecone/retrievedoc",
        {"question": question, "namespace": namespace, "top_k": top_k},
    )
    texts = body.get("texts") or []
    return [str(text).strip() for text in texts if str(text).strip()]
