"""DeepEval end-to-end RAG evaluation for pa-chat-bot."""

import json
import os
from pathlib import Path
from typing import List

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from evals.rag_pipeline import run_rag_for_eval

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_eval_env() -> None:
    """Load .env.{APP_ENV} so GOOGLE_API_KEY is available for the DeepEval judge."""
    app_env = os.getenv("APP_ENV", "dev").lower()
    env_file = _PROJECT_ROOT / ".env.{}".format(app_env)
    load_dotenv(
        dotenv_path=env_file if env_file.exists() else _PROJECT_ROOT / ".env",
        override=True,
    )


_load_eval_env()


def _load_goldens() -> list:
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("golden_dataset.json must be a JSON list")
    return data


def _build_judge_model() -> GeminiModel:
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Export APP_ENV=local (or dev) so "
            ".env.local / .env.dev loads, or set GOOGLE_API_KEY in the shell."
        )
    model_name = (os.getenv("GOOGLE_MODEL") or "gemini-2.5-flash").strip()
    return GeminiModel(model=model_name, api_key=api_key)


def _build_metrics():
    """Create the five industry-standard RAG metrics using Gemini as judge."""
    threshold = 0.5
    judge = _build_judge_model()
    return [
        AnswerRelevancyMetric(threshold=threshold, model=judge),
        FaithfulnessMetric(threshold=threshold, model=judge),
        ContextualRelevancyMetric(threshold=threshold, model=judge),
        ContextualPrecisionMetric(threshold=threshold, model=judge),
        ContextualRecallMetric(threshold=threshold, model=judge),
    ]


def _evaluate_and_report(test_case: LLMTestCase, metrics: list) -> None:
    """Measure each metric, print score+reason, fail with a readable summary."""
    failed_lines: List[str] = []
    print("\n=== DeepEval metric report ===")
    print("input: {!r}".format(test_case.input))
    print("actual_output: {!r}".format((test_case.actual_output or "")[:300]))
    print("retrieval_context chunks: {}".format(len(test_case.retrieval_context or [])))
    if test_case.retrieval_context:
        for i, chunk in enumerate(test_case.retrieval_context[:3]):
            print("  chunk[{}]: {!r}".format(i, str(chunk)[:160]))
    else:
        print("  (empty — retrieve_documents missing or result shape not parsed)")
    for metric in metrics:
        metric.measure(test_case)
        name = metric.__class__.__name__
        score = getattr(metric, "score", None)
        reason = getattr(metric, "reason", "") or "(no reason returned)"
        threshold = getattr(metric, "threshold", None)
        passed = bool(metric.is_successful())
        status = "PASS" if passed else "FAIL"
        line = (
            "[{}] {} score={} threshold={}\n  reason: {}".format(
                status, name, score, threshold, reason
            )
        )
        print(line)
        if not passed:
            failed_lines.append(line)
    print("=== end metric report ===\n")
    if failed_lines:
        raise AssertionError(
            "DeepEval metric(s) failed for input={!r}:\n{}".format(
                test_case.input, "\n".join(failed_lines)
            )
        )


@pytest.mark.parametrize("golden", _load_goldens())
def test_rag_end_to_end(golden):
    """Run one golden question through RAG and score all five metrics."""
    question = str(golden["input"]).strip()
    expected = str(golden["expected_output"]).strip()
    actual_output, retrieval_context = run_rag_for_eval(question)
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected,
    )
    if not question or not expected:
        raise ValueError("golden input and expected_output are required")
    if not isinstance(retrieval_context, list):
        raise TypeError("retrieval_context must be a list of chunk strings")
    assert actual_output.strip(), "RAG pipeline returned an empty answer"
    metrics = _build_metrics()
    _evaluate_and_report(test_case, metrics)
