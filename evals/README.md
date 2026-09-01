# DeepEval RAG evaluation

## What this folder does

Runs your live RAG agent (MCP + Gemini) on questions in
`golden_dataset.json`, then scores each answer with DeepEval's five
standard RAG metrics.

## Install

Create a **project-local** venv (do not use the parent `github/venv` unless you install there too):

```bash
cd pa-chat-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-eval.txt
python -c "import mcp; print('mcp ok')"
```

`requirements-eval.txt` includes `-r requirements.txt`, so `mcp` is installed.

## Run

Always use the **same** venv’s pytest:

```bash
cd pa-chat-bot
source .venv/bin/activate
pytest evals/test_rag_deepeval.py -s
```

Or without activating:

```bash
cd pa-chat-bot
.venv/bin/python -m pytest evals/test_rag_deepeval.py -s
```

If you still see `No module named 'mcp'`, you are not using this venv:

```bash
which python
which pytest
python -c "import sys; print(sys.executable)"
```

Those paths must be under `pa-chat-bot/.venv/`.

## Before you run

1. Fill `evals/golden_dataset.json` with real `input` / `expected_output`.
2. Start `arun-mcp-server` (port from `ARUN_MCP_SERVER_URL`).
3. Ensure gemini-app is reachable (`GEMINI_APP_BASE_URL`).
4. Set `GOOGLE_API_KEY` (and optional `GOOGLE_MODEL`) in `.env.local` / `.env.dev`.
   DeepEval uses Gemini as the judge — no `OPENAI_API_KEY` required.

## Metrics scored

- AnswerRelevancyMetric
- FaithfulnessMetric
- ContextualRelevancyMetric
- ContextualPrecisionMetric
- ContextualRecallMetric

## Reading a failed metric reason

When a test fails, the console prints a **metric report** for that golden question.
Each metric looks like:

```text
[FAIL] FaithfulnessMetric score=0.2 threshold=0.5
  reason: The claim 'Arun is a doctor' is not supported by the retrieval context...
```

### What to read

1. **Status** — `PASS` or `FAIL` (fail means score &lt; threshold).
2. **Metric name** — which quality check failed.
3. **score** — 0.0 to 1.0 (higher is better).
4. **threshold** — pass bar (currently 0.5).
5. **reason** — plain-English explanation from the LLM judge. **This is the most important line.**

### What each failure usually means

| Failed metric | Usually means | Where to fix |
|---------------|---------------|--------------|
| AnswerRelevancyMetric | Answer is off-topic or padded | Agent / Gemini final prompt |
| FaithfulnessMetric | Answer invents facts not in chunks | Generator prompt; or retrieval too weak |
| ContextualRelevancyMetric | Retrieved chunks are noisy / off-topic | Namespace, query, Pinecone index |
| ContextualPrecisionMetric | Useful chunks ranked too low | Ranking / `top_k` / retriever |
| ContextualRecallMetric | Missing info needed for the gold answer | Retriever coverage; gold may be too strict |

### Example walkthrough

Suppose you see:

```text
[PASS] AnswerRelevancyMetric score=0.9 threshold=0.5
  reason: The answer directly addresses the job title question.
[FAIL] FaithfulnessMetric score=0.1 threshold=0.5
  reason: The output says Arun works at Google, but retrieval_context only mentions Example Corp.
[PASS] ContextualRelevancyMetric score=0.8 threshold=0.5
  reason: Chunks discuss employment history.
```

How to think:

1. Relevancy passed → the answer is on-topic.
2. Contextual relevancy passed → retrieval found related text.
3. Faithfulness failed → the **generator** added “Google”, which was **not** in the chunks.

So you do **not** first blame Pinecone. You check why Gemini invented a company name (prompt, tool results empty, or model ignoring context).

If instead **ContextualRecall** fails and Faithfulness passes, retrieval probably missed facts that your `expected_output` requires — improve search / namespace / `top_k`, or soften an overly specific gold answer.

### Tip

Always run with `-s` so pytest does not hide the metric report:

```bash
pytest evals/test_rag_deepeval.py -s
```
