# idp_test

Property & Regulatory Document Intelligence Platform — ingests, parses, indexes, and semantically searches regulatory PDF documents (MahaRERA, with MHADA/CIDCO planned). See [docs/PRD.md](docs/PRD.md) for the full product spec and phased roadmap.

## Status

**Phase 1: Document Intelligence Foundation — complete.**
**Phase 2: Search Foundation — complete.**
**Phase 3: Enterprise Retrieval Layer — complete.**
**Phase 4: AI Copilot — complete.**
**Phase 5: Agentic Layer — complete.**
**Phase 6: Evaluation — complete.**

Ingestion, parsing (Docling primary / MarkItDown fallback), metadata schema, document registry, PostgreSQL storage, and the batch processing pipeline are implemented and tested. Search (PostgreSQL full-text, BM25, and metadata filtering/pagination) is implemented and tested. Phase 3 adds chunking, BGE-M3 embeddings, Qdrant vector storage, semantic search, hybrid BM25+vector search with Reciprocal Rank Fusion, and BGE cross-encoder reranking. Phase 4 adds RAG (Retrieval-Augmented Generation) on top: an OpenAI-powered Q&A copilot with inline citations, document summarisation, multi-document comparison, SSE streaming, and a FastAPI REST API. Phase 5 adds five LangGraph-based specialist agents (Document Analyst, Comparison, Compliance, Research, Report) with a keyword-based router. Phase 6 adds RAGAS + DeepEval metrics, LangSmith + OpenAI tracing, and an EvaluationPipeline orchestrator. Phase 7 (enterprise features) not started.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker Desktop (for PostgreSQL and Qdrant).

```bash
uv sync
cp .env.example .env
docker compose up -d postgres qdrant
```

## Running the batch pipeline

Drop PDFs into `data/raw/<source>/<category>/...pdf` (categories: `acts`, `circulars`, `orders`, `regulations`, `reports`, `rules`), then:

```bash
uv run python -m property_intel.pipeline
```

This scans `data/raw`, parses each document (Docling, falling back to MarkItDown on failure), writes generated markdown to `data/processed/<category>/`, and records each document's lifecycle state (`uploaded` → `processing` → `completed`/`failed`) in PostgreSQL, including its plain-text content for search indexing. Re-running skips documents already completed (detected by content hash).

## Indexing for retrieval (Phase 3)

After the batch pipeline has completed documents in PostgreSQL, chunk and embed them into Qdrant:

```bash
uv run python -m property_intel.retrieval index
uv run python -m property_intel.retrieval index --reindex   # wipe and rebuild
```

This splits each document's text content into 512-token overlapping chunks (using a markdown-aware chunker), generates BGE-M3 embeddings for each chunk, and stores the vectors in Qdrant's `document_chunks` collection alongside chunk metadata (document_id, section_title, content).

## Semantic and hybrid search (Phase 3)

```bash
uv run python -m property_intel.retrieval search "cancellation rights of homebuyers"
uv run python -m property_intel.retrieval search "registration deadline" --mode semantic --no-rerank
uv run python -m property_intel.retrieval search "promoter obligations" --limit 5
```

Three retrieval modes, all returning ranked `ScoredChunk` results (`src/property_intel/retrieval/models.py`):

* **`hybrid`** (default) — Combines chunk-level BM25 keyword ranking with Qdrant ANN vector search, fused with Reciprocal Rank Fusion (RRF). Results appearing in both lists are boosted; neither raw score scale is assumed comparable.
* **`semantic`** — Pure vector search via Qdrant ANN on BGE-M3 embeddings. Best for synonym and paraphrase matching.

Both modes apply a BGE cross-encoder reranker by default (`--no-rerank` to skip). The cross-encoder re-scores the top retrieved candidates by processing each (query, chunk) pair together with full attention — higher accuracy at the cost of ~30 forward passes.

`RetrievalService` (`src/property_intel/retrieval/service.py`) is the single entry point wiring all components.

## AI Copilot — question answering, summarisation, comparison (Phase 4)

Set `OPENAI_API_KEY` in your `.env`, then:

```bash
# Ask a question (streams by default)
uv run python -m property_intel.copilot ask "What are the registration rules for builders?"

# Non-streaming (wait for full answer)
uv run python -m property_intel.copilot ask --no-stream "What is the penalty for delayed possession?"

# Summarise documents relevant to a topic
uv run python -m property_intel.copilot summarize "homebuyer rights"

# Compare two sets of regulations
uv run python -m property_intel.copilot compare "2020 registration rules" "2023 registration rules"
```

Each answer includes inline citations (`[1]`, `[2]`…) referencing the exact document chunks it drew from.

### REST API (Phase 4)

```bash
uvicorn property_intel.api.app:app --reload
# then open http://localhost:8000/docs for interactive API docs
```

Endpoints:
* `POST /api/v1/ask` — answer a question (JSON response with answer + citations)
* `POST /api/v1/summarize` — summarise by topic
* `POST /api/v1/compare` — compare two topics
* `POST /api/v1/ask/stream` — SSE streaming answer
* `POST /api/v1/summarize/stream` — SSE streaming summary
* `POST /api/v1/compare/stream` — SSE streaming comparison
* `GET /health` — liveness check

`CopilotService` (`src/property_intel/copilot/service.py`) is the single entry point that wires Phase 3 retrieval with OpenAI generation.

## Agentic Workflows (Phase 5)

Phase 5 moves beyond the single-pass copilot into **multi-step agentic workflows** built with [LangGraph](https://github.com/langchain-ai/langgraph). Each agent runs a ReAct loop — the LLM reasons about what to do, calls a retrieval tool, observes the result, and continues until it has enough information to produce a final answer.

### Agents

| Agent | Class | Capability |
|---|---|---|
| Document Analyst | `DocumentAnalystAgent` | Summarise a topic, extract structured facts (rules, deadlines, amounts) |
| Comparison | `ComparisonAgent` | Compare two regulatory subjects, detect material changes |
| Compliance | `ComplianceAgent` | Validate a situation: COMPLIANT / NON-COMPLIANT / NEEDS REVIEW |
| Research | `ResearchAgent` | Multi-document research brief across the corpus |
| Report | `ReportAgent` | Generate a professional 6-section regulatory report |

### Tools available to agents

* `retrieve_documents(query, limit)` — hybrid vector + BM25 search over indexed chunks
* `summarize_topic(topic)` — retrieves and condenses context for a broad topic
* `extract_entities(query)` — retrieves passages for fact extraction (rules, amounts, dates)

### Router

`AgentRouter` classifies the user's task by keyword matching and dispatches to the correct agent:

```python
from property_intel.agents import AgentRouter
from property_intel.agents.document_analyst import DocumentAnalystAgent
from property_intel.agents.comparison import ComparisonAgent
# ... wire up retrieval and llm ...

router = AgentRouter({
    "document_analyst": DocumentAnalystAgent.from_retrieval(retrieval, llm),
    "comparison": ComparisonAgent.from_retrieval(retrieval, llm),
    # ... other agents
})

result = router.route("Compare the 2016 and 2019 MahaRERA regulations.")
```

All agents accept a `from_retrieval(retrieval, llm)` factory that wires the existing `RetrievalService` — no duplicate retrieval code.

## Evaluation (Phase 6)

Phase 6 adds measurement tools to quantify the quality of the RAG pipeline and agents.

### Components

| Component | Class | What it measures |
|---|---|---|
| RAGAS | `RagasEvaluator` | Faithfulness, Answer Relevancy, Context Precision (0–1) |
| DeepEval | `DeepEvaluator` | Correctness, Answer Relevancy via LLM-as-judge G-Eval |
| LangSmith | `LangSmithTracer` | Traces every LLM call: inputs, outputs, latency, tokens |
| OpenAI Traces | `OpenAITracer` | Spans sent to OpenAI's platform tracing dashboard |

### Usage

```python
from property_intel.evaluation import EvalDataset, EvalSample, EvaluationPipeline
from property_intel.evaluation.ragas_eval import RagasEvaluator
from property_intel.evaluation.deepeval_eval import DeepEvaluator

# Build a dataset of question/answer/context triples
dataset = EvalDataset.from_list([
    {"question": "What is the registration fee?", "answer": "Rs. 10,000.",
     "contexts": ["Fee is Rs. 10,000 per Section 4."], "ground_truth": "Rs. 10,000"},
])

# Run all evaluators
pipeline = EvaluationPipeline({
    "ragas": RagasEvaluator(llm=chat_openai, embeddings=openai_embeddings),
    "deepeval": DeepEvaluator(model="gpt-4o-mini"),
})
report = pipeline.run(dataset)
print(report.summary())
report.to_json("eval_report.json")
```

The `EvaluationReport` scores are 0–1 (higher = better) and can be exported to JSON for tracking improvement over time.

### Tracing

```python
from property_intel.evaluation.tracer import LangSmithTracer, OpenAITracer

# LangSmith: records full traces in LangSmith dashboard
tracer = LangSmithTracer(project_name="property-intel-eval")
traced_ask = tracer.trace(copilot_service.ask, name="copilot_ask")

# OpenAI: records spans in OpenAI's platform
oai_tracer = OpenAITracer()
with oai_tracer.span("agent_run"):
    result = agent.run("What are the builder registration rules?")
```

Both tracers are no-ops when the relevant env vars (`LANGCHAIN_API_KEY` / `OPENAI_API_KEY`) are absent — safe in CI.

## Legacy keyword search (Phase 2)

```bash
uv run python -m property_intel.search "registration deadline" --mode fulltext --category circulars
uv run python -m property_intel.search "registration deadline" --mode bm25
uv run python -m property_intel.search --mode metadata --category acts --source maharera
```

Three document-level backends: `fulltext` (PostgreSQL `tsvector` + GIN index + `ts_rank`), `bm25` (in-memory keyword ranking), and `metadata` (filter/browse with no ranking).

## Testing

```bash
uv run pytest --cov=src --cov-report=term-missing
uv run mypy src
uv run pytest -m "not slow"          # skip real model inference tests
uv run pytest -m "not slow and not qdrant"   # skip model + Qdrant tests
```

Tests are marked:
* `slow` — real Docling or BGE model inference (heavy, downloads models)
* `db` — requires a live PostgreSQL connection (auto-skipped if unreachable)
* `qdrant` — requires a live Qdrant instance (auto-skipped if unreachable)
* `integration` — requires the real `data/raw` dataset (auto-skipped if absent)
