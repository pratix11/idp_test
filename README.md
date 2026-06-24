# idp_test

Property & Regulatory Document Intelligence Platform — ingests, parses, indexes, and semantically searches regulatory PDF documents (MahaRERA, with MHADA/CIDCO planned). See [docs/PRD.md](docs/PRD.md) for the full product spec and phased roadmap.

## Status

**Phase 1: Document Intelligence Foundation — complete.**
**Phase 2: Search Foundation — complete.**
**Phase 3: Enterprise Retrieval Layer — complete.**

Ingestion, parsing (Docling primary / MarkItDown fallback), metadata schema, document registry, PostgreSQL storage, and the batch processing pipeline are implemented and tested. Search (PostgreSQL full-text, BM25, and metadata filtering/pagination) is implemented and tested. Phase 3 adds chunking, BGE-M3 embeddings, Qdrant vector storage, semantic search, hybrid BM25+vector search with Reciprocal Rank Fusion, and BGE cross-encoder reranking. Phases 4+ (RAG, AI Copilot, agents) are not started.

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
