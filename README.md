# idp_test

Property & Regulatory Document Intelligence Platform — ingests, parses, and indexes regulatory PDF documents (MahaRERA, with MHADA/CIDCO planned). See [docs/PRD.md](docs/PRD.md) for the full product spec and phased roadmap.

## Status

**Phase 1: Document Intelligence Foundation — complete.**
**Phase 2: Search Foundation — complete.**

Ingestion, parsing (Docling primary / MarkItDown fallback), metadata schema, document registry, PostgreSQL storage, and the batch processing pipeline are implemented and tested. Search (PostgreSQL full-text, BM25, and metadata filtering/pagination) is implemented and tested. Phases 3+ (embeddings, Qdrant, RAG, agents) are not started.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and Docker Desktop (for PostgreSQL).

```bash
uv sync
cp .env.example .env
docker compose up -d postgres
```

## Running the batch pipeline

Drop PDFs into `data/raw/<source>/<category>/...pdf` (categories: `acts`, `circulars`, `orders`, `regulations`, `reports`, `rules`), then:

```bash
uv run python -m property_intel.pipeline
```

This scans `data/raw`, parses each document (Docling, falling back to MarkItDown on failure), writes generated markdown to `data/processed/<category>/`, and records each document's lifecycle state (`uploaded` → `processing` → `completed`/`failed`) in PostgreSQL, including its plain-text content for search indexing. Re-running skips documents already completed (detected by content hash).

## Searching documents

Once the batch pipeline has populated PostgreSQL, search it ad hoc:

```bash
uv run python -m property_intel.search "registration deadline" --mode fulltext --category circulars
uv run python -m property_intel.search "registration deadline" --mode bm25
uv run python -m property_intel.search --mode metadata --category acts --source maharera
```

Three interchangeable backends, all returning the same paginated `SearchResultPage` shape (`src/property_intel/search/schema.py`):

* **`fulltext`** — PostgreSQL full-text search (`websearch_to_tsquery` against a generated, GIN-indexed `tsvector` column over title + content), ranked with `ts_rank`, with `ts_headline` snippets.
* **`bm25`** — in-memory BM25 ranking (`rank_bm25`) over completed documents, useful as a keyword-ranking baseline independent of Postgres' ranking function.
* **`metadata`** — pure filter/browse: title substring match plus exact category/source/document_type and date-range filters, no ranking.

All three support `--category`, `--source`, `--document-type`, `--page`, and `--page-size`. `SearchService` (`src/property_intel/search/service.py`) is the single entrypoint dispatching to whichever backend `--mode` selects.

## Testing

```bash
uv run pytest --cov=src --cov-report=term-missing
uv run mypy src
```

Tests are marked `slow` (real Docling model inference), `db` (requires a live PostgreSQL connection — auto-skipped if unreachable), and `integration` (requires the real `data/raw` dataset — auto-skipped if absent).
