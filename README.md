# idp_test

Property & Regulatory Document Intelligence Platform — ingests, parses, and indexes regulatory PDF documents (MahaRERA, with MHADA/CIDCO planned). See [docs/PRD.md](docs/PRD.md) for the full product spec and phased roadmap.

## Status

**Phase 1: Document Intelligence Foundation — complete.**

Ingestion, parsing (Docling primary / MarkItDown fallback), metadata schema, document registry, PostgreSQL storage, and the batch processing pipeline are implemented and tested. Phases 2+ (search, embeddings, RAG, agents) are not started.

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

This scans `data/raw`, parses each document (Docling, falling back to MarkItDown on failure), writes generated markdown to `data/processed/<category>/`, and records each document's lifecycle state (`uploaded` → `processing` → `completed`/`failed`) in PostgreSQL. Re-running skips documents already completed (detected by content hash).

## Testing

```bash
uv run pytest --cov=src --cov-report=term-missing
uv run mypy src
```

Tests are marked `slow` (real Docling model inference), `db` (requires a live PostgreSQL connection — auto-skipped if unreachable), and `integration` (requires the real `data/raw` dataset — auto-skipped if absent).
