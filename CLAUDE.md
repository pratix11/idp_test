# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Property & Regulatory Document Intelligence Platform — ingests, parses, and indexes regulatory PDF documents (MahaRERA first; MHADA/CIDCO planned). Full spec and phase breakdown: `docs/PRD.md`. That file is the source of truth for scope — always check which phase is active before adding functionality from a later phase.

**Status:** Phases 1–5 complete. Phase 4 (AI Copilot) adds OpenAI RAG, `CopilotService` facade, FastAPI REST API with SSE streaming, and a copilot CLI. Phase 5 (Agentic Layer) adds five LangGraph agents (DocumentAnalystAgent, ComparisonAgent, ComplianceAgent, ResearchAgent, ReportAgent) with a keyword-based AgentRouter. Phase 6+ (evaluation, enterprise) not started.

## Commands

```bash
uv sync                                              # install/update deps into .venv
docker compose up -d postgres                        # start local Postgres (required for db-marked tests and the real pipeline)
uv run pytest --cov=src --cov-report=term-missing    # full test suite + coverage (must stay >=80%)
uv run pytest -m "not slow"                          # skip real-Docling tests (faster iteration)
uv run pytest tests/test_task5_docling_parser.py -k test_markdown_generated_for_real_pdf  # single test
uv run mypy src                                      # type check (must be clean)
uv run python -m property_intel.pipeline             # run the batch pipeline over data/raw
```

Test markers (`pyproject.toml`): `slow` (real Docling inference, no mocks), `db` (needs live Postgres, auto-skips if unreachable), `integration` (needs the real `data/raw` dataset, auto-skips if absent).

`data/raw/**/*.pdf` and `data/processed/` are gitignored — the real 73-PDF MahaRERA corpus lives on disk locally but is never committed.

## Architecture

Pipeline shape: `DatasetOrganizer` (scans/categorizes `data/raw/<source>/<category>/**/*.pdf`, SHA-256 dedup) → `ParserRouter` (tries `DoclingParser` first, falls back to `MarkItDownParser` on `ParserError`) → `DocumentProcessor` (drives the `DocumentState` lifecycle `uploaded → processing → completed/failed` via `DocumentRegistry`'s transition rules, writes markdown to `data/processed/`, persists via `DocumentRepository`/Postgres) → `BatchProcessor` (orchestrates the full corpus, `run()`/`retry_failed()`).

All concrete parsers and the router share one exception family (`ParserError`, `CorruptedFileError`, `UnsupportedFileTypeError` in `parsing/base.py`) so the router only ever has to catch `ParserError`.

`DocumentCategory` (`metadata/schema.py`) is a superset enum (`acts, circulars, orders, regulations, reports, rules`) — it covers both the PRD's stated categories and the real on-disk folder names, which don't fully match each other.

## Development rules (from the PRD)

1. Build phase by phase — don't add Phase 5+ functionality (LangGraph agents) while Phase 4 is still active.
2. Every feature needs tests before merge; keep coverage >=80%.
3. Typed Python throughout — `mypy src` must stay clean.
4. One PR per phase (or major task), not direct pushes to `main`, except for small config-only changes (e.g. `.claude/settings.json`) where pushing straight to `main` is fine.
5. Update `README.md` after each completed phase.

## Git/commit conventions

- Never add a `Co-Authored-By: Claude` trailer to commits or a "Generated with Claude Code" line to PR descriptions — this is enforced by `attribution: { commit: "", pr: "" }` in `.claude/settings.json`, not just this instruction.
- Commit per task/feature with a clear message, not one giant commit.

## Session rules — enforce these every session, no exceptions

These are standing rules. Never ask for confirmation on them.

1. **Git/GitHub — just do it.** Run all git commands (commit, push, branch, PR create/merge) and all `gh` commands without asking for approval first. No "should I commit?", no "ready to push?", no "shall I open the PR?". Just run it.

2. **Tests and type checks — always run after every task.** After any code change, run `uv run pytest` and `uv run mypy src` immediately without prompting. Report results. Coverage must stay ≥ 80%.

3. **Proceed task-to-task automatically.** After a task is committed and tests pass, move straight into the next task. Do not ask "ready for the next one?" or "shall I proceed?". Only stop if there is a genuine architectural decision the user needs to make (not routine sequencing).

4. **Layman explanations — mandatory format every task.**
   - **Before writing any code:** give a plain-English explanation of what we're about to build and the core idea behind the technology. Assume zero prior knowledge. Example: "What is RAG?", "What is a cross-encoder?", "What is SSE?".
   - **After the task is committed:** give a "what we built and why" breakdown — what each class/method does, why that design decision was made, why the tests are structured that way. Plain English, no jargon without definition.
   - Goal: user should be able to answer "explain X to me" in an AI engineering interview after reading these explanations.
