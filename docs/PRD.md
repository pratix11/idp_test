# PRD: Property & Regulatory Document Intelligence Platform

## Version

v1.0

## Status

Planning → Phase 1 Execution

## Goal

Build an enterprise-grade Document Intelligence Platform that can ingest, process, search, analyze, monitor, and generate insights from large collections of regulatory and government documents.

Initial domain:

* MahaRERA
* MHADA
* CIDCO
* Maharashtra Government Circulars
* Property Regulations
* Acts
* Orders
* Notifications

Long-term objective:

The architecture should remain domain-agnostic so it can later support:

* Legal
* Healthcare
* Government Schemes
* Tender Intelligence
* Compliance
* Corporate Knowledge Bases

---

# Problem Statement

Government and regulatory documents are:

* Large
* Difficult to understand
* Spread across multiple sources
* Poorly searchable
* Frequently updated

Users struggle to:

* Find relevant regulations
* Understand implications
* Compare policy changes
* Track amendments
* Generate compliance reports
* Extract actionable insights

The platform will solve these problems through enterprise search, AI reasoning, automation, and agentic workflows.

---

# Current Dataset

Collected:

* 73 MahaRERA PDF documents

Document Types:

* Circulars
* Acts
* Orders
* Regulations
* Reports

Future Sources:

* MHADA
* CIDCO
* Government Notifications
* Development Control Regulations
* MRTP Act Documents

---

# Success Criteria

Phase 1

* 73 PDFs processed successfully
* Metadata extracted
* Markdown generated
* Documents indexed in PostgreSQL

Phase 2

* Search across all documents

Phase 3

* Hybrid Retrieval
* Semantic Search

Phase 4

* AI Copilot

Phase 5

* Agentic Workflows

Phase 6

* Enterprise Monitoring & Evaluation

---

# High-Level Architecture

Document Sources
↓
Ingestion
↓
Parsing
↓
Metadata Extraction
↓
Document Registry
↓
Storage Layer
↓
Search Layer
↓
Retrieval Layer
↓
AI Copilot
↓
Agents
↓
Reports / Compliance / Monitoring

---

# Phase 1: Document Intelligence Foundation

## Objective

Convert raw documents into structured knowledge.

### Deliverables

* Document ingestion
* Parsing pipeline
* Metadata extraction
* Registry
* PostgreSQL storage
* Markdown generation

### Out of Scope

* Embeddings
* Qdrant
* LangGraph
* RAG
* Agents

---

# Task Breakdown

## Task 1: Repository Initialization

### Deliverables

Create repository structure.

```text
property-intelligence-platform/

data/
src/
tests/
docs/
notebooks/

README.md
pyproject.toml
.env.example
```

### Unit Tests

* Verify directories exist
* Verify project boots successfully
* Verify config loading

---

## Task 2: Dataset Organization

### Deliverables

Categorize documents.

```text
raw/

acts/
circulars/
orders/
regulations/
reports/
```

### Unit Tests

* Verify category assignment
* Verify no missing files
* Verify duplicate detection

---

## Task 3: Metadata Registry

### Deliverables

Create metadata schema.

Fields:

```python
title
source
category
document_type
date
pages
file_path
markdown_path
```

### Unit Tests

* Schema validation
* Required field validation
* Date validation
* Path validation

---

## Task 4: Parsing Layer

### Deliverables

Implement parser abstraction.

Interface:

```python
parse_document()
extract_text()
extract_tables()
extract_metadata()
```

Supported:

* PDF

Future:

* DOCX
* XLSX
* PPTX

### Unit Tests

* Parse sample PDF
* Extract text
* Extract page count
* Handle invalid files
* Handle corrupted PDFs

---

## Task 5: Docling Integration

### Deliverables

Primary parser.

Output:

```text
markdown
metadata
tables
images
```

### Unit Tests

* Markdown generated
* Tables extracted
* Images detected
* Empty document handling

---

## Task 6: MarkItDown Integration

### Deliverables

Fallback parser.

### Unit Tests

* Conversion success
* Output validation
* Fallback execution path

---

## Task 7: Document Registry

### Deliverables

Track processing lifecycle.

States:

```text
uploaded
processing
completed
failed
```

### Unit Tests

* State transitions
* Duplicate prevention
* Failure recovery

---

## Task 8: PostgreSQL Integration

### Deliverables

Documents table.

### Unit Tests

* Insert document
* Retrieve document
* Update document
* Delete document
* Duplicate handling

---

## Task 9: Batch Processing Pipeline

### Deliverables

Process all 73 PDFs.

### Unit Tests

* Single file processing
* Batch processing
* Failure isolation
* Retry behavior

---

## Task 10: Logging

### Deliverables

Centralized logging.

### Unit Tests

* Log creation
* Error logging
* Rotation behavior

---

# Phase 2: Search Foundation

## Objective

Enable enterprise search without AI.

### Components

* PostgreSQL Full Text Search
* BM25 Search
* Metadata Search
* Filtering

### Deliverables

Search by:

* Title
* Category
* Keywords
* Source

### Unit Tests

* Search accuracy
* Metadata filters
* Pagination
* Empty result handling

---

# Phase 3: Enterprise Retrieval Layer

## Objective

Semantic retrieval.

### Components

* BGE-M3 Embeddings
* Qdrant
* Hybrid Search
* Re-ranking

### Deliverables

* Semantic search
* Similar documents
* Retrieval pipeline

### Unit Tests

* Embedding generation
* Vector insertion
* Retrieval quality
* Hybrid retrieval
* Reranker scoring

---

# Phase 4: AI Copilot

## Objective

Document understanding.

### Components

* OpenAI
* RAG
* Citations
* Streaming

### Deliverables

* Question answering
* Summaries
* Comparisons

### Unit Tests

* Citation generation
* Context building
* Prompt generation
* Streaming responses

---

# Phase 5: Agentic Layer

## Objective

Move beyond chatbot.

### Agents

Document Analyst Agent

* Summarize
* Extract

Regulation Comparison Agent

* Compare versions
* Detect changes

Compliance Agent

* Validate requirements

Research Agent

* Multi-document analysis

Report Agent

* Generate reports

### Unit Tests

* Agent execution
* Tool calls
* Workflow routing
* Failure handling

---

# Phase 6: Evaluation

## Components

* RAGAS
* DeepEval
* LangSmith
* OpenAI Traces

### Unit Tests

* Evaluation pipeline
* Metric generation
* Trace collection

---

# Phase 7: Enterprise Features

## Components

* RBAC
* Audit Logs
* Alerts
* Drive Sync
* Versioning

### Unit Tests

* Access control
* Audit generation
* Sync validation
* Alert generation

---

# Recommended Tech Stack

## Phase 1

* Python
* Docling
* MarkItDown
* Pydantic
* PostgreSQL
* SQLAlchemy

## Phase 2

* PostgreSQL FTS
* BM25

## Phase 3

* BGE-M3
* Qdrant
* BGE Reranker

## Phase 4

* OpenAI
* FastAPI
* SSE Streaming

## Phase 5

* LangGraph
* OpenAI SDK

## Phase 6

* RAGAS
* DeepEval
* LangSmith
* OpenAI Traces

## UI

Development:

* Gradio

Production:

* Lovable
* Next.js
* React

---

# Development Rules For Claude Code

1. Build phase by phase.
2. Never skip unit tests.
3. Every feature requires tests before merge.
4. No embeddings before Phase 1 completion.
5. No agents before Phase 3 completion.
6. Maintain >80% test coverage.
7. Use typed Python throughout.
8. Follow repository structure strictly.
9. Create PR for every major task.
10. Update README after every completed phase.
