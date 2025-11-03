
### 🧾 `CHANGELOG.md`

# 🧩 Changelog

## [v0.2.1] — 2025-11-02 
###Refactor: Separation of Ingest and Validate

This milestone refactor introduces a cleaner project architecture, splitting document ingestion from validation.  
It eliminates redundant index rebuilding during RAG runs, improving both clarity and performance.

### 🔧 Key Changes
- 🧩 **`ingest_index()`**  
  Handles loading, splitting, embedding, and persisting documents into Chroma.  
  Optional `--force` flag for rebuilding from scratch.  
  Added preview output (source length, chunk count, example chunk).

- 🔍 **`validate_from_store()`**  
  New function for store-only checks.  
  Runs semantic retrieval tests without fetching or re-splitting the source.  
  Ideal for quick verification after ingestion.

- 🚫 **RAG no longer rebuilds indexes**  
  Retrieval runs only load existing Chroma collections (`get_vector_store()`).  
  Prevents duplicate “Starting index build…” logs.

- 🧠 **Lifecycle Simplified**
      uv run python scripts/main.py --ingest [--force]
      uv run python scripts/main.py --validate
      uv run python scripts/main.py --rag --question "..."

- 🧰 **New CLI Modes**
  | Flag | Description |
  |------|--------------|
  | `--ingest` | Load, split, embed, and persist to Chroma |
  | `--force` | Force rebuild existing collection |
  | `--validate` | Validate the existing store only |
  | `--rag` | Run RAG (Retrieve → Augment → Generate) |
  | `--provider` | Choose `openai` or `bedrock` |
  | `--question` | Set the query for RAG mode |
  | `--k` | Optional override for retrieval depth (default: 3) |

---

## [v0.2.0] — 2025-10-30
### Added
- Full Retrieval-Augmented Generation (RAG) implementation using LangChain’s `Runnable` graph.
- `rag.py` module with:
  - Lazy, thread-safe vector store initialization.
  - Context formatter with `chunk_index` citations.
  - Customizable retrieval `k` for flexible grounding.
- LLM provider factory with OpenAI and AWS Bedrock support.
- Command-line flags:
  - `--provider [openai|bedrock]`
  - `--question "<query>"`
  - `--k N`
- Integrated `.env` support for Bedrock and OpenAI credentials.
- Support for **Claude Haiku 4.5 via Bedrock**.

### Improved
- Clear module separation: `indexer.py`, `rag.py`, and `main.py`.
- Detailed inline documentation and consistent logging format.
- Added LangSmith environment variable hooks.

---

## [v0.1.1] — 2025-10-28
### Added
- Refactored `indexer.py` for modularity and readability.
- Introduced `main.py` CLI interface.
- Added flags:
  - `--validate` → ingestion + semantic search test.
  - `--rag` → prepares for RAG integration.
- Implemented provider detection (`_make_llm`) and factory pattern.
- Added `.env` variable support for embedding and model configuration.

### Improved
- Stronger docstrings and structured output formatting.
- Persistent Chroma storage path handling.
- Clean separation of configuration, validation, and runtime modes.

---

## [v0.1.0] — 2025-10-27
### Added
- Base semantic search pipeline:
  - Web-based document loading (Lilian Weng’s “LLM-Powered Autonomous Agents”).
  - Recursive chunking via `RecursiveCharacterTextSplitter`.
  - Embedding and persistence using `HuggingFaceEmbeddings` + Chroma.
- Initial test harness to validate ingestion and vector search.

### Notes
- Established foundational structure for future RAG implementation.
- Introduced `.env` configuration and idempotent indexing logic.

# 🧩 Version Summary

| Version | Highlights |
|----------|-------------|
| **v0.2.1** | Refactor: ingestion and validation separated; RAG now attach-only. |
| **v0.2.0** | Initial RAG pipeline (Retrieve → Augment → Generate). |
| **v0.1.1** | CLI refactor; added flags and LLM provider factory. |
| **v0.1.0** | Initial indexer and semantic search demo. |

---
