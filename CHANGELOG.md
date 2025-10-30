
### 🧾 `CHANGELOG.md`

# 🧩 Changelog

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
