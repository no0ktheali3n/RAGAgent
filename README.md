# 🧠 RAGAgent Tutorial — Retrieval-Augmented Generation with LangChain

RAGAgent demonstrates a **modular Retrieval-Augmented Generation (RAG)** pipeline built with **LangChain**, **Chroma**, and **Hugging Face embeddings**, with optional integration for **OpenAI** and **AWS Bedrock / Anthropic Claude** models.

This project is the baseline for the upcoming **ContextAI** system, which will extend RAGAgent to index and reason over Wikipedia and larger corpora.

---

## 🚀 Overview

RAGAgent evolves from a simple semantic search indexer into a full RAG workflow capable of **document retrieval** and **grounded generation**.

### System Components

1. **Indexer** — Loads, chunks, embeds, and persists documents locally using Chroma.  
2. **Validator** — Verifies data integrity and semantic search performance.  
3. **RAG Engine** — Retrieves relevant chunks and synthesizes an answer using an LLM.

### RAG Flow Diagram

~~~
Question
   │
   ▼
Retriever (Chroma Vector Store)
   │ → Top-k similar chunks
   ▼
Context Formatter
   │ → Annotates context with chunk indices
   ▼
Prompt Template (System + User)
   │
   ▼
LLM (OpenAI or Bedrock)
   │
   ▼
Final Grounded Answer
~~~

---

## 📁 Project Structure

~~~
RAGAgent/
│
├── scripts/
│   ├── main.py         # CLI entrypoint (validate | rag)
│   ├── indexer.py      # Loads, chunks, embeds, and persists documents
│   └── rag.py          # Implements Retrieve → Augment → Generate chain
│
├── data/chroma/        # Chroma vector store persistence
├── .env                # Provider credentials and environment configuration
└── requirements.txt
~~~

---

## 🧱 Dependencies

> **Note:** This project uses [`uv`](https://github.com/astral-sh/uv) — a fast Python package manager and runtime for dependency management and script execution.  
> You must have `uv` installed before running this project. Choose one of the following installation methods:

### 🪟 Windows

**Option 1: Chocolatey (Recommended)**
~~~powershell
choco install uv
~~~

**Option 2: PowerShell (Official Installer)**
~~~powershell
iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex
~~~

**Verify Installation**
~~~powershell
uv --version
~~~

### 🐧 macOS / Linux

~~~bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~~~

**Verify Installation**
~~~bash
uv --version
~~~

Install core packages:

~~~bash
uv add langchain-core langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-aws chromadb
~~~

Optional OpenAI:

~~~bash
uv add langchain-openai
~~~

---

## ⚙️ Configuration

Create a `.env` file in the project root with the following keys:

~~~powershell
# Embeddings
EMBED_MODEL=BAAI/bge-small-en-v1.5
CHROMA_DIR=./data/chroma
COLLECTION=RAGAgent_tutorial

# OpenAI (optional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.0

# AWS Bedrock (optional)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
BEDROCK_MODEL=global.anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_REGION=us-east-1
BEDROCK_TEMPERATURE=0.0

# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=RAGAgent
~~~

> **Note:** Bedrock uses standard AWS credentials (IAM/SigV4). The Bedrock “API key” mechanism is not used by LangChain’s `ChatBedrock`.

---

## 🧪 Usage

### 1) Validate Mode — Build and Inspect the Index

Validates document ingestion, chunking, and persistence while running a simple semantic search test.

~~~powershell
uv run python scripts/main.py --validate
~~~

Expected highlights:

~~~
✅ Loaded document — 43,047 characters
✅ Split into 63 chunks
✅ Vector store currently holds 63 embeddings
🔎 Retrieval test for query: 'What is Task Decomposition?'
~~~

### 2) RAG Mode — Retrieve and Generate an Answer

Runs the complete RAG chain using your chosen LLM provider.

**Bedrock:**
~~~bash
uv run python scripts/main.py --rag --provider bedrock --question "How do agents plan and decompose tasks?"
~~~

**OpenAI:**
~~~bash
uv run python scripts/main.py --rag --provider openai --question "Explain the difference between planning and reacting in agents."
~~~

**Optional: override retrieval depth (k):**
~~~bash
uv run python scripts/main.py --rag --question "What is task decomposition?" --k 5
~~~

---

## 🔧 Internals

### `scripts/indexer.py`
Responsibilities:
- **Load** web content with `WebBaseLoader` (custom User-Agent to avoid 403s)  
- **Split** content using `RecursiveCharacterTextSplitter` (overlap keeps coherence)  
- **Embed** chunks with `HuggingFaceEmbeddings` (`BAAI/bge-small-en-v1.5`)  
- **Persist** vectors to **Chroma** on disk  
- **Idempotence**: skips re-ingest if vectors already exist, but still re-loads/splits for validation

### `scripts/rag.py`
Implements RAG:
- **Lazy, thread-safe init**: builds/loads the vector store on first use (no heavy work at import; guarded by a `Lock`)  
- **Context formatter**: converts retrieved `Document` chunks into a single block with inline labels like `(chunk_index=##)` for transparent citation  
- **Runnable chain**:
  ~~~
  {"question"} → retriever(k) → _format_context → PROMPT → llm → StrOutputParser()
  ~~~
- **k override**: default `k=3`, optionally overridden at runtime (`--k N`)

### `scripts/main.py`
CLI controller:
- `--validate` → ingestion/chunking/persistence checks + semantic search  
- `--rag` → runs full Retrieve → Augment → Generate answer  
- `--provider` → `"openai"` or `"bedrock"` (auto-detects if omitted)  
- `--k` → number of retrieved chunks  
- `_make_llm()` factory builds either:
  - `ChatOpenAI` (OpenAI) or
  - `ChatBedrock` (AWS Bedrock; supports Claude Haiku 4.5)

---

## 🎛️ Tuning Cheat-Sheet (RAG vs Generation)

- **Retrieval `k` (rag.py)**: how many **chunks** to retrieve from Chroma. More can improve coverage but may add noise.  
- **Temperature (LLM)**: randomness of **wording**. For faithful RAG, keep **`0.0`** (deterministic).  
- **Top-p / Top-k (LLM sampling)**: only matter when temperature > 0. For creative or multi-answer tasks, try `temp=0.3, top_p=0.9`.

---

## 🧮 Version Summary

| Version | Highlights |
|--------:|------------|
| **v0.2.0** | Implemented full RAG pipeline (Retrieve → Augment → Generate). Added Bedrock + OpenAI LLM support, LangSmith hooks, lazy & thread-safe vector store init, and `--k` override. |
| **v0.1.1** | Refactor for modularity; introduced `scripts/main.py` CLI with `--validate` and `--rag`; provider factory; `.env` configuration. |
| **v0.1.0** | Initial indexer: web loading, chunking, embeddings, local Chroma persistence, and semantic search validation. |

---

## 🧭 Roadmap

- [ ] Agent / Tool integration
- [ ] Wikipedia ingestion for **ContextAI**  
- [ ] LangSmith tracing + prompt metadata  
- [ ] Reranking / hybrid search  
- [ ] RAG evaluation (RAGAS / TruLens)  
- [ ] Caching + parallelization for throughput

---

## 📜 License

**MIT License** — free to use, modify, and extend.

---

## 💡 Acknowledgements

- [LangChain](https://www.langchain.com/) — Framework for building RAG and agentic systems  
- [Chroma](https://www.trychroma.com/) — Local vector database  
- [Hugging Face](https://huggingface.co/) — Embedding models (BAAI/bge-small-en-v1.5)  
- [Anthropic Claude](https://www.anthropic.com/) — LLM via AWS Bedrock  
- [OpenAI GPT](https://platform.openai.com/docs/models) — Optional model backend
