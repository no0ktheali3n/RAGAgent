"""
main.py — CLI entrypoint for the RAGAgent tutorial

Two modes:
  1) --validate
     - Builds/loads the index
     - Prints source length, chunk count, a chunk preview
     - Checks persisted vector count
     - Runs a quick semantic search (no generation)

  2) --rag [--question "..."] [--k 3] [--provider openai|bedrock]
     - Builds/loads the index (rag.py handles retriever)
     - Calls the RAG chain (Retrieve → Augment → Generate) with the chosen LLM
     - Prints a grounded answer with inline chunk citations

Notes:
  - Provider selection:
      * --provider openai  -> uses langchain_openai.ChatOpenAI
      * --provider bedrock -> uses langchain_aws.ChatBedrock
      * If omitted, tries to auto-detect (OPENAI_API_KEY -> OpenAI).
  - LangSmith tracing:
      Set in .env if desired:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=...
        LANGCHAIN_PROJECT=RAGAgent
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment early (OPENAI_API_KEY, AWS creds, LangSmith vars, etc.)
load_dotenv()

# Indexing/validation utilities
from indexer import build_index, COLLECTION

# RAG chain helper (Retrieve → Augment → Generate)
from rag import rag_answer

# -------------------- LLM FACTORY --------------------
def _make_llm(provider: str | None = None):
    """
    Create a chat LLM based on provider selection or environment.
    - provider can be 'openai' or 'bedrock'
    - if None, auto-detect using env vars
    """
    # Auto-detect if not explicitly provided
    effective = (provider or "").strip().lower()
    if not effective:
        effective = "openai" if os.getenv("OPENAI_API_KEY") else "bedrock"

    if effective == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAI provider requested but langchain-openai is not installed. "
                "Install with: uv pip install langchain-openai"
            ) from e
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        return ChatOpenAI(model=model, temperature=temperature)

    if effective == "bedrock":
        try:
            from langchain_aws import ChatBedrock
        except ImportError as e:
            raise RuntimeError(
                "Bedrock provider requested but langchain-aws is not installed. "
                "Install with: uv pip install langchain-aws"
            ) from e
        # Default to Claude Sonnet; override with BEDROCK_MODEL if desired
        model_id = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20240620-v1:0")
        temperature = float(os.getenv("BEDROCK_TEMPERATURE", "0.0"))
        return ChatBedrock(model_id=model_id, temperature=temperature)

    raise ValueError("Unknown provider. Use 'openai' or 'bedrock'.")

# -------------------- VALIDATION MODE --------------------
def run_validate():
    """
    Validates ingestion, chunking, persistence, and runs a quick semantic search.
    Mirrors your previous validation flow with the refactored indexer.
    """
    # builds vector store, splits and docs index
    vs, splits, docs = build_index()

    # Step 1: basic validation
    assert len(docs) == 1, f"Expected one document, got {len(docs)}"
    print(f"✅ Total characters in source: {len(docs[0].page_content)}")

    # Step 2: chunk check
    print(f"✅ Total chunks created: {len(splits)}")
    print(f"✅ Example chunk preview:\n{splits[0].page_content[:400]}...\n")

    # Step 3: check persistence
    try:
        count = len(vs.get()["ids"])
        print(f"✅ Vector store currently holds {count} embeddings in collection '{COLLECTION}'")
    except Exception as e:
        print(f"⚠️ Could not fetch vector count: {e}")

    # Step 4: quick retrieval test (semantic search only; no LLM generation)
    query = "What is Task Decomposition?"
    results = vs.similarity_search(query, k=2)
    print(f"\n🔎 Retrieval test for query: '{query}'")
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(f"Source: {doc.metadata}")
        print(f"Snippet: {doc.page_content[:300]}...")

    print("\n✅ Index test completed.")

# -------------------- RAG MODE --------------------
def run_rag(question: str, provider: str | None, k: int | None):
    """
    Executes the RAG chain (Retrieve → Augment → Generate) with an LLM.
    """
    # Build the model (OpenAI or Bedrock)
    llm = _make_llm(provider)
    print(f"\nQ: {question}\n")
    answer = rag_answer(llm, question, k=k)
    print(answer)

# -------------------- CLI --------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAgent Tutorial CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true",
                      help="Validate ingestion/chunking/persistence + semantic search.")
    mode.add_argument("--rag", action="store_true",
                      help="Run RAG (Retrieve → Augment → Generate) for a question.")
    parser.add_argument("--question", type=str, default="What is Task Decomposition?",
                        help="Question to ask in RAG mode.")
    parser.add_argument("--provider", type=str, choices=["openai", "bedrock"],
                        help="LLM provider to use (overrides auto-detect).")
    parser.add_argument("--k", type=int, default=None,
                        help="(Optional) Top-k docs for retrieval (if exposed later).")
    return parser.parse_args(argv)

def main():
    args = parse_args(sys.argv[1:])

    if args.validate:
        run_validate()
        return

    if args.rag:
        # default k value is 3 set in rag.py
        run_rag(question=args.question, provider=args.provider, k=args.k)
        return

if __name__ == "__main__":
    main()
