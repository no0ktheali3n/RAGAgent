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
from indexer import ingest_index, validate_from_store, COLLECTION

# -------------------- LLM FACTORY --------------------
# scripts/main.py
import os

#decide whether to use OpenAI or Bedrock
def _detect_provider(explicit: str | None) -> str:
    """
    Decide which provider (OpenAI or Bedrock) to use.

    Priority:
      1. explicit argument if passed
      2. OPENAI_API_KEY present → "openai"
      3. otherwise → "bedrock"

    Example:
        _detect_provider(None) -> "openai"  # if OPENAI_API_KEY is set
        _detect_provider("bedrock") -> "bedrock"
    """
    if explicit:
        return explicit.strip().lower()
    return "openai" if os.getenv("OPENAI_API_KEY") else "bedrock"

#create ChatOpenAI object from environment config
def _make_openai_llm():
    """
    Initialize an OpenAI chat model via langchain-openai.

    Environment vars used:
        OPENAI_MODEL        default "gpt-4o-mini"
        OPENAI_TEMPERATURE  default "0.0"

    Raises:
        RuntimeError if langchain-openai not installed.

    Returns:
        ChatOpenAI instance ready for use in RAG or agent chains.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise RuntimeError(
            "OpenAI selected but langchain-openai is not installed.\n"
            "Install: uv pip install langchain-openai"
        ) from e

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    return ChatOpenAI(model=model, temperature=temperature)

#safely determine AWS region for bedrock
def _resolve_region() -> str:
    """
    Pick a region for Bedrock in order of precedence.

    Search order:
        1. BEDROCK_REGION
        2. AWS_REGION
        3. AWS_DEFAULT_REGION
        4. "us-east-1" fallback

    Returns:
        The first non-empty region string found.

    Example:
        -> "us-east-1"
    """
    return (
        os.getenv("BEDROCK_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )

#create ChatBedrock client with correct model + region
def _make_bedrock_llm():
    """
    Initialize a Bedrock chat model via langchain-aws.

    Environment vars used:
        BEDROCK_MODEL        default "global.anthropic.claude-haiku-4-5-20251001-v1:0"
        BEDROCK_REGION       (or AWS_REGION / AWS_DEFAULT_REGION)
        BEDROCK_TEMPERATURE  default "0.0"

    Raises:
        RuntimeError if langchain-aws not installed.

    Returns:
        ChatBedrock instance configured for chosen model and region.
    """
    try:
        from langchain_aws import ChatBedrock
    except ImportError as e:
        raise RuntimeError(
            "Bedrock selected but langchain-aws is not installed.\n"
            "Install: uv pip install langchain-aws"
        ) from e

    # Use the regional model id (not the 'global.' prefix)
    model_id = os.getenv("BEDROCK_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
    region   = _resolve_region()
    temperature = float(os.getenv("BEDROCK_TEMPERATURE", "0.0"))
    return ChatBedrock(model_id=model_id, region_name=region, temperature=temperature)

#orchestrates detection + creation
def _make_llm(provider: str | None = None):
    """
    Public factory to create an LLM client (OpenAI or Bedrock).

    Args:
        provider: "openai" | "bedrock" | None
                  If None, auto-detects via _detect_provider().

    Returns:
        ChatOpenAI or ChatBedrock instance.

    Raises:
        ValueError for unknown provider.

    Example:
        llm = _make_llm("bedrock")
        llm = _make_llm()  # auto-detect
    """
    effective = _detect_provider(provider)
    if effective == "openai":
        return _make_openai_llm()
    if effective == "bedrock":
        return _make_bedrock_llm()
    raise ValueError("Unknown provider. Use 'openai' or 'bedrock'.")


# -------------------- INGEST/VALIDATION MODE --------------------
def run_ingest(force: bool):
    ingest_index(force=force)

def run_validate():
    validate_from_store()  # uses store only; no network

# -------------------- RAG MODE --------------------
def run_rag(question: str, provider: str | None, k: int | None):
    """
    Executes the RAG chain (Retrieve → Augment → Generate) with an LLM.
    """
    from rag import rag_answer

    # Build the model (OpenAI or Bedrock)
    llm = _make_llm(provider)
    print(f"\nQ: {question}\n")
    answer = rag_answer(llm, question, k=k)
    print(answer)

# -------------------- CLI --------------------
def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAgent CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ingest", action="store_true", 
                        help="Load/split/embed and persist vectors.")
    mode.add_argument("--validate", action="store_true",
                        help="Validate ingestion/chunking/persistence + semantic search.")
    mode.add_argument("--rag", action="store_true",
                        help="Run (Retrieve → Augment → Generate) for a question (requires ingested store).")
    parser.add_argument("--force", action="store_true", 
                        help="With --ingest, rebuild vectors from scratch.")
    parser.add_argument("--question", type=str, default="What is Task Decomposition?",
                        help="Question to ask in RAG mode.")
    parser.add_argument("--provider", type=str, choices=["openai", "bedrock"],
                        help="LLM provider to use (overrides auto-detect).")
    parser.add_argument("--k", type=int, default=None,
                        help="(Optional) Top-k docs for retrieval (if exposed later).")
    return parser.parse_args(argv)

def main():
    args = parse_args(sys.argv[1:])
    if args.ingest:
        run_ingest(force=args.force) 
        return
    if args.validate:
        run_validate() 
        return
    if args.rag:
        run_rag(question=args.question, provider=args.provider, k=args.k)
        return

if __name__ == "__main__":
    main()
