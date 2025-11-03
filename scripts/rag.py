# rag.py
import threading
from typing import Sequence
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from indexer import get_vector_store  # returns (vector_store, splits, docs)

# 1) Build / load once, keep store and docs available if needed
_VECTORSTORE = None

# A simple process-local lock to make first-time initialization thread-safe.
# This is enough for CLI runs and typical single-process servers.
_STORE_LOCK = threading.Lock()

# Default k, can be overridden
DEFAULT_K = 3

# 2) Prompt for the GENERATION step (RAG = Retrieve → Augment → Generate)
SYSTEM = (
    "You are a precise assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know. "
    "Cite chunks inline like (chunk_index=##)."
)
USER = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)
PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", USER)])

def _ensure_store() -> None:
    """
    Ensure the vector store (and companion splits/docs) is built exactly once.
    Load the persisted vector store once (no ingest, no network).

    Concurrency:
      - Uses a process-local threading.Lock to guard the first build.
      - Double-check pattern prevents needless locking after initialization.
    """
    global _VECTORSTORE
    if _VECTORSTORE is not None:       # fast path (already initialized)
        return
    with _STORE_LOCK:                   # only the first caller enters
        if _VECTORSTORE is None:        # second check in case another thread won the race
            vs = get_vector_store()
            # sanity: ensure it actually has vectors
            try:
                count = len(vs.get()["ids"])
            except Exception:
                count = 0
            if count == 0:
                raise RuntimeError(
                    "Vector store is empty. Run the ingest step first:\n"
                    "  uv run python scripts/main.py --ingest"
                )
            _VECTORSTORE = vs

def _format_context(docs: Sequence) -> str:
    """
    Convert retrieved Documents into a readable context block for the LLM.

    Input:
      docs: an ordered sequence of LangChain `Document` objects
            (each has .page_content and .metadata)

    Output:
      A single string where each chunk is labeled with its chunk_index for
      transparent inline citation, e.g.:

        [chunk_index=12]
        ...chunk text...

        [chunk_index=37]
        ...chunk text...

    Notes:
      - Keeping labels in the text lets the model reference them easily when
        following the “(chunk_index=##)” citation rule in the system prompt.
      - Keep this formatting simple and deterministic so LangSmith traces are clean.
    """
    return "\n\n".join(
        f"[chunk_index={d.metadata.get('chunk_index')}]\n{d.page_content}"
        for d in docs
    )


def build_rag_chain(llm, k: int = DEFAULT_K):
    """
    Build a Runnable graph for RAG:
      question → retrieve top-k → format → prompt → llm → text

    Args:
      llm: a LangChain chat model (ChatOpenAI, ChatBedrock, etc.)
      k:   how many chunks to retrieve (default 3)

    Lazily ensures the vector store exists, then constructs a retriever using k.
    """
    _ensure_store()
    retriever = _VECTORSTORE.as_retriever(search_kwargs={"k": k})
    return (
        {"question": RunnablePassthrough()}
        | {"context": retriever | _format_context, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

def rag_answer(llm, question: str, k: int | None = None) -> str:
    """Run RAG end-to-end for a single question with optional k override."""
    chain = build_rag_chain(llm, k=k or DEFAULT_K)
    return chain.invoke(question)
