"""
indexer.py — builds and persists a local vector index
using Hugging Face embeddings + a Chroma vector store.

This module is intentionally limited to:
  - Loading source documents
  - Chunking/splitting
  - Embedding + persisting to Chroma
  - Returning objects needed for validation

"""

import os
import bs4
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# -------------------- CONFIG --------------------
EMBED_MODEL_ID = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")  # Real HF model id
EMBED_MODEL_NS = EMBED_MODEL_ID.replace("/", "_")                    # FS-safe suffix
COLLECTION     = os.getenv("COLLECTION", "RAGAgent_tutorial")
BASE_CHROMA    = os.getenv("CHROMA_DIR", "./data/chroma")
CHROMA_DIR     = f"{BASE_CHROMA}/{COLLECTION}_{EMBED_MODEL_NS}"
USER_AGENT     = os.getenv("USER_AGENT", "RAGAgent/0.1")

SOURCE_URLS: list[str] = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
]

print(f"[cfg] EMBED_MODEL_ID={EMBED_MODEL_ID}")
print(f"[cfg] CHROMA_DIR={CHROMA_DIR}")

# -------------------- PRIVATE HELPERS --------------------
def _load_docs() -> list[Document]:
    """Load source documents with explicit headers to avoid 403s."""
    loader = WebBaseLoader(
        web_paths=tuple(SOURCE_URLS),
        requests_kwargs={"headers": {"User-Agent": USER_AGENT}},
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))),
    )
    docs = loader.load()
    # For this tutorial we expect a single long article; assert to surface surprises early.
    assert len(docs) == 1, f"Expected one document, got {len(docs)}"
    return docs

def _split_docs(docs: list[Document]) -> list[Document]:
    """Split into overlapping chunks and enrich metadata for traceability."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = splitter.split_documents(docs)

    for i, d in enumerate(splits):
        d.metadata = {
            **d.metadata,
            "collection": COLLECTION,
            "chunk_index": i,
            "char_start": d.metadata.get("start_index"),
            # You can optionally preserve source URL/title if present in loader metadata:
            # "source": d.metadata.get("source"),
            # "title": d.metadata.get("title"),
        }
    return splits

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        encode_kwargs={"normalize_embeddings": True},
    )

def _get_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Open (or create) a Chroma collection on disk."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

# -------------------- PUBLIC API --------------------
def build_index() -> tuple[Chroma, list[Document], list[Document]]:

    """
    Build (or load) the vector index and return:
      (vector_store, splits, docs)

    Why return all three?
      - vector_store: use this to run similarity search (semantic search).
      - splits: lets you validate chunking (count, preview) without hitting the store.
      - docs: lets you validate original document stats (character count, etc.).

    Idempotence:
      - If the Chroma collection already has vectors, we skip re-ingest, but still
        load + split the source so you can validate state consistently.
    """
    print("🚀 Starting index build/load...")
    embeddings   = _get_embeddings()
    vector_store = _get_vector_store(embeddings)

    # Check whether vectors already exist
    try:
        existing = len(vector_store.get()["ids"])
    except Exception:
        existing = 0

    # Always load and split for validation, but only ingest if empty
    docs   = _load_docs()
    splits = _split_docs(docs)

    if existing == 0:
        vector_store.add_documents(splits)
        vector_store.persist()  # persist once after ingest
        print(f"✅ Loaded document — {len(docs[0].page_content)} characters")
        print(f"✅ Split into {len(splits)} chunks")
        print(f"✅ Added {len(splits)} chunks to vector store and persisted to disk")
    else:
        print(f"⚠️ Skipped re-ingest — {existing} vectors already present")
        print(f"✅ Loaded document — {len(docs[0].page_content)} characters")
        print(f"✅ (Re)split into {len(splits)} chunks for validation (no store writes)")

    print("🏁 Index build/load complete.")
    return vector_store, splits, docs
