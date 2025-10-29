"""
indexer.py — builds and persists a local vector index
using Hugging Face embeddings + Chroma vector store.
"""
import os
import bs4
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool


# ------------------------------------------------------------
# CONFIGURATION - pull from env with defaults
# ------------------------------------------------------------
embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
embed_model_ns = f"{embed_model}".replace("/","_") #for chroma dir namespace
collection = os.getenv("COLLECTION", "RAGAgent_tutorial")
base_chroma_dir = os.getenv("CHROMA_DIR", "./data/chroma")
chroma_dir = f"{base_chroma_dir}/{collection}_{embed_model_ns}" #uses cleaned embed model
user_agent = os.getenv("USER_AGENT", "RAGAgent/0.1")

# ------------------------------------------------------------
# CORE INDEXING FUNCTION
# ------------------------------------------------------------
def build_index():
    """Loads, chunks, embeds, and persists the article into Chroma."""
    print("🚀 Starting index build...")

    # Step 1. Embeddings setup
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    # Step 2. Initialize Chroma vector store
    vector_store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    # Step 3. Load document
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        header_template={"User-Agent": user_agent},
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    assert len(docs) == 1, f"Expected one document, got {len(docs)}"
    print(f"✅ Loaded document — {len(docs[0].page_content)} characters")

    # Step 4. Chunk document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"✅ Split into {len(all_splits)} chunks")

    # Step 5. Enrich metadata
    for i, d in enumerate(all_splits):
        d.metadata = {
            **d.metadata,
            "chunk_index": i,
            "char_start": d.metadata.get("start_index", None),
            "collection": collection,
        }

    # Step 6. Prevent duplicate ingestion
    existing = 0
    try:
        existing = len(vector_store.get()["ids"])
    except Exception:
        pass

    if existing == 0:
        vector_store.add_documents(all_splits)
        vector_store.persist()
        print(f"✅ Added {len(all_splits)} chunks to vector store")
    else:
        print(f"⚠️ Skipped re-ingest — {existing} vectors already present")

    # Step 7. Return handle for downstream use
    print("🏁 Index build complete.")
    return vector_store, all_splits, docs


# ------------------------------------------------------------
# RETRIEVAL TOOL
# ------------------------------------------------------------
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    vs, *_ = build_index()
    retrieved_docs = vs.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# ------------------------------------------------------------
# MAIN GUARD
# ------------------------------------------------------------
if __name__ == "__main__":
    vector_store, all_splits, docs = build_index()
    print("\n🔍 Sample retrieval test:")
    query = "What is Task Decomposition?"
    results = vector_store.similarity_search(query, k=2)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:\n{doc.page_content[:300]}...")