from indexer import build_index, COLLECTION

#validates ingestion, chunking and store and tests semantic search
def main():

    #builds vector store, splits and docs index
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

    # Step 4: quick retrieval test
    query = "What is Task Decomposition?"
    results = vs.similarity_search(query, k=2)
    print(f"\n🔎 Retrieval test for query: '{query}'")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Source: {doc.metadata}")
        print(f"Snippet: {doc.page_content[:300]}...")

    print("\n✅ Index test completed.")

if __name__ == "__main__":
    main()
