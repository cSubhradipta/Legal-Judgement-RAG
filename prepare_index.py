from rag_pipeline import ingestion, CHUNK_SIZE, CHUNK_OVERLAP

if __name__ == "__main__":
    result = ingestion(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print("Index ready:", result["vector_store"].collection.count(), "chunks")