from app.services.rag_indexer import RAGIndexer
import os

def test_rag_indexing():
    # 1. Setup paths
    docs_folder = "./data/AI Knowledge Base WMS"  # Folder chứa SRS mẫu
    db_path = "./rag_db_test"
    
    # 2. Initialize Indexer
    print("--- Initializing RAG Indexer ---")
    indexer = RAGIndexer(persist_path=db_path)
    
    # 3. Run Indexing
    print(f"--- Indexing documents from '{docs_folder}' ---")
    indexer.build_index(docs_folder)
    
    # 4. Verification Check
    print("\n--- Verifying Index ---")
    count = indexer.collection.count()
    print(f"Total chunks in DB: {count}")
    
    if count > 0:
        # Lấy thử 1 chunk để xem metadata
        sample = indexer.collection.peek(limit=1)
        print("\nSample Chunk Metadata:")
        print(sample['metadatas'][0])
        print("\nSample Chunk Content (First 100 chars):")
        print(sample['documents'][0][:100] + "...")
        print("\n✅ Indexing verified successfully!")
    else:
        print("\n❌ Indexing failed or no documents found.")

if __name__ == "__main__":
    test_rag_indexing()
