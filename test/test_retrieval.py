from src.app.services.rag_retriever import RAGRetriever
import time

def test_retrieval():
    print("--- Initializing Retriever ---")
    start_time = time.time()
    retriever = RAGRetriever(persist_path="./rag_db_test")
    print(f"Init done in {time.time() - start_time:.2f}s")
    
    # Test queries related to the indexed documents
    queries = [
        "Quy tắc đặt mã sản phẩm",
        "Quy trình nhập kho", 
        "Làm sao để hủy phiếu nhập?",
        "Bảo mật hệ thống"
    ]
    
    for query in queries:
        print(f"\n==================================================")
        print(f"QUERY: {query}")
        print(f"==================================================")
        
        # 1. Test Semantic Search Only
        print("\n--- [1] Semantic Search Only (Top 3) ---")
        semantic_results = retriever._semantic_search(query, k=3)
        for i, res in enumerate(semantic_results):
            print(f"Rank {i+1}: {res['content'][:100]}... (Score: {res['initial_score']:.4f})")

        # 2. Test Keyword Search Only
        print("\n--- [2] Keyword Search Only (Top 3) ---")
        keyword_results = retriever._keyword_search(query, k=3)
        for i, res in enumerate(keyword_results):
            print(f"Rank {i+1}: {res['content'][:100]}... (BM25 Score: {res['initial_score']:.4f})")
            
        # 3. Test Full Retrieval with Rerank
        print("\n--- [3] Full Hybrid Retrieval (Top 3 with Rerank) ---")
        reranked_results = retriever.retrieve(query, top_k=3, rerank=True)
        for i, res in enumerate(reranked_results):
            source = res.get('search_type', 'mixed')
            print(f"Rank {i+1}: {res['content'][:100]}... [Source: {source}] (Rerank Score: {res.get('rerank_score', 0):.4f})")

if __name__ == "__main__":
    test_retrieval()
