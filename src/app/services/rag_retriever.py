import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
import torch
from src.app.utils.logger import logger

class RAGRetriever:
    """
    RAG Retriever Service.
    Nhiệm vụ: Tìm kiếm context liên quan nhất từ Vector DB cho câu query của user.
    
    Quy trình chuẩn (Pipeline):
    1. Retrieve: Lấy tập ứng viên rộng (Candidate Generation) bằng Vector Search.
    2. Rerank: Sắp xếp lại danh sách ứng viên bằng Cross-Encoder (chính xác hơn embedding).
    3. Filter: Lọc theo ngưỡng điểm (Score Threshold).
    """

    def __init__(self, persist_path: str = "./rag_db_test"):
        """
        Khởi tạo các models.
        
        Gợi ý:
        1. Load ChromaDB Client & Collection "srs_knowledge_base".
        2. Dùng embedding model y hệt lúc Indexer: "paraphrase-multilingual-MiniLM-L12-v2".
        3. Load CrossEncoder model: "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" (nhỏ, nhanh, tốt).
        """
        """Khởi tạo với sự tối ưu hóa phần cứng."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Khởi tạo ChromaDB
        self.client = chromadb.PersistentClient(path=persist_path)
        
        # 2. Embedding Function (Bi-Encoder) - Phải khớp với Indexer
        # Senior Tip: Dùng chính xác model đã dùng để Index để đảm bảo vector space đồng nhất
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            device=self.device
        )
        
        self.collection = self.client.get_or_create_collection(
            name="srs_knowledge_base",
            embedding_function=self.embedding_fn
        )

        # 3. Load Cross-Encoder để Rerank
        self.reranker = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", 
            device=self.device
        )
        
        # 4. [NEW] Build In-Memory BM25 Index for Hybrid Search
        # Note: Với dữ liệu lớn (>100k chunks), nên dùng ElasticSearch/Solr thay vì build RAM như này.
        logger.info("--- Building BM25 Index (Hybrid Search) ---")
        try:
            all_docs = self.collection.get() # Fetch all data
            if all_docs['documents']:
                self.bm25_corpus = all_docs['documents']
                self.bm25_ids = all_docs['ids']
                self.bm25_metadatas = all_docs['metadatas']
                
                # Simple tokenization by splitting spaces (Production nên dùng ViTokenizer)
                tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]
                from rank_bm25 import BM25Okapi
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.success(f"BM25 Index built successfully with {len(self.bm25_corpus)} docs.")
            else:
                self.bm25 = None
                logger.warning("Warning: Database is empty, BM25 skipped.")
        except Exception as e:
            logger.error(f"Error building BM25: {e}")
            self.bm25 = None

    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True) -> List[Dict[str, Any]]:
        """Hàm chính: Hybrid Search (Vector + Keyword) -> Rerank."""
        
        # 1. Semantic Search (Vector)
        vector_results = self._semantic_search(query, k=top_k * 2)
        
        # 2. Keyword Search (BM25)
        keyword_results = self._keyword_search(query, k=top_k * 2)
        
        # 3. Merge Results (Reciprocal Rank Fusion - RRF)
        unified_results = self._merge_results_rrf(vector_results, keyword_results, k=top_k * 3)
        
        if not unified_results:
            return []
            
        # 4. Rerank
        if rerank:
            ranked_results = self._rerank_results(query, unified_results)
            return ranked_results[:top_k]
        
        return unified_results[:top_k]

    def _keyword_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Tìm kiếm từ khóa chính xác bằng BM25."""
        if not self.bm25:
            return []
            
        tokenized_query = query.lower().split()
        # Lấy top k docs có điểm BM25 cao nhất
        # rank_bm25 trả về list documents, ta cần map ngược lại ID và Metadata
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top n indices
        import numpy as np
        top_n_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0: # Chỉ lấy nếu có điểm
                results.append({
                    "id": self.bm25_ids[idx],
                    "content": self.bm25_corpus[idx],
                    "metadata": self.bm25_metadatas[idx] if self.bm25_metadatas else {},
                    "initial_score": float(scores[idx]), # BM25 score (không chuẩn hóa 0-1 nhưng RRF không quan tâm)
                    "search_type": "keyword"
                })
        return results

    def _merge_results_rrf(self, vector_results: List[Dict], keyword_results: List[Dict], k: int = 60) -> List[Dict]:
        """Gộp kết quả bằng thuật toán Reciprocal Rank Fusion (cân bằng cả 2)."""
        
        # Map ID -> {doc_info, rrf_score}
        fusion_scores = {}
        
        # Constant k for RRF (thường là 60)
        c = 60
        
        # Process Vector Results
        for rank, doc in enumerate(vector_results):
            doc_id = doc['id']
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {**doc, "rrf_score": 0}
            fusion_scores[doc_id]["rrf_score"] += 1 / (c + rank + 1)
            
        # Process Keyword Results
        for rank, doc in enumerate(keyword_results):
            doc_id = doc['id']
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {**doc, "rrf_score": 0}
            fusion_scores[doc_id]["rrf_score"] += 1 / (c + rank + 1)
            
        # Sort by RRF score desc
        sorted_results = sorted(fusion_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
        return sorted_results[:k]

    def _semantic_search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Tìm kiếm không gian vector và format lại dữ liệu."""
        if self.collection.count() == 0:
             return []
             
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if not results['ids'] or not results['ids'][0]:
            return []

        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "initial_score": 1 - results['distances'][0][i], # Convert distance to similarity
                "search_type": "vector"
            })
        
        return formatted_results

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sử dụng Cross-Encoder để đánh giá lại mức độ liên quan thực tế."""
        if not results:
            return []

        # Tạo cặp [Query, Passages] cho Cross-Encoder
        pairs = [[query, r['content']] for r in results]
        
        # Tính toán scores (logits)
        scores = self.reranker.predict(pairs)

        # Gán lại score và sắp xếp
        for i, score in enumerate(scores):
            results[i]['rerank_score'] = float(score)

        # Sắp xếp giảm dần theo rerank_score
        return sorted(results, key=lambda x: x['rerank_score'], reverse=True)
