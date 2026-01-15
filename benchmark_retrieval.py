import json
import numpy as np
import time
from tqdm import tqdm
from src.app.services.rag_retriever import RAGRetriever
from src.app.utils.logger import logger

# Config
TESTSET_FILE = "./data/synthetic_testset.json"
TOP_K = 5

def calculate_metrics(retriever, dataset):
    hits = 0
    reciprocal_ranks = []
    total_time = 0
    
    print(f"🔍 Benchmarking Retrieval Model with {len(dataset)} queries...")
    
    for item in tqdm(dataset, desc="Retrieving"):
        query = item['question']
        target_source = item['ground_truth_source']
        
        start = time.time()
        # Retrieve results
        results = retriever.retrieve(query, top_k=TOP_K, rerank=True)
        total_time += (time.time() - start)
        
        # Check for Hit
        found = False
        rank = 0
        
        for i, doc in enumerate(results):
            # Check if source filename matches
            # Note: doc['metadata']['source'] might be full path, need to check basename logic
            retrieved_source = doc['metadata'].get('source', '')
            if target_source in retrieved_source: # Loose match to handle paths
                hits += 1
                reciprocal_ranks.append(1 / (i + 1))
                found = True
                break
        
        if not found:
            reciprocal_ranks.append(0)

    # Calculate final metrics
    hit_rate = hits / len(dataset)
    mrr = np.mean(reciprocal_ranks)
    avg_latency = total_time / len(dataset)
    
    return hit_rate, mrr, avg_latency

def main():
    # Load Testset
    try:
        with open(TESTSET_FILE, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ Dataset not found at {TESTSET_FILE}. Run generate_testset.py first!")
        return

    # Init Retriever
    # Setting persist_path same as app default
    retriever = RAGRetriever(persist_path="./rag_db_test")
    
    # Run Benchmark
    hit_rate, mrr, latency = calculate_metrics(retriever, dataset)
    
    print("\n" + "="*40)
    print("📊 RETRIEVAL BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Queries: {len(dataset)}")
    print(f"Top-K:         {TOP_K}")
    print("-" * 40)
    print(f"🎯 Hit Rate:       {hit_rate:.2%}")
    print(f"🥇 MRR:            {mrr:.4f}")
    print(f"⏱️ Avg Latency:    {latency:.4f}s")
    print("="*40)
    
    # Save results for Streamlit
    results_data = {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "avg_latency": latency,
        "total_queries": len(dataset),
        "top_k": TOP_K,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("benchmark_retrieval.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"✅ Metrics saved to 'benchmark_retrieval.json'")

    if hit_rate < 0.7:
        print("💡 Suggestion: Improve Chunking strategy or try Hybrid Search weights.")
    elif hit_rate > 0.9:
        print("🌟 Excellent Retrieval Performance!")

if __name__ == "__main__":
    main()
