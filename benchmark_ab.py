import requests
import json
import pandas as pd
import time
from tabulate import tabulate

# Config
API_URL_GENERATE = "http://127.0.0.1:8000/generate-srs"
API_URL_EVALUATE = "http://127.0.0.1:8000/evaluate-srs"
HEADERS = {"Content-Type": "application/json"}

# Test Cases: Các kịch bản cần so sánh
TEST_CASES = [
    {
        "id": "TC01",
        "description": "Xây dựng hệ thống WMS cho kho bán lẻ. Yêu cầu chi tiết về quy tắc đặt mã SKU và quy trình nhập kho (PO, Receipt, Putaway)."
    },
    {
        "id": "TC02",
        "description": "Phần mềm quản lý kho dược phẩm (GSP). Yêu cầu chặt chẽ về FEFO và theo dõi lô/date."
    }
]

def call_generate(project_desc, use_rag):
    try:
        start = time.time()
        payload = {"project_description": project_desc, "use_rag": use_rag}
        resp = requests.post(API_URL_GENERATE, json=payload, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        duration = time.time() - start
        return data.get("srs_content"), data.get("rag_context"), duration
    except Exception as e:
        print(f"Error Generate (RAG={use_rag}): {e}")
        return None, None, 0

def call_evaluate(srs_content, rag_context=None):
    try:
        payload = {"srs_content": srs_content, "rag_context": rag_context}
        resp = requests.post(API_URL_EVALUATE, json=payload, headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        return data.get("evaluation_result", {})
    except Exception as e:
        print(f"Error Evaluate: {e}")
        return {}

def run_benchmark():
    print("🚀 Starting A/B Testing: RAG vs. Vanilla Mode...\n")
    results = []

    for case in TEST_CASES:
        desc = case["description"]
        cid = case["id"]
        print(f"🧪 Processing Case [{cid}]: {desc[:50]}...")

        # 1. Run RAG
        print(f"   ► Running RAG Mode...")
        srs_rag, ctx_rag, time_rag = call_generate(desc, True)
        eval_rag = call_evaluate(srs_rag, ctx_rag)
        
        # 2. Run Vanilla (No RAG)
        print(f"   ► Running Vanilla Mode...")
        srs_van, _, time_van = call_generate(desc, False)
        eval_van = call_evaluate(srs_van, None)

        # Helper to extract raw scores
        def get_raw(data, key): return data.get("score", {}).get(key, {}).get("raw", 0)
        
        # Calculate Quality Score (Normalized on 4 shared criteria: Completeness, Consistency, Accuracy, Format)
        # Weights normalized: 0.25+0.2+0.2+0.15 = 0.8
        # Quality = (Comp*0.25 + Cons*0.2 + Acc*0.2 + Fmt*0.15) / 0.8
        def calc_quality(data):
            score_obj = data.get("score", {})
            raw_sum = (
                score_obj.get("completeness", {}).get("weighted", 0) + 
                score_obj.get("consistency", {}).get("weighted", 0) +
                score_obj.get("accuracy", {}).get("weighted", 0) +
                score_obj.get("format_tone", {}).get("weighted", 0)
            )
            return raw_sum / 0.8

        qual_rag = calc_quality(eval_rag)
        qual_van = calc_quality(eval_van)
        total_rag = eval_rag.get("score", {}).get("total_weighted_score", 0)
        total_van = eval_van.get("score", {}).get("total_weighted_score", 0)

        # 3. Log Result
        print(f"   🏁 Result: RAG (Quality={qual_rag:.2f}, Total={total_rag:.2f}) | Vanilla (Quality={qual_van:.2f}, Total={total_van:.2f})")
        print("-" * 60)

        results.append({
            "Case ID": cid,
            "Mode": "RAG",
            "Time (s)": round(time_rag, 2),
            "Total Score": total_rag,
            "Quality Score (Shared)": round(qual_rag, 2),
            "Faithfulness": get_raw(eval_rag, "faithfulness"),
            "Accuracy": get_raw(eval_rag, "accuracy"),
            "Completeness": get_raw(eval_rag, "completeness")
        })
        results.append({
            "Case ID": cid,
            "Mode": "Vanilla",
            "Time (s)": round(time_van, 2),
            "Total Score": total_van,
            "Quality Score (Shared)": round(qual_van, 2),
            "Faithfulness": "N/A (Def 8.0)",
            "Accuracy": get_raw(eval_van, "accuracy"),
            "Completeness": get_raw(eval_van, "completeness")
        })

    # Output Summary
    df = pd.DataFrame(results)
    print("\n📊 BENCHMARK SUMMARY (Quality Score = Appples-to-Apples comparison excluding Faithfulness):")
    print(tabulate(df, headers="keys", tablefmt="grid"))
    
    # Save to CSV
    df.to_csv("benchmark_results.csv", index=False)
    print("\n✅ Results saved to 'benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark()
