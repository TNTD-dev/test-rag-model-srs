import requests
import json
import time

# Configuration
API_URL = "http://127.0.0.1:8000/generate-srs" # Mặc định FastAPI chạy port 8000. Nếu bạn chạy port khác, hãy sửa lại.
HEADERS = {"Content-Type": "application/json"}

# Test Data: Câu hỏi liên quan đến kiến thức đã Index (WMS, Quy tắc nhập kho, SKU...)
PROJECT_DESC = "Xây dựng hệ thống quản lý kho (WMS) cho ngành bán lẻ. Yêu cầu chi tiết về quy tắc đặt mã SKU và quy trình nhập kho (Inbound)."

def test_rag_generation():
    print(f"🚀 Starting E2E RAG Test...")
    print(f"Target URL: {API_URL}")
    
    payload = {
        "project_description": PROJECT_DESC,
        "use_rag": True # Quan trọng: Bật chế độ RAG
    }
    
    start_time = time.time()
    try:
        print(f"\n📤 Sending request with RAG=True...")
        print(f"Payload: {json.dumps(payload, ensure_ascii=False)}")
        
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        
        print(f"\n📥 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            srs_content = data.get('srs_content', '')
            rag_context = data.get('rag_context', '')
            duration = time.time() - start_time
            
            print(f"✅ Success! Response received in {duration:.2f}s")
            print(f"📄 RAG Context Length: {len(rag_context) if rag_context else 0} chars")
            print("\n" + "="*50)
            print("SRS OUTPUT PREVIEW (First 1000 chars):")
            print("="*50)
            print(srs_content[:1000] + "...")
            print("="*50)
            
            # Simple Assertion for RAG Evidence
            print("\n🔍 Checking for RAG Evidence (Citations)...")
            keywords = ["Source:", "DATA_Master_Data_Rules", "General Best Practice", "fifo"]
            found_keywords = [k for k in keywords if k.lower() in srs_content.lower()]
            
            if found_keywords:
                print(f"✅ Found RAG indicators: {found_keywords}")
                print("Conclusion: RAG pipeline is ACTIVE and influencing the output.")
            else:
                print("⚠️ No specific RAG citations found. Check if the retrieved context was relevant or if the model ignored instructions.")
                
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Tip: Make sure the FastAPI server is running (uv run uvicorn src.app.main:app --reload)")

if __name__ == "__main__":
    test_rag_generation()
