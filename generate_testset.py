import os
import glob
import json
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Config
DATA_DIR = "./data/AI Knowledge Base WMS"
OUTPUT_FILE = "./data/synthetic_testset.json"
NUM_QUESTIONS_PER_FILE = 2

# Init Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

def generate_questions(text, filename):
    prompt = f"""
    Bạn là một chuyên gia tạo dữ liệu kiểm thử.
    Dựa trên văn bản sau (được trích từ file '{filename}'), hãy đặt {NUM_QUESTIONS_PER_FILE} câu hỏi cụ thể mà người dùng có thể hỏi.
    
    YÊU CẦU:
    - Câu hỏi phải liên quan trực tiếp đến nội dung văn bản.
    - Câu hỏi phải đóng vai là Business Analyst hoặc PM hỏi về nghiệp vụ.
    - Output trả về dạng JSON List thuần túy: ["Câu hỏi 1", "Câu hỏi 2", ...]
    
    VĂN BẢN:
    {text[:4000]} (cắt ngắn để vừa context)
    """
    try:
        response = model.generate_content(prompt)
        text_resp = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text_resp)
    except Exception as e:
        print(f"Error generating questions for {filename}: {e}")
        return []

def main():
    print(f"🚀 Starting Synthetic Testset Generation from {DATA_DIR}...")
    dataset = []
    
    # Get all .md files
    files = glob.glob(os.path.join(DATA_DIR, "*.md"))
    
    if not files:
        print(f"❌ No markdown files found in {DATA_DIR}")
        return

    for filepath in tqdm(files, desc="Processing files"):
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            if not content.strip():
                continue
                
            questions = generate_questions(content, filename)
            
            for q in questions:
                dataset.append({
                    "question": q,
                    "ground_truth_source": filename,
                    "ground_truth_content_snippet": content[:200]
                })
                
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ Generated {len(dataset)} pairs. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
