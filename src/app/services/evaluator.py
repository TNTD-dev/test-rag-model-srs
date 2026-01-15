import os
import google.generativeai as genai
from dotenv import load_dotenv
from src.app.utils.logger import logger

load_dotenv()


EVALUATION_PROMPT_TEMPLATE = """
Bạn là một "Giám khảo AI" chuyên nghiệp. Hãy đánh giá bản SRS sau đây dựa trên các tiêu chí nghiêm ngặt.

BỐI CẢNH (RAG CONTEXT - TÀI LIỆU GỐC):
{rag_context}
(Lưu ý: Nếu Bối cảnh trống, hãy bỏ qua tiêu chí "Độ trung thực" và đánh giá dựa trên tiêu chuẩn chung).

### Bộ tiêu chí chấm điểm (Scale 1-10):
1. Tính đầy đủ (Completeness - Trọng số 25%): Có đủ các mục Giới thiệu, Workflow, Module, DB Schema, Phi chức năng không?
2. Tính nhất quán (Consistency - Trọng số 20%): Logic giữa các phần có mâu thuẫn không?
3. Độ chính xác nghiệp vụ (Accuracy - Trọng số 20%): Thuật ngữ và quy trình có chuyên nghiệp, thực tế không?
4. Định dạng & Văn phong (Format - Trọng số 15%): Format Markdown, Mermaid có chuẩn không?
5. Độ trung thực (Faithfulness - Trọng số 20%): [QUAN TRỌNG] Bài viết có tuân thủ thông tin trong phần "BỐI CẢNH" không? 
   - Nếu có bịa đặt thông tin trái ngược với Bối cảnh -> Điểm thấp (<5).
   - Nếu trích dẫn chuẩn xác -> Điểm cao.
   - Nếu không có Bối cảnh -> Đặt mặc định 8.0 (N/A).

### Phân loại (group) mức điểm đánh giá:
Dựa trên tổng điểm trọng số sẽ phân loại mô hình để đưa ra quyết định triển khai:
* **9 - 10 điểm (Xuất sắc):** Bản SRS hoàn hảo, có thể gửi trực tiếp cho khách hàng hoặc chuyển xuống đội ngũ lập trình.
* **7 - 8 điểm (Tốt):** Cấu trúc tốt, thông tin chính xác nhưng cần con người chỉnh sửa nhẹ về văn phong hoặc bổ sung các chi tiết nhỏ.
* **5 - 6 điểm (Trung bình):** Đủ ý chính nhưng logic còn lỏng lẻo, định dạng chưa chuẩn.
* **Dưới 5 điểm (Kém):** AI bị hiện tượng "bịa đặt" (Hallucination) hoặc sai lệch quy trình nghiệp vụ nghiêm trọng. Phải kiểm tra lại chất lượng dữ liệu đầu vào.

### JSON OUTPUT SCHEMA (BẮT BUỘC):
{{
  "score": {{
    "completeness": {{"raw": 0, "weight": 0.25, "weighted": 0.0}},
    "consistency": {{"raw": 0, "weight": 0.2, "weighted": 0.0}},
    "accuracy": {{"raw": 0, "weight": 0.2, "weighted": 0.0}},
    "format_tone": {{"raw": 0, "weight": 0.15, "weighted": 0.0}},
    "faithfulness": {{"raw": 0, "weight": 0.2, "weighted": 0.0}},
    "total_weighted_score": 0.0
  }},
  "group": "",
  "comment": {{
    "summary": "",
    "strengths": [],
    "issues": [
      {{
        "criterion": "completeness|consistency|accuracy|format_tone|faithfulness",
        "problem": "Mô tả ngắn gọn vấn đề",
        "evidence_quote": "Trích dẫn đoạn văn bản gặp lỗi (nếu có)",
        "impact": "Ảnh hưởng (Nghiêm trọng/Vừa/Nhẹ)",
        "recommendation": "Đề xuất sửa đổi"
      }}
    ],
    "quick_fixes": []
  }}
}}

--- 
BẢN SRS CẦN ĐÁNH GIÁ:
{srs_content}
"""

class Evaluator:
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    def evaluate_srs(self, srs_content: str, rag_context: str = None) -> dict:
        logger.info(f"[Evaluator] Assessing SRS (Context Provided: {bool(rag_context)})...")
        try:
            # Handle empty context gracefully
            context_display = rag_context if rag_context else "Không có ngữ cảnh (Đánh giá ở chế độ General Knowledge)"
            
            full_eval_prompt = EVALUATION_PROMPT_TEMPLATE.format(
                srs_content=srs_content,
                rag_context=context_display
            )
            response = self.model.generate_content(full_eval_prompt)
            
            # Clean and Parse JSON
            cleaned_response = response.text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            import json
            result = json.loads(cleaned_response)
            logger.success(f"[Evaluator] Score: {result.get('score', {}).get('total_weighted_score', 'N/A')}")
            return result
        except Exception as e:
            logger.error(f"[Evaluator Error] {e}")
            raise e
