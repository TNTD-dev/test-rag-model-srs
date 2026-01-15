import os
from openai import OpenAI
from dotenv import load_dotenv
from src.app.services.rag_retriever import RAGRetriever
from src.app.utils.logger import logger

load_dotenv()

# Prompt tạo sinh SRS gốc 
SRS_SYSTEM_PROMPT = """
Bạn là một Senior Business Analyst và Technical Architect với 10 năm kinh nghiệm.
Nhiệm vụ: Viết tài liệu Đặc tả yêu cầu phần mềm (SRS) dựa trên cấu trúc chuẩn chuyên nghiệp.

Cấu trúc bắt buộc tuân thủ:
1. Giới thiệu: Mục đích, Phạm vi (Table), Thuật ngữ.
2. Mô tả tổng quan: Sơ đồ kiến trúc (Mermaid), Nhóm người dùng (Table).
3. Khung Workflow Framework (Tái sử dụng): 
   - Định nghĩa các Engine (Approval, Notification, Automation, Scheduling).
   - Các Workflow Patterns (Sequential, Parallel, Matrix...).
   - Database Schema mẫu cho Framework.
4. Các Module chức năng: 
   - Mỗi module phải có: Business Flow (Mermaid), Activity Diagram, DB Schema, và Business Rules (đánh mã XXX-BR-01).
5. Yêu cầu phi chức năng: Hiệu năng, Bảo mật, Độ tin cậy.

Văn phong: Chuyên nghiệp, dùng đúng thuật ngữ chuyên ngành IT/BA. Tránh bịa đặt (Anti-Hallucination).
"""

class SRSGenerator:
    def __init__(self):
        # Init LLM Client
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.getenv("HF_TOKEN"),
        )
        self.model = "Qwen/Qwen3-Coder-30B-A3B-Instruct:nebius"
        
        # [NEW] Init RAG Retriever
        # Gợi ý: Khởi tạo instance của RAGRetriever tại đây
        self.retriever = RAGRetriever()

    def generate_srs(self, project_description: str, use_rag: bool = True) -> tuple[str, str]:
        """
        Hàm tạo SRS chính với logic Plan-then-Generate.
        Args:
            project_description: Input của user.
            use_rag: Có bật RAG hay không.
        """
        import time
        start_time = time.time()
        logger.info(f"[SRS Generator] Start: {project_description} | RAG Mode: {use_rag}")
        
        context_str = ""
        context_sources = []
        
        if use_rag:
            try:
                logger.debug("[Retrieval] Searching Knowledge Base...")
                # 1. Retrieve Context (Hybrid Search + Rerank)
                retrieved_docs = self.retriever.retrieve(project_description, top_k=5, rerank=True)
                
                if retrieved_docs:
                    logger.success(f"[Retrieval] Found {len(retrieved_docs)} documents.")
                    
                    # 2. Build Context String with Metadata
                    context_parts = []
                    for i, doc in enumerate(retrieved_docs):
                        source_name = doc['metadata'].get('source', 'Unknown')
                        score = doc.get('rerank_score', doc.get('initial_score', 0))
                        
                        # Format context block
                        doc_block = f"""
                        [DOCUMENT {i+1}]
                        - Source: {source_name}
                        - Relevance Score: {score:.4f}
                        - Content:
                        {doc['content']}
                        """
                        context_parts.append(doc_block)
                        context_sources.append(source_name)
                    
                    context_str = "\n".join(context_parts)
                else:
                    logger.warning("[Retrieval] No relevant documents found. Switching to General Knowledge mode.")
            except Exception as e:
                logger.error(f"[Retrieval Error] {e}. Proceeding without RAG.")
                use_rag = False

        # 3. Construct Final Prompt Strategy: "Evidence-Based Generation"
        if context_str:
            system_prompt = f"""
            {SRS_SYSTEM_PROMPT}

            --------------------------------------------------
            KB CONTEXT (TÀI LIỆU NỘI BỘ ĐÃ ĐƯỢC XÁC THỰC):
            {context_str}
            --------------------------------------------------

            CHỈ DẪN QUAN TRỌNG (RAG RULES):
            1. ƯU TIÊN TUYỆT ĐỐI thông tin trong [KB CONTEXT] để định nghĩa Business Rules, Workflow và Logic.
            2. TRÍCH DẪN NGUỒN (Citation) ngay sau mỗi khẳng định quan trọng. 
               Ví dụ: "Quy tắc nhập kho tuân theo FIFO [Source: DATA_Master_Data_Rules.md]".
            3. Nếu Context không đủ thông tin cho một mục nào đó (ví dụ "Bảo mật"), hãy dùng kiến thức chuyên gia của bạn để điền vào nhưng đánh dấu bằng [General Best Practice].
            4. KHÔNG được bịa đặt thông tin mâu thuẫn với Context.
            """
        else:
            system_prompt = SRS_SYSTEM_PROMPT

        # 4. Call LLM
        logger.info(f"[Generation] Sending prompt to LLM (Context Len: {len(context_str)} chars)...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Yêu cầu dự án: {project_description}\n\nHãy viết bản SRS chi tiết ngay bây giờ."}
                ],
                temperature=0.2, # Low temp for factual accuracy
                max_tokens=4096  # Ensure enough space for full SRS
            )
            
            result = response.choices[0].message.content
            print(f"[Generation] Done in {time.time() - start_time:.2f}s")
            return result, context_str
            
        except Exception as e:
            print(f"[Generation Error] {e}")
            raise e
